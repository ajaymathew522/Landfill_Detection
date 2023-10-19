#Importing required libraries
import geopandas as gpd
from pystac.extensions.eo import EOExtension as eo
import pystac_client
import planetary_computer
import rasterio
from rasterio import windows
from rasterio import features
from rasterio import warp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Read the GeoJSON training file
gdf_train = gpd.read_file('data/train.geojson')
gdf_test = gpd.read_file('data/test.geojson')
# Display the first few rows of the GeoDataFrame
print(gdf_train.shape)
print(gdf_test.shape)
# Initialising the pystac library for accessing Plantery Computer's Blob storage containers 

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


# Function to pull satellite image data
def getimgdata(area_of_interest, time_of_interest):
    search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=area_of_interest,
    datetime=time_of_interest,
    query={"eo:cloud_cover": {"lt": 10}},
    )

    items = search.item_collection()
    

    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

    asset_href = least_cloudy_item.assets["visual"].href
    
    with rasterio.open(asset_href) as ds:
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        img_data = ds.read(window=aoi_window)
    
    return img_data.transpose(1, 2, 0) 

# Function to get Indexes data
def getIndexesdata(area_of_interest, time_of_interest):
    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=area_of_interest,
    datetime=time_of_interest,
    query={"eo:cloud_cover": {"lt": 10}},
    )

    items = search.item_collection()
    item = min(items, key=lambda item: eo.ext(item).cloud_cover)


    red_band = item.get_assets()["B04"]  
    nir_band = item.get_assets()["B08"] 

    L = 0.5
    m = 0.5
    with rasterio.open(red_band.href) as red_src, rasterio.open(nir_band.href) as nir_src:
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", red_src.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(transform=red_src.transform, *warped_aoi_bounds)    
        red_data = red_src.read(window=aoi_window).astype(float)
        nir_data = nir_src.read(window=aoi_window).astype(float)


        ndvi = (nir_data - red_data) / (nir_data + red_data)
        savi = ((nir_data-red_data)/(nir_data + red_data + L))* (1+L)
        msavi = 2 * nir_data + 1 - np.sqrt((2 * nir_data + 1) ** 2 - 8 * (nir_data - red_data)) / 2
    return [ndvi[0], savi[0], msavi[0]]

# Setting the time of interest
time_of_interest = "2022-01-01/2022-12-30"

# Gathering satellite images for training data
img_data_train = [getimgdata(row['geometry'], time_of_interest) for _, row in gdf_train.iterrows()]
img_data_test =  [getimgdata(row['geometry'], time_of_interest) for _, row in gdf_test.iterrows()]


# Gathering NVDI, SAVI and MSAVI indexes for the images
#index_data_train = [getIndexesdata(row['geometry'], time_of_interest)  for _, row in gdf_train.iterrows()]
#index_data_test = [getIndexesdata(row['geometry'], time_of_interest) for _, row in gdf_test.iterrows()]

def resize_images(img_data, target_size=(128, 128)):
    """
    Resize a list of images to the target size.

    Args:
        img_data (list): A list of image-label pairs, where each pair is [image, label].
        target_size (tuple): The target size for the images in the format (width, height).

    Returns:
        list: A list of resized images.
    """
    resized_img_data = []

    for img in img_data:
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(target_size)
        resized_img = np.array(pil_image)
        resized_img_data.append([resized_img])

    return resized_img_data

resized_data_train = resize_images(img_data_train, target_size=(128, 128))
resized_data_test = resize_images(img_data_test, target_size=(128, 128))

# CNN model 
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),


    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])  


model.compile(optimizer='adam',
              loss='binary_crossentropy',  
              metrics=['accuracy'])



images_train = [pair[0] for pair in resized_data_train]
labels_train = [row['label'] for _, row in gdf_train.iterrows()]


images_train = np.array(images_train)   #Converting to numpy array
labels_train = np.array(labels_train)


images_train = images_train / 255.0     #Normalising the data to fit between 0 and 1


model.fit(images_train, labels_train, epochs=10, batch_size=32)



images_test = [pair[0] for pair in resized_data_test]
labels_test = [row['label'] for _, row in gdf_test.iterrows()]


images_test = np.array(images_test)
labels_test = np.array(labels_test)


images_test = images_test / 255.0


test_loss, test_accuracy = model.evaluate(images_test, labels_test)
print("Test Accuracy:", test_accuracy)

# Testing performance of model using test data
predictions = model.predict(images_test)

# Convert predictions to binary labels (0 or 1)
binary_predictions = (predictions > 0.5).astype(int)

# Calculate precision and recall
precision = precision_score(labels_test, binary_predictions)
recall = recall_score(labels_test, binary_predictions)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(labels_test, predictions)

# Calculate ROC-AUC score
roc_auc = roc_auc_score(labels_test, predictions)


print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

model.save('CNN_model.keras')
