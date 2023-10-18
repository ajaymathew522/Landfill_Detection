import json
import tensorflow as tf

from flask import Flask, flash, redirect,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
from pystac.extensions.eo import EOExtension as eo
import pystac_client
import planetary_computer
import rasterio
from rasterio import windows
from rasterio import features
from rasterio import warp
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from io import BytesIO

ALLOWED_EXTENSIONS = {'json','geojson'}

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

time_of_interest = "2022-01-01/2022-12-30"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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



app=Flask(__name__)
## Load the model
loaded_model = tf.keras.models.load_model('CNN_model.keras')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    json_data=request.json['features']
    gdf = gpd.GeoDataFrame.from_features(json_data)
    img_data =  [getimgdata(row['geometry'], time_of_interest) for _, row in gdf.iterrows()]
    resized_data = resize_images(img_data, target_size=(128, 128))
    images_test = [pair[0] for pair in resized_data]
    images_test = np.array(images_test)
    images_test = images_test / 255.0
    predictions = loaded_model.predict(images_test)
    binary_predictions = (predictions > 0.5).astype(int)
    print(binary_predictions)
    return jsonify(binary_predictions.tolist())


@app.route('/predict',methods=['POST'])
def predict():
    # Check if a file is uploaded in the request
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # Check if the user submitted an empty part
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # Check if the file has a valid extension
    if file and allowed_file(file.filename):
        # Process the uploaded file without saving it
        file_contents = file.read()
        file_bytesio = BytesIO(file_contents)
        gdf = gpd.read_file(file_bytesio)
        img_data =  [getimgdata(row['geometry'], time_of_interest) for _, row in gdf.iterrows()]
        resized_data = resize_images(img_data, target_size=(128, 128))
        images_test = [pair[0] for pair in resized_data]
        images_test = np.array(images_test)
        images_test = images_test / 255.0
        predictions = loaded_model.predict(images_test)
        binary_predictions = (predictions > 0.5).astype(int)
        predictions_text = ["Landfill Presence Found" if i == 1 else "Landfill Presence Not Found" for i in binary_predictions]
        cols = gdf.columns

        if "label" in gdf.columns:
            labels = gdf['label']
            test_loss, test_accuracy = loaded_model.evaluate(images_test, labels)
            print("Test Accuracy:", test_accuracy)
            # Calculate precision and recall
            precision = precision_score(labels, binary_predictions)
            recall = recall_score(labels, binary_predictions)
            # Calculate ROC-AUC score
            roc_auc = roc_auc_score(labels, predictions)
            results = [test_accuracy, precision, recall, roc_auc]
            return render_template('testing.html', results=results)
        else:
            results = [prediction for prediction in predictions_text]
            return render_template("prediction.html", results=results)

    else:
        flash('Invalid file extension')
        return redirect(request.url)
    



if __name__=="__main__":
    app.run(debug=True)
   
     