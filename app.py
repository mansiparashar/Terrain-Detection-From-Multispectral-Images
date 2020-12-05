# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from gevent.pywsgi import WSGIServer

# deep learning utilities
import numpy as np
from util import base64_to_pil
import keras
from keras import models
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *

# Declare a flask app
app = Flask(__name__)

#Declaring the classes
categories=['AnnualCrop',
 'Forest',
 'HerbaceousVegetation',
 'Highway',
 'Industrial',
 'Pasture',
 'PermanentCrop',
 'Residential',
 'River',
 'SeaLake',
]

# Importing the pretrained model. You can use model.json to load model architecture, rather than declaring it once again
vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(64,64,3))
model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
NUM_TRAINABLE_LAYERS = 10
for layer in model.layers[:-NUM_TRAINABLE_LAYERS]:
    layer.trainable = False
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))


# Model saved with Keras model.save()
MODEL_PATH = 'D:\Downloads\cbirProjectModel.h5'

# Load your own trained model
# model = load_model(MODEL_PATH)
model.load_weights(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving at http://127.0.0.1:5000/')


#routes

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")
        
        #pre-processing
        img = image.load_img("./uploads/image.png", target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        
        # Make prediction
        classe = model.predict_classes(images)
        #print(str(classe[0]))
        
        # Serialize the result, you can add additional fields
        return jsonify(result=categories[classe[0]])

        

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
