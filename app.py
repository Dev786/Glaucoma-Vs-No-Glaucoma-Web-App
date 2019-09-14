from flask import Flask
from flask import render_template
from flask import request, send_from_directory
import json
import io
import cv2
import numpy as np
import base64
from flask import jsonify
from PIL import Image
from model.reusable_model.predicter import getPrediction
import os

export_path = os.getcwd()+"/model/cnn/saved_model"
app = Flask(__name__)


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert('RGB')
    image = image.resize(target_size)
    return image


@app.route('/', methods=["GET", "POST"])
def homepage():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def uploadImage():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    image = preprocess_image(image, (128, 128))
    nparray = np.reshape(np.array(image.getdata()), (1, 128, 128, 3))
    prediction = np.argmax(getPrediction(nparray, export_path)[0])
    print(prediction)
    response = dict({
        "glaucoma": "Yes" if prediction == 1 else 'No',
        "noGlaucoma": "Yes" if prediction == 0 else 'No'
    })
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
