from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from scripts.inference import Evaluator
# import time

from db import write_output_to_db
MODEL_VERSION='v0.1'
model = Evaluator("model/COVID-Netv1/")
app = Flask(__name__)


@app.route("/predict", methods=['GET'])
def predict():
    image_loc = request.args.get('image_loc')
    img_resp = requests.get(image_loc, stream=True).raw
    image = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    resp = model.evaluate(image)
    write_output_to_db({'img_url':image_loc,'model_version':MODEL_VERSION,'model_output':resp})
    return jsonify({'result': resp})


if __name__ == "__main__":
    app.run(debug=True)
