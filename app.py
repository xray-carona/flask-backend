from flask import Flask,request,jsonify
import requests
import numpy as np
import cv2
from scripts.inference import Evaluator
from db import write_output_to_db
model=Evaluator("model/COVID-Netv1/")
app=Flask(__name__)
@app.route("/predict",methods=['GET'])
def predict():
    image_loc=request.args['image_loc']
    img_resp=requests.get(image_loc,stream=True).raw
    image = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    resp=model.evaluate(image)
    return jsonify(resp)

if __name__=="__main__":
    app.run(debug=True)