from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from scripts.inference import Evaluator
from datetime import datetime
import logging
from db import write_output_to_db
from aws_functions import get_xray_image
MODEL_VERSION = 'v0.1'
covid_model = Evaluator("model/COVID-Netv1/")
sess, x, op_to_restore = covid_model.export()
app = Flask(__name__)


@app.route("/test", methods=["GET"])
def main():
    app.logger.info("test endpoint hit")
    return jsonify({'reuslt':'Test','current_time':datetime.now()})


@app.route("/predict", methods=['GET'])
def predict():
    # a=time.time()
    app.logger.info("predict endpoint hit")
    node_env = request.headers.get('node-env')
    if node_env == 'dev':
        return jsonify({'result': 'COVID-19 Viral','dummy':True})
    elif node_env == 'prod':
        image_loc = request.args.get('image_loc')
        model=request.args.get('model')
        # img_resp = requests.get(image_loc, stream=True).raw
        img_resp=get_xray_image(image_loc)
        image = np.asarray(bytearray(img_resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        resp = covid_model.predict(image, sess, x, op_to_restore)

        # resp = model.evaluate(image)
        write_output_to_db({'img_url': image_loc, 'model_version': MODEL_VERSION, 'model_output': resp})
        # b = time.time()
        # print(b - a)
        return jsonify({'result': resp})
    else:
        return jsonify({'result':'invalid node-env','node_env':node_env})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
