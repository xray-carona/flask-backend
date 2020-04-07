from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from scripts.inference import CovidEvaluator, ChesterAiEvaluator
from datetime import datetime
import logging
from db import write_output_to_db
from aws_functions import get_xray_image

MODEL_VERSION = 'v0.1'
covid_model = CovidEvaluator("model/COVID-Netv1/")
sess, x, op_to_restore = covid_model.export()
chester_ai=ChesterAiEvaluator("model/ChesterAI/")
chester_ai_model=chester_ai.export()
app = Flask(__name__)


@app.route("/test", methods=["GET"])
def main():
    app.logger.info("test endpoint hit")
    return jsonify({'reuslt': 'Test', 'current_time': datetime.now()})


@app.route("/predict", methods=['GET'])
def predict():
    # a=time.time()
    app.logger.info("predict endpoint hit")
    node_env = request.headers.get('node-env')
    if node_env == 'dev':
        return jsonify({'result': 'COVID-19 Viral', 'dummy': True})
    elif node_env == 'prod':
        image_loc = request.args.get('image_loc')
        model = request.args.get('model')
        # img_resp = requests.get(image_loc, stream=True).raw
        img_resp = get_xray_image(image_loc)
        image = np.asarray(bytearray(img_resp.read()), dtype="uint8")
        # app.logger.info(image_np.shape)
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        app.logger.info(image.shape)
        covid_resp = covid_model.predict(image, sess, x, op_to_restore)
        app.logger.info(covid_resp)
        image=chester_ai.preprocess(image)
        # chester_resp=chester_ai_model.predict(image)
        chester_resp = chester_ai.predict_chest_conditions(chester_ai_model,image)
        app.logger.info(chester_resp)
        # resp = model.evaluate(image)
        # write_output_to_db({'img_url': image_loc, 'model_version': MODEL_VERSION, 'model_output': covid_resp})
        # b = time.time()
        # print(b - a)
        return jsonify({'result': {'covid':covid_resp,'chest':chester_resp}})
    else:
        return jsonify({'result': 'invalid node-env', 'node_env': node_env})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
