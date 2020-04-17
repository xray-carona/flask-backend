from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from scripts.inference import CovidEvaluator, ChesterAiEvaluator,UNetCTEvaluator
from datetime import datetime
import logging
from db import write_output_to_db
from aws_functions import get_xray_image,upload_to_s3
import json
MODEL_VERSION = 'v0.1'

# covid_model = CovidEvaluator("model/COVID-Netv1/")
# sess_covid, x_covid, op_to_restore_covid = covid_model.export()
chester_ai_model=ChesterAiEvaluator("model/ChesterAI/")
sess_chester,x_chester,op_to_restore_chester=chester_ai_model.export()
unet_ct_model=UNetCTEvaluator("model/U-Net-CT/")
sess_unet,x_unet,op_to_restore_unet=unet_ct_model.export()
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
        patient_info = request.args.get('patientInfo')
        # img_resp = requests.get(image_loc, stream=True).raw
        img_resp = get_xray_image(image_loc)
        image = np.asarray(bytearray(img_resp.read()), dtype="uint8")
        # app.logger.info(image_np.shape)
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        app.logger.info(image.shape)
        covid_resp = covid_model.predict(image, sess_covid, x_covid, op_to_restore_covid)
        app.logger.info(covid_resp)

        chester_resp = chester_ai_model.predict(image,sess_chester,x_chester,op_to_restore_chester)
        #Added result_boolean
        chester_resp=chester_ai_model.modify_prediction_dict(chester_resp)
        app.logger.info(chester_resp)
        # unet_resp=unet_ct_model.predict(image,sess_unet,x_unet,op_to_restore_unet)
        # upload_to_s3(unet_resp)


        write_output_to_db({'img_url': image_loc, 'model_version': MODEL_VERSION,
                            'model_output': json.dumps({'covid': covid_resp, 'chest': chester_resp}),
                            'patient_info': patient_info})
        # b = time.time()
        # print(b - a)
        return jsonify({'result': {'covid':covid_resp,'chest':chester_resp}})
    else:
        return jsonify({'result': 'invalid node-env', 'node_env': node_env})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
