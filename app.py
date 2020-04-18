from flask import Flask, request, jsonify, redirect
import numpy as np
from scripts.inference import CovidEvaluator, ChesterAiEvaluator, UNetCTEvaluator
from datetime import datetime
import logging
from db import write_output_to_db
from aws_functions import get_xray_image, upload_to_s3
import json
import uuid

MODEL_VERSION = 'v0.1'

covid_model = CovidEvaluator("model/COVID-Netv1/")
sess_covid, x_covid, op_to_restore_covid = covid_model.export()
chester_ai_model = ChesterAiEvaluator("model/ChesterAI/")
sess_chester, x_chester, op_to_restore_chester = chester_ai_model.export()
unet_ct_model = UNetCTEvaluator("model/U-Net-CT/")
sess_unet, x_unet, op_to_restore_unet = unet_ct_model.export()
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return redirect("/test")


@app.route("/test", methods=["GET"])
def test():
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
        model = request.args.get('model_type','xray')  # So that FrontEnd doesnt break
        patient_info = request.args.get('patientInfo')
        user_id = request.args.get('user_id', 100)
        # img_resp = requests.get(image_loc, stream=True).raw
        img_resp = get_xray_image(image_loc)
        image = np.asarray(bytearray(img_resp.read()), dtype="uint8")
        app.logger.info(image.shape)
        if model == 'xray':
            # app.logger.info(image_np.shape)
            # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            covid_resp = covid_model.predict(image, sess_covid, x_covid, op_to_restore_covid)
            app.logger.info(covid_resp)

            chester_resp = chester_ai_model.predict(image, sess_chester, x_chester, op_to_restore_chester)
            # Added result_boolean
            chester_resp = chester_ai_model.modify_prediction_dict(chester_resp)
            app.logger.info(chester_resp)
            write_output_to_db({'img_url': image_loc, 'model_version': 'xray_' + MODEL_VERSION,
                                'model_output': json.dumps({'covid': covid_resp, 'chest': chester_resp}),
                                'patient_info': patient_info, 'user_id': user_id})

            return jsonify({'result': {'covid': covid_resp, 'chest': chester_resp}})

        if model == 'ct':
            unet_resp = unet_ct_model.predict(image, sess_unet, x_unet, op_to_restore_unet)
            filename = f"{user_id}_{uuid.uuid4()}"
            image_url = upload_to_s3(unet_resp, filename)
            write_output_to_db({'img_url': image_loc, 'model_version': 'ct_' + MODEL_VERSION,
                                'model_output': json.dumps({'image_url': image_url}),
                                'patient_info': patient_info, 'user_id': user_id})

            return jsonify({'result': {'image_url': image_url}})

    else:
        return jsonify({'result': 'invalid node-env', 'node_env': node_env})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
