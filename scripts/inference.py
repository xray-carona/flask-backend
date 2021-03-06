import cv2
import numpy as np

# If tensorflow 2.0 is installed
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# If tensorflow 1.0 is installed
import tensorflow as tf
from tensorflow.io import gfile
from config import CHEST_CONDITION_THRESHOLD, CHEST_CONDITION_NO_FINDINGS


def image_hash(image, hashSize=8):
    #  dhash pyimagesearch
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


class CovidEvaluator(object):
    def __init__(self, model_dir):
        self.labels = ["Normal", "Non-COVID19 Viral", "COVID-19 Viral"]
        self.INPUT_SIZE = (224, 224)
        self.MODEL_GRAPH = model_dir + "model.meta_eval"
        self.MODEL = model_dir + "model-2069"

    def load_img(self, img):
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return image  # Keeping image in memory

    def pre_process(self, img):
        img_arr = cv2.resize(img, self.INPUT_SIZE)  # resize
        img_arr = img_arr.astype('float32') / 255.0
        img_arr = img_arr.reshape(1, self.INPUT_SIZE[0], self.INPUT_SIZE[1], 3)
        return img_arr

    def export(self):
        ses = tf.Session()
        saver = tf.train.import_meta_graph(self.MODEL_GRAPH)
        saver.restore(ses, self.MODEL)
        graph = tf.get_default_graph()
        # print('COVID')
        # print(graph.get_operations())
        x = graph.get_tensor_by_name("input_1:0")
        op_to_restore = graph.get_tensor_by_name("dense_3/Softmax:0")
        return ses, x, op_to_restore

    def evaluate(self, img):
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.MODEL_GRAPH)
            saver.restore(sess, self.MODEL)
            graph = tf.get_default_graph()

            # Get tensor names
            x = graph.get_tensor_by_name("input_1:0")
            op_to_restore = graph.get_tensor_by_name("dense_3/Softmax:0")

            # Preprocess image input
            img = self.load_img(img)
            processed_img = self.pre_process(img)
            feed_dict = {x: processed_img}
            result_index = sess.run(op_to_restore, feed_dict)
            return self.labels[np.argmax(result_index)]
        sess.close()

    def predict(self, img, sess, x, op_to_restore):
        img = self.load_img(img)
        processed_img = self.pre_process(img)
        feed_dict = {x: processed_img}
        result_index = sess.run(op_to_restore, feed_dict)
        return self.labels[np.argmax(result_index)]


class ChesterAiEvaluator(object):
    def __init__(self, model_dir):
        self.INPUT_SIZE = (224, 224)
        self.model = model_dir + "14_label_model.h5"
        self.model_tf = model_dir + "chester.pb"
        self.labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                       'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
                       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    def export(self):
        ses = tf.Session()
        f = gfile.GFile(self.model_tf, 'rb')
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        # ses.graph.as_default()
        tf.import_graph_def(graph_def, name="chester")
        # print('ChestAi')
        # print(ses.graph.get_operations())
        tensor_output = ses.graph.get_tensor_by_name('chester/dense_1/Sigmoid:0')
        tensor_input = ses.graph.get_tensor_by_name('chester/conv2d_1_input:0')
        return ses, tensor_input, tensor_output

    def predict(self, img, sess, in_to_restore, op_to_restore):
        # img = self.load_img(img)
        processed_img = self.preprocess(img)
        predictions = sess.run(op_to_restore, {in_to_restore: processed_img})
        pred = [i * 100 for i in predictions[0]]
        prediction = dict(zip(self.labels, pred))
        return prediction

    def export_keras(self):

        exported_model = tf.keras.models.load_model(self.model)
        print(exported_model.summary())

        return exported_model

    def evaluate(self, img=None):
        from tensorflow.io import gfile
        with tf.Session() as sess:
            # load model from pb file
            with gfile.GFile(self.model_tf, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                g_in = tf.import_graph_def(graph_def)
            # write to tensorboard (check tensorboard for each op names)
            # writer = tf.summary.FileWriter(wkdir + '/log/')
            # writer.add_graph(sess.graph)
            # writer.flush()
            # writer.close()
            # print all operation names
            print('\n===== ouptut operation names =====\n')
            ops = sess.graph.get_operations()
            print(sess.graph.get_operations())
            # for op in ops:
            # print(op)
            # for op in sess.graph.get_operations():
            #     print(op)
            # inference by the model (op name must comes with :0 to specify the index of its output)
            tensor_output = sess.graph.get_tensor_by_name('import/dense_1/Sigmoid:0')
            tensor_input = sess.graph.get_tensor_by_name('import/conv2d_1_input:0')
            # print(tensor_input,tensor_input)
            predictions = sess.run(tensor_output, {tensor_input: img})
            pred = [i * 100 for i in predictions[0]]
            prediction = dict(zip(self.labels, pred))
            print('\n===== output predicted results =====\n')
            print(prediction)

    def preprocess(self, img):
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.INPUT_SIZE)
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img

    def predict_chest_conditions(self, exported_model, img):
        img = self.preprocess(img)
        pred = exported_model.predict(img)[0]
        pred = [i * 100 for i in pred]
        prediction = dict(zip(self.labels, pred))
        return prediction

    def modify_prediction_dict(self, prediction):
        result = []
        nofindings_bool = True if prediction['No Finding'] >= CHEST_CONDITION_NO_FINDINGS else False
        for k, v in prediction.items():
            temp = {k: v, 'user_agree_result': None}
            if nofindings_bool:
                if k == 'No Finding':
                    temp.update({'result_boolean': nofindings_bool})
                else:
                    temp.update({'result_boolean': False})
            elif not nofindings_bool:
                if k == 'No Finding':
                    temp.update({'result_boolean': nofindings_bool})
                elif v >= CHEST_CONDITION_THRESHOLD:
                    temp.update({'result_boolean': True})
                elif v < CHEST_CONDITION_THRESHOLD:
                    temp.update({'result_boolean': False})
            result.append(temp)
        return result


class UNetCTEvaluator:
    def __init__(self, model_dir):
        self.INPUT_SIZE = (512, 512)
        self.model = model_dir + "model.h5"
        self.model_tf = model_dir + "unetct.pb"
        self.color_mapping = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
        self.labels = {1: 'Ground Glass', 2: 'Consolidations', 3: 'Pleural effusion'}

    def export(self):
        ses = tf.Session()
        f = gfile.GFile(self.model_tf, 'rb')
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        # ses.graph.as_default()
        tf.import_graph_def(graph_def, name='unet')
        # print('UNet')
        # print(ses.graph.get_operations())
        tensor_input = ses.graph.get_tensor_by_name('unet/input_1:0')
        tensor_output = ses.graph.get_tensor_by_name('unet/activation_22/Sigmoid:0')
        return ses, tensor_input, tensor_output

    def predict(self, img, sess, in_to_restore, op_to_restore):
        processed_img = self.preprocess(img)
        predictions = sess.run(op_to_restore, {in_to_restore: [processed_img]})
        prediction = predictions[0]
        prediction = prediction.reshape((512, 512, 4)).argmax(axis=2)
        output_image, output_dict = self.post_process(prediction, img)
        return output_image, output_dict

    def preprocess(self, img):
        # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.INPUT_SIZE,interpolation=cv2.INTER_NEAREST)
        img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32, copy=False)
        return img

    def post_process(self, predicted, original_img):
        count = {1: 0, 2: 0, 3: 0}  # To get Area
        result = np.zeros((512, 512,3))
        for i in range(self.INPUT_SIZE[0]):
            for j in range(self.INPUT_SIZE[1]):
                if predicted[i][j] != 0:
                    result[i][j] = self.color_mapping[predicted[i][j]]
                    count[predicted[i][j]] += 1
        # original_img = cv2.imdecode(original_img, cv2.IMREAD_COLOR)
        original_img = cv2.resize(original_img, (512, 512),interpolation=cv2.INTER_NEAREST)
        original_img= cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
        dst = cv2.addWeighted(original_img, 0.7, result, 0.3, 0, dtype=cv2.CV_8U)
        # cv2.imshow('okay',dst)
        # cv2.waitKey(0)
        # area_percentage = {k: 100 * v / 512.0 for k, v in count.items()}
        # TODO , instead of taking full image, find only inside lung cavity
        detailed_dict = [
            {'color': list(reversed(self.color_mapping[k])), 'area_percentage': round(100 * count[k] / (512.0 * 512), 3),
             'legend': self.labels[k], 'count': count[k], 'key': k} for k in count]
        output_image = cv2.imencode('.jpg', dst)[1].tostring()
        return output_image, detailed_dict


if __name__ == "__main__":
    # import time
    # a=time.time()
    image = '/home/ronald/xray_corona/flask_backend/data/test/covid-19-pneumonia-15-PA.jpg'
    # covid_model = CovidEvaluator("/home/ronald/xray_corona/flask_backend/model/COVID-Netv1/")
    # sess, x, op_to_restore = covid_model.export()
    # covid_resp = covid_model.predict(image, sess, x, op_to_restore)
    # print(time.time()-a,covid_resp)
    # chester_model = ChesterAiEvaluator("/home/ronald/xray_corona/flask_backend/model/ChesterAI/")
    # chester_model_d=chester_model.export_keras()
    # model=tf.keras.models.load_model(chester_model.model)
    # print(model.summary())
    # pred=chester_model.evaluate(img=chester_model.preprocess(img=image))
    # pred=chester_model.predict_chest_conditions(chester_model_d,image)
    # print(pred)
    # sess,inp,outp=chester_model.export()
    # pred=chester_model.predict(image,sess,inp,outp)
    # pred = {'Atelectasis': 0.4010319709777832, 'Cardiomegaly': 16.25364124774933, 'Consolidation': 2.338424324989319,
    #         'Edema': 0.00020563602447509766, 'Effusion': 2.881249785423279, 'Emphysema': 3.8743019104003906e-05,
    #         'Fibrosis': 0.0001817941665649414, 'Hernia': 0.0008046627044677734, 'Infiltration': 5.490788444876671,
    #         'Mass': 42.957788705825806, 'No Finding': 65.832532048225403, 'Nodule': 10.899960994720459,
    #         'Pleural_Thickening': 0.053374317940324545, 'Pneumonia': 66.225700303912163,
    #         'Pneumothorax': 70.27559681329876184}
    # # print(pred)
    # predictions =chester_model.modify_prediction_dict(pred)
    # print(predictions)
    from aws_functions import upload_to_s3,get_xray_image
    import uuid
    user_id=1
    image_loc = '/home/ronald/Downloads/dicom_00000001_000.dcm'
    # image_loc='/home/ronald/Downloads/Exported20200425.dcm'
    # image_loc='/home/ronald/Downloads/ID_0000_AGE_0060_CONTRAST_1_CT.dcm'
    # image_url='https://xray-corona-ds.s3.ap-south-1.amazonaws.com/dataset/dummy_test/dicom_00000001_000.dcm'
    image_url='https://xray-corona-ds.s3.ap-south-1.amazonaws.com/dataset/dummy_test/Exported20200425.dcm'
    image_loc=get_xray_image(image_url)
    dicom_data=image_loc.read()
    print('Read')
    from pydicom import dcmread
    from pydicom.filebase import DicomBytesIO

    dicom_bytes = DicomBytesIO(dicom_data)
    dicom_dataset = dcmread(dicom_bytes)
    image=dicom_dataset
    image=image.pixel_array
    unetModel = UNetCTEvaluator("/home/ronald/xray_corona/flask_backend/model/U-Net-CT/")
    sess, inpu, output = unetModel.export()
    unet_resp, unet_dict=    unetModel.predict(image, sess, inpu, output)
    print(unet_dict)
    filename = f"{user_id}_{uuid.uuid4()}"
    image_url = upload_to_s3(unet_resp, filename)
    print(image_url)
