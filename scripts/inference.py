import cv2
import numpy as np
import glob
import os

# If tensorflow 2.0 is installed
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# If tensorflow 1.0 is installed
import tensorflow as tf


class CovidEvaluator(object):
    def __init__(self, model_dir):
        self.labels = ["Normal", "Bacterial", "Non-COVID19 Viral", "COVID-19 Viral"]
        self.INPUT_SIZE = (224, 224)
        self.MODEL_GRAPH = model_dir + "model.meta"
        self.MODEL = model_dir + "model"

    def load_img(self, img):
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return image  # Keeping image in memory

    def pre_process(self, img):
        img_arr = cv2.resize(img, self.INPUT_SIZE)  # resize
        img_arr = img_arr.astype('float32') / 255.0
        img_arr = img_arr.reshape(1, self.INPUT_SIZE[0], self.INPUT_SIZE[1], 3)
        return img_arr

    def export(self):
        sess = tf.Session()
        saver = tf.train.import_meta_graph(self.MODEL_GRAPH)
        saver.restore(sess, self.MODEL)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("input_1:0")
        op_to_restore = graph.get_tensor_by_name("dense_3/Softmax:0")
        return sess, x, op_to_restore

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
        self.labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                       'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
                       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    def export(self):
        exported_model = tf.keras.models.load_model(self.model)
        print(exported_model.summary())
        return exported_model

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


if __name__ == "__main__":
    # model = CovidEvaluator("model/COVID-Netv1/")
    # sample_test = model.evaluate("data/test/covid-19-pneumonia-15-PA.jpg")
    # print(sample_test)
    chester_model=ChesterAiEvaluator("model/ChesterAI/")
    # model=tf.keras.models.load_model(chester_model.model)
    # print(model.summary())
    chester_model_d=chester_model.export()
    pred=chester_model.predict_chest_conditions(chester_model_d,'data/test/covid-19-pneumonia-15-PA.jpg')
    print(pred)