import cv2
from skimage.metrics import structural_similarity as ssim
import magic
import os


class BasicValidator:
    def __init__(self, image_loc):
        self.image_loc = image_loc
        self.allowed_file_types = ['jpeg', 'jpg', 'png', 'dicom']

    def _file_type_validation(self):
        file_type = magic.from_buffer(self.image_loc, mime=True)
        file_types = file_type.split('/')
        if file_types[0] != 'image':
            return {'result': False, 'reason': 'Invalid file type,please upload an image'}
        if file_types[1] not in self.allowed_file_types:
            return {'result': False, 'reason': f'Please upload {self.allowed_file_types} only'}
        return {'result': True}

    def _ssim(self,gturth_image,image_size):
        #  Compares Strucutral similartiy
        ssmi_image=cv2.imread(gturth_image,cv2.IMREAD_GRAYSCALE)  # Bad design,TODO  pass it as a variable
        test_image=cv2.imdecode(self.image_loc,cv2.IMREAD_GRAYSCALE)
        ssmi_image=cv2.resize(ssmi_image,image_size)
        test_image=cv2.resize(test_image,image_size)
        return ssim(ssmi_image,test_image)

    def _calcHist(self,gtruth_image):
        gtruth_image=cv2.imread(self.gtruth_image,cv2.IMREAD_COLOR) # Bad design,TODO  pass it as a variable
        gtruth_image=cv2.cvtColor(gtruth_image,cv2.COLOR_BGR2RGB)
        test_image=cv2.imdecode(self.image_loc,cv2.IMREAD_COLOR)
        test_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
        grtuh_hist = cv2.calcHist([gtruth_image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        grtuh_hist=cv2.normalize(grtuh_hist,grtuh_hist).flatten()
        test_hist = cv2.calcHist([test_image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        test_hist=cv2.normalize(test_hist,test_hist).flatten()
        result=cv2.compareHist(grtuh_hist,test_hist,cv2.HISTCMP_CORREL)
        return result

    def basic_validate(self):
        return self._file_type_validation()


class ChestCTValidator(BasicValidator):
    def __init__(self,image_loc):
        super().__init__(image_loc)
        self.gtruth_image = 'data/test/kjr-21-e24-g002-l-b.jpg'
        self.image_size = (512, 512)

    def _sssim(self):
        return super()._ssim(self.gtruth_image, self.image_size)

    def _color_histogram(self):
        return super()._calcHist(self.gtruth_image)

    def validate(self):
        # base_result = super().basic_validate()
        # if not base_result['result']:
        #     return base_result
        ssim_value = self._sssim()
        hist_value = self._color_histogram()
        if hist_value < 0.7 or ssim_value < 0.15:
            return {"result": False, "ChestXray": "Pass", "SSIM": ssim_value, "HistCmp": hist_value,
                    "message": "Image Validation Failed, not Chest Xray"}
        return {"result": True}


class ChestXRayValidator(BasicValidator):
    def __init__(self, image_loc):
        super().__init__(image_loc)
        self.gtruth_image='data/test/covid-19-pneumonia-15-PA.jpg'
        self.image_size=(244,244)

    def _sssim(self):
        return super()._ssim(self.gtruth_image,self.image_size)

    def _color_histogram(self):
        return super()._calcHist(self.gtruth_image)

    def validate(self):
        # base_result = super().basic_validate()
        # if not base_result['result']:
        #     return base_result
        ssim_value = self._sssim()
        hist_value = self._color_histogram()
        if hist_value < 0.7 or ssim_value < 0.15:
            return {"result": False, "ChestXray": "Pass", "SSIM": ssim_value, "HistCmp": hist_value,
                    "message": "Image Validation Failed, not Chest Xray"}
        return {"result": True}


#So SSIM has to be somewhere 20+
if __name__=="__main__":
    # image1_loc='/home/ronald/Downloads/COVID-19 (35).png'
    # image2_loc='/home/ronald/Downloads/pierre-broissand-concept02.jpg'
    # image3_loc='/home/ronald/xray_corona/flask_backend/14_label_model.h5'
    # image4_loc='/home/ronald/Downloads/SSE_Protocol.svg'
    # image5_loc='/home/ronald/Downloads/index.jpeg'
    # image6_loc='/home/ronald/Downloads/COVID-19 (35).png'
    # image7_loc='/home/ronald/Downloads/COVID-19 (22).png'
    # image8_loc='//home/ronald/Downloads/nejmoa2001191_f1-L.jpeg'
    # image9_loc='/home/ronald/Downloads/covid-19-pneumonia-15-PA.jpg'
    # image10_loc='/home/ronald/Downloads/vetstreet.brightspotcdn.com.jpeg'
    # image11_loc='/home/ronald/Downloads/index.png'
    # image12_loc='/home/ronald/Downloads/AGNI_58.jpg'
    # image13_loc='/home/ronald/Downloads/kjr-21-e24-g002-l-c.jpg'
    # image14_loc='/home/ronald/Downloads/10000.png'
    # # image1=cv2.imread(image1_loc,0)
    # # image2=cv2.imread(image2_loc,0)
    # # image1=cv2.resize(image1,(244,244))
    # # image2 = cv2.resize(image2, (244, 244))
    # # ss=ssim(image1,image2)
    # # print(ss)
    # # import time
    # # chest1=ChestXRayValidator(image14_loc)
    # # a=time.time()
    # # print(chest1.validate())
    # # print(time.time()-a)
    pass