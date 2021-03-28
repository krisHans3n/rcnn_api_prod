import os, io, sys
from PIL import Image
import numpy as np
import cv2
import base64

THIS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = THIS_DIRECTORY.split('/')
del IMG_DIR[-2:]
IMG_DIR_MASK_1 = './dataset/NIST2016/mask_1/'
IMG_DIR_MASK_2 = './dataset/NIST2016/mask_2/'
IMG_DIR_PROBE = './dataset/NIST2016/probe/'


class VectorLoader:
    def __init__(self, file_name: str = "", path: str = ""):
        self.file_name = file_name
        self.path = path
        self.size = None

    def decode_base64(self, b64str: str = ''):
        try:
            return base64.b64decode(b64str)
        except Exception as ex:
            print('Problem decoding b64 string', ex)
            return False

    def serialise_vector_base64(self):
        pass

    def deserialise_from_base64(self, img_type, b64_str):
        # serialize string to some output image type
        # dependant on type parameter
        pass

    def image_base64(self, file, path):
        directory = os.listdir(path)
        self.size = os.path.getsize(path)
        for fname in directory:
            print('>>>>>', fname)
            print('>>>>>', path)
            if file in fname:
                content = open(path + fname, 'rb')
                return base64.b64encode(content.read()).decode('utf-8')

    def appendB64toJSON(self, dict_):
        for k in list(dict_):
            #  k = os.path.splitext(k)[0]
            #  dict_[k] = []
            dict_[k].append(["mask_1", self.image_base64(k, IMG_DIR_MASK_1)])
            dict_[k].append(["mask_2", self.image_base64(k, IMG_DIR_MASK_2)])
            if self.size > 1000000:
                #  Multi-compressed images will give vague results
                dict_[k].append(["HEAVY_COMPRESSION", 1])

        return dict_
