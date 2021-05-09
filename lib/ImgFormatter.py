import os, io, sys
import PIL
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


def image_base64_ndarray(image):
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="bmp", quality=50, optimize=True)
        image_str = buffer.getvalue()
        return base64.b64encode(image_str).decode('utf-8')
    except Exception as ex:
        print(ex)


def decode_base64(b64str: str = ''):
    try:
        return base64.b64decode(b64str)
    except Exception as ex:
        print('Problem decoding b64 string', ex)
        return False


def return_img_size(img):
    if type(img == str):
        l_ = len(img)
        c_ = img.count('=')
        bytes_ = 3 * (l_ / 4) - c_
        return bytes_


"""
VectorLoader converts pil.image to base64 decoded string

EXtra features:
 - check amount of images processed and modify status when over the cap
 - check memory used and change status if necessary 
"""


class VectorLoader:
    def __init__(self, file_name: str = "", path: str = ""):
        self.file_name = file_name
        self.path = path
        self.size = 0
        self.images_processed = 0

    def track_usage(self):
        if self.size > 5000000:
            return 501  # Reached internal dict limit
        if self.images_processed > 30:
            return 502  # Reached internal processing limit
        return 1

    def serialise_vector_base64(self):
        pass

    def deserialise_from_base64(self, img_type, b64_str):
        # serialize string to some output image type
        # dependant on type parameter
        pass

    def image_base64_path(self, file, path):
        directory = os.listdir(path)
        self.size = os.path.getsize(path)
        for fname in directory:
            if file in fname:
                content = open(path + fname, 'rb')
                return base64.b64encode(content.read()).decode('utf-8')

    def b64_from_image_obj(self, title, image):
        base64_str = image_base64_ndarray(image)
        self.images_processed += 1
        self.size += return_img_size(base64_str)
        return [title, base64_str]

    def b64_from_path(self, dict_):
        for k in list(dict_):
            dict_[k].append(["mask_1", self.image_base64_path(k, IMG_DIR_MASK_1)])
            dict_[k].append(["mask_2", self.image_base64_path(k, IMG_DIR_MASK_2)])
        return dict_

    def compress_image(self):
        pass
