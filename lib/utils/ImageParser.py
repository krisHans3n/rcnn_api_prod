import os, io
from PIL import Image
import base64
import json
from lib.database.RQHook import get_redis_json_cache

"""
This class saves an image represented as a bas64 string or other 
and saves to directory 
Different functions allow compressing images to specific sizes
  or formatting data representing an image 
  Currently base 64 is used since that is what the api2 encodes it with 
"""


class ImageParser:

    def __init__(self, file_name: str = "", path: str = "", redis_ids: list = [], parent_dict: dict = {}, job_id=0):

        self.file_name = file_name
        self.path = path
        self.job_id = job_id
        self._json = None
        self.mergable_objects = []
        self.usable_images = []
        self.images = parent_dict.keys()
        self.parent_dict = parent_dict
        self.redis_ids = redis_ids

        if file_name == "":
            CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
            TARGET_DIR = CURRENT_DIRECTORY.split('/')
            del TARGET_DIR[-2:]
            self.api_img_mask_dir = '/'.join(TARGET_DIR) + '/dataset/NIST2016/async_masks/'
        else:
            self.api_img_mask_dir = file_name

    def file_success_check(self):
        """
        NOTE: only needed if we use guids for individual image files
        uses self.redis_ids and self.parent_dict to determine which images are used
        :return:
        """
        # parent_dict = {"img1": "guid1", "img2":"guid2"}
        # redis_ids = ["guid1", "guid2", "guid3"]
        return

    def merge_api_responses(self):
        """
        Extract json from redis and compile all into one dict for response
        :return:
        """
        new_dict = {}
        for f in self.images:
            f = os.path.splitext(f)[0]  # TODO: switch filename with guids
            new_dict[f] = []
            for id_ in self.redis_ids:
                print(id_)
                api_resp = None
                if id_:
                    api_resp = get_redis_json_cache(id_)
                    if api_resp is False:
                        continue
                else:
                    api_resp = self.parent_dict  # None type refers to parent dict
                if f in api_resp.keys():
                    print(api_resp.keys())
                    for v_ in api_resp[f]:
                         new_dict[f].append(v_)

                # Delete json

                print('objects gathered .........', new_dict.keys())

        return new_dict

    def save_to_directory(self, data):
        """
        Serialize data object into json format for iteration
        Save image in base64
        :param data: str or dict or json
        :return: True
        """
        if isinstance(data, str):
            self.process_str(data)
        elif isinstance(data, dict):
            self.process_json(data)

        if self.is_safe_json(self._json):
            try:
                for key in self._json:
                    print(self.is_safe_json(self._json))
                    for keys_v in self._json[key]:
                        if keys_v[0] == 'img_result' and keys_v[1] is not None:
                            full_pth = self.api_img_mask_dir + key
                            print('Saving to: ', full_pth)
                            self.save_uncompressed(full_pth, base64.b64decode(keys_v[1]))
            except Exception as ex:
                print('Saving failed')
                print(ex)

    def is_safe_json(self, data):
        #  print(type(data))
        if data is None:
            return True
        elif isinstance(data, (bool, int, float, str)):
            return True
        elif isinstance(data, (tuple, list)):
            return all(self.is_safe_json(x) for x in data)
        if isinstance(data, dict):
            return all(isinstance(k, str) and self.is_safe_json(v) for k, v in data.items())
        return False

    def save_uncompressed(self, full_path, img_data):
        try:
            with open(full_path, 'wb') as f:
                f.write(img_data)
                return True
        except Exception as ex:
            print('problem saving base64 string as uncompressed image')
            print(ex)
            return False

    def save_compressed(self, full_path, img_data, quality=30):
        try:
            buf = io.BytesIO(img_data)
            img = Image.open(buf)
            img.save(full_path,
                     optimize=True,
                     quality=quality)
            return True
        except Exception as ex:
            print('problem saving base64 string as compressed image')
            return False

    def process_json(self, json_arg):
        try:
            self._json = json_arg
        except Exception as ex:
            print('Problem loading json', ex)

    def process_dict(self, dictionary):
        try:
            self._json = json.dumps(dictionary)
        except Exception as ex:
            print('Problem loading json from dict', ex)

    def process_str(self, string):
        try:
            self._json = json.loads(string)
        except Exception as ex:
            print('Problem loading json from string', ex)

    def compress_image_max(self):
        pass

    def compress_to_average(self):
        pass
