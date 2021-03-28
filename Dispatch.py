import requests
from lib.utils.ImageParser import ImageParser
from lib.database.RQHook import save_response_json, get_redis_json_cache

"""
Helper functions for asynchronous api calls
"""


def construct_request(url, json):
    return requests.post(url=url,
                         headers={'Content-Type': 'application/json'},
                         json=json)


def send_to_passive_analysis(json, r_id):
    response = construct_request("http://127.0.0.1:5001/imginterface/", json)
    api_img_mask_dir = '/dataset/NIST2016/async_masks/'

    _json = response.json()
    print(len(_json), r_id)
    save_response_json(_json, redis_id=r_id)
    # Below code may not be necessary in productin since
    # json data will be persisted on redis
    #  jsn = get_redis_json_cache(redis_id=r_id)
    # Below for test purposes
    ip = ImageParser()
    ip.save_to_directory(_json)

    return


def send_to_facial_analysis(json):
    # Development url
    response = construct_request("http://127.0.0.1:5002/ganinterface/", json)
    print('%%%%%%% API worker 2 JSON %%%%%%%')
    print(response.text)
    return response


"""
Takes image objects and saves in temporary space
Also gathers JSON metadata and saves object in redis 
"""


def unpack_json(json):
    pass
    # Check JSON format
    # iterate through image names
    # save images to directory using file name (or shared guid)
