import flask
import sys
import logging
import json
import uuid

from conduit.settings import ProdConfig, DevConfig, create_redis_connection
from flask import Flask, request, jsonify, session
import lib.urlValidator as valid
from functools import wraps
from main_det import ProcessImages
from lib.utils.ImageParser import ImageParser

logging.basicConfig(level=logging.DEBUG)


def required_params(required):
    def decorator(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            print(request.get_json())
            _json = request.get_json()
            missing = [r for r in required.keys()
                       if r not in _json]
            print(missing, _json)
            if missing:
                response = {
                    "status": "error",
                    "message": "Request JSON is missing some required params",
                    "missing": missing
                }
                return jsonify(response), 400
            wrong_types = [r for r in required.keys()
                           if not isinstance(_json[r], required[r])]
            if wrong_types:
                response = {
                    "status": "error",
                    "message": "Data types in the request JSON do not match the required format",
                    "param_types": {k: str(v) for k, v in required.items()}
                }
                return jsonify(response), 400
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def create_app(config_object=DevConfig):
    app = Flask(__name__)
    app.url_map.strict_slashes = False
    app.config.from_object(config_object)

    @app.route('/ml_img_det_intrfc/', methods=['POST', 'GET'])
    @required_params({"urls": list})
    def respond():

        urls = request.get_json()
        urls = valid.validate_url_string(urls["urls"])

        '''IDs for where to find JSON responses'''
        redis_ids = [None, str(uuid.uuid4())]

        if urls is None or len(urls) == 0:
            return jsonify("no urls were provided")
        elif urls is not None:
            pi = ProcessImages()
            cumulative = pi.start_processing(urls)
            # img_p = ImageParser(redis_ids=redis_ids, parent_dict=cumulative)
            # result_response = img_p.merge_api_responses()

            return jsonify(cumulative)

    return app
