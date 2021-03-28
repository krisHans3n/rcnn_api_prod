import flask
import sys
import logging
import json
import uuid

import lib.urlValidator as valid
from conduit.settings import ProdConfig, DevConfig, create_redis_connection
from lib.vectorformatter import VectorLoader
from flask import Flask, request, jsonify, session
from functools import wraps
from main_det import ProcessImages
from Dispatch import *
from lib.utils.ImageParser import ImageParser
from rq import Queue

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

    # redis_status() # check redis is running
    # run_worker()   # ensure worker is running before queueing
    # start_queueing()

    q = None

    try:
        q = Queue(connection=create_redis_connection())
    except Exception as ex:
        print('problem establishing queue ', ex)

    @app.route('/ml_img_det_intrfc/', methods=['POST', 'GET'])
    @required_params({"urls": list})
    def respond():
        """
        start environment with --> source isoEnv/bin/activate
        start redis (in new terminal): redis-server
        run file (in new terminal): worker.py

        curl -v -H "Content-Type: application/json" -X POST -d '{"urls": ["https://www.w3schools.com/howto/img_mountains.jpg", "https://www.oxforduniversityimages.com/images/rotate/Image_Spring_17_4.gif"]}' http://127.0.0.1:5000/ml_img_det_intrfc/
        """
        # TODO: refactor so all image/url/response conducting code is inside ImageParser class (pass q, job, guid etc variables to it)
        # TODO: add delete function to redis
        # TODO: add startup file for running redis and worker at runtime
        # TODO: complete configs so they are ready for production and switching to dev
        # TODO: test api with different images
        # TODO: tidy/organise code
        # TODO: move to chrome extension for processing
        # TODO: Add another call to GAN/ Face validity api
        # TODO: add more passive analysis on passive api
        # TODO: switch out current cnn for one that handles compressed images better

        # ImageParser sequence
        # -> instantiate image parser
        # -> determine guids
        # -> instantiate dispatch
        # --> pass q to dispatch
        # --> pass URLs to dispatch
        # --> return job
        # -> pass guids to imageparser
        # -> pass q to image parser
        # -> pass job to image parser (in case it's needed)
        # =====> port all necessary functions to image parser
        # --> return result merged dict
        # IMPORTANT: imageparser needs function to wait for job complete

        urls = request.get_json()
        urls = valid.validate_url_string(urls["urls"])

        redis_ids = [None, str(uuid.uuid4())]  # IDs for where to find JSON responses
        _report = {}
        result_response = {}

        job = None

        if urls is None or len(urls) == 0:
            result_response["Error"] = "no urls were provided"
        elif urls is not None:
            # handle_dispatch # return job
            job = q.enqueue_call(
                func=send_to_passive_analysis,
                args=(request.get_json(), redis_ids[1]),
                result_ttl=1800,
                job_id=redis_ids[1]
            )
            # handle_images(urls) # return _report
            pi = ProcessImages()
            vl = VectorLoader()
            _nu_fls = pi.start_processing(urls)
            _report = vl.appendB64toJSON(_nu_fls)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # job_status()  #  return true
        result = q.fetch_job(job.id)
        if result is None:
            print("Job Incomplete")
        elif result.is_failed:
            print('something went wrong', result.is_failed)
        else:
            print(job.id)
            print('job successful')

        img_p = ImageParser(redis_ids=redis_ids, parent_dict=_report)
        result_response = img_p.merge_api_responses()

        return jsonify(result_response)

    return app
