import logging

from conduit.api import bp
from flask import Flask, request, jsonify, session
from main_det import commence
from conduit.api.content.validation import required_params, validate_url_string
from conduit.api.security.session_domain import required_ip

logging.basicConfig(level=logging.DEBUG)


@bp.route('/ml_img_det_intrfc/', methods=['POST', 'GET'])
@required_params({"urls": list})
def respond():
    urls = request.get_json()
    urls = validate_url_string(urls["urls"])

    if urls is None or len(urls) == 0:
        return jsonify("no urls were provided")
    elif urls is not None:
        cumulative = commence(urls)

        return jsonify(cumulative)
