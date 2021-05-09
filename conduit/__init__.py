from flask import Flask, request, jsonify
from conduit.settings import ProdConfig, DevConfig
from main_det import commence


def create_app(config_object=DevConfig):
    app = Flask(__name__)
    app.url_map.strict_slashes = False
    app.config.from_object(config_object)

    from conduit.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
