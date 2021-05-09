from flask import Blueprint

bp = Blueprint('api', __name__)

from conduit.api import app_routes

