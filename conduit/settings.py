# -*- coding: utf-8 -*-
"""Application configuration."""
import os
import redis

from datetime import timedelta


def create_redis_connection():
    redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

    conn = redis.from_url(redis_url)
    return conn


def redis_listen():
    return ['default']


def runtime_settings():
    # if in development
    return int(os.environ.get('PORT', 5000)), True


class Config(object):
    """Base configuration."""

    # SECRET_KEY = os.environ.get('CONDUIT_SECRET', 'secret-key')  # TODO: Change me
    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
    # DEBUG_TB_INTERCEPT_REDIRECTS = False
    # CACHE_TYPE = 'simple'  # Can be "memcached", "redis", etc.


class ProdConfig(Config):
    """Production configuration."""

    FLASK_ENV = 'production'
    DEBUG = False
    #SERVER_NAME = 'ImageSpotLightAPI'


class DevConfig(Config):
    """Development configuration."""

    FLASK_ENV = 'development'
    DEBUG = True
    # DB_NAME = 'dev.db'
    # Put the db file in project root
    # DB_PATH = os.path.join(Config.PROJECT_ROOT, DB_NAME)
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///{0}'.format(DB_PATH)
    # CACHE_TYPE = 'simple'  # Can be "memcached", "redis", etc.
    # JWT_ACCESS_TOKEN_EXPIRES = timedelta(10 ** 6)


class TestConfig(Config):
    """Test configuration."""

    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
    # For faster tests; needs at least 4 to avoid "ValueError: Invalid rounds"
    BCRYPT_LOG_ROUNDS = 4
