import redis
from redis import ConnectionError
import logging


def check_alive():
    logging.basicConfig()
    logger = logging.getLogger('redis')

    rs = redis.Redis("localhost")
    try:
        rs.ping()
    except ConnectionError:
        logger.error("Redis isn't running. try `/etc/init.d/redis-server restart`")
        exit(0)
