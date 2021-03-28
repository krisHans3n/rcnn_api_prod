"""
For task queing when sending requests to multiple other apis
TODO: Use wrapper to change config from development to production
"""

from rq import Worker, Queue, Connection
from conduit.settings import create_redis_connection, redis_listen

if __name__ == '__main__':
    with Connection(create_redis_connection()):
        worker = Worker(list(map(Queue, redis_listen())))
        worker.work()












