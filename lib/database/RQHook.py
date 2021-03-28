from rq import Queue
from rq.job import Job
import json
import redis
from conduit.settings import create_redis_connection
"""
Wrapper class to interact with rq worker threads
note: strictly inspection class. No changes to runtime processes or results
TODO: Have the same redis connection shared among all APIs and pass IDs 
      for pulling data into one response
"""
r = redis.Redis()

# def save_type_wrapper(response, redis_id: str = ''):
#     if type(response) is dict:
#         save_response_dict()


def save_response_json(response_json, redis_id: str = ''):
    print(type(response_json), ' with id: ', redis_id)

    #  TODO: Ensure object is redis safe

    try:
        data = json.dumps(response_json)
        r.set(redis_id, data)
        print('Successfully saved JSON dictionary', redis_id)
    except Exception as ex:
        print('Something went wrong saving json response ', ex)

    return


def save_response_str(response_dict, redis_id: str = ''):
    data = json.dumps(response_dict)
    r.set(redis_id, data)
    return


def get_redis_json_cache(redis_id: str = ''):
    try:
        return json.loads(r.get(redis_id))
        print('Retrieved redis data')
    except Exception as ex:
        print('Could not retrieve redis data', ex)
        return False


def get_rq_json_cache(redis_id: str = ''):
    # job = Job.fetch(redis_id, create_redis_connection())
    # q = Queue(connection=create_redis_connection())
    # result = q.fetch_job(job_id=redis_id)
    # print('job complete')
    # print(job.result)
    # print(result.result)
    return


class RQHook:

    def __init__(self, job_id, job=Job, q=None):
        self.job = job
        self.q = q
        self.job_id = job_id
        self.api_img_mask_dir = './dataset/NIST2016/async_masks/'

    def pull_json_rq_res(self, redis_obj, r_id):
        pass
        #  use the job object id created when saving Json to redis
        #  return the Json object for each query

    def inspect_job_object(self):
        pass

    def process_job_complete(self):
        if self.job.result:
            _json = json.dumps(self.job.to_dict())
        #  TODO: do alternative checks to make sure process
        #   hasn't failed or is just incomplete
        return

    def wait_until_complete(self):
        pass

    def fetch_job_response(self):
        pass

    def change_timeout_job(self):
        pass
