from lib.database.RQHook import RQHook


class PayloadFormatter:

    def __init__(self, file_list=[]):
        self.file_list

    def merge_api_responses(self):
        RQHook.pull_json_rq_res()
        #  take valid file name list
        #  for each sub root of Json match with file in list
        #     copy elements that don't have images to be dict
        #     if it has images get the base64 string
        # .    And compress if needed
        # .    Append the string to the dict
        # .  Do for each Json







