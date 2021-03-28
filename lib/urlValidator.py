import validators


def validate_url_string(url_arr):
    for url in url_arr:
        if not validators.url(url):
            url_arr.remove(url)
    return url_arr
