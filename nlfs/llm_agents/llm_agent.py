import os

def get_api_key(api_key_var):

    try:
        api_key = os.getenv(api_key_var)
    except Exception:

        raise Exception("Need to set the %s environment variable with the api key" % (
            api_key_var))

    return api_key
