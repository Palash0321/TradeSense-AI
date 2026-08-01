import time

_cache = {}

def get(key):

    value = _cache.get(key)

    if not value:
        return None

    data, expiry = value

    if time.time() > expiry:

        del _cache[key]

        return None

    return data


def set(key, value, ttl):

    _cache[key] = (

        value,

        time.time() + ttl

    )