import json
import copy
import numpy as np

from ezai.util import log_util
l = log_util.get_logger()

def load_dict_from_json(filename):
    cf_dict = json.load(open(filename, 'r'))
    return cf_dict

class NPJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NPJSONEncoder, self).default(obj)

def save_to_json(obj, filename, sort_keys=False, indent=4):
    if not isinstance(obj, dict):
        obj = obj.__dict__
    json.dump(obj, open(filename, 'w'), indent=indent, sort_keys=sort_keys, cls = NPJSONEncoder)

class DictObj(object):
    '''
    Dictionaries that also work like objects

    can be created by passing a dictionary or json filename
    '''

    def __init__(self, d=None):
        if isinstance(d, dict):
            self.__dict__ = d
        elif isinstance(d, str):  # d is filename
            self.__dict__ = load_dict_from_json(d)
        else:
            l.info('Making empty DictObj because not given a dict or a json filename')

    def __getitem__(self, key):
        return self.__dict__[key]

    # Return the value for key if key is in the dictionary, else default.
    # If default is not given, it defaults to None
    def get(self, key, default=None):
        return self.__dict__.get(key,default)

    # If key is in the dictionary, return its value.
    # If key is not in, insert key with a value of default and return default.
    def setdefault(self, key,default=None):
        return self.__dict__.setdefault(key,default)

    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def update(self, obj):
        if isinstance(obj, dict):
            self.__dict__.update(obj)
        else:
            self.__dict__.update(obj.__dict__)
        return self

    def save_to_json(self, filename, sort_keys=False, indent=4):
        save_to_json(self, filename, sort_keys, indent)
        return self

    def load_from_json(self, filename):
        self.__dict__ = load_dict_from_json(filename)
        return self

    def dumps_json(self, sort_keys=False, indent=4):
        obj = self.__dict__
        return json.dumps(obj, indent=indent, sort_keys=sort_keys, cls = NPJSONEncoder)

    def deepcopy(self):
        return copy.deepcopy(self)

    def values(self):
        return list(self.__dict__.values())

    def keys(self):
        return list(self.__dict__.keys())

