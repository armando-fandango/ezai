import json
import yaml
import copy
import numpy as np
import cattr
from .json import NPJSONEncoder

# from . import log_util
# logger = log_util.get_logger()

def json_or_yaml(filename):
    """
    This function would be obsolete when pyyaml supports yaml 1.2
    With yaml 1.2 pyyaml can also read json files
    :return:
    """
    import re
    from pathlib import Path

    commas = re.compile(r',(?=(?![\"]*[\s\w\?\.\"\!\-\_]*,))(?=(?![^\[]*\]))')
    """
    Find all commas which are standalone 
     - not between quotes - comments, answers
     - not between brackets - lists
    """
    file_path = Path(filename)
    signs = commas.findall(file_path.open('r').read())
    return "json" if len(signs) > 0 else "yaml"


def load_dict_from_json_file(filename):
    dict_obj = json.load(open(filename, 'r'))
    return dict_obj


def load_dict_from_yaml_file(filename):
    dict_obj = yaml.safe_load(open(filename, 'r'))
    return dict_obj


def load_dict_from_json_str(string):
    dict_obj = json.loads(string)
    return dict_obj


def to_nested_objdict(obj):
    """ Converts to Nested DictObj

    Args:
        obj: a Dict kind of Object

    Returns:
        Converts dict into nested ObjDict

    """
    if not isinstance(obj, ObjDict):
        if isinstance(obj, dict):
            obj = ObjDict(obj)
        else:
            raise ValueError(f"You passed {type(obj)}, which is not a Dict or ObjDict")

    for key in obj:
        if isinstance(obj[key], dict):
            obj[key] = to_nested_objdict(obj[key])
    return obj

def to_nested_dict(obj):
    """ Converts to Nested Dict

    Args:
        obj: a Dict kind of Object

    Returns:
        Converts ObjDict into nested dict

    """
    if not isinstance(obj, dict):
        if isinstance(obj, ObjDict):
            obj = obj.__dict__
        else:
            raise ValueError(f"You passed {type(obj)}, which is not a Dict or ObjDict")

    for key in obj:
        if isinstance(obj[key], ObjDict):
            obj[key] = to_nested_dict(obj[key])
    return obj


def to_yaml(obj):
    #if not isinstance(obj, dict):
    #    obj = obj.__dict__
    #return yaml.dump(obj, default_flow_style=False)
    return yaml.dump(json.loads(json.dumps(obj)))





def save_to_json_file(obj, filename, sort_keys=False, indent=4):
    if not isinstance(obj, dict):
        obj = obj.__dict__
    json.dump(obj, open(filename, 'w'), indent=indent, sort_keys=sort_keys, cls=NPJSONEncoder)


def save_to_yaml_file(obj, filename):
    if not isinstance(obj, dict):
        obj = obj.__dict__
    #yaml.dump(obj, open(filename, 'w'), default_flow_style=False)
    yaml.dump(json.loads(json.dumps(obj)), open(filename, 'w') )

class ObjDict(dict):
    """A data structure that inherits from dict and adds object style member access

    TODO:
        * Create an init method that converts nested dicts in this object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        to_nested_objdict(self)

    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    # def __str__(self):
    #    return self.json_dumps()

    def to_json(self, sort_keys=False, indent=2):
        """convert to json

        Args:
            sort_keys: Sort the keys of dict or not, default False
            indent: indentation for JSON struct, default 2 spaces

        Returns:
            json string

        """
        return json.dumps(self, indent=indent, sort_keys=sort_keys, cls=NPJSONEncoder)

    def to_dict(self):
        """convert to dict

        Returns:
            dict object
        """
        return cattr.unstructure(self)

    def to_yaml(self):
        """convert to yaml representation

        Returns:
            yaml string
        """
        return to_yaml(self)

    def save_to_json_file(self, filename, sort_keys=False, indent=2):
        """Save to json file

        Args:
            filename: path or name of the file
            sort_keys: whether to sort the keys
            indent: indentation of the spaces

        Returns:

        """
        save_to_json_file(self, filename, sort_keys, indent)
        return self

    def save_to_yaml_file(self, filename):
        """Save to yaml file

        Args:
            filename: path or name of the file

        Returns:

        """
        save_to_yaml_file(self, filename)
        return self

    @staticmethod
    def load_from_file(filename):
        """load objdict from a file

        Args:
            filename: path or name of the file

        Returns:
            ObjDict object
        """
        filetype = json_or_yaml(filename)
        if filetype == "json":
            obj = ObjDict.load_from_json_file(filename)
        elif filetype == "yaml":
            obj = ObjDict.load_from_yaml_file(filename)
        return obj

    @staticmethod
    def load_from_json_file(filename):
        """load objdict from a file

        Args:
            filename: path or name of the file

        Returns:
            ObjDict object
        """
        obj = ObjDict(load_dict_from_json_file(filename))
        return obj

    @staticmethod
    def load_from_yaml_file(filename):
        """load objdict from a file

        Args:
            filename: path or name of the file

        Returns:
            ObjDict object
        """
        obj = ObjDict(load_dict_from_yaml_file(filename))
        return obj

    def deepcopy(self):
        """Make a deep copy of itself

        Returns:
            ObjDict object
        """
        return copy.deepcopy(self)
