import attr
import os
import json
import toml


def load_config(config_file):
    ext = os.path.splitext(config_file)[1]
    with open(config_file, 'r') as f:
        if ext == '.json':
            config = json.load(f)
        elif ext == '.toml':
            config = toml.load(f)
        elif ext == '':
            try:
                config = toml.load(f)
            except ValueError:
                try:
                    config = json.load(f)
                except ValueError:
                    raise ValueError("No file extension provided "
                                     "and cannot be loaded with json or toml")
        else:
            raise ValueError("Only json and toml config files supported,"
                             " not %s" % ext)
    return config


def ensure_cls(cl):
    """If the attribute is an instance of cls, pass, else try constructing."""
    def converter(val):
        if isinstance(val, cl) or val is None:
            return val
        else:
            return cl(**val)
    return converter


def ensure_cls_list(cl):
    """If the attribute is an list of instances of cls, pass, else try constructing."""
    def converter(vals):
        if vals is None:
            return None

        assert isinstance(vals, list), "list of {} expected ({})".format(cl, vals)
        converted = []
        for val in vals:
            if isinstance(val, cl) or val is None:
                converted.append(val)
            else:
                converted.append(cl(**val))

        return converted
    return converter


def _check_nd_shape(ndims):

    def _check_shape(self, attribute, value):
        if len(value) != ndims:
            raise ValueError("{} must be 4d".format(attribute))
    return _check_shape

_int_list_validator = attr.validators.deep_iterable(
    member_validator=attr.validators.instance_of(int),
    iterable_validator=attr.validators.instance_of(list))

_list_int_list_validator = attr.validators.deep_iterable(
    member_validator=_int_list_validator,
    iterable_validator=attr.validators.instance_of(list))
