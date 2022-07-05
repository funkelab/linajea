"""Contains some utility functions to load configuration
"""
import json
import logging
import os
import time

import attr
import toml

logger = logging.getLogger(__name__)


def load_config(config_file):
    """Load toml or json config file into dict

    Args
    ----
    config_file: str
        path to config file, in json or toml format

    Raises
    ------
    ValueError
        If file not in json or toml format
    """
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


def dump_config(config):
    """Write config (as class or dict) to toml file

    Args
    ----
    config: TrackingConfig or dict
    """
    if not isinstance(config, dict):
        config = attr.asdict(config)
    path = os.path.join(config["general"]["setup_dir"],
                        "tmp_configs",
                        "config_{}.toml".format(
                            time.time()))
    logger.debug("config dump path: %s", path)
    with open(path, 'w') as f:
        toml.dump(config, f, encoder=toml.TomlNumpyEncoder())
    return path


def ensure_cls(cl):
    """attrs convert to ensure type of value

    If the attribute is an instance of cls or None, pass, else try
    constructing. This way an instance of an attrs config object can be
    passed or a dict that can be used to construct such an instance.
    """
    def converter(val):
        if isinstance(val, str) and val.endswith(".toml"):
            val = load_config(val)
        if isinstance(val, cl) or val is None:
            return val
        else:
            return cl(**val)
    return converter


def ensure_cls_list(cl):
    """attrs converter to ensure type of values in list

    If the attribute is an list of instances of cls, pass, else try constructing.
    This way a list of instances of an attrs config object can be passed
    or a list of dicts that can be used to construct such instances.

    Raises
    ------
    RuntimeError
        If passed value is not a list
    """
    def converter(vals):
        if vals is None:
            return None

        if isinstance(vals, str) and vals.endswith(".toml"):
            vals = load_config(vals)
            assert len(vals) == 1, "expects dict with single entry"
            vals = list(vals.values())[0]

        if not isinstance(vals, list):
            vals = [vals]
        converted = []
        for val in vals:
            if isinstance(val, cl) or val is None:
                converted.append(val)
            else:
                converted.append(cl(**val))

        return converted
    return converter


def _check_nd_shape(ndims):
    """attrs validator to verify length of list

    Verify that lists representing nD shapes or size have the correct
    length.
    """
    def _check_shape(self, attribute, value):
        if len(value) != ndims:
            raise ValueError("{} must be {}d".format(attribute, ndims))
    return _check_shape

def _check_nested_nd_shape(ndims):
    """attrs validator to verify length of list

    Verify that lists representing nD shapes or size have the correct
    length.
    """
    def _check_shape(self, attribute, value):
        for v in value:
            if len(v) != ndims:
                raise ValueError("{} must be {}d".format(attribute, ndims))
    return _check_shape

"""
_int_list_validator:
    attrs validator to validate list of ints
_list_int_list_validator:
    attrs validator to validate list of list of ints
"""
_int_list_validator = attr.validators.deep_iterable(
    member_validator=attr.validators.instance_of(int),
    iterable_validator=attr.validators.instance_of(list))

_list_int_list_validator = attr.validators.deep_iterable(
    member_validator=_int_list_validator,
    iterable_validator=attr.validators.instance_of(list))

def _check_possible_nested_lists(self, attribute, value):
    """attrs validator to verify list of ints or list of lists of ints
    """
    try:
        attr.validators.deep_iterable(
            member_validator=_int_list_validator,
            iterable_validator=attr.validators.instance_of(list))(self,attribute, value)
    except:
        attr.validators.deep_iterable(
            member_validator=_list_int_list_validator,
            iterable_validator=attr.validators.instance_of(list))(self, attribute, value)


def maybe_fix_config_paths_to_machine_and_load(config):
    """Automatically adapt paths in config to machine the code is run on

    Notes
    -----
    Expects a file "linajea_paths.toml" in the home directory of the users
    with two entries:
    HOME: root of experiments directory
    DATA: root of data directory

    Returns
    -------
    dict
        config with paths adapted to local machine
    """
    config_dict = toml.load(config)
    config_dict["path"] = config

    if os.path.isfile(os.path.join(os.environ['HOME'], "linajea_paths.toml")):
        paths = load_config(os.path.join(os.environ['HOME'], "linajea_paths.toml"))

        if "general" in config_dict:
            config_dict["general"]["setup_dir"] = \
                config_dict["general"]["setup_dir"].replace(
                    "/groups/funke/home/hirschp/linajea_experiments",
                    paths["HOME"])
        if "predict" in config_dict:
            if "output_zarr_dir" in config_dict["predict"]:
                config_dict["predict"]["output_zarr_dir"] = \
                    config_dict["predict"]["output_zarr_dir"].replace(
                        "/nrs/funke/hirschp",
                        paths["DATA"])
        dss = []
        dss += ["train_data"] if "train_data" in config_dict else []
        dss += ["test_data"] if "test_data" in config_dict else []
        dss += ["validate_data"] if "validate_data" in config_dict else []
        dss += ["inference_data"] if "inference_data" in config_dict else []
        for ds in dss:
            ds = (config_dict[ds]["data_sources"]
                  if "data_sources" in config_dict[ds]
                  else [config_dict[ds]["data_source"]])
            for sample in ds:
                if "datafile" in sample:
                    sample["datafile"]["filename"] = \
                        sample["datafile"]["filename"].replace(
                            "/nrs/funke/hirschp",
                            paths["DATA"])
    return config_dict
