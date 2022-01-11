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

def _check_possible_nested_lists(self, attribute, value):
    try:
        attr.validators.deep_iterable(
            member_validator=_int_list_validator,
            iterable_validator=attr.validators.instance_of(list))(self,attribute, value)
    except:
        attr.validators.deep_iterable(
            member_validator=_list_int_list_validator,
            iterable_validator=attr.validators.instance_of(list))(self, attribute, value)

def maybe_fix_config_paths_to_machine_and_load(config):
    config_dict = toml.load(config)
    config_dict["path"] = config

    if os.path.isfile(os.path.join(os.environ['HOME'], "linajea_paths.toml")):
        paths = load_config(os.path.join(os.environ['HOME'], "linajea_paths.toml"))
        # if paths["DATA"] == "TMPDIR":
            # paths["DATA"] = os.environ['TMPDIR']
        config_dict["general"]["setup_dir"] = config_dict["general"]["setup_dir"].replace(
            "/groups/funke/home/hirschp/linajea_experiments",
            paths["HOME"])
        config_dict["model"]["path_to_script"] = config_dict["model"]["path_to_script"].replace(
            "/groups/funke/home/hirschp/linajea_experiments",
            paths["HOME"])
        config_dict["train"]["path_to_script"] = config_dict["train"]["path_to_script"].replace(
            "/groups/funke/home/hirschp/linajea_experiments",
            paths["HOME"])
        config_dict["predict"]["path_to_script"] = config_dict["predict"]["path_to_script"].replace(
            "/groups/funke/home/hirschp/linajea_experiments",
            paths["HOME"])
        config_dict["predict"]["path_to_script_db_from_zarr"] = config_dict["predict"]["path_to_script_db_from_zarr"].replace(
            "/groups/funke/home/hirschp/linajea_experiments",
            paths["HOME"])
        config_dict["predict"]["output_zarr_prefix"] = config_dict["predict"]["output_zarr_prefix"].replace(
            "/nrs/funke/hirschp/linajea_experiments",
            paths["DATA"])
        for dt in [config_dict["train_data"]["data_sources"],
                   config_dict["test_data"]["data_sources"],
                   config_dict["validate_data"]["data_sources"]]:
            for ds in dt:
                ds["datafile"]["filename"] = ds["datafile"]["filename"].replace(
                    "/nrs/funke/hirschp",
                    paths["DATA"])
    return config_dict
