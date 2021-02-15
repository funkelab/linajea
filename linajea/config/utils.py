import attr


def ensure_cls(cl):
    """If the attribute is an instance of cls, pass, else try constructing."""
    def converter(val):
        if isinstance(val, cl) or val is None:
            return val
        else:
            return cl(**val)
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
