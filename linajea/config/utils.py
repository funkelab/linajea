def ensure_cls(cl):
    """If the attribute is an instance of cls, pass, else try constructing."""
    def converter(val):
        if isinstance(val, cl):
            return val
        else:
            return cl(**val)
    return converter
