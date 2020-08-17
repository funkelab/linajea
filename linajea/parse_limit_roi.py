import gunpowder as gp

def parse_limit_roi(**kwargs):
    if 'limit_to_roi_offset' in kwargs:
        assert 'limit_to_roi_shape' in kwargs,\
            "Must specify both shape and offset"
        offset = [o if o >= 0 else None \
                  for o in kwargs['limit_to_roi_offset']]
        shape = [s if s >= 0 else None \
                 for s in kwargs['limit_to_roi_shape']]
        limit_to_roi = gp.Roi(offset, shape)
    else:
        limit_to_roi = None
    return limit_to_roi
