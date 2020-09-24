import gunpowder as gp

def parse_limit_roi(config):
    if 'limit_to_roi_offset' in config:
        assert 'limit_to_roi_shape' in config,\
            "Must specify both shape and offset"
        offset = [o if o >= 0 else None \
                  for o in config['limit_to_roi_offset']]
        shape = [s if s >= 0 else None \
                 for s in config['limit_to_roi_shape']]
        limit_to_roi = gp.Roi(offset, shape)
    else:
        limit_to_roi = None
    return limit_to_roi
