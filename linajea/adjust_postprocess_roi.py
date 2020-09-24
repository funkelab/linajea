import daisy

from linajea import parse_limit_roi

def adjust_postprocess_roi(config, roi, use_context=False):
    if 'limit_to_roi_offset' in config['postprocessing'] or \
       'frames' in config['postprocessing']:
        if 'frames' in config['postprocessing']:
            frames = config['postprocessing']['frames']
            begin, end = frames
            if use_context and 'frame_context' in config['extract_edges']:
                begin -= config['extract_edges']['frame_context']
                end += config['extract_edges']['frame_context']
            frames_roi = daisy.Roi(
                    (begin, None, None, None),
                    (end - begin, None, None, None))
            roi = roi.intersect(frames_roi)
        limit_to_roi = parse_limit_roi(config['postprocessing'])
        if limit_to_roi is not None:
            roi = roi.intersect(limit_to_roi)

        if 'limit_to_roi_hard' not in config['postprocessing'] or \
           not config['postprocessing']['limit_to_roi_hard']:
            roi = roi.grow(
                daisy.Coordinate(config['solve']['context']),
                daisy.Coordinate(config['solve']['context']))
    return roi
