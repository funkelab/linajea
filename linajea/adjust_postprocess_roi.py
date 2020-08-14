import daisy

def adjust_postprocess_roi(roi, use_context=False, **kwargs):
    if 'limit_to_roi_offset' in kwargs['postprocessing'] or \
       'frames' in kwargs['postprocessing']:
        if 'frames' in kwargs['postprocessing']:
            frames = kwargs['postprocessing']['frames']
            begin, end = frames
            if use_context and 'frame_context' in kwargs['extract_edges']:
                begin -= kwargs['extract_edges']['frame_context']
                end += kwargs['extract_edges']['frame_context']
            frames_roi = daisy.Roi(
                    (begin, None, None, None),
                    (end - begin, None, None, None))
            roi = roi.intersect(frames_roi)
        if 'limit_to_roi_offset' in kwargs['postprocessing']:
            assert 'limit_to_roi_shape' in kwargs['postprocessing'],\
                    "Must specify shape and offset in config file"
            limit_to_roi = daisy.Roi(
                daisy.Coordinate(kwargs['postprocessing']['limit_to_roi_offset']),
                daisy.Coordinate(kwargs['postprocessing']['limit_to_roi_shape']))
            roi = roi.intersect(limit_to_roi)

        if 'limit_to_roi_hard' not in kwargs['postprocessing'] or \
           not kwargs['postprocessing']['limit_to_roi_hard']:
            roi = roi.grow(
                daisy.Coordinate(kwargs['solve']['context']),
                daisy.Coordinate(kwargs['solve']['context']))
    return roi
