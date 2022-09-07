"""Provides a gunpowder node that does nothing
"""
import gunpowder as gp


class NoOp(gp.BatchFilter):
    """Gunpowder node that does nothing, passes through data

    Can be used to create optional nodes in pipeline:

    start_of_pipeline +
    (gp.SomeNode(...) if flag else gp.NoOp) +
    rest_of_pipeline
    """
    def __init__(self):
        pass

    def process(self, batch, request):
        pass
