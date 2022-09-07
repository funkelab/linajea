"""Provides gunpowder node to set flag accessible in remaining pipeline
"""
import gunpowder as gp


class SetFlag(gp.BatchFilter):
    """Gunpowder node to create a new array and set it to a specific value

    Attributes
    ----------
    key: gp.ArrayKey
        create a new array with this key
    value: object
        assign this value to new array
    """
    def __init__(self, key, value):

        self.key = key
        self.value = value

    def setup(self):

        self.provides(self.key, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        pass

    def process(self, batch, request):
        batch.arrays[self.key] = \
            gp.Array(self.value,
                     gp.ArraySpec(nonspatial=True))
