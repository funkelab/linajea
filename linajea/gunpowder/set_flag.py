import gunpowder as gp


class SetFlag(gp.BatchFilter):

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
