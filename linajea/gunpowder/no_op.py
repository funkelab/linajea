import gunpowder as gp


class NoOp(gp.BatchFilter):

    def __init__(self):
        pass

    def process(self, batch, request):
        pass
