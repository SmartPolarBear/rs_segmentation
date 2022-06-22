def freeze_component(comp):
    for p in comp.parameters():
        p.stop_gradient=True

def unfreeze_component(comp):
    for p in comp.parameters():
        p.stop_gradient=False

class Freezer:
    def __init__(self, cmps):
        self.cmps = cmps

    def __enter__(self):
        print("Freeze {} components".format(len(self.cmps)))
        for c in self.cmps:
            freeze_component(c)
        return self

    def __exit__(self, exceptionType, exceptionVal, trace):
        print("Unfreeze {} components".format(len(self.cmps)))
        for c in self.cmps:
            unfreeze_component(c)