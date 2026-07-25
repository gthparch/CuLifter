class SaSSTransform:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def apply(self, module):
        self.module = module
