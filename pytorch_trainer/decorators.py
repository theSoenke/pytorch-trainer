def data_loader(fn):
    '''Decorator that makes a data loader lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _data_loader(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _data_loader
