class BasePreprocessor(object):
    def __init__(self, config):
        self.config = config

    def preprocess(self):
        raise NotImplementedError
