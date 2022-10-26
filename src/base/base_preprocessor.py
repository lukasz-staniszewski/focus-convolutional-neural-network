from utils.logger import setup_logging
import logging
import os


class BasePreprocessor(object):
    def __init__(self, *args, **kwargs):
        self.img_in_dir_path = kwargs["img_in_dir_path"]
        self.ann_file_path = kwargs["ann_file_path"]
        self.out_dir_path = kwargs["out_dir_path"]
        self.log_dir = os.path.join(self.out_dir_path, "logs")
        self._setup_logging()

    def _setup_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger()
        setup_logging(self.log_dir)

    def preprocess(self):
        raise NotImplementedError
