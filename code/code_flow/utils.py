"""Module with some utility classes"""

import logging
import logging.config

from sklearn.metrics import (accuracy_score,
                             recall_score,
                             f1_score)
from torch.functional import Tensor


class Log:

    def __init__(self, name, dirpath) -> None:
        self.name = name
        self.logging_dirpath = dirpath
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        file_handler = logging.FileHandler(
            self.logging_dirpath/"logs.log", 'w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s\n%(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter(
            "%(name)s - %(levelname)s\n%(message)s")

        file_handler.setFormatter(file_formatter)
        stream_handler.setFormatter(stream_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.setLevel(logging.DEBUG)

    @ classmethod
    def from_logger(self, dirpath, logger) -> None:
        self.logging_dirpath = dirpath
        self.logger = logger

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def error(self, msg):
        self.logger.error(msg)


class Results():
    """A class to store the results"""

    def __init__(self,
                 predictions: Tensor,
                 labels: Tensor,
                 loss: float) -> None:

        self.pred = predictions.cpu().detach().numpy()
        self.labels = labels.cpu().detach().numpy()
        self.loss = loss

    @property
    def accuracy(self):
        return accuracy_score(self.labels, self.pred)

    @property
    def recall(self):
        return recall_score(self.labels, self.pred)

    @property
    def multiclass_recall(self):
        return recall_score(self.labels, self.pred, average='micro')
    @property
    def multiclass_f1_score(self):
        return f1_score(self.labels, self.pred, average='micro') 
    @property
    def f1_score(self):
        return f1_score(self.labels, self.pred)
