import abc
from typing import Dict, Any
from torch.utils.data import DataLoader


"""
Copied from distillation:
    logs = {
        "ep": 0,                            # current training epoch id
        "step": 0,                          # current step in an epoch
        "iter": 0,                          # current iteration
        "total_epoch": epoch,               # total training epoch
        "total_step": len(train_loader),    # total steps in an epoch
    }
    tensors = {
        "x": None,              # the same input of student and teacher models
        "y_true": None,         # ground truth
        "y_s": None,            # student output(undetached)
        "y_t": None,            # teacher output(no grad)
        "loss": None            # loss of a batch(undetached)
    }
    states = {
        "student": self._s.model,   # student model
        "optimizer": optimizer      # optimizer of student model
    }
"""


class BaseCallback(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def on_train_begin(self, logs: Dict[str, Any], states: Dict[str, Any]):
        pass

    @abc.abstractmethod
    def on_train_end(self):
        pass

    @abc.abstractmethod
    def on_epoch_begin(self, logs: Dict[str, Any], states: Dict[str, Any]):
        pass

    @abc.abstractmethod
    def on_epoch_end(self, logs: Dict[str, Any], states: Dict[str, Any], valid_loader: DataLoader):
        pass

    @abc.abstractmethod
    def on_batch_begin(self, logs: Dict[str, Any]):
        pass

    @abc.abstractmethod
    def on_batch_end(
        self,
        logs: Dict[str, Any],
        tensors: Dict[str, Any],
        states: Dict[str, Any]
    ):
        pass


