import abc
import torch.nn as nn
from torch.utils.data import DataLoader
import model_wrapper


class BaseCallback(abc.ABC):
    """
    logs contains keys: "ep", "step", "use_cuda", "total_epoch", "total_step"
    tensors contains keys: "x", "y_true", "y_s", "y_t", "loss"
    student_wrapper: studen model wrapper
    states contain keys: "optimizer"
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def on_train_begin(self, logs: dict):
        pass

    @abc.abstractmethod
    def on_train_end(
        self, 
        logs: dict, 
        student_wrapper: model_wrapper.BaseStudentWrapper, 
        valid_loader: DataLoader
    ):
        pass

    @abc.abstractmethod
    def on_epoch_begin(self, logs: dict, states: dict):
        pass

    @abc.abstractmethod
    def on_epoch_end(
        self, 
        logs: dict, 
        student_wrapper: model_wrapper.BaseStudentWrapper, 
        valid_loader: DataLoader
    ):
        pass

    @abc.abstractmethod
    def on_batch_begin(self, logs: dict, tensors: dict):
        pass

    @abc.abstractmethod
    def on_batch_end(self, logs: dict, tensors: dict, student_wrapper: model_wrapper.BaseStudentWrapper):
        pass


