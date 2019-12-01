import abc


class BaseCallback(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def on_train_begin(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_train_end(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_epoch_begin(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_epoch_end(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_batch_begin(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_batch_end(self, **kwargs):
        pass


