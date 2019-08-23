import abc


# base class of model wrapper
class BaseModelWrapper(abc.ABC):
    def __init__(self, model):
        self._model = model
    
    # get model
    @property
    def model(self):
        return self._model
    
    def __call__(self, *input):
        """simply return model output"""
        return self._model(*input)

    @abc.abstractmethod
    def detached_call(self, *input):
        """simply return model detached output"""
        pass

    @abc.abstractmethod
    def get_true_predict(self, predit):
        """return true prediction from its output"""
        pass

    @abc.abstractclassmethod
    def get_detached_true_predict(self, predit):
        """return true detached model prediction from graph"""
        pass


class BaseTeacherWrapper(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)


class BaseStudentWrapper(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    # loss function used when distilling
    @abc.abstractmethod
    def distill_loss_function(self, *args, **kwargs):
        pass
    
    # loss function used when evaluating
    @abc.abstractclassmethod
    def eval_loss_function(self, *args, **kwargs):
        pass