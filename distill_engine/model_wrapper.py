import abc
from typing import Iterable, Dict

from torch import Tensor
from torch.nn import Module


# base class of model wrapper
class BaseModelWrapper(abc.ABC):
    def __init__(self, model: Module):
        self.model = model

    def __call__(self, *x):
        """simply return model output"""
        return self._model(*x)

    @abc.abstractmethod
    def get_true_predict(self, raw_output):
        """return true prediction from its output"""
        pass


class BaseTeacherWrapper(BaseModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)


class BaseStudentWrapper(BaseModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)

    # loss function used when distilling
    @abc.abstractmethod
    def distill_loss_function(self, *args, **kwargs):
        pass

    # loss function used when evaluating
    @abc.abstractclassmethod
    def eval_loss_function(self, *args, **kwargs):
        pass


class BaseBlockTeacherWarpper(BaseModelWrapper):
    def __init__(self, model: Module, divide_layers: Iterable[str]):
        """
        Args:
            divide_layers: output of all layer in `divide_layer` will be recorded in
            `self.outputs` after each forward call of `self.model`
        """
        super().__init__(model)
        self.divide_layers = divide_layers
        model_dict: Dict[str, Module] = dict(model.named_modules())

        # register hooks
        self.outputs: Dict[str, Tensor] = dict()
        for layer_name in self.divide_layers:
            module = model_dict[layer_name]
            setattr(module, "name", layer_name)

            def hook(module, input, output):
                self.outputs[module.name] = output

            module.register_forward_hook(hook)


class BaseBlockStudentWarpper(BaseStudentWrapper):
    def __init__(self, model: Module):
        super().__init__(model)



