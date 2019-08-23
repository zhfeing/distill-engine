import callback
import model_wrapper
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import utils


class Distillation(object):
    def __init__(
        self, 
        teacher_wrapper: model_wrapper.BaseTeacherWrapper, 
        student_wrapper: model_wrapper.BaseStudentWrapper, 
        train_loader: DataLoader, 
        valid_loader: DataLoader, 
        optimizer: Optimizer,
        epoch: int, 
        cb: callback.BaseCallback
    ):
        self._t = teacher_wrapper
        self._s = student_wrapper
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._logs = {"ep": None, "step": None, "total_epoch": epoch, "total_step": len(train_loader), "use_cuda": False}
        self._tensors = {"x": None, "y_true": None, "y_s": None, "y_t": None, "loss": None}
        self._states = {"optimizer": optimizer}
        self._cb = cb
    
    def train(
        self, 
        use_cuda=True,       # useless when use_cuda==False
        cuda_device=None, 
    ):  
        to_device = utils._identical_mapping
        if use_cuda:
            self._logs["use_cuda"] = True
            to_device = lambda x: x.cuda(cuda_device)
            # move models to gpu
            self._t.model.eval()
            self._t.model.cuda(cuda_device)
            self._s.model.cuda(cuda_device)

        self._cb.on_train_begin(self._logs)

        for self._logs["ep"] in range(self._logs["total_epoch"]):
            self._cb.on_epoch_begin(self._logs, self._states)
            for self._logs["step"], (self._tensors["x"], self._tensors["y_true"]) \
                    in enumerate(self._train_loader):
                self._cb.on_batch_begin(self._logs, self._tensors)
                self._s.model.train()
                
                self._tensors["x"] = to_device(self._tensors["x"])
                self._tensors["y_true"] = to_device(self._tensors["y_true"])

                self._tensors["y_t"] = self._t(self._tensors["x"])
                self._tensors["y_s"] = self._s(self._tensors["x"])
                self._tensors["loss"] = self._s.distill_loss_function(
                    x=self._tensors["x"],
                    y_t=self._t.get_detached_true_predict(self._tensors["y_t"]), 
                    y_s=self._s.get_true_predict(self._tensors["y_s"]), 
                    y_true=self._tensors["y_true"]
                )
                self._states["optimizer"].zero_grad()
                self._tensors["loss"].backward()
                self._states["optimizer"].step()

                self._cb.on_batch_end(self._logs, self._tensors, self._s)
            self._cb.on_epoch_end(self._logs, self._s, self._valid_loader)
        self._cb.on_train_end(self._logs, self._s, self._valid_loader)


