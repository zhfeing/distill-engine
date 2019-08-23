import callback
import model_wrapper
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import utils


class Distillation(object):
    def __init__(
        self, 
        teacher_wrapper: model_wrapper.BaseTeacherWrapper,  # wrapper of teacher model on cpu
        student_wrapper: model_wrapper.BaseStudentWrapper,  # wrapper of student model on cpu
        train_loader: DataLoader,       # DataLoader of train set, by default (x, y) on cpu
        valid_loader: DataLoader,       # DataLoader of valid set, by default (x, y) on cpu
        optimizer: Optimizer,           # optimizer of student model
        epoch: int,                     # total training epoch
        cb: callback.BaseCallback       # callback class
    ):
        self._t = teacher_wrapper
        self._s = student_wrapper
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._logs = {
            "ep": None,                         # current training epoch id
            "step": None,                       # current step in an epoch
            "total_epoch": epoch,               # total training epoch
            "total_step": len(train_loader),    # total steps in an epoch
            "use_cuda": False                   # whether to use gpu when training
        }
        self._tensors = {
            "x": None,              # the same input of student and teacher models
            "y_true": None,         # ground truth
            "y_s": None,            # student output(undetached)
            "y_t": None,            # teacher output(undetached)
            "loss": None            # loss of a batch(undetached)
        }
        self._states = {
            "optimizer": optimizer  # optimizer of student model
        }
        self._cb = cb
    
    def train(
        self, 
        use_cuda=True, 
        cuda_device=None,           # useless when use_cuda==False
    ):  
        to_device = utils._identical_mapping
        if use_cuda:
            self._logs["use_cuda"] = True
            to_device = lambda x: x.cuda(cuda_device)
            # move models to gpu
            self._t.model.eval()
            self._t.model.cuda(cuda_device)
            self._s.model.cuda(cuda_device)

        # call on_train_begin
        self._cb.on_train_begin(self._logs)

        for self._logs["ep"] in range(self._logs["total_epoch"]):
            # call on_epoch_begin
            self._cb.on_epoch_begin(self._logs, self._states)
            for self._logs["step"], (self._tensors["x"], self._tensors["y_true"]) \
                    in enumerate(self._train_loader):
                # call on_batch_begin
                self._cb.on_batch_begin(self._logs, self._tensors)
                self._s.model.train()
                
                # map to gpu when enabled
                self._tensors["x"] = to_device(self._tensors["x"])
                self._tensors["y_true"] = to_device(self._tensors["y_true"])

                # get raw undetached output of teacher and student
                self._tensors["y_t"] = self._t(self._tensors["x"])
                self._tensors["y_s"] = self._s(self._tensors["x"])

                # calculate undetached distill loss
                self._tensors["loss"] = self._s.distill_loss_function(
                    x=self._tensors["x"],
                    y_t=self._t.get_detached_true_predict(self._tensors["y_t"]),    # detached y_t
                    y_s=self._s.get_true_predict(self._tensors["y_s"]),             # undetached y_s
                    y_true=self._tensors["y_true"]
                )
                # bp
                self._states["optimizer"].zero_grad()
                self._tensors["loss"].backward()
                self._states["optimizer"].step()

                # call on_batch_end
                self._cb.on_batch_end(self._logs, self._tensors, self._s)
            # call on_epoch_end
            self._cb.on_epoch_end(self._logs, self._s, self._valid_loader)
        # call on_train_end
        self._cb.on_train_end(self._logs, self._s, self._valid_loader)


