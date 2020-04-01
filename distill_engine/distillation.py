from distill_engine import callback
from distill_engine.model_wrapper import BaseStudentWrapper, BaseTeacherWrapper
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch
from model_zoo import total_params


class Distillation(object):
    def __init__(
        self,
        teacher_wrapper: BaseTeacherWrapper,  # wrapper of teacher model on cpu
        student_wrapper: BaseStudentWrapper,  # wrapper of student model on cpu
        train_loader: DataLoader,       # DataLoader of train set, by default (x, y) on cpu
        valid_loader: DataLoader,       # DataLoader of valid set, by default (x, y) on cpu
        optimizer: Optimizer,           # optimizer of student model
        epoch: int,                     # total training epoch
        cb: callback.BaseCallback,      # callback class
        device: torch.device            # device for tensors mapping
    ):
        self._t = teacher_wrapper
        self._s = student_wrapper
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._logs = {
            "ep": 0,                            # current training epoch id
            "step": 0,                          # current step in an epoch
            "iter": 0,                          # current iteration
            "total_epoch": epoch,               # total training epoch
            "total_step": len(train_loader),    # total steps in an epoch
        }
        self._tensors = {
            "x": None,              # the same input of student and teacher models
            "y_true": None,         # ground truth
            "y_s": None,            # student output(undetached)
            "y_t": None,            # teacher output(no grad)
            "loss": None            # loss of a batch(undetached)
        }
        self._states = {
            "student": self._s.model,   # student model
            "optimizer": optimizer      # optimizer of student model
        }
        self._cb = cb
        self._device = device
        print(
            "[info] initial distill engine done, ",
            "teacher model has {} parameters and student model has {} parameters".format(
                total_params(self._t.model),
                total_params(self._s.model)
            )
        )

    def train(self):
        # call on_train_begin
        self._cb.on_train_begin(
            logs=self._logs,
            states=self._states
        )

        # map model to device
        self._t.model.to(self._device)
        self._t.model.eval()
        self._s.model.to(self._device)

        while self._logs["ep"] < self._logs["total_epoch"]:
            # call on_epoch_begin
            self._cb.on_epoch_begin(
                logs=self._logs,
                states=self._states
            )
            for self._logs["step"], (self._tensors["x"], self._tensors["y_true"]) \
                    in enumerate(self._train_loader):
                # call on_batch_begin
                self._cb.on_batch_begin(
                    logs=self._logs
                )
                self._s.model.train()

                # map to gpu when enabled
                self._tensors["x"] = self._tensors["x"].to(self._device)
                self._tensors["y_true"] = self._tensors["y_true"].to(self._device)

                # get raw undetached output of teacher and student
                with torch.no_grad():
                    self._tensors["y_t"] = self._t(self._tensors["x"])
                self._tensors["y_s"] = self._s(self._tensors["x"])

                # calculate undetached distill loss
                self._tensors["loss"] = self._s.distill_loss_function(
                    x=self._tensors["x"],
                    y_t=self._t.get_true_predict(self._tensors["y_t"]),    # detached y_t
                    y_s=self._tensors["y_s"],                              # undetached y_s
                    y_true=self._tensors["y_true"]
                )
                # bp
                self._states["optimizer"].zero_grad()
                self._tensors["loss"].backward()
                self._states["optimizer"].step()
                # update iteration
                self._logs["iter"] += 1

                # call on_batch_end
                self._cb.on_batch_end(
                    logs=self._logs,
                    tensors=self._tensors,
                    states=self._states,
                    valid_loader=self._valid_loader
                )
            # call on_epoch_end
            self._cb.on_epoch_end(
                logs=self._logs,
                states=self._states
            )
            self._logs["ep"] += 1
        # call on_train_end
        self._cb.on_train_end()


