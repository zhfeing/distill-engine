import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from distill_engine import callback
from distill_engine.distillation import Distillation
from distill_engine.model_wrapper import BaseStudentWrapper, BaseTeacherWrapper


class BlockDistillation(Distillation):
    def __init__(
        self,
        teacher_wrapper: BaseTeacherWrapper,  # wrapper of teacher model on cpu
        student_wrapper: BaseStudentWrapper,  # wrapper of student model on cpu
        train_loader: DataLoader,       # DataLoader of train set, by default (x, y) on cpu
        valid_loader: DataLoader,       # DataLoader of valid set, by default (x, y) on cpu
        optimizer: Optimizer,           # optimizer of student model
        epoch: int,                     # total training epoch
        cb: callback.BaseCallback,      # callback class
        use_cuda: bool                  # whether to use cuda
    ):
        super().__init__(
            teacher_wrapper,
            student_wrapper,
            train_loader,
            valid_loader,
            optimizer,
            epoch,
            cb,
            use_cuda
        )

    def train(self):
        # call on_train_begin
        self._cb.on_train_begin(
            logs=self._logs,
            states=self._states
        )
        self._t.model.eval()

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
                if self._use_cuda:
                    self._tensors["x"] = self._tensors["x"].cuda()
                    self._tensors["y_true"] = self._tensors["y_true"].cuda()

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
                    states=self._states
                )
            # call on_epoch_end
            self._cb.on_epoch_end(
                logs=self._logs,
                states=self._states,
                valid_loader=self._valid_loader
            )
            self._logs["ep"] += 1
        # call on_train_end
        self._cb.on_train_end()


