import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from distill_engine.callback import BaseBlockCallback
from distill_engine.distillation import Distillation
from distill_engine.model_wrapper import BaseBlockStudentWrapper, BaseBlockTeacherWrapper


class BlockDistillation(Distillation):
    def __init__(
        self,
        teacher_wrapper: BaseBlockTeacherWrapper,  # wrapper of teacher model on cpu
        student_wrapper: BaseBlockStudentWrapper,  # wrapper of student model on cpu
        train_loader: DataLoader,       # DataLoader of train set, by default (x, y) on cpu
        valid_loader: DataLoader,       # DataLoader of valid set, by default (x, y) on cpu
        optimizer: Optimizer,           # optimizer of student model
        epoch: int,                     # total training epoch
        cb: BaseBlockCallback,          # callback class
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
        self._tensors["teacher_block_outputs"] = None

    def train(self):
        # call on_train_begin
        self._cb.on_train_begin(
            logs=self._logs,
            states=self._states,
            train_loader=self._train_loader
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
                    y_t = self._t(self._tensors["x"])
                    self._tensors["teacher_block_outputs"] = self._t.outputs
                    self._tensors["teacher_block_outputs"].append(y_t)
                    self._tensors["y_t"] = self._tensors["teacher_block_outputs"][self._s.train_block]

                self._tensors["y_s"] = self._s(self._tensors["x"])

                # calculate undetached distill loss
                self._tensors["loss"] = self._s.distill_loss_function(self._tensors)
                # bp
                self._states["optimizer"].zero_grad()
                self._tensors["loss"].backward()
                self._states["optimizer"].step()
                # update iteration
                self._logs["iter"] += 1
                # clear teacher outputs
                self._t.outputs.clear()
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


