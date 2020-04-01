import torch.nn as nn
import torch

from distill_engine.model_wrapper import BaseStudentWrapper, BaseTeacherWrapper
from distill_engine import callback
from distill_engine import utils
from utils import VisdomPlotLogger, MessageLogger


# define teacher and student wrapper
class GoogLeNetStudentWrapper(BaseStudentWrapper):
    def __init__(self, model, alpha):
        super().__init__(model)
        self._distill_loss = utils.MyDistillLoss(alpha)

    def get_true_predict(self, predit):
        """return true prediction from its output"""
        return predit[0]

    @torch.no_grad()
    def eval_loss_function(self, *args, **kwargs):
        """Required parameters:
        y_s: detached student true predict,
        y_true: detached ground truth"""
        y_s = kwargs["y_s"]
        y_true = kwargs["y_true"]
        return nn.CrossEntropyLoss(reduction='mean')(y_s, y_true)

    def distill_loss_function(self, *args, **kwargs):
        """Required parameters:
        y_t: detached teacher output,
        y_s: student output,
        y_true: ground truth"""
        y_t = kwargs["y_t"]
        y_s = kwargs["y_s"]
        y_true = kwargs["y_true"]
        loss_p1 = self._distill_loss(y_s[0], y_t, y_true)
        loss_p2 = self._distill_loss(y_s[1], y_t, y_true)
        loss_p3 = self._distill_loss(y_s[2], y_t, y_true)
        loss = loss_p1 + 0.3 * loss_p2 + 0.3 * loss_p3
        return loss


class ResnetStudentWrapper(BaseStudentWrapper):
    def __init__(self, model, alpha):
        super().__init__(model)
        self._distill_loss = utils.MyDistillLoss(alpha)

    def get_true_predict(self, predit):
        """return true prediction from its output"""
        return predit

    @torch.no_grad()
    def eval_loss_function(self, *args, **kwargs):
        """Required parameters:
        y_s: detached student true predict,
        y_true: detached ground truth"""
        y_s = kwargs["y_s"]
        y_true = kwargs["y_true"]
        return nn.CrossEntropyLoss(reduction='mean')(y_s, y_true)

    def distill_loss_function(self, *args, **kwargs):
        """Required parameters:
        y_t: detached teacher output,
        y_s: student output,
        y_true: ground truth"""
        y_t = kwargs["y_t"]
        y_s = kwargs["y_s"]
        y_true = kwargs["y_true"]
        loss = self._distill_loss(y_s, y_t, y_true)
        return loss


class ResnetTeacherWrapper(BaseTeacherWrapper):
    def __init__(self, model):
        super().__init__(model)

    def get_true_predict(self, predit):
        """return true prediction from its output"""
        return predit


# define callbacks
class TrainCallback(callback.BaseCallback):
    """Callback requires keys:
        "logs" contains keys: "ep", "step", "iter", "total_epoch", "total_step"
        "tensors" contains keys: "x", "y_true", "y_s", "y_t", "loss"
        "student_wrapper": studen model wrapper
        "states" contain keys: "optimizer"
    """
    def __init__(
        self,
        save_model_dir: str,
        version: str,
        check_freq: int,
        check_valid_freq: int,
        recover_checkpoint: dict,
        message_logger: MessageLogger,
        port=9870,
    ):
        """Args:
            save_model_dir: directory to save models
            version: training version
            check_freq: check train loss, acc
            check_valid_freq: check valid loss, acc
            save_checkpoint_freq: freq to save checkpoint
            recover_checkpoint: checkpoint to recover, can be None
            port: visdom port
        """
        super().__init__()
        self._loss = list()                         # loss history
        self._acc = list()                          # accuracy history

        self._save_model_dir = save_model_dir
        self._version = version
        self._check_freq = check_freq
        self._check_valid_freq = check_valid_freq
        self._recover_ckpt = recover_checkpoint

        env = "distill-version-{}".format(version)

        self._message_logger = message_logger
        self._loss_logger = VisdomPlotLogger(
            plot_type="line",
            env=env,
            win="loss",
            port=port,
            opts=dict(title="loss")
        )
        self._acc_logger = VisdomPlotLogger(
            plot_type="line",
            env=env,
            win="acc",
            port=port,
            opts=dict(title="acc")
        )

    def on_train_begin(self, **kwargs):
        logs = kwargs["logs"]
        student_wrapper = kwargs["student_wrapper"]
        states = kwargs["states"]
        if self._recover_ckpt is None:
            self._message_logger.log("[info] start training with {} epochs".format(logs["total_epoch"]))
        else:
            logs["ep"] = self._recover_ckpt["ep"] + 1
            logs["iter"] = self._recover_ckpt["iter"] + 1
            states["optimizer"] = self._recover_ckpt["optimizer"]
            student_wrapper.model.load_state_dict(self._recover_ckpt["student_model"])

    def on_train_end(self, **kwargs):
        self._message_logger.log("[info] train done!")

    def on_epoch_begin(self, **kwargs):
        logs = kwargs["logs"]
        states = kwargs["states"]
        self._message_logger.log("[info] start epoch {:4}".format(logs["ep"]))
        if logs["ep"] % 80 == 0 and logs["ep"] > 0:
            self._message_logger.log("[info] learning rate decreased by 10")
            for param_group in states["optimizer"].param_groups:
                param_group['lr'] /= 10

    def on_epoch_end(self, **kwargs):
        logs = kwargs["logs"]
        student_wrapper = kwargs["student_wrapper"]
        states = kwargs["states"]
        # save checkpoint
        checkpoint = dict(
            ep=logs["ep"],
            iter=logs["iter"],
            optimizer=states["optimizer"],
            student_model=student_wrapper.model.state_dict()
        )
        torch.save(
            checkpoint,
            utils.join(
                self._save_model_dir,
                "ckpt_epoch_{}_version_{}.pth".format(logs["ep"], self._version)
            )
        )

    def on_batch_begin(self, **kwargs):
        logs = kwargs["logs"]
        # take a look per check_freq
        if logs["iter"] % self._check_freq == self._check_freq - 1:
            self._message_logger.log("[info] epoch: {:2}/{:2} step: {:4}/{:4} iteration: {:5}".format(
                logs["ep"],
                logs["total_epoch"],
                logs["step"],
                logs["total_step"],
                logs["iter"]
            ))

    def on_batch_end(self, **kwargs):
        logs = kwargs["logs"]
        tensors = kwargs["tensors"]
        student_wrapper = kwargs["student_wrapper"]
        valid_loader = kwargs["valid_loader"]
        # have a check per check_freq
        if logs["iter"] % self._check_freq == self._check_freq - 1:
            batch_size = tensors["x"].size()[0]
            pred = torch.max(student_wrapper.get_true_predict(tensors["y_s"]), 1)[1]
            acc = (pred == tensors["y_true"]).sum().float() / batch_size
            self._message_logger.log("loss: {:.5f}, acc: {:.4f}".format(
                tensors["loss"],
                acc
            ))
            self._loss_logger.log(logs["iter"], tensors["loss"].item(), name="train loss")
            # self._acc_logger.log(logs["iter"], acc.item(), name="train acc")
        if logs["iter"] % self._check_valid_freq == self._check_valid_freq - 1:
            loss, acc = utils.eval_model(
                student_wrapper=student_wrapper,
                data_loader=valid_loader,
                device=tensors["x"].device
            )
            self._message_logger.log("[info] validate loss: {:.5f}, acc: {:.4f}".format(
                loss,
                acc
            ))
            self._loss_logger.log(logs["iter"], loss, name="valid loss")
            self._acc_logger.log(logs["iter"], acc, name="valid acc")
