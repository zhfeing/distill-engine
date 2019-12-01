import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model_wrapper import BaseStudentWrapper
import tqdm


join = os.path.join


def _identical_mapping(x):
    return x


@torch.no_grad()
def eval_model(
    student_wrapper: BaseStudentWrapper, 
    data_loader: DataLoader, 
    device: torch.device
):
    """
    eval_loss_function: calculate loss from output and ground truth, in the function, it will be called as :
        eval_loss_function(pred, label, x), which pred comes from model output
    get_true_pred: get true prediction from output, useful especially when model output a tuple, it will be called as
        true_pred = get_true_pred(module_output)
    detach_pred: detach pred from output, useful when model output a tuple, called as detach_pred(module_output)
    """
    student_wrapper.model.eval()
    student_wrapper.model.to(device)

    acc = 0
    loss = 0
    num = len(data_loader)
    for x, y in tqdm.tqdm(data_loader):
        batch_size = x.size()[0]
        x = x.to(device)
        y = y.to(device)
        # get detached output
        pred = student_wrapper(x)
        pred = student_wrapper.get_detached_true_predict(pred)
        # get loss and acc
        loss += student_wrapper.eval_loss_function(y_s=pred, y_true=y)
        pred = torch.max(pred, 1)[1]
        acc += (pred == y).sum().float() / batch_size

    loss /= num
    acc /= num
    return loss.to("cpu").item(), acc.to("cpu").item()


class MyDistillLoss(object):
    def __init__(self, T, alpha):
        """Use high temperature for soft target"""
        self._T = T
        self._alpha = alpha

    # calculate KL Divergence
    def _kl_div(self, pred, target):
        R = nn.Softmax(dim=1)(target/self._T)
        Q = nn.Softmax(dim=1)(pred/self._T)
        loss = R*(R.log() - Q.log())
        loss = loss.sum(dim=1)
        loss = loss.mean()
        return loss
    
    def __call__(self, y_s, y_t, y_true):
        hard_loss = nn.CrossEntropyLoss(reduction='mean')(y_s, y_true)
        soft_loss = self._T * self._T * self._kl_div(y_s, y_t)
        if torch.isnan(soft_loss):
            raise Exception("[Exception] soft loss is nan")
        if torch.isnan(hard_loss):
            raise Exception("[Exception] soft loss is nan")
        return self._alpha * hard_loss + (1 - self._alpha) * soft_loss
