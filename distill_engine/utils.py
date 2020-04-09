import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from distill_engine.model_wrapper import BaseStudentWrapper
import tqdm


join = os.path.join


def _identical_mapping(x):
    return x


@torch.no_grad()
def eval_model(
    student_wrapper: BaseStudentWrapper,
    data_loader: DataLoader,
    use_cuda: bool
):
    """
    eval_loss_function: calculate loss from output and ground truth, in the function, it will be called as :
        eval_loss_function(pred, label, x), which pred comes from model output
    get_true_pred: get true prediction from output, useful especially when model output a tuple, it will be called as
        true_pred = get_true_pred(module_output)
    detach_pred: detach pred from output, useful when model output a tuple, called as detach_pred(module_output)
    """
    if use_cuda:
        student_wrapper.model.cuda()
    student_wrapper.model.eval()

    acc = 0
    loss = 0
    num = len(data_loader)
    for x, y in tqdm.tqdm(data_loader):
        batch_size = x.size()[0]
        x = x.cuda()
        y = y.cuda()
        # get detached output
        pred = student_wrapper(x)
        pred = student_wrapper.get_true_predict(pred)
        # get loss and acc
        loss += student_wrapper.eval_loss_function(y_s=pred, y_true=y)
        pred = torch.max(pred, 1)[1]
        acc += (pred == y).sum().float() / batch_size

    loss /= num
    acc /= num
    return loss.to("cpu").item(), acc.to("cpu").item()


class DistillLoss(object):
    def __init__(self, alpha):
        self._alpha = alpha

    # calculate KL Divergence
    # def _kl_div(self, pred, target):
    #     R = torch.softmax(target/self._T, dim=1)
    #     Q = torch.log_softmax(pred/self._T, dim=1)
    #     loss = F.kl_div(input=Q, target=R, reduction="batchmean")
    #     return self._T * self._T * loss

    def _l2_loss(self, pred, target):
        loss = F.mse_loss(pred, target, reduction="mean")
        return loss

    def __call__(self, y_s, y_t, y_true):
        hard_loss = F.cross_entropy(input=y_s, target=y_true, reduction="mean")
        # soft_loss = self._kl_div(y_s, y_t)
        soft_loss = self._l2_loss(y_s, y_t)
        return self._alpha * hard_loss + (1 - self._alpha) * soft_loss
