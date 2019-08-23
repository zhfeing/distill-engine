import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import model_wrapper


join = os.path.join


def _identical_mapping(x):
    return x


def eval_model(
    model_wrapper: model_wrapper.BaseStudentWrapper, 
    data_loader: DataLoader, 
    use_cuda=True
):
    """
    eval_loss_function: calculate loss from output and ground truth, in the function, it will be called as :
        eval_loss_function(pred, label, x), which pred comes from model output
    get_true_pred: get true prediction from output, useful especially when model output a tuple, it will be called as
        true_pred = get_true_pred(module_output)
    detach_pred: detach pred from output, useful when model output a tuple, called as detach_pred(module_output)
    """
    model_wrapper.model.eval()
    if use_cuda:
        model_wrapper.model.cuda()
        map_to_cuda = lambda x: x.cuda()
    else:
        model_wrapper.model.cpu()
        map_to_cuda = lambda x: x

    acc = 0
    loss = 0
    num = len(data_loader)
    for step, (x, y) in enumerate(data_loader):
        batch_size = x.size()[0]
        x = map_to_cuda(x)
        y = map_to_cuda(y)
        pred = model_wrapper.detached_call(x)
        loss += model_wrapper.eval_loss_function(y_s=pred, y_true=y)
        pred = model_wrapper.get_detached_true_predict(pred)
        pred = torch.max(pred, 1)[1]
        acc += (pred == y).sum().float() / batch_size

    loss /= num
    acc /= num
    return loss.item(), acc.item()


class MyDistillLoss(object):
    def __init__(self, T, alpha):
        """Use high temperature for soft target"""
        self._T = T
        self._alpha = alpha

    def _kl_div(self, pred, target):
        R = nn.Softmax(dim=1)(target/self._T)
        Q = nn.Softmax(dim=1)(pred/self._T)
        loss = R*(R.log() - Q.log())
        loss = loss.sum(dim=1)
        loss = loss.mean()

        if torch.isnan(loss):
            print("{}, {}, {}".format(pred, pred.max(), pred.min()))
            raise Exception("[Exception] hard loss is nan")

        return loss
    
    def __call__(self, y_s, y_t, y_true):
        hard_loss = nn.CrossEntropyLoss(reduction='mean')(y_s, y_true)
        soft_loss = self._T * self._T * self._kl_div(y_s, y_t)
        if torch.isnan(soft_loss):
            raise Exception("[Exception] soft loss is nan")

        return self._alpha * hard_loss + (1 - self._alpha) * soft_loss


# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
