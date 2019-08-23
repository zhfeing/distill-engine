import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def eval_model(model, data_loader, eval_loss_function, get_true_pred, detach_pred):
    """
    eval_loss_function: calculate loss from output and ground truth, in the function, it will be called as :
        eval_loss_function(pred, label, x), which pred comes from model output
    get_true_pred: get true prediction from output, useful especially when model output a tuple, it will be called as
        true_pred = get_true_pred(module_output)
    detach_pred: detach pred from output, useful when model output a tuple called as detach_pred(module_output)
    """
    model.eval()
    acc = 0
    loss = 0
    num = len(data_loader)
    for step, (x, y) in enumerate(data_loader):
        batch_size = x.size()[0]
        pred = model(x)
        pred = detach_pred(pred)
        loss += eval_loss_function(pred, y, x)
        pred = get_true_pred(pred)
        pred = torch.max(pred, 1)[1]
        acc += (pred == y).sum().float() / batch_size

    loss /= num
    acc /= num
    return loss.item(), acc.item()