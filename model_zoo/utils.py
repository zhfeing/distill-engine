import torch


def total_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())
