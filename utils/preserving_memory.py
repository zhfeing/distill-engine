import os
import torch

def check_mem(gpu_id):
    
    mem = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split("\n")
    
    return mem[gpu_id].split(", ")

def preserve(gpu_id):
    print("[info] preserving memory")
    total, used = check_mem(gpu_id)
    
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    device = torch.device("cuda:{}".format(gpu_id))
    x = torch.zeros(256, 1024, block_mem).to(device)
    x = torch.rand((2,2)).to(device)
