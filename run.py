import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch import optim
import distill_engine
import get_data
import model_zoo
import utils
import distill_engine
import example


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store', type=str, default="0")
parser.add_argument('--lr', action='store', type=float, default=1e-2)
parser.add_argument('--epoch', action='store', type=int, default=100)
parser.add_argument('--batch_size', action='store', type=int, default=32)
parser.add_argument('--check_freq', action='store', type=int, default=50)
parser.add_argument('--version', action='store', type=str, default="distill-1.0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

lr = args.lr
weight_decay = 1e-4
epoch = args.epoch
version = args.version
save_model_dir = "model_zoo/model"
batch_size = args.batch_size
check_freq = args.check_freq

# get data
train_set, valid_set, test_set = get_data.import_dataset(
    load_dir="./get_data/data", 
    train_to_cuda=False, 
    test_to_cuda=False
)

train_loader = DataLoader(
    dataset=train_set, 
    batch_size=batch_size, 
    shuffle=True
)
valid_loader = DataLoader(
    dataset=valid_set, 
    batch_size=batch_size, 
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_set, 
    batch_size=batch_size, 
    shuffle=True
)

student_model = model_zoo.Resnet(
    n=7,
    in_channels=3,
    channel_base=16
)
teacher_model = model_zoo.Resnet(
    n=7,
    in_channels=3,
    channel_base=16
)
teacher_model.load_state_dict(torch.load(
    "./model_zoo/model/well-trained-model/model_weights_resnet-tiny-n7.pkl", 
    map_location='cuda:0'
    ))

student_wrapper = example.StudentWrapper(student_model)
teacher_wrapper = example.TeacherWrapper(teacher_model)

optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
my_callback = example.MyCallback(
    save_model_dir=save_model_dir, 
    version=version, 
    check_freq=check_freq
)

dist = distill_engine.Distillation(
    teacher_wrapper=teacher_wrapper, 
    student_wrapper=student_wrapper, 
    train_loader=train_loader, 
    valid_loader=valid_loader, 
    optimizer=optimizer, 
    epoch=epoch,
    cb=my_callback
)

dist.train()
