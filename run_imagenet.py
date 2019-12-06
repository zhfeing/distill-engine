import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import optim

import distill_engine
import get_data
import model_zoo
import distill_on_imagenet
from utils import MessageLogger, preserve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store", type=int, default=-1)
    parser.add_argument("--num_workers", action="store", type=int, default=0)

    parser.add_argument("--lr", action="store", type=float, default=1e-3)
    parser.add_argument("--weight_decay", action="store", type=float, default=4e-5)
    parser.add_argument("--epoch", action="store", type=int, default=100)
    parser.add_argument("--batch_size", action="store", type=int, default=64)
    parser.add_argument("--use_percentage", action="store", type=float)
    parser.add_argument("--T", action="store", type=float)
    parser.add_argument("--alpha", action="store", type=float)

    parser.add_argument("--check_freq", action="store", type=int, default=10)
    parser.add_argument("--check_valid_freq", action="store", type=int, default=10)

    parser.add_argument("--save_model_dir", action="store", type=str)
    parser.add_argument("--imagenet_root", action="store", type=str)
    parser.add_argument("--teacher_filepath", action="store", type=str)
    parser.add_argument("--student_structure", action="store", type=str)
    parser.add_argument("--recover_checkpoint", action="store", type=str, default="")
    parser.add_argument("--logger_filepath", action="store", type=str, default="logger.txt")

    parser.add_argument("--version", action="store", type=str)
    parser.add_argument("--port", action="store", type=int)
    args = parser.parse_args()

    message_logger = MessageLogger(args.logger_filepath)

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:{}".format(args.gpu))
        preserve(args.gpu)

    # get data
    train_dataset = get_data.imagenet.ImagenetDataset(
        imagenet_root=args.imagenet_root, 
        key="train", 
        train_use_percentage=args.use_percentage
    )
    valid_dataset = get_data.imagenet.ImagenetDataset(
        imagenet_root=args.imagenet_root, 
        key="val"
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        shuffle=False
    )

    teacher_model = torchvision.models.resnet152()
    teacher_model.load_state_dict(torch.load(args.teacher_filepath, map_location=device))
    teacher_wrapper = distill_on_imagenet.ResnetTeacherWrapper(teacher_model)

    if args.student_structure == "my_googLeNet":
        student_model = model_zoo.googLeNet.my_googLeNet(class_num=1000)
        student_wrapper = distill_on_imagenet.GoogLeNetStudentWrapper(
            model=student_model, 
            alpha=args.alpha
        )
    elif args.student_structure == "resnet18":
        student_model = torchvision.models.resnet18()
        student_wrapper = distill_on_imagenet.ResnetStudentWrapper(
            model=student_model, 
            alpha=args.alpha
        )
    else:
        raise Exception("student structure {} does not support".format(args.student_structure))

    recover_checkpoint = None
    if args.recover_checkpoint != "":
        recover_checkpoint = torch.load(args.recover_checkpoint, map_location="cpu")
        message_logger.log("[info] load recover check point: {}".format(args.recover_checkpoint))

    optimizer = optim.SGD(
        params=student_model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=args.weight_decay
    )
    my_callback = distill_on_imagenet.TrainCallback(
        save_model_dir=args.save_model_dir, 
        version=args.version, 
        check_freq=args.check_freq, 
        check_valid_freq=args.check_valid_freq, 
        recover_checkpoint=recover_checkpoint, 
        port=args.port, 
        message_logger=message_logger
    )

    dist = distill_engine.Distillation(
        teacher_wrapper=teacher_wrapper, 
        student_wrapper=student_wrapper, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        optimizer=optimizer, 
        epoch=args.epoch,
        cb=my_callback, 
        device=device
    )

    dist.train()
