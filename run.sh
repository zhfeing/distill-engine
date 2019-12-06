#!/bin/bash
# python run_imagenet.py --gpu=1 --num_workers=40 --lr=8e-3 --weight_decay=4e-5 --epoch=100 \
#     --batch_size=512 --alpha=0.3 --check_freq=2 --check_valid_freq=30 \
#     --save_model_dir=/disk3/zhfeing/project/distill-models \
#     --imagenet_root=/nfs2/zhfeing/dataset/ILSVRC \
#     --teacher_filepath=/nfs2/zhfeing/modelzoo/resnet152-b121ed2d.pth \
#     --student_structure=resnet18 --version=v2.0 \
#     --logger_filepath=/disk3/zhfeing/project/distill-models/v2.0_logger.txt \
#     --port=23456 --recover_checkpoint=/disk3/zhfeing/project/distill-models/ckpt_epoch_29_version_v2.0.pth --use_percentage=10


# 208
env CUDA_VISIBLE_DEVICES=1 python run_imagenet.py --gpu=0 --num_workers=40 --lr=8e-3 --weight_decay=4e-5 --epoch=100 \
    --batch_size=512 --alpha=0.3 --check_freq=2 --check_valid_freq=30 \
    --save_model_dir=/home/disk2/zhfeing/project/distill-engine \
    --imagenet_root=/nfs2/zhfeing/dataset/ILSVRC \
    --teacher_filepath=/nfs2/zhfeing/modelzoo/resnet152-b121ed2d.pth \
    --student_structure=resnet18 --version=v2.0 \
    --logger_filepath=/home/disk2/zhfeing/project/distill-engine/v2.0_logger.txt \
    --port=23456 --recover_checkpoint=/home/disk2/zhfeing/project/distill-engine/ckpt_epoch_29_version_v2.0.pth --use_percentage=10

