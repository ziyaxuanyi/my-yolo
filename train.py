from __future__ import division     # 导入精确除法

from utils.logger import *
from utils.parse_config import *

import os      # 必要库
import sys
import time
import datetime
import argparse

import torch      # pytorch相关
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()     # 命令行参数解析
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")     # 训练epochs，在整个数据集上训练一次为一个epoch
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")    # 一次正向和反向传播样本数量，即批次大小
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file") # 模型定义cfg文件路径
    parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")    # 数据集配置文件路径
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")       # 预训练模型权重路径
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")    # 生成batch批次，所用cpu线程数
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")     # 输入图片w,h
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")     # 保存模型间隔
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")    # 在valid数据集上评估一次的间隔
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")    # 是否每10个batch计算一次map
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")     # 是否多尺度训练
    opt = parser.parse_args()
    print(opt)

    # logger = Logger('logs')    # 训练日志

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 判断是否使用gpu

    os.makedirs("output", exist_ok=True)               # 创建相关文件夹
    os.makedirs("checkpoints", exist_ok=True)

    # 数据集配置
    data_config = parse_data_config(opt.data_config)
    train_path = data_config['train']
    valid_path = data_config['valid']

    print('haha')

    
