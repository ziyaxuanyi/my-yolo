import numpy as np
from PIL import Image

import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.augmentations import horisontal_flip

def resize(image, size):
    '''
    双线性插值调整图片尺寸
    '''
    image = F.interpolate(image.unsqueeze(0), size=size, mode='nearest').squeeze(0)
    return image

def pad_to_square(img, pad_value):
    '''
    将图片用pad_value：int的值填充至正方形
    '''
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2  # 上下左右需要填充的数目
    
    # 填充，具体查看pytorch的pad()函数
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, 'constant', value=pad_value)

    return img, pad

class ListDataset(Dataset):
    '''
    读取图片和标注文件，并做成pytorch数据类
    '''
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True, dataset_format='VOC'):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()    # 读取图片文件路径
        
        if dataset_format == 'VOC':  # 标注文件路径和图片路径相似，替换掉差别
            self.label_files = [
                path.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                for path in self.img_files
            ]
        elif dataset_format == 'COCO':
            self.label_files = [     # 标注文件路径和图片路径相似，替换掉差别
                path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                for path in self.img_files
            ]

        self.img_size = img_size   # 图片尺寸
        self.max_objects = 100    # 一张图片上最多标注的物体数目，目前没有用到
        self.augment = augment   # 是否数据增强
        self.multiscale = multiscale   # 是否多尺度训练
        self.normalized_labels = normalized_labels  # 是否归一化label
        self.min_size = self.img_size - 3*32  # 多尺度图片最小尺寸
        self.max_size = self.img_size + 3*32  # 多尺度图片最大尺寸
        self.batch_count = 0   # 当前第多少个batch的，主要用于控制多尺度训练
    
    def __getitem__(self, index):
        '''
        函数重载，根据index，取出一个样本图片和其对应的label
        '''
        ###############
        # 图片
        img_path = self.img_files[index % len(self.img_files)].rstrip()  # 图片路径
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))  # PIL库读取图片，并转为Tensor
        if len(img.shape) != 3:   # 如果图片小于3通道，即并非RGB图像，填充至3个通道
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)   # 是否归一化label
        # 填充至正方形，等比例缩放
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        ################
        # 标签
        label_path = self.label_files[index % len(self.img_files)].rstrip()   # 标签路径
        targets = None
        if os.path.exists(label_path):
            # [num_object, classes_index, x, y, w, h] 其中x,y,w,h为yolo格式的归一化数据：(x,y)为框中心坐标，w,h为框的宽度和高度
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # 获取原始图像上的框坐标
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # 填充
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # 填充后框的归一化yolo格式的[x,y,w,h]
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        
        # 使用数据增强
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        
        return img_path, img, targets
    
    def collate_fn(self, batch):
        '''
        组装一个batch的数据，自定义读取的数据格式
        返回：
        paths: 图片路径
        imgs：[batch_size, C, W, H]
        targets: [sample_index, classes_index, x, y, w, h]
        '''
        paths, imgs, targets = list(zip(*batch))   # 获取一个batch的path, imgs, targets
        # 去掉空的目标
        targets = [boxes for boxes in targets if boxes is not None]
        
        # 记录每一个样本的index
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        
        # 如果开启多尺度训练，则每10各batch，重新选择一次图片尺寸
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))  # 从[min_size, max_size]随机选择一个图片尺寸
        
        # 将图片resize到输入的尺寸
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        
        return paths, imgs, targets
    
    def __len__(self):
        return len(self.img_files)