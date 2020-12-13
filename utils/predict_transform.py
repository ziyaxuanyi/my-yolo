from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import cv2
import matplotlib.pyplot as plt

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
    '''
    prediction:网络输出特征图 [B,C,W,H]
    inp_dim:输入图像维度
    anchor:所选取的anchor框维度
    num_classes:识别类别数
    CUDA：是否cuda运算

    返回[B, bbox_number， bbox_attrs]，每一个预测框形式为(x,y,w,h,obj_cls,......)
    '''

    batch_size = prediction.size(0)      # batch_size
    stride = inp_dim // prediction.size(2)    # 步长
    grid_size = inp_dim // stride    # 网格尺寸
    bbox_attrs = 5+num_classes       # [x, y, w, h] 框坐标4个数,1个是否有物体，num_class个分类概率
    num_anchors = len(anchors)       # anchor数量

    # [b, c, h, w] -> [b, bbox_number, bbox_attrs]
    # 将输出结果转化为一个三维矩阵，第一维是batch_size,对应每张图片的结果
    # 每张图片结果是一个2维矩阵，每一行是一个bbox框，预测的是框的bbox_attrs个参数,形式为(x,y,w,h,obj_cls,......)
    # 总共预测bbox_number=grid_size * grid_size * num_anchors个参数
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # 按照比例缩放anchor维度
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # 对(x, y)坐标和obj分数执行sigmoid函数
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # 向坐标中心添加网格偏移
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # 将anchor应用到边界框维度
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:, 2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    # 类别分数使用sigmoid函数
    prediction[:,:,5:5+num_classes] = torch.sigmoid(prediction[:,:,5:5+num_classes])

    # 将坐标框结果调整到与输入图像一致
    prediction[:,:,:4] *= stride

    return prediction



if __name__ == "__main__":

    l = np.arange(19)
    a, b = np.meshgrid(l, l)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    x_y_offset = torch.cat((x_offset, y_offset), 1)  #[361, 2]
    x_y_offset = x_y_offset.repeat(1, 3)   # [361, 6]
    x_y_offset = x_y_offset.view(-1, 2)   #[1083, 2]
    x_y_offset = x_y_offset.unsqueeze(0) #[1, 1083, 2]
    print('haha')

