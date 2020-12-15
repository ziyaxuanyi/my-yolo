from __future__ import division

import _init_paths

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


from predict_transform import predict_transform

def parse_cfg(cfgfile):
    """
    解析cfg文件
    将某一个层用block字典存储，并返回整个blcoks网络参数和结构列表 
    """

    file = open(cfgfile)
    lines = file.read().split('\n')  # 将配置文件一行一行读取，并存储为列表
    lines = [x for x in lines if len(x) > 0] # 去掉空行
    lines = [x for x in lines if x[0] != '#'] # 去掉注释

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':   # 遇到新的block
            if len(block) != 0: # block非空，清除上一个block的内容
                blocks.append(block) # 清除之前，先将上一个block内容保存
                block = {}    # 清空
            block['type'] = line[1:-1].rstrip()  # 记录该block的类型，如net等，rstrip()去除字符串末尾的空格
        else:    # 说明正在读取某一个block内容
            key, value = line.split("=")    # 取出等号两边的字符串，如 batch = 16
            block[key.rstrip()] = value.lstrip()   # 以字典键值对的形式存储，同时去掉空格
    blocks.append(block)    # 存储最后一个block

    return blocks

# 一个空层
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        '''
        一个空层，用以取代route和shortcut，而具体操作直接在整个网络模型的nn.Module对象的forward函数中实现
        '''

# 检测层，用于保存anchors
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
    根据blocks，为每一个block(层)创建nn.Module对象，
    返回一个存储了每一个层nn.Module对象的nn.ModuleList列表
    """
    net_info = blocks[0]  # 存储输入和预处理信息
    module_list = nn.ModuleList()   # 返回的结果
    index = 0  # blocks的索引，帮助实现跨层连接
    prev_filters = 3   # 输入通道数，默认为rgb图像的3通道
    output_filters = []  # 记录每一层输出的filter的数量

    for x in blocks[1:]:
        module = nn.Sequential()

        # 如果是convolutional层
        if(x['type'] == 'convolutional'):
            
            # 获取层参数
            activation = x['activation']  # 获取激活函数类型
            try:
                batch_normalize = int(x['batch_normalize'])  # 是否batch_norm
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x['filters'])   # 输出通道数量
            paddinng = int(x['pad'])   # pad
            kernel_size = int(x['size'])   # 巻积核尺寸
            stride = int(x['stride'])  # 巻积步长

            if paddinng:   # 计算pad
                pad = (kernel_size-1) // 2
            else:
                pad = 0

            # 添加convolutional巻积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias) # pytorch基操
            module.add_module('conv_{0}'.format(index), conv)   # 添加到该层的nn.Sequential()
            
            # 如果有，添加Batch Norm层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)  # pytorch 基操
                module.add_module('batch_norm_{0}'.format(index), bn)
            
            # 如果有，添加激活函数
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True) # pytorch 基操,inplace=True直接对数据修改，节省内存
                module.add_module('leak_{0}'.format(index), activn)
        
        # 如果是upsampling上采样层
        elif(x['type'] == 'upsample'):
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module('unsammple_{0}'.format(index), upsample)
        
        # 如果是route层
        # route层要么就是直接复制某一层输出结果
        # 要么将若干层输出结果在通道维度上拼接
        elif(x['type'] == 'route'):
            
            # 获取拼接的各层索引，数据形式例如:layer = -1,36
            # 如果是负数，例如-3表示从当前层往前倒推3层
            # 如果是正数，例如4表示从最开始向前4层
            # 对于yolov3最多只有两层拼接
            x['layers'] = x['layers'].split(',')

            # 拼接的第一个层
            start = int(x['layers'][0])

            # 如果只有一个层，则结束，相当于直接复制一份
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start -= index
            
            if end > 0:
                end -= index
            
            route = EmptyLayer()   # 空层,拼接操作直接在整个网络模型的nn.Module对象的forward函数中实现
            module.add_module('route_{0}'.format(index), route)

            # 计算输出filters数量
            if end < 0: # 说明拼接两个通道
                filters = output_filters[index + start] + output_filters[index + end]
            else:   # 只是复制一份，并无拼接操作
                filters = output_filters[index + start]
        
        # 如果是shortcut层
        elif(x['type'] == 'shortcut'):
            from_ = int(x['from'])   # 获取shortcut层相对索引，例如-3，表示倒数向前3层的结果
            shortcut = EmptyLayer()   # 空层，具体操作直接在整个网络模型的nn.Module对象的forward函数中实现
            module.add_module('shortcut_{0}'.format(index), shortcut)

            # shortcut不改变输出通道数
        
        # 如果是yolo层，yolo层是检测层
        elif(x['type'] == 'yolo'):
            mask = x['mask'].split(',')  # 获取要使用的anchor的索引。例如mask = 0,1,2表示使用anchor的前三个索引
            mask = [int(m) for m in mask]

            anchors = x['anchors'].split(',') # 获取anchor
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(index), detection)
    
        index += 1
        module_list.append(module)   # 加入到module_list中
        prev_filters = filters
        output_filters.append(filters)
    
    return net_info, module_list, output_filters

class Darknet(nn.Module):
    '''
    yolo darknet 模型定义
    '''
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)  # 读取配置文件
        self.net_info, self.module_list, _ = create_modules(self.blocks) # 生成module_list
    
    def forward(self, x, CUDA=False):
        modules = self.blocks[1:]  # 第一个net块不参与前向传播
        outputs = {}  # 实现跨层连接，需要暂存各层的输出

        write = 0
        for i in range(len(modules)):
            module_type = modules[i]['type']    # 层类型

            # 巻积或者上采样
            if module_type == 'convolutional' or module_type == 'upsample':
                m = self.module_list[i]
                x = self.module_list[i](x)
            
            # route层
            elif module_type == 'route':
                layers = modules[i]['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] -= i
                
                # 只是复制一份数据，而不拼接
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else: # 在通道维度上拼接
                    if layers[1] > 0:
                        layers[1] -= i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)   # (B, C, H, W)
                
            # shortcut层
            elif module_type == 'shortcut':
                from_ = int(modules[i]['from'])
                x = outputs[i-1] + outputs[i + from_]
    
            
            # yolo层
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])  # yolo输入图片w,h一致
                num_classes = int(modules[i]['classes'])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if not write:   # 写入第一个yolo检测头部的结果
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            
            outputs[i] = x
        
        return detections


if __name__ == "__main__":

    # blocks = parse_cfg('cfg/yolov3.cfg')
    # net_info, model_list, output_filters = create_modules(blocks)
    # print(model_list)
    # print(output_filters[-3])

    model = Darknet('cfg/yolov3.cfg')

    inp = torch.rand((1, 3, 416, 416))

    pred = model(inp)

    print(pred)
