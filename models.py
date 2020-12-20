import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu

def create_modules(module_defs):
    '''
    从模型定义中构建模型
    '''
    hyperparams = module_defs.pop(0)   # 列表中首项存储的是[net]，网络模型的超参数
    output_filters = [int(hyperparams['channels'])]    # 存储每一层输出的filters，方便构建卷积，pytorch卷积需要知道上一层输出的filter的大小
    module_list = nn.ModuleList()   # 将建立的网络结构存储在列表中，这个是pytorch模型列表，只有这样存储，pytorch才能进行正向反向传播参数更新等操作
    for module_i, module_def in enumerate(module_defs):   #逐步构建，注意pop()已经将[net]删除出列表
        modules = nn.Sequential()     # 每一个block可能包含多个操作，因此存储成nn.Sequential()

        if module_def['type'] == 'convolutional':  # convolutional block
            bn = int(module_def['batch_normalize'])   # 是否batch_normalize
            filters = int(module_def['filters'])    # 卷积通道数
            kernel_size = int(module_def['size'])   # 卷积核尺寸
            pad = (kernel_size - 1) // 2   # 计算填充
            modules.add_module(
                f'conv_{module_i}',
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def['stride']),
                    padding=pad,
                    bias= not bn,
                ),
            )
            if bn:   # batch_normalize
                modules.add_module(f'batch_norm_{module_i}', nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module(f'leaky_{module_i}', nn.LeakyReLU(0.1, inplace=True))
        
        elif module_def['type'] == 'maxpool':   # maxpool
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f'_debug_padding_{module_i}', nn.ZeroPad2d(0, 1, 0, 1))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f'maxpool_{module_i}', maxpool)
        
        elif module_def['type'] == 'upsample':  # unsample
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')   # 使用自定义的unsample
            modules.add_module(f'upsample_{module_i}', upsample)
        
        elif module_def['type'] == 'route':  # route
            layers = [int(x) for x in module_def['layers'].split(',')]  # eg: layers = -1, 61  -1表示往上倒一层，61表示正数第61层
            filters = sum([output_filters[1:][i] for i in layers])  # route为各层在通道维度上拼接，因此输出filters为拼接各layers输出filters之和
            modules.add_module(f'route_{module_i}', EmptyLayer())   # 空层，占位层
        
        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]   # shortcut为对应通道相加，不改变通道数
            modules.add_module(f'shortcut_{module_i}', EmptyLayer())   # 空层占位层
        
        elif module_def['type'] == 'yolo':
            anchor_idx = [int(x) for x in module_def['mask'].split(',')]  # 所选取的anchor索引
            
            # 解析anchors
            anchors = [int(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idx]
            
            num_classes = int(module_def['classes'])  # 分类数
            img_size = int(hyperparams['height'])  # 输入图像尺寸
            ignore_thresh = module_def['ignore_thresh']   # 阈值

            # 定义yolo detection层
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f'yolo_{module_i}', yolo_layer)
        module_list.append(modules)   # 将当前block块建立的modules存入module_list中
        output_filters.append(filters)  # 记录每一层的输出filters，主要作为下一层的建立模型的输入参数
    return hyperparams, module_list

class EmptyLayer(nn.Module):
    '''
    空层，route和shortcut层的占位层，forward实现在darknet的forward中
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(nn.Module):
    '''
    nn.Upsample在pytorch高版本中被弃用，这里插值实现
    '''
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor  # 缩放因子
        self.mode = mode   # 插值方式
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLOLayer(nn.Module):
    '''
    yolo 检测头
    '''
    def __init__(self, anchors, num_classes, ignore_thresh=0.5, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors   # 预设框
        self.num_anchors = len(anchors)  # 预设框个数
        self.num_classes = num_classes   # 分类数
        self.ignore_thres = ignore_thresh    # 阈值
        self.mse_loss = nn.MSELoss()   # 均方损失
        self.bce_loss = nn.BCELoss()  # 二元交叉熵误差损失
        self.obj_scale = 1    # 含有物体的损失权重
        self.noobj_scale = 100  # 不含有物体的损失权重
        self.metrics = {}   # 训练时，用于存储评价指标
        self.img_dim = img_dim
        self.grid_size = 0      # 网格尺寸,初始化为0,在forward执行时，在compute_grid_offsets()函数中正确赋值
    
    def compute_grid_offsets(self, grid_size, cuda=True):
        '''
        计算网格偏移
        '''
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        
        # 对每一个网格计算偏移
        # 生成网格点坐标
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  # 解释见mytest.py
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # 计算放缩anchors
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])  # anchors按比例放缩
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))  # 调整形状
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # 放到cuda上计算
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        #LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        #ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)   # batch大小
        grid_size = x.size(2)   # 网格大小，即特征图大小

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)   # 调整输出形状
            .permute(0, 1, 3, 4, 2)  # ---> (num_samples, self.num_anchors, grid, grid, self.num_classes + 5)
            .contiguous()     # 在内存中变为连续区域
        )

        # 获取输出
        x = torch.sigmoid(prediction[..., 0])  # 中心坐标 x,归一化[0,1]
        y = torch.sigmoid(prediction[..., 1])  # 中心坐标 y,归一化[0,1]
        w = prediction[..., 2]   # 宽
        h = prediction[..., 3]   # 高
        pred_conf = torch.sigmoid(prediction[..., 4])   # 是否含有物体概率
        pred_cls = torch.sigmoid(prediction[..., 5:])   # 各类别的概率

        # 如果gird_size不匹配,重新计算偏移
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # 添加偏移
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # [num_samples, self.num_anchors, grid, grid, 4]
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w   # 论文中的公式
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # --> [num_samples, self.num_anchors*grid*grid, 4] 并扩大到原尺寸
                pred_conf.view(num_samples, -1, 1),  # --> [num_samples, self.num_anchors*grid*grid, 1]
                pred_cls.view(num_samples, -1, self.num_classes),  # --> [num_samples, self.num_anchors*grid*grid, self.num_classes]
            ),
            -1
        )  # --> [num_samples, self.num_anchors*grid*grid, 4+1+self.num_classes]

        # 测试模式
        if targets is None:
            return output, 0
        else:  # 训练，计算loss
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # loss 计算
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # 坐标误差
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  # 是否含有物体误差
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj   # 平衡正负样本的权重
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])   # 分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls  # 总损失

            # 指标
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss

class Darknet(nn.Module):
    '''
    Darknet网络模型
    '''
    
    def __init__(self, config_path, img_size=416):
        '''
        初始化网络
        config_path:网络配置文件路径
        img_size:输入网络图片尺寸
        '''
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)   # 读取cfg文件，返回各block块定义
        self.hyperparams, self.module_list = create_modules(self.module_defs)   # 根据block定义建立module_list
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], 'metrics')]  # 记录yolo层
        self.img_size = img_size # 图片尺寸
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
    
    def forward(self, x, targets=None):
        '''
        利用self.module_list里面定义好的module进行前向传播
        '''
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []   # 记录结果
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':  # 在通道上拼接
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def['layers'].split(',')], 1)
            elif module_def['type'] == 'shortcut':  # 层结果相加
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss   # 累计yolo层的loss
                yolo_outputs.append(x)   # 记录yolo输出
            layer_outputs.append(x)  # 记录各层输出
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))  # 合并输出
        return yolo_outputs if targets is None else (loss, yolo_outputs)