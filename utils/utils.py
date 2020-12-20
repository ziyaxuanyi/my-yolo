from __future__ import division
import torch

def load_classes(path):
    '''
    加载类别标签
    path:类别文件路径
    '''
    fp = open(path, 'r')
    names = fp.read().split('\n')[:]   # 读取每一行
    return names

def to_cpu(tensor):
    return tensor.detach().cpu()

def weights_init_normal(m):
    '''
    权重初始化
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bbox_wh_iou(wh1, wh2):
    '''
    计算iou,不考虑相对位置，相当于计算两个框的相似度，计算时将左上角对齐
    '''
    wh2 = wh2.t()  # 转置
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area   # 这里防止分母为0
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    '''
    给定两个矩形框坐标，计算iou，考虑相对位置的
    '''
    if not x1y1x2y2:   # 如果给定的是中心坐标(x,y)和宽高(w,h),那么计算出四个角的坐标
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:  # 直接给定的就是四个角的坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # 计算两个矩形框交集和并集的面积，画图就知道了
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 交集面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  # 第一个矩形框面积  
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)  # 第二个矩形框面积

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)   # 防止分母为0

    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    '''
    pred_boxes:[num_samples, num_anchors, grid_size, grid_size, 4]
    pred_cls:[num_samples, num_anchors, grid_size, grid_size, num_classes]
    target: [num_object, 6] 其中每一行的6个数据分别为：sample_index, classes_index, x, y, w, h详情见datasets.py中的ListDataset类，sample_index表示属于那个样本图片
    anchors:[num_anchors, 2]  当前尺度情况下的
    ignore_thres: int 忽略阈值
    '''

    # cuda
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)   # num_samples
    nA = pred_boxes.size(1)   # num_anchors
    nC = pred_cls.size(-1)    # num_classes
    nG = pred_boxes.size(2)   # grid_size

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)   # [num_samples, num_anchors, grid_size, grid_size] 全0填充
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)  # 全1填充
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)   # 全0
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)   # 全0
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)    
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)  # [num_samples, num_anchors, grid_size, grid_size, num_classes]  全0填充

    # 转换成相对于方框的位置
    target_boxes = target[:, 2:6] * nG   # [num_samples, 4] 原本是归一化尺度，放大到当前尺度
    gxy = target_boxes[:, :2]      # [num_object, 2]  中心坐标x,y
    gwh = target_boxes[:, 2:]      # [num_object, 2]  框的w,h
    
    # 计算iou最大者
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # 计算每个anchor和目标框的iou，这里用到了广播机制  [num_anchors, num_object, 1]
    best_ious, best_n = ious.max(0)   # iou最大值和位置

    # 获取相应值，调整格式，方便后面计算
    b, target_labels = target[:, :2].long().t()   # [num_object, 2] -->  [2, num_object], 每一列为sample_index, classes_index, b为sample_index，target_labels为classes_index
    gx, gy = gxy.t()    # [2, num_object]  中心坐标
    gw, gh = gwh.t()    # [2, num_object]  w,h
    gi, gj = gxy.long().t()

    # 设置mask
    obj_mask[b, best_n, gj, gi] = 1 # iou最大的那个gird负责预测
    noobj_mask[b, best_n, gj, gi] = 0

    # 当iou超过忽略阈值时，将noobj mask设置为零 ????
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # 论文公式
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # label one-hot编码
    tcls[b, best_n, gj, gi, target_labels] = 1

    #
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
