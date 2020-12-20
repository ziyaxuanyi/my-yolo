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

def xywh2xyxy(x):
    '''
    x,y,w,h --> x,y,x.y
    '''
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

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

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


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
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # 计算每个anchor和目标框的iou，这里用到了广播机制  [num_anchors, num_object]
    best_ious, best_n = ious.max(0)   # 根据iou，计算每个真值框由哪个预设anchor预测，best_n为anchors的索引值，best_ious为相应的最大iou值，形状均为[num_object]

    # 获取相应值，调整格式，方便后面计算
    b, target_labels = target[:, :2].long().t()   # [num_object, 2] -->  [2, num_object], 每一列为sample_index, classes_index, b为sample_index，target_labels为classes_index
    gx, gy = gxy.t()    # [2, num_object]  中心坐标
    gw, gh = gwh.t()    # [2, num_object]  w,h
    gi, gj = gxy.long().t()  # [2, num_object] 取正，取到网格格点

    # 设置mask
    obj_mask[b, best_n, gj, gi] = 1 # iou最大的那个gird负责预测
    noobj_mask[b, best_n, gj, gi] = 0   # 与obj_mask意义相反

    # 当iou超过忽略阈值时，将noobj mask设置为零，表示该网格有物体
    for i, anchor_ious in enumerate(ious.t()):    # ious.t() ---> [num_object, num_anchors]
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    
    # 论文公式 实际预测的是tx,ty,为偏移格点的偏移量，tw,th为宽高和anchor宽高比的对数，这里计算的是真值
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # label one-hot编码
    tcls[b, best_n, gj, gi, target_labels] = 1

    # class_mask 表示分类正确的那个位置置为1
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # 计算预测的框和真实框的iou值
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
