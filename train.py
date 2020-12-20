from __future__ import division     # 导入精确除法

from utils.logger import *
from utils.parse_config import *
from utils.utils import *
from utils.datasets import *
from models import *
from test import evaluate

from terminaltables import AsciiTable

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
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")    # 一次正向和反向传播样本数量，即批次大小
    parser.add_argument("--gradient_accumulations", type=int, default=8, help="number of gradient accums before step")   # 更新梯度前累计梯度数，有效缓解计算机内存小的问题
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
    data_config = parse_data_config(opt.data_config)    # 读取数据集配置文件，主要是路径，其中backup为训练时存放中间结果
    train_path = data_config['train']   # 训练集路径
    valid_path = data_config['valid']   # 测试集路径
    class_names = load_classes(data_config['names'])    # 读取类别标签

    # 建立模型
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)  # 模型参数初始化


    # x = torch.rand(4, 3, 416, 416).to(device)
    # x = model(x)
    # print(x)

    # 加载数据集
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,  # 打乱数据顺序
        num_workers=opt.n_cpu,   # 多进程读取
        pin_memory=True,     # 将生成的Tensor数据存于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些，对内存大小要求较高
        collate_fn=dataset.collate_fn,   # 自定义读取的数据格式，为函数
    )

    optimizer = torch.optim.Adam(model.parameters())

    # 评价指标

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            # print(imgs.size())
            # print(targets.size())
            batches_done = len(dataloader) * epoch + batch_i  # 已经完成batch次迭代

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad = False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # 更新之前计算梯度
                optimizer.step()
                optimizer.zero_grad()
            
            # ----------------
            #   训练日志
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # 每个yolo layer的评价指标日志,并打印
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                # tensorboard_log = []
                # for j, yolo in enumerate(model.yolo_layers):
                #     for name, metric in yolo.metrics.items():
                #         if name != "grid_size":
                #             tensorboard_log += [(f"{name}_{j+1}", metric)]
                # tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # 确定剩下大约多长时间
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)  # 记录已经训练过多少张图片
        
        if epoch % opt.evaluation_interval == 0:
            print('\n---- Evaluating Model ----')
            # 在验证集上评估
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=2,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

    
    
    print('haha')

    
