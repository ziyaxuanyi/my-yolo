from __future__ import division

import numpy as np
import torchvision
import torch
import cv2

from torch.utils.data import Dataset

def convert(size, box):

    x_center = (box[0]+box[1])/2.0
    y_center = (box[2]+box[3])/2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    
    # print(x, y, w, h)
    return [x,y,w,h]

class Vocdatasets(Dataset):

    def __init__(self, file_path='data/VOC', year='2012', image_set='train', image_size=416, download=False): 

        self.dataes = torchvision.datasets.VOCDetection(file_path, year, image_set, download)    # pytorch自带的读取voc格式的接口
        self.image_shape = [image_size, image_size]
        self.max_objects = 50    # 最大检测对象数
        self.dataes_size = len(self.dataes)
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor', 'person']
    
    def __getitem__(self, index):

        cur_data = self.dataes[index % self.dataes_size]
        img = np.array(cur_data[0])   # 读取的图片是PIL，RGB格式
        
        while len(img.shape) != 3:
            index += 1
            cur_data = self.dataes[index % self.dataes_size]
            img = np.array(cur_data[0])
        
        h, w, _ = img.shape
        # 填充图片至正方形 https://blog.csdn.net/weixin_45482843/article/details/109427426
        # dim_diff = np.abs(h - w)
        # pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else (
        #     (0, 0), (pad1, pad2), (0, 0))
        # np.pad函数见 https://blog.csdn.net/qq_36332685/article/details/78803622
        # 这里pad两位指的是,第几轴,头尾增加pad1,pad2位数值
        input_img = img / 255. # 归一化到[0, 1]
        
        # 这里注意的是,图片填充和resize(),标签也需要做相应操作,不然对不上
        #padded_h, padded_w, _ = input_img.shape
        # cv2.resize()输出默认是3通道
        input_img = cv2.resize(input_img, (416, 416))
        # input_img = np.transpose(input_img, (2, 0, 1))  BGR->RGB
        input_img = torch.from_numpy(input_img).float()

        # 制作标签
        objects = cur_data[1]['annotation']['object']
        obj_list = []
        if not isinstance(objects, list):
            obj_index = self.classes.index(objects['name'])
            x_min = int(objects['bndbox']['xmin'])
            y_min = int(objects['bndbox']['ymin'])
            x_max = int(objects['bndbox']['xmax'])
            y_max = int(objects['bndbox']['ymax'])
            box = [obj_index] + convert((w, h), (x_min, x_max, y_min, y_max))
            obj_list.append(box)
        else:  
            for obj in objects:
                obj_index = self.classes.index(obj['name'])
                x_min = int(obj['bndbox']['xmin'])
                y_min = int(obj['bndbox']['ymin'])
                x_max = int(obj['bndbox']['xmax'])
                y_max = int(obj['bndbox']['ymax'])
                box = [obj_index] + convert((w, h), (x_min, x_max, y_min, y_max))
                obj_list.append(box)
        labels = np.array(obj_list)

        # 初始化标签结果
        filled_labels = np.zeros((self.max_objects, 5))
        # 存储标签,如果没有就为零,超过50就舍弃
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]
                          ] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return input_img, filled_labels
    
    def __len__(self):
        return self.dataes_size

if __name__ == "__main__":

    train_data = Vocdatasets()

    print(train_data)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, num_workers=1, shuffle=True)

    for i, (image, label) in enumerate(train_loader):
        
        print('haha')







        

        

