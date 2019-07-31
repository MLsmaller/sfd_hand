#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import preprocess

EGOHAND_CLASSES=('left','right')

class HandDetection(data.Dataset):   #torch.utils.data.Dataset
    """docstring for WIDERDetection"""
    #自定义Dataset时需要继承data.Dataset并实现__getitem__()和__len__()这两个
    #成员函数,分别是读取数据和返回数据集的长度
    def __init__(self, list_file, mode='train'):
        super(HandDetection, self).__init__()   #self()是一个对象
        self.mode = mode
        self.fnames = []
        self.boxes = []
        #self.h_types=[]
        self.labels = []   #标签,训练集为1,测试集为0

        with open(list_file) as f:
            lines = f.readlines()  #读取所有行的内容,每次读一行
            
        class_id=dict(zip(EGOHAND_CLASSES,range(len(EGOHAND_CLASSES))))
        for line in lines:
            line = line.strip().split()  #line.strip()参数为空,则删除\n,\t等字符,只删除字符串开头或结尾的空格,不删除中间的
            num_hands = int(line[1])    #split()返回列表,以空格作为分割
            box = []
            #h_type=[]
            label = []
            
            for i in range(num_hands):
                xmin = float(line[3 + 5 * i])
                ymin = float(line[4 + 5 * i])
                xmax = float(line[5 + 5 * i])
                ymax = float(line[6 + 5 * i])
                box.append([xmin, ymin, xmax, ymax]) #多少个[]就是几维数组,append之后box就是二维
                #h_type.append(line[2 + 5 * i])  #左上角和右下角坐标
                h_type=line[2 + 5 * i]
                label.append(int(int(class_id[h_type])+1))

            self.fnames.append(line[0])
            self.boxes.append(box)         #boxes中每一个元素为一个list,转换为np.array后都是一个二维数组[[]]or[[],[]]
            #self.h_types.append(h_type)
            self.labels.append(label)     #训练数据中的Img都是包含手的,区别只是左手和右手
            
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):   #__getitem__()函数的参数为index,下标索引
        img, target, h, w = self.pull_item(index)
        return img, target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)   #一般在这里用from PIL import Image和Image.open()来读取图片
            if img.mode == 'L':    #img.mode=='L'表示灰度图片,转为彩色
                img = img.convert('RGB')

            im_width, im_height = img.size
            #对boxes中坐标进行归一化操作
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)  
            #np.array()将list对象转换成为了数组
            label = np.array(self.labels[index])  #np.newaxis()在数组中添加新轴
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            #np.hstack()水平把数组给堆叠起来,与vstack()函数相反,拼接数组的行要相同,拼接后数组行数不变,列增加,tolist()将数组转换为list对象
            #
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()  #xmax>xmin
                assert (target[:, 3] > target[:, 1]).any()  #ymax>ymin
                break 
            else:
                index = random.randrange(0, self.num_samples)

        '''
        #img = Image.fromarray(img)        
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in target:
            bbox = (bbox[:-1] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.show()
        '''
        return torch.from_numpy(img), target, im_height, im_width
        
    #对box进行了归一化操作
    def annotransform(self, boxes, im_width, im_height):  #将box的位置除以宽和高
        boxes[:, 0] /= im_width   #xmin/width ,即x占width的比例为多少
        boxes[:, 1] /= im_height  #y_min/height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


    def pull_image(self,index):
        img_path = self.fnames[index]
        img = Image.open(img_path)
        if img.mode=='L':
            img.convert('RGB')
        img = np.array(img)
        return img


if __name__ == '__main__':
    from data.config import cfg
    dataset = HandDetection(cfg.TRAIN_FILE)
    #for i in range(len(dataset)):
    dataset.pull_item(2)
