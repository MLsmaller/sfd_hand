#-*-coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import csv

from data.config import cfg

if not os.path.exists('./data'):
    os.makedirs('./data')

TRAIN_ROOT = os.path.join(cfg.HAND.DIR, 'images', 'train')
TEST_ROOT = os.path.join(cfg.HAND.DIR, 'images', 'test')


def generate_file(csv_file, target_file, root):
    filenames = []
    bboxes = []
    num=0
    with open(csv_file, 'rb') as sd:
        lines = csv.DictReader(sd)     #以字典形式读取
        for line in lines:
            filenames.append(os.path.join(root, line['filename']))
            #print(filenames)
            bbox = [int(line['xmin']), int(line['ymin']),
                    int(line['xmax']), int(line['ymax'])]
            bboxes.append(bbox)
            num+=1
            #print(bboxes)
            #print(num)
  
    #转换为数组形式
    filenames = np.array(filenames)  #存放图片名称
    bboxes = np.array(bboxes)
    #print((bboxes[4]))
    uniq_filenames = np.unique(filenames)  #np.unique()函数去除其中重复的元素并按元素大小返回一个新的无元素重复的元组或列表
    #上面for line 每读一行数据就存入一次filename,因此有几个手就存进去了几次图片名字,然后通过np.where()找到array中图片位置的索引
    fout = open(target_file, 'w')

    for name in uniq_filenames:
        idx = np.where(filenames == name)[0]   #filenames中图片有重复,[0]是只取索引中第一个数组
        print(idx)
        bbox = bboxes[idx]  #bboxes中存放的手的位置与filanames数组中的位置对应
        #print(bbox)
        fout.write('{} '.format(name))
        fout.write(('{} ').format(len(bbox)))  #有多少个手就有多少个坐标
        for loc in bbox:
            x1, y1, x2, y2 = loc
            fout.write('{} {} {} {} '.format(x1, y1, x2, y2))
        fout.write('\n')
    fout.close()


if __name__ == '__main__':
    train_csv_file = os.path.join(TRAIN_ROOT, 'train_labels.csv')
    test_csv_file = os.path.join(TEST_ROOT, 'test_labels.csv')
    generate_file(train_csv_file, cfg.HAND.TRAIN_FILE, TRAIN_ROOT)
    generate_file(test_csv_file, cfg.HAND.VAL_FILE, TEST_ROOT)
