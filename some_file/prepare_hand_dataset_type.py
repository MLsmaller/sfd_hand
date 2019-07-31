#-*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import numpy as np
import json
import decimal

from data.config import cfg

root=os.getcwd()
train_json = os.path.join(root, 'json_file/train.json')
val_json = os.path.join(root, 'json_file/val.json')
train_txt = os.path.join(root, 'json_file/train.txt')
val_txt = os.path.join(root, 'json_file/val.txt')
if os.path.exists(train_txt):
    os.remove(train_txt)
if os.path.exists(val_txt):
    os.remove(val_txt)
def read_data(json_name, txt_name):
    img_list = []
    whole_type=[]
    whole_box=[]
    with open(json_name,'r') as f:
        data = json.load(f)
        annotaions = [x for x in data if x['regions']]
        for a in annotaions:
            img_type = []
            img_box=[]
            img_list.append(a['filename'])
            polygons=[x['shape_attribute'] for x in a['regions'].values()]
            for polygon in polygons:
                img_type.append(polygon['type'])
                loc_x = polygon['loc_x']
                loc_y = polygon['loc_y']
                max_x = max(loc_x)
                min_x = min(loc_x)
                min_y = min(loc_y)
                max_y = max(loc_y)
                #对float数据精确的进行四舍五入(round不精确),不保留小数,若保留两位则是("0.00")
                #quantize()函数用于保留小数,先存储为str()类型,然后再转换为int类型
                max_x = decimal.Decimal(str(max_x)).quantize(decimal.Decimal("0"))
                max_y = decimal.Decimal(str(max_y)).quantize(decimal.Decimal("0"))
                min_x = decimal.Decimal(str(min_x)).quantize(decimal.Decimal("0"))
                min_y = decimal.Decimal(str(min_y)).quantize(decimal.Decimal("0"))
                box = [int(min_x), int(min_y), int(max_x), int(max_y)]
                img_box.append(box)
            whole_box.append(img_box)
            whole_type.append(img_type)

    fout = open(txt_name, 'w')
    
    for i in range(len(img_list)):
        fout.write('{} '.format(img_list[i]))
        c_box = whole_box[i]
        c_type = whole_type[i]
        fout.write('{}'.format(len(c_type)))
        for j, box in enumerate(c_box):   #以后为了同时出现下标索引和数值,可以用enumerate(下标可以对应到其他List中的对应元素)
            fout.write(' {} '.format(c_type[j]))
            x1,y1,x2,y2=box
            fout.write('{} {} {} {}'.format(x1, y1, x2, y2))
        fout.write('\n')
    fout.close()
    print("the txt file has done")

if __name__ == '__main__':
    read_data(train_json, train_txt)
    read_data(val_json, val_txt)
    