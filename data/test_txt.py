#-*-coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np

train_path = '/home/lj/projects/detection/detectHand/sfd_hand/json_file/train.txt'
with open(train_path, 'r') as f:
    lines = f.readlines()

boxs = []
labels = []

for line in lines:
    line = line.strip().split()
    num_hands = int(line[1])
    box = []
    label = []
    hand_type=[]
    for i in range(num_hands):
        x = int(line[3 + i * 5])
        y = int(line[4 + i * 5])
        w = int(line[5 + i * 5])
        h = int(line[6 + i * 5])
        box.append([x, y, x + w, y + h])
        hand_type.append(line[2 + i * 5])
    labels.append(hand_type)
    boxs.append(box)
    
        


print(len(line))
