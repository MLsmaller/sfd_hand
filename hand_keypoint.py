#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import shutil
import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr
from utils.detect_hand1 import detect
from utils.detect_type import detect_hand
from utils.convert_loc import hand_right, hand_left
from utils.shift_img import find_change
from utils.to_json import to_json,saveJsonFile
from utils.keypoint_detect import keypoint_det,draw_point
from utils.with_shake import detect_frame

parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--save_dir', type=str, default='/home/lj/projects/detection/detectHand/sfd_hand/img_keypoint/test1_remove/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/weights1/sfd_hand_60000.pth', help='trained model')
parser.add_argument('--cuda', type=bool, default=True,
                    choices=[False, True], help='use gpu')
parser.add_argument('--hand_thresh', default=0.90, type=float,
                    help='Final confidence threshold')
parser.add_argument('--keypoint_thresh', default=0.1, type=float,
                    help='threshold to show the keypoint')
parser.add_argument('--img_dir', type=str, default="/home/lj/cy/openpose/piano/test_piano/image/temp_dir/",
                    help='Directory for img to detect')
parser.add_argument('--dataset', type=str, default='hand',
                    help='decide how many classes in the network input')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
use_cuda = torch.cuda.is_available() and args.cuda
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
RootPath=os.getcwd()

FrameTostop=1000

if __name__ == '__main__':
    if args.dataset == 'hand':
        net = build_s3fd('test', cfg.HAND.NUM_CLASSES)
    else:
        net = build_s3fd('test', cfg.NUM_CLASSES)
        
    net.load_state_dict(torch.load(args.model))
    net.eval()
    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    file_list=[x for x in os.listdir(args.img_dir) if x.endswith('jpg')]
    #file_list.sort(key=lambda x:int(x[:-4]))  #有时候得到的list里面图片不是顺序排列,x[:-4]表示以倒数第四位.为分割线,按照.左边的数字从小到大排序
    img_list = [os.path.join(args.img_dir, x) for x in file_list ]
    currentFrame = 0
    axis = 0
    test_list = [os.path.join(args.img_dir, x)
                 for x in os.listdir(args.img_dir) if x.endswith(('png','jpg', '089', '09'))]
    t = time.time()
    point_loc_l = []
    point_loc_r = []
    false_num_l = []
    false_num_r = []
    change_img_l = []
    change_img_r = []
    ori_point_l=[]
    ori_point_r=[]
    # width=[0]   #因为在下面的for循环中修改了width,若想在后面的程序使用修改后的width,可以定义为列表[]
    # height=[0]     #不能直接width=0,这样无法修改,也可在前面全局定义global width(if__name__外面),因为if_name__内部也算一个作用域也算是一个变量
    values=0
    res=[]
    point_l=[]
    point_r = []
    h_type=[]   #返回值为一个list的时候,首先要创建一个list变量
    for path in test_list:
        print(path)
        print('\n')
        img = cv2.imread(path)
        if img is None:
            print("error read image")
        width = img.shape[1]
        height = img.shape[0]
        box_l, box_r, h_type = detect_hand(net, path, args.hand_thresh,
                              use_cuda)  # 检测到的左/右手框
        print(box_l, box_r,h_type)
  
'''         point_loc_l, false_num_l, change_img_l, ori_point_l = hand_left(box_l, false_num_l, change_img_l,
                                                                 point_loc_l, img, args.keypoint_thresh, path,
                                                                 args.save_dir, currentFrame,ori_point_l)
        point_loc_r, false_num_r, change_img_r, ori_point_r = hand_right(box_r, false_num_r, change_img_r,
                                                            point_loc_r, img, args.keypoint_thresh, path,
                                                            args.save_dir, currentFrame,ori_point_r) '''
