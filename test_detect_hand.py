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
from utils.convert_loc import hand_right, hand_left
from utils.shift_img import find_change
from utils.to_json import to_json,saveJsonFile
from utils.keypoint_detect import keypoint_det,draw_point
from utils.with_shake import detect_frame

parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--save_dir', type=str, default='/home/lj/projects/detection/detectHand/sfd_hand/img_keypoint/test1_remove/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/sfd_hand_95000.pth', help='trained model')
parser.add_argument('--cuda', type=bool, default=True,
                    choices=[False, True], help='use gpu')
parser.add_argument('--hand_thresh', default=0.90, type=float,
                    help='Final confidence threshold')
parser.add_argument('--keypoint_thresh', default=0.1, type=float,
                    help='threshold to show the keypoint')
parser.add_argument('--img_dir', type=str, default='/home/lj/cy/openpose/some_code/video_to_image/res/test2_whole/',
                    help='Directory for img to detect')
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
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_list = [os.path.join(args.img_dir, x)
                for x in os.listdir(args.img_dir) if x.endswith('jpg')]
    currentFrame = 0
    axis = 0
    test_list = [os.path.join(args.img_dir, x)
                 for x in os.listdir(args.img_dir) if x.startswith(('086', '087', '088', '089', '09'))]
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
    for path in img_list:
        if(values<247):
            values+=1
            continue
        print(path)
        img = cv2.imread(path)
        width = img.shape[1]
        height = img.shape[0]

        box_l, box_r = detect(net, path, args.hand_thresh,
                              use_cuda)  # 检测到的左/右手框
        point_loc_l, false_num_l, change_img_l, ori_point_l = hand_left(box_l, false_num_l, change_img_l,
                                                                 point_loc_l, img, args.keypoint_thresh, path,
                                                                 args.save_dir, currentFrame,ori_point_l)
        point_loc_r, false_num_r, change_img_r, ori_point_r = hand_right(box_r, false_num_r, change_img_r,
                                                            point_loc_r, img, args.keypoint_thresh, path,
                                                            args.save_dir, currentFrame,ori_point_r)
        if box_l:
            cv2.rectangle(img,box_l[0],box_l[1],(255,0,0),2)
        if box_r:
            cv2.rectangle(img,box_r[0],box_r[1],(255,0,0),2)
        cv2.imwrite(os.path.join('/home/lj/projects/detection/detectHand/sfd_hand/img_keypoint/test1_box/',os.path.basename(path)),img)
        point_l.append(ori_point_l)
        point_r.append(ori_point_r)
        #path为绝对路径
        #res=to_json(ori_point_l,ori_point_r,#,path)   
        currentFrame += 1
        second = round((time.time() - t) / 1000, 4)  # 4是保留4位小数
        print('the img of : {} has took {} s'.format(path, second))
        if currentFrame > FrameTostop:
            break


    #change_frame=[]
    fps=15
    point_l=detect_frame(point_l,fps)
    point_r=detect_frame(point_r,fps)

    values1=0
    point_num=0
    currentFrame1=0
    for path in img_list:
        if(values1<247):
            values1+=1
            continue
        points=[]
        points1=point_l[point_num]
        points2=point_r[point_num]
        points.append(points1)
        points.append(points2)
        img1 = cv2.imread(path)
        if (len(points[0])>0)or((len(points[1])>0)):
            img1=draw_point(img1,points)
            #cv2.imwrite(os.path.join(args.save_dir,os.path.basename(path)),img1)
        point_num+=1
        currentFrame1 += 1
        if currentFrame1 > FrameTostop:
            break


    #json_name=os.path.join(RootPath,"hand_point.json")
    #saveJsonFile(#,json_name)
    num_r = find_change(false_num_r, point_loc_r, change_img_r, width,
                        height, args.save_dir)
    num_l = find_change(false_num_l, point_loc_l, change_img_l, width,
                        height, args.save_dir)

    print("the number of error detect img about hand_right is {}".format(num_r))
    print("the number of error detect img about hand_left is {}".format(num_l))
