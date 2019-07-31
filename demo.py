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
from utils.detect_hand import detect
from utils.shift_img import shift
from utils.keypoint_detect import keypoint_det,draw_point


parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--save_dir', type=str, default='./tm_output',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/sfd_hand_150000.pth', help='trained model')
parser.add_argument('--cuda', type=bool, default=True,
                    choices=[False, True], help='use gpu')
parser.add_argument('--hand_thresh', default=0.86, type=float,
                    help='Final confidence threshold')
parser.add_argument('--keypoint_thresh',default=0.1,type=float,
                    help='threshold to show the keypoint')
parser.add_argument('--img_dir',type=str,default='./tm_img',
                    help='Directory for img to detect')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
use_cuda = torch.cuda.is_available() and args.cuda
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


if __name__ == '__main__':
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_list = [os.path.join(args.img_dir, x)
                for x in os.listdir(args.img_dir) if x.endswith('jpg')]
    currentFrame=0
    axis=0
    for path in img_list:
        if path=='./tm_img/0340.jpg':
            img=cv2.imread(path)
            t=time.time()
            box_l,box_r=detect(net, path, args.hand_thresh,use_cuda)   #检测到的左/右手框
            if not(box_l is None):
                hand_left=img[box_l[0][1]:box_l[1][1],box_l[0][0]:box_l[1][0]]
                hand_right=img[box_r[0][1]:box_r[1][1],box_r[0][0]:box_r[1][0]]
                width_r=box_r[1][0]-box_r[0][0]
                height_r=box_r[1][1]-box_r[1][0]
                left,offset=shift(hand_left)
                right,offset=shift(hand_right)
                right = np.array(right).astype(np.uint8)   #将数据类型转换为uint8类型,因为opencv读取的图片中数据类型就是Uint8,    #区分数据类型和变量类型
                finger_point,prb=keypoint_det(right,args.keypoint_thresh)
                ori_point=[]            #转换为原始图中的坐标
                
                for i in range(len(finger_point)):
                    if not(finger_point[i] is None):   #等于None,是finger_point[i]直接就是None
                        ori_point.append((int(finger_point[i][0]+box_r[0][0]-offset),int(finger_point[i][1]+box_r[0][1])))
                    else:
                        ori_point.append(None)
                    cv2.circle(img,ori_point[i],4,(0,0,255),thickness=-1,lineType=cv2.FILLED)
                
                
                img=draw_point(img,ori_point)

                finger_num=[13,17]
                currentloc=[]
                nextloc=[]
                #frame_num=[]
                
                error_dir='./error_img'
                frame_num=-10
                if ((finger_point[3] is None)and((finger_point[4] is None))):
                    currentloc.append(ori_point[13])
                    currentloc.append(ori_point[17])
                    axis+=2
                    print("the error img is {}".format(path))
                    #shutil.copy(img,error_dir)
                    frame_num=currentFrame
                
                if(currentFrame==(frame_num+1)):
                    nextloc.append(ori_point[13])
                    nextloc.append(ori_point[17])
                    #判断当前帧和上一帧之间横坐标位置的变化
                    if(((nextloc[0][0]-currentloc[axis-2][0])>10)and((nextloc[1][0]-currentloc[axis-1][0])>10)):
                        cv2.putText(img,'change',(box_r[0][0]+int(0.5*width_r),box_r[0][1]+height_r),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

                cv2.imwrite(os.path.join(args.save_dir,os.path.basename(path)),img)

                if ((finger_point[3] is None)and((finger_point[4] is None))):
                    shutil.copy(os.path.join(args.save_dir,os.path.basename(path)),error_dir)
                #cv2.imwrite('./tmp/final.jpg',img)
                print('the img of : {} has took {} s'.format(path,time.time()-t))
                currentFrame+=1
        

