#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import numpy as np
from PIL import Image
import time,os

from utils.augmentations import to_chw_bgr
from torch.autograd import Variable
from data.config import cfg

def detect(net, img_path, thresh,use_cuda):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    height, width, _ = img.shape
    #max_im_shrink = np.sqrt(1400 * 1200 / (img.shape[0] * img.shape[1]))
    max_im_shrink=1.8
    image = cv2.resize(img, None, None, fx=max_im_shrink,fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)    #这种是对原图像width,height进行缩放,fx,fy表示x和y轴的缩放比例
    #image = cv2.resize(img, (800, 600))
    
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x) 
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img1=img.copy()
    left=[]
    right=[]
    score=[]
    for i in range(detections.size(1)):     
        j = 0
        #print(detections[0, i, j, 0])
        while detections[0, i, j, 0] >= 0.8:
            score.append((detections[0, i, j, 0],j))
            j += 1
        if len(score)>0:
            score=sorted(score,key=lambda value:value[0],reverse=True)  #以list中tuple的第一个元素进行降序排列
            #sorted()函数不改变原list,sort()函数会改变,默认升序,reverse=True实现降序
        num=0
        for score in score:    
            if num>1:         #检测到的包含手的概率最高的两个框
                break
            index=int(score[1])    
            score=score[0]

            pt = (detections[0, i, index, 1:] * scale).cpu().numpy()
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            #j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)   #左上和右下角坐标
            left.append(left_up)
            right.append(right_bottom)
            conf = "{:.3f}".format(score)    #保留3位小数
            point = (int(left_up[0]), int(left_up[1] - 5))
            cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 255, 0), 1)
            num+=1
        #else:
            #continue  #对于for..else循环,执行完里面的for循环后再执行else,continue结束当前for循环(外面的),执行下次for循环
        #break

    hand_left=None
    hand_right=None
    #得到左手的框
    if (len(left)==2):
        if left[0][0]<left[1][0]:
            left_point=left[0]
        else:
            left_point=left[1]
        for i in range(len(left)):
            if left_point==left[i]:
                left_axis=i
            else:
                right_axis=i

        left_width=right[left_axis][0]-left_point[0]
        left_height=right[left_axis][1]-left_point[1]
        #print('坐标是({},{})'.format(left_width,left_height))
        #左上坐标
        if(int(left_point[0]-0.3*left_width)<0):
            new_left_x=0
        else:
            new_left_x=int(left_point[0]-0.3*left_width)
        if(int(left_point[1]-0.3*left_height)<0):
            new_left_y=0
        else:
            new_left_y=int(left_point[1]-0.3*left_height)
        new_left_point=(new_left_x,new_left_y)  
        #左下坐标
        #一开始左手和右手的框就有重叠,右边框就不扩充了
        if not(int(right[left_axis][0]>int(left[right_axis][0]))):
            if(int(right[left_axis][0]+0.3*left_width)>int(left[right_axis][0])):
                new_left_x1=int(left[right_axis][0])
            else:
                new_left_x1=int(right[left_axis][0]+0.3*left_width)
        else:
            new_left_x1=int(right[left_axis][0])
            
        if(int(right[left_axis][1]+0.3*left_height)>height):
            new_left_y1=height
        else:
            new_left_y1=int(right[left_axis][1]+0.3*left_height)
        new_left_point1=(new_left_x1,new_left_y1)
        #得到右手的框
        right_point=left[right_axis]   #左上角坐标
        right_width=right[right_axis][0]-right_point[0]
        right_height=right[right_axis][1]-right_point[1]
        #print('坐标是({},{})'.format(right_width,right_height))
        #一开始左手和右手的框就有重叠,左边框就不扩充了
        if not(int(right_point[0])<int(right[left_axis][0])):
            if(int(right_point[0]-0.3*right_width)<int(right[left_axis][0])):
                new_right_x=int(right[left_axis][0])
            else:
                new_right_x=int(right_point[0]-0.3*right_width)
        else:
            new_right_x=int(right_point[0])

        if(int(right_point[1]-0.4*right_height)<0):
            new_right_y=0
        else:
            new_right_y=int(right_point[1]-0.4*right_height)
        new_right_point=(new_right_x,new_right_y)  
        #右下坐标
        if(int(right[right_axis][0]+0.3*right_width)>width):
            new_right_x1=width
        else:
            new_right_x1=int(right[right_axis][0]+0.3*right_width)
        if(int(right[right_axis][1]+0.4*right_height)>height):
            new_right_y1=height
        else:
            new_right_y1=int(right[right_axis][1]+0.4*right_height)
        new_right_point1=(new_right_x1,new_right_y1)
        box1=(new_left_point,new_left_point1)
        box2=(new_right_point,new_right_point1)
        t2 = time.time()
        #print('detect:{} timer:{}'.format(img_path, t2 - t1))
        #cropImg_right=img1[new_right_point[1]:new_right_point1[1],new_right_point[0]:new_right_point1[0]]
        #cv2.imwrite(os.path.join('.', os.path.basename(img_path)), cropImg_right)
        return box1,box2
    elif len(left==1):
        



