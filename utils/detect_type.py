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

def detect_hand(net, img_path, thresh,use_cuda):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    height, width, _ = img.shape
    #print(height,width)
    #max_im_shrink = np.sqrt(1700 * 1200 / (img.shape[0] * img.shape[1]))
    max_im_shrink=1.5
    image = cv2.resize(img, None, None, fx=max_im_shrink,fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)    #这种是对原图像width,height进行缩放,fx,fy表示x和y轴的缩放比例
    #image = cv2.resize(img, (640, 640))
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
    scores = []
    h_type=[]
    num_class=[]
    num_class.append('left')
    num_class.append('right')
    #i=0的时候检测到的是background class
    #i=1是左手,2是右手
    for i in range(1,detections.size(1)):     
        j = 0
        print('the probability of {} is {}'.format(num_class[i-1],detections[0, i, j, 0]))
        #detections[0, i, j, 0]为检测到的手的box的概率,j表示有多少个框(box)
        while detections[0, i, j, 0] >= 0.2:
            scores.append((detections[0, i, j, 0],j))
            j += 1
            #print(scores)
        if len(scores)>0:
            scores=sorted(scores,key=lambda value:value[0],reverse=True)  #以list中tuple的第一个元素进行降序排列
            #sorted()函数不改变原list,sort()函数会改变,默认升序,reverse=True实现降序
        num = 0
        for score in scores:    
            if num>0:      #检测到的包含左/右手的概率最高的那个框
                break
            index=int(score[1])    
            score=score[0]
            pt = (detections[0, i, index, 1:] * scale).cpu().numpy()
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            #j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)   #左上和右下角坐标
            left.append(left_up)
            right.append(right_bottom)
            h_type.append(num_class[i - 1])
            conf = "{:.3f}".format(score)    #保留3位小数
            point = (int(left_up[0]), int(left_up[1] - 5))
            point1=(int(right_bottom[0]-20), int(left_up[1] - 5))
            cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)
            cv2.putText(img, str(num_class[i - 1]), point1, cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            num += 1
        #cv2.imwrite(os.path.join('/home/lj/cy/openpose/piano/test_piano/image/point_dir',os.path.basename(img_path)),img)
        #else:
            #continue  #对于for..else循环,执行完里面的for循环后再执行else,continue结束当前for循环(外面的),执行下次for循环
        #break

    #双手覆盖的问题先不考虑,比较复杂,看其他的
    hand_left=None
    hand_right = None
    #h_type = None
    #再考虑一个手的情况兄弟 
    if (len(left) == 2):
        #下面是假设左手一直在左边,在右手的左边,没有考虑左手在右边(两手交叉),框扩充的时候需要变一下
        #而且这种两个手是假定弹钢琴是横着的情况,如果是竖着的话坐标轴变了,判断两个框是否重合的时候进行比较是不一样的
        for i, left1 in enumerate(left):
            hand_type = h_type[i]
            if (hand_type == 'left'):
                
                left_width = right[i][0] - left1[0]
                left_height = right[i][1] - left1[1]
            #左手框左上坐标(将检测到的框进行扩充,尽量使得手在中心位置,便于检测关键点) ,而且有时候检测到的手不完整
                if(int(left1[0]-0.3*left_width)<0):
                    new_left_x=0
                else:
                    new_left_x=int(left1[0]-0.3*left_width)
                if(int(left1[1]-0.3*left_height)<0):
                    new_left_y=0
                else:
                    new_left_y=int(left1[1]-0.3*left_height)
                new_left_point = (new_left_x, new_left_y)
            #左手框右下坐标
            #一开始左手和右手的框就有重叠,右边框就不扩充了,以防止框中包含两个手
                if (i == 0):
                    j = 1
                else:
                    j = 0
                if not(int(right[i][0]>int(left[j][0]))):  #左手框右下大于右手框左上
                    if(int(right[i][0]+0.3*left_width)>int(left[j][0])):
                        new_left_x1=int(left[j][0])     #如果没重叠左手框右边界最多扩充到右手框的左边界
                    else:
                        new_left_x1=int(right[i][0]+0.3*left_width)
                else:
                    new_left_x1=int(right[i][0])
                
                if(int(right[i][1]+0.3*left_height)>height):
                    new_left_y1=height
                else:
                    new_left_y1=int(right[i][1]+0.3*left_height)
                new_left_point1 = (new_left_x1, new_left_y1)
                  
            if (hand_type == 'right'):
                #right_point=left[i]    #左上角坐标
                right_width = right[i][0] - left1[0]
                right_height = right[i][1] - left1[1]               
                if (i == 0):
                    j = 1
                else:
                    j = 0
            #右手框左上坐标     
            #一开始左手和右手的框就有重叠,左边框就不扩充了    
                if not (int(left1[0]) < int(right[j][0])):
                    if(int(left1[0]-0.3*right_width)<int(right[j][0])):
                        new_right_x=int(right[j][0])
                    else:
                        new_right_x=int(left1[0]-0.3*right_width)
                else:
                    new_right_x=int(left1[0])

                if(int(left1[1]-0.4*right_height)<0):
                    new_right_y=0
                else:
                    new_right_y=int(left1[1]-0.4*right_height)
                new_right_point = (new_right_x, new_right_y)
            #右手框右上坐标 
            #右下坐标
                if(int(right[i][0]+0.3*right_width)>width):
                    new_right_x1=width
                else:
                    new_right_x1=int(right[i][0]+0.3*right_width)
                if(int(right[i][1]+0.4*right_height)>height):
                    new_right_y1=height
                else:
                    new_right_y1=int(right[i][1]+0.4*right_height)
                new_right_point1=(new_right_x1,new_right_y1)
        box1 = (new_left_point, new_left_point1)  #左上和右下角坐标
        box2 = (new_right_point, new_right_point1)
        cv2.rectangle(img, box1[0], box1[1], (0, 0, 255), 2)
        cv2.rectangle(img, box2[0], box2[1], (0, 0, 255), 2)
        cv2.imwrite(os.path.join('/home/lj/projects/detection/detectHand/sfd_hand/test_img/test_res',os.path.basename(img_path)),img)     
        return box1, box2, h_type
        
    elif (len(left) == 1):
        hand_type = h_type[0]
        if (hand_type == 'left'):

            left_width = right[0][0] - left[0][0]
            left_height = right[0][1] - left[0][1]
        #左手框左上坐标
            if(int(left[0][0]-0.3*left_width)<0):
                new_left_x=0
            else:
                new_left_x=int(left[0][0]-0.3*left_width)
            if(int(left[0][1]-0.3*left_height)<0):
                new_left_y=0
            else:
                new_left_y=int(left[0][1]-0.3*left_height)
            new_left_point = (new_left_x, new_left_y)

        #左手框右下坐标
            if(int(right[0][0]+0.3*left_width)>width):
                new_left_x1=width
            else:
                new_left_x1 = int(right[0][0] + 0.3 * left_width)
            
            if(int(right[0][1]+0.3*left_height)>height):
                new_left_y1 = height
            else:
                new_left_y1=int(right[0][1]+0.3*left_height)
            new_left_point1 = (new_left_x1, new_left_y1)            
            box1 = (new_left_point, new_left_point1)
            cv2.rectangle(img, box1[0], box1[1], (0, 0, 255), 2)
            cv2.imwrite(os.path.join('/home/lj/projects/detection/detectHand/sfd_hand/test_img/test_res',os.path.basename(img_path)),img)     
            return box1, hand_right, h_type
            
        else:

            right_width = right[0][0] - left[0][0]
            right_height = right[0][1] - left[0][1]
        #右手框左上坐标
            if(int(left[0][0]-0.3*right_width)<0):
                new_right_x = 0
            else:
                new_right_x=int(left[0][0]-0.3*right_width)
            if(int(left[0][1]-0.3*right_height)<0):
                new_right_y=0
            else:
                new_right_y=int(left[0][1]-0.3*right_height)
            new_right_point = (new_right_x, new_right_y)
        #右手框右下坐标
            if (int(right[0][0] + 0.3 * right_width) > width):
                new_right_x1=width
            else:
                new_right_x1 = int(right[0][0] + 0.3 * right_width)
            
            if(int(right[0][1]+0.3*right_height)>height):
                new_right_y1=height
            else:
                new_right_y1=int(right[0][1]+0.3*right_height)
            new_right_point1 = (new_right_x1, new_right_y1)    
            box2 = (new_right_point, new_right_point1)
            cv2.rectangle(img, box2[0], box2[1], (0, 0, 255), 2)
            cv2.imwrite(os.path.join('/home/lj/projects/detection/detectHand/sfd_hand/test_img/test_res',os.path.basename(img_path)),img)   
            return hand_left, box2, h_type
    else:
        return hand_left, hand_right, h_type
        

