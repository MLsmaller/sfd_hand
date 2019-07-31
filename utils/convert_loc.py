#-*- coding:utf-8 -*-

import cv2
import numpy as np
import os
import shutil
from utils.shift_img import shift
from utils.keypoint_detect import keypoint_det,draw_point
#--- convert the location to the initial img
def convert_loc(finger_point,img,box,offset):
    ori_point=[]
    for i in range(len(finger_point)):
        if not(finger_point[i] is None):   #等于None,是finger_point[i]直接就是None
            ori_point.append((int(finger_point[i][0]+box[0][0]-offset),int(finger_point[i][1]+box[0][1])))
        else:
            ori_point.append(None)
        #cv2.circle(img,ori_point[i],4,(0,0,255),thickness=-1,lineType=cv2.FILLED)
    return ori_point

#--- find the operator to swap finger
def hand_left(box,false_num,change_img,point_loc,
                    img,keypoint_thresh,path,save_dir,currentFrame,ori_point):
    if not(box is None):
        hand=img[box[0][1]:box[1][1],box[0][0]:box[1][0]]
        w=box[1][0]-box[0][0]
        h=box[1][1]-box[1][0]
        center_img,offset=shift(hand)
        center_img = np.array(center_img).astype(np.uint8)   #将数据类型转换为uint8类型,因为opencv读取的图片中数据类型就是Uint8,    #区分数据类型和变量类型
        center_img=cv2.flip(center_img,1)  #将图片翻转,翻转后的坐标相当于沿box的中线翻转了一下
        finger_point,prb=keypoint_det(center_img,keypoint_thresh)
        box_center=w/2+offset   #检测Keypoint时用的box是加上offset的,因此算中线也要加上
        initial_point=[]
        for point in finger_point:
            if not(point is None):
                point=list(point)
            if point is None:
                initial_point.append(None)
            elif point[0]<=box_center:
                point[0]+=int(2*(box_center-point[0]))
                initial_point.append((point[0],point[1]))
            else:
                point[0]-=int(2*(point[0]-box_center))
                initial_point.append((point[0],point[1]))
        ori_point=convert_loc(initial_point,img,box,offset)
        #img=draw_point(img,ori_point)

        point_loc.append((ori_point[13],currentFrame))            
        #cv2.imwrite(os.path.join(save_dir,os.path.basename(path)),img)
        error_dir='./error_img/error1'
        if initial_point.count(None)<10:   #可排除一下模糊或者没有手的图片
            if ((initial_point[3] is None)and((initial_point[4] is None))):
                #currentloc.append(ori_point[14])
                false_num.append(currentFrame)
                change_img.append(path)
                print("the error img is {}".format(path))
                shutil.copy(path,error_dir)
                #cv2.imwrite(os.path.join(args.save_dir,os.path.basename(path)),img)
    else:
        point_loc.append((None,currentFrame))
    return point_loc,false_num,change_img,ori_point  #多返回一个img1便于将两个手的关键点画在同一张图上,另一个手的关键点在此基础上画


def hand_right(box,false_num,change_img,point_loc,
                    img,keypoint_thresh,path,save_dir,currentFrame,ori_point):
    if not(box is None):
        hand=img[box[0][1]:box[1][1],box[0][0]:box[1][0]]
        w=box[1][0]-box[0][0]
        h=box[1][1]-box[1][0]
        center_img,offset=shift(hand)
        center_img = np.array(center_img).astype(np.uint8)   #将数据类型转换为uint8类型,因为opencv读取的图片中数据类型就是Uint8,    #区分数据类型和变量类型
        finger_point,prb=keypoint_det(center_img,keypoint_thresh)
        ori_point=convert_loc(finger_point,img,box,offset)
        #img=draw_point(img,ori_point)
        point_loc.append((ori_point[13],currentFrame))            
        #cv2.imwrite(os.path.join(save_dir,os.path.basename(path)),img)
        error_dir='./error_img/error1'
        if finger_point.count(None)<10:   #可排除一下模糊或者没有手的图片
            if ((finger_point[3] is None)and((finger_point[4] is None))):
                #currentloc.append(ori_point[14])
                false_num.append(currentFrame)
                change_img.append(path)
                print("the error img is {}".format(path))
                shutil.copy(path,error_dir)
                #cv2.imwrite(os.path.join(args.save_dir,os.path.basename(path)),img)
    else:
        point_loc.append((None,currentFrame))
    return point_loc,false_num,change_img,ori_point

