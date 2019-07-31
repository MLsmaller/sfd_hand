#-*- coding:utf-8 -*-

import cv2
import numpy as np
import os 


def detect_frame(hand_point,fps):
    change_frame=[]
    for i in range(len(hand_point)):    #循环每一帧
        if not(i%fps==0):
            continue
        current_loc=hand_point[i]
        index=i+1
        change_frame.append(i)
        for p in range(fps):       #检测后面的15帧
            index_num=[]
            next_loc_list=[]
            if(len(hand_point)-index>0):
                next_loc_list.append(hand_point[index])
                if not((current_loc is None)or(next_loc_list is None)):    #手中有关键点
                    for j in range(len(current_loc)):    #循环每个关键点
                        next_loc=next_loc_list[0]
                        if not((current_loc[j] is None)or(next_loc[j] is None)):         #某个关键点被检测到
                            if ((abs(current_loc[j][0]-next_loc[j][0])<5)and(abs(current_loc[j][1]-next_loc[j][1])<5)):
                                index_num.append(j)
                    if len(index_num)>0:
                        change_frame.append((index,index_num))               
            index+=1

    for i in range(len(change_frame)):
        number=i
        loc=change_frame[i]
        if type(loc) is int and type(change_frame[i+1]) is tuple:
            base_num=loc
        if type(loc) is int:
            continue
        num=loc[0]
        ax=loc[1]
        #initial_point=point_l[num-1]
        #change_point=point_l[num]
        for ax_num in ax:
            hand_point[num][ax_num]=hand_point[base_num][ax_num]
    
    return hand_point
        #print("the base img keypoint is {}".format(point_l[base_num]))
        #print("the change img keypoint is {}".format(point_l[num]))