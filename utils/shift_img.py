#-*- coding:utf-8 -*-
#----shift the hand to the center of img,which can be 
#----easily detect 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import math,os

#---shift the hand_image to the center of img
def shift(img):
    if img is None:
        print('img is None')
        return None
    width=img.shape[1]
    height=img.shape[0]
    #imageToTest = cv2.resize(img,(int(1.3*width),height))
    expand_offset=int(0.3*width)
    output_img = np.ones((height, width+expand_offset, 3)) * 128
    final_img=np.copy(output_img)
    img_h = output_img.shape[0]
    img_w = output_img.shape[1]
    
    save_path='./tmp/'
    num=0
    offset = width % 2   #这个偏移量算的是img的,不是output_img,因为是math.floor(width/2)
    output_img[:, int(img_w / 2 - math.floor(width / 2)):int(
        img_w / 2 + math.floor(width / 2) + offset), :] = img
    #cv2.imwrite('{}pic_{}.jpg'.format(save_path,num),output_img)
    add_loc=expand_offset/2
    num+=1
    return output_img,add_loc

#---locate which frame should be cv2.putText
def find_change(false_num,point_loc,change_img,width,
                    height,save_dir):
    frames=0
    num=0  #有多少帧图片显示了"change"
    for i in false_num:
        cur_loc=point_loc[i]
        index=i+1
        for j in range(20):    #出现多次读取图片再保存的原因是因为该帧后面(20帧)有多帧图片的位置与其相差较大       #可设置第一次读到后就停下来
            nextloc=[]
            if(len(point_loc)-index)>0:
                nextloc.append(point_loc[index])
                if not((cur_loc[0] is None)or(nextloc[0][0] is None)):  #如果该帧没检测到该关键点,则不行
                    if (abs(nextloc[0][0][0]-cur_loc[0][0])>int(0.03*width)):
                        print("the subtract is {}".format(abs(nextloc[0][0][0]-cur_loc[0][0])))
                        img_name=os.path.basename(change_img[frames])
                        img_list = [os.path.join(save_dir, x)
                                        for x in os.listdir(save_dir) if x.endswith('jpg')]
                        for n_img in img_list:
                            if img_name==os.path.basename(n_img):
                                final_img=cv2.imread(n_img)    
                                cv2.putText(final_img,"change",(int(0.5*width),int(0.5*height)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                                cv2.imwrite(os.path.join(save_dir,os.path.basename(n_img)),final_img)
                                print(n_img)
                        num+=1
                        break
            index+=1
        frames+=1
    return num