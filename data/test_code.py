#-*- coding:utf-8 -*-

import cv2
import os
import numpy as np
import math
from PIL import Image

class bbox():
    
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax



def expand_img(img, img_width, img_height,boxes_label):
    
    img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype(
    'float32')
    expand_ratio = np.random.uniform(1, 4)
    height = int(img_height * expand_ratio)
    width = int(img_width * expand_ratio)
    h_off = math.floor(np.random.uniform(0, height - img_height))
    w_off = math.floor(np.random.uniform(0, width - img_width))
    expand_bbox = bbox(-w_off / img_width, -h_off / img_height,  #bbox为一个类
                        (width - w_off) / img_width,
                        (height - h_off) / img_height)
    expand_img = np.ones((height, width, 3))
    expand_img=np.uint8(expand_img * np.squeeze(img_mean))
    expand_img=Image.fromarray(expand_img)
    expand_img.save("expand.jpg")
    expand_img.paste(img, (int(w_off), int(h_off)))
    return expand_img,width,height


label = np.array([1,2])
boxes = np.array([[200, 300, 500, 460],[500,500,500,500]])
boxes_label = np.hstack((label[: , np.newaxis], boxes))

img=Image.open('/home/lj/projects/detection/detectHand/sfd_hand/test_json.jpg')
width, height=img.size

scale=np.array([width, height, width, height])
print(scale.shape)
boxes=boxes_label[:, 1:5] * scale
print(boxes)
expand_im, w, h=expand_img(img, width, height,boxes_label)
expand_im.save("new.jpg")


