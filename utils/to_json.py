#-*- coding:utf-8 -*-

import json
import numpy as np
import os
import cv2

#-----save the keypoint of finger to json


def to_json(point_l,point_r,res,path):
    num=0
    dict={}
    dict['filename']=path
    dict['keypoint_l']=[]
    for point in point_l:
        if point is None:
            dict['keypoint_l'].append(None)
        else:
            dict['keypoint_l'].append(point)
    dict['keypoint_r']=[]
    for point_r in point_r:
        if point_r is None:
            dict['keypoint_r'].append(None)
        else:
            dict['keypoint_r'].append(point_r)
    #还要区分左右手
    res.append(dict)
    return res

#------可以存数组进去,也可以存字典进去----
def saveJsonFile(res,json_name):
    dict={}
    dict['value']=[]
    for img in res:
        dict['value'].append(img)
    with open(json_name,'w') as f:
        json.dump(dict,f,sort_keys=True,indent=4,separators=(', ',': '))
        print("the json file has done")