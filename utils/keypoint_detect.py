#-*- coding:utf-8 -*-
from __future__ import division
import cv2
import time
import numpy as np
import os
import random
import glob as gb


def keypoint_det(frame,threshold):
    protoFile = "./caffe_model/pose_deploy.prototxt"
    weightsFile = "./caffe_model/pose_iter_102000.caffemodel"
    nPoints = 21
    POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)  

    currentFrame=0
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    t = time.time()
    save_path='./tmp'
    inHeight = 368                                         
    inWidth = int(((aspect_ratio*inHeight)*8)//8)   
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)   
    net.setInput(inpBlob)
    output = net.forward()    
    # Empty list to store the detected keypoints
    points = []
    framenum=0
    prb=[]   #存放关键点的概率
    img_axis=[]    #存放哪一帧是换手

    finger_point=[]
    for i in range(nPoints):    
    # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]   
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))  
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap) 
            prb.append(prob)
            #print("the prob of the{} is {}".format(i,prob))
            if prob > threshold :
                    #cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    framenum+=1
                    points.append((int(point[0]), int(point[1])))
            else :
                    points.append(None)


    currentFrame+=1
    #print("img Total time taken : {:.3f}".format(time.time() - t))
    return points,prb
  

def draw_point(img,whole_points):    #同时画两个手的
        POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
        for points in whole_points:
                for pair in POSE_PAIRS:     
                        partA = pair[0]
                        partB = pair[1]
                        if points[partA] and points[partB]: 
                                cv2.line(img, points[partA], points[partB], (0, 255, 255), 2)
                                cv2.circle(img, points[partA], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                                cv2.circle(img, points[partB], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                                
        return img
