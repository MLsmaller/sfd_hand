#-*-coding:utf-8 -*-

import csv
import numpy as np
import os
import cv2
import scipy.io as scio

RootPath="/home/data/lj/hand_dataset/training_dataset/training_data/"
dataPath=RootPath+"annotations/"
#dataFile=os.path.join(dataPath,"annotations/Buffy_1.mat")
'''
data=scio.loadmat(dataFile)
print(data)
print(type(data))
print(data["boxes"][0][0])
print(type(data["boxes"][0][0]))

'''
#创建csv文件
fileHeader=["filename","width","height","class","xmin","ymin","xmax","ymax"]
csvFile=open("instance.csv","w")
dict_writer=csv.DictWriter(csvFile,fileHeader)
dict_writer.writeheader()

img_path=RootPath+"images/"
img_list=[os.path.join(img_path,x)
            for x in os.listdir(img_path) if x.endswith('.jpg')]
data_list=[os.path.join(dataPath,x)
            for x in os.listdir(dataPath) if x.endswith('.mat')] 
        

for annotation in data_list:
    data=scio.loadmat(annotation)
    imgFile=os.path.join(img_path,"Buffy_1.jpg")
    img=cv2.imread(imgFile)
    Height=img.shape[0]
    Width=img.shape[1]

    loc=data.get('boxes')
    location=[]
    bbox=[]

    for location in loc[0]:
        x1=[]
        y1=[]
        for point in location[0][0]:
            if len(point):
                if(isinstance(point[0][0],(int,long,float))):
                    x1.append(point[0][0])
                    y1.append(point[0][1])
        xmin=int(min(x1))
        xmax=int(max(x1))
        ymin=int(min(y1))
        ymax=int(max(y1))
        img_name,_=os.path.splitext(os.path.basename(annotation))
        img_name=img_name+'.jpg'
        dict_writer.writerow({"filename":img_name, "width":Width, 
            "height":Height, "class":"hand", "xmin":xmin, "ymin":ymin,
            "xmax":xmax, "ymax":ymax})

csvFile.close()
