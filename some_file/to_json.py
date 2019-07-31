#-*- coding:utf-8 -*-

import numpy as np
import cv2
import os
import scipy.io as scio
import skimage.draw
import skimage.io
import json
import random
import shutil

#-----change the egohand dataset to the json file
RootPath=os.getcwd()
save_path='/home/data/cy/egohand'
class_list = [x for x in os.listdir(save_path) if not (x.endswith(('train', 'val')))]
#class_list.sort(key=lambda x:str(x[0:2]))   #文件夹也按照字母顺序排一下
class_list.sort()     #默认是按照字典进行排序,就是从a,b,c这样排序
#对于endswith和startswith()函数,当里面要判断多个字符串时,需要用括号来圈住eg: endswith('jpg'),endswith(('jpg','png'))
train_json=os.path.join(save_path,'train.json')
val_json=os.path.join(save_path,'val.json')
train=[]
val=[]
img_num=0
ori_img=[]
#os.walk()函数首先遍历根目录,再遍历其中的子目录
for imgDir in class_list:
    #print(imgDir)
    c_dir=os.path.join(save_path,imgDir)
    imgList=[os.path.join(c_dir,x) for x in os.listdir(c_dir)
                if x.endswith('.jpg')]
    img_num+=len(imgList)
    for each_img in imgList:
        ori_img.append(each_img)

rate=0.2
picknumber=int(img_num*0.2)
sample=random.sample(ori_img,picknumber)   #随机选取的验证集

toStop=0
for img_dir in class_list:  #遍历每个文件夹
    toStop+=1
    current_dir=os.path.join(save_path,img_dir)
    annotation_path=os.path.join(current_dir,'polygons.mat')
    img_sorted=[]
    img_sorted=os.listdir(current_dir)
    img_sorted.sort(key=lambda x:str(x[:-4]))  #-4是索引到文件名的倒数第四个字符,即.(按照后缀名前面的进行排序)
    img_list=[os.path.join(current_dir,x) for x in img_sorted
                if x.endswith('.jpg')]
    data = scio.loadmat(annotation_path)  #dic类型
    #print(data.keys())   #data中包含'version','polygons','__header__','__globals__'4个key
    #.values()函数可以去除dict中key对应的值
    loc = data.get('polygons')
    for i in range(len(loc[0])):       #每一帧图片
        polygons=[]
        img_name = img_list[i]
        hand_type = []
        hand=[]
        hand_index=0
        #len(loc[0][i])=4,此数据集最多有两个人,4只手,如果没到4只手,则point中有数组为空[]
        #loc[0][i]中顺序分别是myleft,myright,yourleft,yourright，my是以自我为中心
        for point in loc[0][i]:  #每个手的坐标
            #print("the data is {}".format(point))
            #if hand_index == 0:
                #print("my_left")
            #elif hand_index == 1:
                #print("my_right")
            #elif hand_index == 2:
                #print("your_left")
            #else:
                #print("your_right") 
            if len(point)>1:   #如果包含了手其len会大于0，否则为空则len(point)=1
                axix_x=[]
                axix_y=[]
                for x in point:
                    axix_x.append(x[0])
                    axix_y.append(x[1])
                polygons.append((axix_x, axix_y))
                if ((hand_index == 0) or (hand_index == 2)):
                    hand_type.append('left')
                else:
                    hand_type.append('right')
            hand_index+=1
        #print(loc[0][i])
        
        print(img_name)
        img_dic={}
        img_dic['filename']=img_name
        if not(len(polygons)):          #图片中不包含手
            img_dic['regions']=None
            if img_name in sample:
                val.append(img_dic)
            else:
                train.append(img_dic) 
        else:
            img_dic['regions']={}
            num=0             #int转换为str
            for points in polygons:
                img_dic['regions'][str(num)]={}
                img_dic['regions'][str(num)]['shape_attribute'] = {}
                img_dic['regions'][str(num)]['shape_attribute']['type'] = hand_type[num]
                img_dic['regions'][str(num)]['shape_attribute']['loc_x']=points[0]
                img_dic['regions'][str(num)]['shape_attribute']['loc_y'] = points[1] 
                num+=1
            if img_name in sample:
                val.append(img_dic)
            else:
                train.append(img_dic)
    #print(data)
    #brek
print(len(sample))
T_json_name = './json_file/train.json'
V_json_name='./json_file/val.json'
if os.path.exists(T_json_name):
    os.remove(T_json_name)
if os.path.exists(V_json_name):
    os.remove(V_json_name)
with open(T_json_name,'w') as f:
    json.dump(train,f,sort_keys=True,indent=4,separators=(', ',': '))
    print('the train_json file has done')
with open(V_json_name, 'w') as f:
    json.dump(val, f, sort_keys=True, indent=4, separators=(', ', ': '))
    print("the val_json file has don")


#-----验证json文件中的坐标及手分类是否正确
""" with open(json_name,'r') as f:
    data = json.load(f)

annotation = [a for a in data if a['regions']]  #annotaion中的元素分别代表一张图片,如果该图片包含手则存入list中
filename = []
polygons=[]
for a in annotation:
    polygo=[r['shape_attribute'] for r in a['regions'].values()]   #a['regions'].values()即是shape_attribute,然后再r['shape_attribute']即是取出来
    polygons.append(polygo)
    filename.append(a['filename'])       #len()不是100因为前面分了一些给val


#test=annotation[0]['regions'].values()
#polygons = test[0]['shape_attribute']
test_path = "/home/lj/projects/detection/detectHand/sfd_hand/test_img/test_json/"
if os.path.exists(test_path):
    shutil.rmtree(test_path)    #先删除目录及其文件(os.rmdir只能删除空目录)
    os.mkdir(test_path)

for i in range(len(filename)):
    img = skimage.io.imread(filename[i])
    polygon = polygons[i]
    for poly in polygon:
        rr, cc = skimage.draw.polygon(poly['loc_y'], poly['loc_x'])

        #for polygon in polygons循环读取polygons的时候并不是按照存入时myleft,myright,yourleft,yourright来的
        min_x = min(poly['loc_x'])
        min_y = min(poly['loc_y'])
        cv2.putText(img,poly['type'],(int(min_x),int(min_y)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        skimage.draw.set_color(img, [rr, cc], (255, 0, 0))
        #break
    
    cv2.imwrite(os.path.join(test_path,os.path.basename(filename[i])), img) """



