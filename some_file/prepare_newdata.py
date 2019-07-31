#-*- coding:utf-8 -*-

import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import random

#-----将自己标的新数据.xml文件保存为txt文件------
data_root = '/home/data/cy/piano_hand/'
#os.path.join()函数前面的路径结尾是/,后面要接上的路径开头不加/
xml1_path = os.path.join(data_root, 'xml/xml1')
xml2_path = os.path.join(data_root, 'xml/xml2')
print(xml1_path)
file_list1 = os.listdir(xml1_path)
file_list1.sort(key=lambda x: int(x[:-4]))
xml1_list = [os.path.join(xml1_path, x) for x in file_list1
                if x.endswith('.xml')]
file_list2 = os.listdir(xml2_path)
file_list2.sort(key=lambda x: int(x[:-4]))
xml2_list = [os.path.join(xml2_path, x) for x in file_list2
                if x.endswith('.xml')]                

save_path = './json_file'
new_txt = os.path.join(save_path, 'new.txt')
img1_root = os.path.join(data_root, 'img/img1')
img2_root = os.path.join(data_root, 'img/img2')

#-----将xml文件中的路径标签存入txt文件中------
def parse_xml(xml_list, new_txt, img_root, flag):
    boxes = []
    labels = []
    filenames = []
    for path in xml_list:
        target = ET.parse(path).getroot()
        filename = target.find('filename').text
        sizes = target.find('size')
        size = (sizes.find('width').text, sizes.find('height').text)
        num = 0
        box = []
        label=[]
        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            temp_box = [int(bbox.find(pts[0]).text) - 1, int(bbox.find(pts[1]).text) - 1,
                        int(bbox.find(pts[2]).text) - 1, int(bbox.find(pts[3]).text) - 1]             
            temp_label=[num]
            box.append(temp_box)
            label.append(temp_label)
            num += 1
        boxes.append(box)
        labels.append(label)
        filenames.append(filename)

#-----如果是open('w'),则是在原来的txt文件中追加内容------
    if (flag == 'w'):
        if os.path.exists(new_txt):
            os.remove(new_txt)
        fout = open(new_txt, 'w')
    elif (flag == 'a+'):
        fout = open(new_txt, 'a+')
    
    types = []
    types.append('left')
    types.append('right')
    
    for i, img in enumerate(filenames):
        fout.write('{} '.format(os.path.join(img_root,img)))
        box = boxes[i]
        label = labels[i]
        fout.write('{}'.format(len(box)))
        for j, bbox in enumerate(box):
            fout.write(' {} '.format(types[j]))
            x1, y1, x2, y2 = bbox
            fout.write('{} {} {} {}'.format(x1, y1, x2, y2))
        fout.write('\n')
    fout.close()
    print('the txt file has done')

#-----将txt文件中的图片分为训练集和测试集------
def loadData(txt_name, ratio):
    trainData = []
    valData = []
    with open(txt_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if random.random() < ratio:
                trainData.append(line)
            else:
                valData.append(line)
    return trainData, valData

#-----将训练集和测试集分别存入train/val.txt文件中------
def saveData(txt_name, Data):
    if os.path.exists(txt_name):
        os.remove(txt_name)
    fout = open(txt_name, 'w')
    for data in Data:
        for label in data:
            fout.write('{} '.format(label))
        fout.write('\n')
    fout.close()
    print('the new txt file has done')

#-----将新增加的数据集的txt文件追加到egohand.txt文件夹中------
def final_data(initial_txt, subsequent_txt):
    data = []
    with open(subsequent_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            data.append(line)
        f.close()
    fout = open(initial_txt, 'a+')
    for labels in data:
        for label in labels:
            fout.write('{} '.format(label))
        fout.write('\n')
    fout.close()
        

#-----测试txt文件中的数据是否正确
def verify_data(txt_name):
    EGOHAND_CLASSES = ('left', 'right')
    class_id=dict(zip(EGOHAND_CLASSES,range(len(EGOHAND_CLASSES))))
    with open(txt_name, 'r') as f:
        lines = f.readlines()

    fnames = []
    boxes1 = []
        #self.h_types=[]
    labels1 = []   #标签,左手为1,右手为0
    for line in lines:
        line = line.strip().split()
        num_hands = int(line[1])    #split()返回列表,以空格作为分割
        box = []
        #h_type=[]
        label = []
        
        for i in range(num_hands):
            xmin = float(line[3 + 5 * i])
            ymin = float(line[4 + 5 * i])
            xmax = float(line[5 + 5 * i])
            ymax = float(line[6 + 5 * i])
            box.append([xmin, ymin, xmax, ymax]) #多少个[]就是几维数组,append之后box就是二维
            #h_type.append(line[2 + 5 * i])  #左上角和右下角坐标
            h_type=line[2 + 5 * i]
            label.append(int(int(class_id[h_type])+1))
        fnames.append(line[0])
        boxes1.append(box)         #boxes中每一个元素为一个list,转换为np.array后都是一个二维数组[[]]or[[],[]]
        #h_types.append(h_type)
        labels1.append(label)   

    tostop = 0
    test_path = '/home/lj/projects/detection/detectHand/sfd_hand/test_img/text_txt2/'
    if not (os.path.exists(test_path)):
        os.mkdir(test_path)
    for i in range(len(fnames)):
        img_path = fnames[i]
        img = cv2.imread(img_path)
        box = boxes1[i]
        type = labels1[i]
        for j, boxx in enumerate(box):
            cv2.rectangle(img, (int(boxx[0]), int(boxx[1])), (int(boxx[2]), int(boxx[3])), (0, 0, 255), 3)
            cv2.putText(img,str(type[j]),(int(boxx[0]), int(boxx[1]-5)),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        print(os.path.join(test_path, os.path.basename(img_path)))

        cv2.imwrite(os.path.join(test_path, os.path.basename(img_path)), img)
        tostop += 1
        if (tostop > 200):
            break


if __name__ == '__main__':
    parse_xml(xml1_list, new_txt, img1_root, 'w')
    parse_xml(xml2_list, new_txt, img2_root, 'a+')
    ratio = 0.8
    trainData, valData = loadData(new_txt, ratio)
    test_train = './json_file/new_train.txt'
    test_val = './json_file/new_val.txt'
    saveData(test_train, trainData)
    saveData(test_val, valData)
    
    #追加写入txt文件追加一次就好啦
    #initial_txt = './json_file/train.txt'
    #final_data(initial_txt, test_train)
    #verify_data(test_val)






