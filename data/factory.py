#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


from .egohand import HandDetection
from .widerface import WIDERDetection
from .vochead import VOCDetection, VOCAnnotationTransform
from .config import cfg

import torch


def dataset_factory(dataset):
    if dataset == 'face':
        train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
        val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
    if dataset == 'hand':
        train_dataset = HandDetection(cfg.HAND.TRAIN_FILE, mode='train')
        val_dataset = HandDetection(cfg.HAND.VAL_FILE, mode='val')
    if dataset == 'head':
        train_dataset = VOCDetection(cfg.HEAD.DIR, image_sets=[
                                     ('PartA', 'trainval'), ('PartB', 'trainval')],
                                     target_transform=VOCAnnotationTransform(),
                                     mode='train',
                                     dataset_name='VOCPartAB')
        val_dataset = VOCDetection(cfg.HEAD.DIR, image_sets=[('PartA', 'test'), ('PartB', 'test')],
                                   target_transform=VOCAnnotationTransform(),
                                   mode='test',
                                   dataset_name='VOCPartAB')
    return train_dataset, val_dataset


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    #用于处理具有不同数量关联对象注释(包围框)的成批图像的自定义排序fn
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])   #sample[0]是图片
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
