#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python hand_keypoint.py --img_dir './test_img/' --dataset 'hand' --model 'weights/weights1/sfd_hand_60000.pth'