#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:01:35 2022

@author: pc2041
"""

# import os

# The new config inherits a base config to highlight the necessary modification
_base_ = '../../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
#_base_ = '/home/pc2041/VIP_lab/openMM/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('balloon',)

#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('present working directory: ',os.getcwd())
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
data = dict(
    train=dict(
        img_prefix='/home/m32patel/projects/def-dclausi/share/whale/balloon_data_delete_later/balloon/train',
        classes=classes,
        ann_file='/home/m32patel/projects/def-dclausi/share/whale/balloon_data_delete_later/annotations/train_coco_annotations.json'),
    val=dict(
        img_prefix='/home/m32patel/projects/def-dclausi/share/whale/balloon_data_delete_later/balloon/val',
        classes=classes,
        ann_file='/home/m32patel/projects/def-dclausi/share/whale/balloon_data_delete_later/annotations/test_coco_annotations.json'),
    test=dict(
        img_prefix='/home/m32patel/projects/def-dclausi/share/whale/balloon_data_delete_later/balloon/val',
        classes=classes,
        ann_file='/home/m32patel/projects/def-dclausi/share/whale/balloon_data_delete_later/annotations/test_coco_annotations.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
