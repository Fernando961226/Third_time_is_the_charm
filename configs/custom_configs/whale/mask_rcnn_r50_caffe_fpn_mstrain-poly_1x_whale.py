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
classes = ('whale',)

data_root = '../'
# img_root = '/home/pc2041/VIP_lab/Sahi/whale/'
img_root='/home/m32patel/projects/def-dclausi/share/whale/sahi_whale/sahi/whale/'
# classes = ('whale',)
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('present working directory: ',os.getcwd())
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

data = dict(
    train=dict(
        img_prefix=img_root+ "train/",
        classes=classes,
        ann_file=img_root+"train/sahi_coco.json"),
        # ann_file='whale_datasets/2014_only_cc/group_8/train/annotation_coco.json'),
    val=dict(
        img_prefix=img_root+ "val/",
        classes=classes,
        ann_file=img_root+"val/sahi_coco.json"),
        # ann_file='whale_datasets/2014_only_cc/group_8/val/annotation_coco.json'),
    test=dict(
        img_prefix=img_root+"test/",
        classes=classes,
        ann_file=img_root+"test/sahi_coco.json"),
        # ann_file='whale_datasets/2014_only_cc/group_8/test/annotation_coco.json'),
)
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2000,  # same as burn-in in darknet
#     warmup_ratio=0.01,
#     step=[218, 246])
runner = dict(type='EpochBasedRunner', max_epochs=80)
# evaluation = dict(interval=1, metric=['bbox'])
# # base_batch_size = ((1) GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# >>>>>>> local_machine
#
# # We can use the pre-trained Mask RCNN model to obtain higher performance
# # load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
#
