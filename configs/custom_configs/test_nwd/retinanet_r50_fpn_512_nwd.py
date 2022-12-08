_base_ = ['../../retinanet/retinanet_r50_fpn_1x_coco.py']
            # '../../_base_/schedules/schedule_1x.py',
            # '../../_base_/default_runtime.py',
            # ]
# model = dict(
#     bbox_head=dict(num_classes=1,)
# )

# _base_ = [
#     #'../_base_/models/retinanet_r50_fpn_aitod.py',
#     '../_base_/datasets/aitodv2_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]


# optimizer
model = dict(
    type='RetinaNet',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='RankingAssigner',
            ignore_iof_thr=-1,
            gpu_assign_thr=512,
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='nwd',
            topk=3),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))

dataset_type = 'COCODataset'
# classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

classes=('whale',)
#
# data_root = '../'
# # img_root = '../'

# data_root = '../'
# img_root='/home/m32patel/projects/def-dclausi/share/whale/sahi_whale/sahi/whale/'

path_dataset = '/home/fernando/Documents/Graduate Studies/Databases'
path_annotation = '/home/fernando/Documents/Graduate Studies/Databases/whale_data/'
path_train_data = path_dataset
path_train_anno = path_annotation+'train/annotation_coco.json'
path_val_data = path_dataset
path_val_anno = path_annotation+'val/annotation_coco.json'
path_test_data = path_dataset
path_test_anno = path_annotation+'test/annotation_coco.json'



img_norm_cfg=dict(
    mean=[148.22004452, 173.58450995, 161.69282048],
    std=[44.85728789, 34.87804841, 28.33359094] , to_rgb=True
)
 

# img_norm_cfg = dict(
#     mean=[0.59360199, 0.69523434, 0.64578167],
#     std=[0.17142864, 0.13084096, 0.10729129],
#     to_rgb=True,
# )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# img_root = '/home/pc2041/VIP_lab/Sahi/whale/'

# classes = ('whale',)
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('present working directory: ',os.getcwd())
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix=path_train_data,
        classes=classes,
        ann_file=path_train_anno),
        # ann_file='whale_datasets/2014_only_cc/group_8/train/annotation_coco.json'),
    val=dict(
        img_prefix=path_val_data,
        classes=classes,
        ann_file=path_val_anno),
        # ann_file='whale_datasets/2014_only_cc/group_8/val/annotation_coco.json'),
    test=dict(
        img_prefix=path_test_data,
        classes=classes,
        ann_file=path_test_anno),
        # ann_file='whale_datasets/2014_only_cc/group_8/test/annotation_coco.json'),
)

optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=1.0,
    step=[218, 246])


# custom_imports = dict(
#     imports=['custom_modules.custom_hooks.wandb_customHook'],
#     allow_failed_imports=False)

runner = dict(type='EpochBasedRunner', max_epochs=1)
# evaluation = dict(interval=1, metric=['bbox'])
# # base_batch_size = ((1) GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# >>>>>>> local_machine
#
# # We can use the pre-trained Mask RCNN model to obtain higher performance
# # load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
#
