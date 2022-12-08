_base_ = '../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# model=dict(roi_head = dict(
#     bbox_head=dict(
#         num_classes=1)))

# _base_ = [
#     #'../_base_/models/faster_rcnn_r50_fpn_aitod.py',
#     '../_base_/datasets/aitodv2_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]



model = dict(
    type='FasterRCNN',
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
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='RankingAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='nwd',
                topk=2),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000) # If you want to cut down on the inference times, you can set max_per_img to a smaller number 
    ))

# _base_ = [
#     '../../_base_/models/faster_rcnn_r50_fpn_aitod.py',
#     # '../../_base_/datasets/aitod_detection.py',
#     # '../../_base_/schedules/schedule_1x.py', 
#     # '../../_base_/default_runtime.py'
# ]


# model = dict(
#     roi_head = dict(
#     bbox_head=dict(
#         num_classes=1)),
#     rpn_head=dict(
#         reg_decoded_bbox=True,
#         loss_bbox=dict(type='WassersteinLoss', loss_weight=10.0)),
#     train_cfg=dict(
#         rcnn=dict(
#             assigner=dict(
#                 gpu_assign_thr=512)),
#         rpn_proposal=dict(
#             nms_pre=3000,
#             max_per_img=3000,
#             nms=dict(type='wasserstein_nms', iou_threshold=0.85)),
#         rpn=dict(
#             assigner=dict(
#                 gpu_assign_thr=512, 
#                 iou_calculator=dict(type='BboxDistanceMetric'),
#                 ))),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=3000,
#             max_per_img=3000,
#             nms=dict(type='wasserstein_nms', iou_threshold=0.85),
#             min_bbox_size=0)
#         # soft-nms is also supported for rcnn testing
#         # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
#     ))


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
    dict(type='Resize', img_scale=(256,256), keep_ratio=True),
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
        img_scale=(256,256),
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


custom_imports = dict(
    imports=['custom_modules.custom_hooks.wandb_customHook'],
    allow_failed_imports=False)

runner = dict(type='EpochBasedRunner', max_epochs=100)
# evaluation = dict(interval=1, metric=['bbox'])
# # base_batch_size = ((1) GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# >>>>>>> local_machine
#
# # We can use the pre-trained Mask RCNN model to obtain higher performance
# # load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
#
