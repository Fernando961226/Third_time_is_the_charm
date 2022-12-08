_base_ = '../../retinanet/retinanet_r50_fpn_1x_whale_sahi_coco.py'
# The new config inherits a base config to highlight the necessary modification
#_base_ = '/home/pc2041/VIP_lab/openMM/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         ))

# <<<<<<< HEAD
# model=dict(
# bbox_head=dict(type='RetinaHead',
#         num_classes=3,)
# )
# =======
model=dict(
bbox_head=dict(num_classes=3,)
)
# >>>>>>> local_machine
#
#
# # im_scale=(7360, 4912)
# # Modify dataset related settings
# <<<<<<< HEAD
# # dataset_type = 'COCODataset'
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

# =======
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

classes=('person', 'bicycle', 'car',)
#
# data_root = '../coco' # for compute canada
# ann_root='../coco' # for compute canada

data_root = '/media/pc2041/data/vip_lab/data/coco' # for local machine
ann_root='/media/pc2041/data/vip_lab/data/coco/annotations_trainval2017' # for local machine
# img_root = '../'
# classes = ('whale',)
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('present working directory: ',os.getcwd())
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
data = dict(
    train=dict(
        img_prefix=data_root+'/train2017',
        classes=classes,
        ann_file=ann_root+'/annotations/instances_train2017.json'),
    val=dict(
        img_prefix=data_root+'/val2017',
        classes=classes,
        ann_file=ann_root+'/annotations/instances_val2017.json'),
    test=dict(
        img_prefix=data_root+'/test2017',
        classes=classes,
        ann_file=''),)
#
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
#
# <<<<<<< HEAD
# # runner = dict(type='EpochBasedRunner', max_epochs=200)
# evaluation = dict(interval=1, metric=['bbox'])
# # base_batch_size = ((2) GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=16)
# =======
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.01,
    step=[218, 246])

runner = dict(type='EpochBasedRunner', max_epochs=12)
# evaluation = dict(interval=1, metric=['bbox'])
# # base_batch_size = ((1) GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
# >>>>>>> local_machine
#
# # We can use the pre-trained Mask RCNN model to obtain higher performance
# # load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
#
