_base_ = '../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model=dict(roi_head = dict(
    bbox_head=dict(
        num_classes=1)))

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

path_dataset = '/home/m32patel/scratch/whale_sahi/whale_256_256_0.2_0.2_patches/'
path_annotation = '/home/m32patel/scratch/whale_sahi/whale_256_256_0.2_0.2_patches/'
path_train_data = path_dataset+'train'
path_train_anno = path_annotation+'train/train_coco.json'
path_val_data = path_dataset+'val'
path_val_anno = path_annotation+'val/val_coco.json'
path_test_data = path_dataset+'test'
path_test_anno = path_annotation+'test/test_coco.json'



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
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
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
        img_scale=(1024, 1024),
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        img_prefix=path_train_data,
        classes=classes,
        ann_file=path_train_anno,
        pipeline=train_pipeline),
        # ann_file='whale_datasets/2014_only_cc/group_8/train/annotation_coco.json'),
    val=dict(
        img_prefix=path_val_data,
        classes=classes,
        ann_file=path_val_anno,
        pipeline=test_pipeline),
        # ann_file='whale_datasets/2014_only_cc/group_8/val/annotation_coco.json'),
    test=dict(
        img_prefix=path_test_data,
        classes=classes,
        ann_file=path_test_anno),
        # ann_file='whale_datasets/2014_only_cc/group_8/test/annotation_coco.json'),
)

optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=1.0,
    step=[218, 246])

workflow = [('train', 1), ('val', 1)]

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
