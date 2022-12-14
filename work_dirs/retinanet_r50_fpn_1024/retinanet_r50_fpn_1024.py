model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
dataset_type = 'COCODataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[148.22004452, 173.58450995, 161.69282048],
    std=[44.85728789, 34.87804841, 28.33359094],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[148.22004452, 173.58450995, 161.69282048],
        std=[44.85728789, 34.87804841, 28.33359094],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(
                type='Normalize',
                mean=[148.22004452, 173.58450995, 161.69282048],
                std=[44.85728789, 34.87804841, 28.33359094],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/train/train_coco.json',
        img_prefix=
        '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[148.22004452, 173.58450995, 161.69282048],
                std=[44.85728789, 34.87804841, 28.33359094],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('whale', )),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/val/val_coco.json',
        img_prefix=
        '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[148.22004452, 173.58450995, 161.69282048],
                        std=[44.85728789, 34.87804841, 28.33359094],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('whale', )),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/test/test_coco.json',
        img_prefix=
        '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('whale', )))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0,
    step=[218, 246])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='Custom_MMdetWandbHook',
            init_kwargs=dict(
                project='mmwhale',
                name='retinanet_r50_fpn_1024.py',
                entity='mmwhale'),
            interval=10)
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
classes = ('whale', )
path_dataset = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/'
path_annotation = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/'
path_train_data = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/train'
path_train_anno = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/train/train_coco.json'
path_val_data = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/val'
path_val_anno = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/val/val_coco.json'
path_test_data = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/test'
path_test_anno = '/home/m32patel/scratch/whale_sahi/whale_1024_1024_0.2_0.2_patches/test/test_coco.json'
custom_imports = dict(
    imports=['custom_modules.custom_hooks.wandb_customHook'],
    allow_failed_imports=False)
work_dir = './work_dirs/retinanet_r50_fpn_1024'
auto_resume = False
gpu_ids = [0]
