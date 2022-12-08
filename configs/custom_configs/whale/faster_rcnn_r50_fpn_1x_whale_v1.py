_base_ = '../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# im_scale = (3680, 2456)


# im_scale = (7360, 4912)


im_scale = (500, 500)

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[5],  # 100% IoU at full overlap on P2 at half resolution (imresize)
            ratios=[1.0],  # 1:1 aspect ratios only
            strides=[4]  # All larger strides (levels) have max IoU < 0.3
        )
    ),
    roi_head = dict(
        bbox_head=dict(num_classes=1)
    ),
    train_cfg = dict(
        rpn = dict(
            sampler=dict(
                type='RandomSampler',
                num=64,  # From 256; reduce total examples
                pos_fraction=0.5,
                neg_pos_ub=2,  # Use max 2:1 -ve to +ve examples
                add_gt_as_proposals=False),
            pos_weight=5,  # upweigh positive examples
            debug=True
        )
    )
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=im_scale, keep_ratio=True),
    # dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=im_scale,
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
]
data_root = ''
img_root = '../'
dataset_type = 'COCODataset'
classes = ('whale',)
data = dict(
    samples_per_gpu=1,  # default 2
    workers_per_gpu=1,  # default 2
    train=dict(
        img_prefix=img_root,
        classes=classes,
        ann_file='whale_datasets/2014_only_cc/group_8/train/annotation_coco.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=im_scale, keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
    ),
    val=dict(
        img_prefix=img_root,
        classes=classes,
        ann_file='whale_datasets/2014_only_cc/group_8/val/annotation_coco.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=im_scale,
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
    ),
    test=dict(
        img_prefix=img_root,
        classes=classes,
        ann_file='whale_datasets/2014_only_cc/group_8/test/annotation_coco.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=im_scale,
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
    )
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # default_lr = 0.02 (8x2)
runner = dict(type='EpochBasedRunner', max_epochs=50)
log_level = 'DEBUG'
