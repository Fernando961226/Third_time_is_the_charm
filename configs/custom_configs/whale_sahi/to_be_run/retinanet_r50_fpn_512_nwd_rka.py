_base_ = ['../../nwd_rka/retinanet_r50_fpn_1x_coco_nwd_rka.py']
            # '../../_base_/schedules/schedule_1x.py',
            # '../../_base_/default_runtime.py',
            # ]

model = dict(
    bbox_head=dict(num_classes=1,)
)




dataset_type = 'COCODataset'

classes=('whale',)
#
# data_root = '../'
# # img_root = '../'

# data_root = '../'
# img_root='/home/m32patel/projects/def-dclausi/share/whale/sahi_whale/sahi/whale/'

path_dataset = '/home/m32patel/scratch/whale_sahi/whale_512_512_0.2_0.2_patches/'
path_annotation = '/home/m32patel/scratch/whale_sahi/whale_512_512_0.2_0.2_patches/'
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



data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
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

optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)

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
