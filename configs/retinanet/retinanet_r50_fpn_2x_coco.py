_base_ = './retinanet_r50_fpn_1x_whale_sahi_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
