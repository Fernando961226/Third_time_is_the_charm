_base_ = './retinanet_r50_fpn_1x_whale_sahi_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)
