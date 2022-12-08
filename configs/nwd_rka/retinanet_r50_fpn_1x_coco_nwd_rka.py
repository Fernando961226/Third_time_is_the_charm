_base_ = ['../retinanet/retinanet_r50_fpn_1x_coco.py']


# optimizer
model = dict(
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='RankingAssigner',
            ignore_iof_thr=-1,
            gpu_assign_thr=512,
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='nwd',
            topk=3),),
    test_cfg=dict(
        nms_pre=3000,
        max_per_img=3000))
