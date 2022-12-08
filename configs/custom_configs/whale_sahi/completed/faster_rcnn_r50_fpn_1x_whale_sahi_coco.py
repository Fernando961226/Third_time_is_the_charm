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

path_dataset = '/home/m32patel/scratch/whale_sahi/whale_512_512_0.2_0.2_patches/'
path_annotation = 'annotation/'
path_train_data = path_dataset+'train'
path_train_anno = path_annotation+'train/train_coco.json'
path_val_data = path_dataset+'val'
path_val_anno = path_annotation+'val/val_coco.json'
path_test_data = path_dataset+'test'
path_test_anno = path_annotation+'test/test_coco.json'



# img_root = '/home/pc2041/VIP_lab/Sahi/whale/'

# classes = ('whale',)
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('present working directory: ',os.getcwd())
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

data = dict(
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

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.01,
    step=[218, 246])

runner = dict(type='EpochBasedRunner', max_epochs=60)
# evaluation = dict(interval=1, metric=['bbox'])
# # base_batch_size = ((1) GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# >>>>>>> local_machine
#
# # We can use the pre-trained Mask RCNN model to obtain higher performance
# # load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
#
