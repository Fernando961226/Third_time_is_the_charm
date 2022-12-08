import os
from sahi.slicing import slice_coco
from sahi.utils.file import load_json

cwd = os.getcwd()



coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="/home/fernando/Documents/Graduate Studies/Python/mmwhale/whale_datasets/2014_only_cc/group_8/test/annotation_coco.json",
    image_dir="/home/fernando/Documents/Graduate Studies/Databases/",
    output_coco_annotation_file_name="fer_sliced_coco.json",
    ignore_negative_samples=False,
    output_dir="/home/fernando/Documents/Graduate Studies/Databases/whale_data/whale_only_images/sahi_test",
    slice_height=768,
    slice_width=768,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True
)