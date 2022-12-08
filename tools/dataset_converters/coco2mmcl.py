'''
The following file is used to convert coco dataset to mmclassification annotation file. 

Update Date: 2022-11-09
Created by Fernando J. Pena Cantu 
'''

import json
import pandas as pd
import os

def coco_to_mmclassification(json_path,out_path,name):
    '''
    Function: Create an annotation file for mmclassification from the coco dataset provided.
    The output txt file looks like this. 

    140810_Cam2_18295_0_2665_256_2921.jpg 0
    140810_Cam2_18295_205_2665_461_2921.jpg 1
    140810_Cam2_18295_410_2665_666_2921.jpg 0
    140810_Cam2_18295_615_2665_871_2921.jpg 1

    Where the first string is the image file name, and the second value is 1 if the image contains whales otherwise is 0. 

    Inputs:
        - json_path: The path to the coco json file. 
        - out_path: The output directory
        - name: The name of the output file. Eg "data.txt"
    
    '''
   
    with open(json_path) as json_file:
        data = json.load(json_file)

    images = data['images']

    for img in images:
        img.pop('height')
        img.pop('width')
        img['whale_num'] = 0

    whales = data['annotations']

    for whale in whales:
        images[whale['image_id']-1]['whale_num']+=1

    for img in images:
        img.pop('id')
        if img['whale_num'] > 0:
            img['is_there_whale'] = 1
        else:
            img['is_there_whale'] = 0
        img.pop('whale_num')
    
    pd_images = pd.DataFrame.from_dict(images)
    
    output_path = os.path.join(out_path,name)
    pd_images.to_csv(output_path,header=None, index=None, sep=' ', mode='a')

def class_count(pd_file):

  
        
    pd_whale = pd.read_csv(pd_file,sep=' ',header=None)

    print(pd_whale[1].value_counts())


if __name__ == "__main__":
    json_path = 'whale_datasets/2014_only_cc/group_8/val/fer_sliced_coco.json_coco.json'
    out_path = '/home/fernando/Documents/Graduate Studies/Python/mmclassification'
    name = 'val_mmcl.txt'
    coco_to_mmclassification(json_path,out_path,name)
    class_count(os.path.join(out_path,name))




