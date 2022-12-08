# Copyright (c). All rights reserved by kirtan
import os.path as osp
import os

import mmcv
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 
import math 

from ..builder import PIPELINES

@PIPELINES.register_module()
class GenerateImageWithRatio(object):

    def __init__(self, path_dir,path_anno,threshold = 0.05):
        self.path = path_dir
        self.path_anno = path_anno
        self.df = pd.read_csv(path_anno,sep = " ",names = ["path","category"])

        # need to confirm 
        self.whale = self.df[self.df["category"] == 1]
        self.no_whale = self.df[self.df["category"] == 0]
        self.threshold = threshold
        self.whale = self.whale.reset_index(drop=True)
        self.no_whale = self.no_whale.reset_index(drop = True)
        self.whale_placeholder = 0
        self.no_whale_placeholder = 0
        self.distribution = np.concatenate( (np.zeros(math.floor((len(self.no_whale) * threshold) )),
                                            np.ones(math.floor(len(self.whale))) ) );
        np.random.shuffle(self.distribution)


        
        self.current_index = 0 
        #need to confirm 

    def __call__(self,results):

        if(self.distribution[self.current_index] == 1):
            index = self.whale_placeholder
            category = self.whale.iloc[index]["category"]
            filename = self.whale.iloc[index]["path"]
            self.whale_placeholder+=1
            if(self.whale_placeholder >= len(self.whale)):
                self.whale_placeholder = 0
        else:
            index = self.no_whale_placeholder
            filename = self.no_whale.iloc[index]["path"]
            category = self.no_whale.iloc[index]["category"]
            self.no_whale_placeholder += 1
            if(self.no_whale_placeholder >= len(self.no_whale)):
                self.no_whale_placeholder = 0
        
        self.current_index += 1
        if(self.current_index >= len(self.distribution)):
            self.current_index  = 0
            np.random.shuffle(self.distribution)
        
        full_path = os.path.join(self.path,filename)
        
        results = {'img_prefix': self.path}
        results['img_info'] = {'filename': filename}
        results['gt_label'] = np.array(category, dtype=np.int64)
        results['filename'] = filename
        results['ori_filename'] = full_path
        img = plt.imread(full_path)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

            
    def __repr__():
        pass