from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch 
from glob import glob 

#from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class PandaDataset(MonoDataset):

    def __init__(self, *args, **kwargs):
        super(PandaDataset, self).__init__(*args, **kwargs)

        # print("filenames")
        # print(self.filenames)

        # print("data_path")
        # print(self.data_path)

        # print('frame_idxs')
        # print(self.frame_idxs)

        #this is an intrinsic array 
        self.K = np.array( [[(1427.6329526399998 / 2), 0, (1280.0 / 2), 0],
                            [0, (1393.8135290999999 / 2), (960.0 / 2), 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]).astype(np.float32)
        
        self.full_res_shape = (1280,960)
        #we don't have side map because not stereo
        #self.count = 0

    def check_depth(self):
        return False
    
    def get_color(self, folder, frame_index, side, do_flip):
        #print("real_frame_index: " + str(frame_index))
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def get_image_path(self, folder, frame_index, side):
        image_path = os.path.join(
            self.data_path, folder + ".png")
        # Convert backslashes to forward slashes
        image_path = image_path.replace("\\", "/")
        #print(image_path)
        #print("count: " + str(self.count))
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        return False
    

        