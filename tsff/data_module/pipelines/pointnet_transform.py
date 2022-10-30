from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd 

from spconv.utils import Point2VoxelCPU1d
from cumm import tensorview as tv

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class pointnet_transform:
    def __init__(self,selected_cols,encoder_length, decoder_length,
                time_interval,pairs):
        self.selected_cols = selected_cols
        self.encoder_length =encoder_length
        self.decoder_length = decoder_length
        self.time_interval = time_interval
        self.pairs = pairs


    def __call__(self,data):
    # cpu transform

        x_data = data['x_data']
        start_time = data['start_time']

        time_interval = self.time_interval # 1 second interval not 1 ms
        start_time *= 0.001
        # multiplied by 60, since we used minute as input
        end_time = (self.encoder_length + self.decoder_length ) * 60 * time_interval
        voxel_generator = Point2VoxelCPU1d(vsize_xyz=[float(time_interval)],
                                            coors_range_xyz=[0,end_time],
                                            #num_point_features = len(self.pairs)*3, # neads to be calculated later 
                                            num_point_features = 3, # neads to be calculated later 
                                            max_num_voxels= int(end_time/time_interval),
                                            max_num_points_per_voxel =500)

        voxels, coords, num_points = [],[],[]        
        for datum in x_data.values():
            datum = datum.to_numpy()
            datum[:,0] = datum[:,0] / 1000 # value between [0,time_lasped]
            datum[:,0] = datum[:,0] - start_time

            # add sample points into datum - to have fixed size output
            artf_time = np.arange(0.1,end_time+0.1,time_interval)
            artf_array = np.c_[artf_time,np.zeros_like(artf_time),np.zeros_like(artf_time)]
            datum = np.r_[datum,artf_array]

            datum = np.ascontiguousarray(datum)
            tv_datum = tv.from_numpy(datum)
            voxels_tv, indices_tv, num_p_in_voxel = voxel_generator.point_to_voxel(tv_datum)
            voxels_tv, indices_tv, num_p_in_voxel = voxels_tv.numpy(), indices_tv.numpy(), num_p_in_voxel.numpy() 
            voxels.append(voxels_tv)
            coords.append(indices_tv)
            num_points.append(num_p_in_voxel)
        
        voxels = np.concatenate(voxels, axis = -1)
        coords = np.concatenate(coords, axis = -1)
        num_points = np.concatenate(num_points, axis = -1)
        
        data['voxels'] = voxels
        data['coords'] = coords 
        data['num_points'] = num_points


        return data