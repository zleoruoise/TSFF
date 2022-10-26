from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd 

from spconv.utils import Point2VoxelCPU1d
from cumm import tensorview as tv

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class pointnet_transform:
    def __init__(self,selected_cols,time_interval,pairs):
        self.selected_cols = selected_cols
        self.time_interval = time_interval
        self.pairs = pairs


    def __call__(self,data):
    # cpu transform

        x_data = data['x_data']
        start_time = data['start_time']
        end_time = data['end_time']

        time_interval = self.time_interval * 1000 # 1 second interval not 1 ms
        #start_time *= 0.001
        #end_time *= 0.001
        voxel_generator = Point2VoxelCPU1d(vsize_xyz=[float(time_interval)],
                                            coors_range_xyz=[start_time,end_time],
                                            #num_point_features = len(self.pairs)*3, # neads to be calculated later 
                                            num_point_features = 3, # neads to be calculated later 
                                            max_num_voxels= int((end_time-start_time)/time_interval),
                                            max_num_points_per_voxel = 1000)

        voxels, coords, num_points = [],[],[]        
        for datum in x_data.values():
            datum = datum.to_numpy()
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