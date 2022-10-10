def pointnet_transform(self,data:List[pd.DataFrame],start_time,end_time):
    # cpu transform
    time_interval = self.time_interval # 1 second interval not 1 ms
    start_time *= 0.001
    end_time *= 0.001
    voxel_generator = Point2VoxelCPU1d(vsize_xyz=[time_interval],
                                        coors_range_xyz=[start_time,end_time],
                                        num_point_features = len(self.pairs)*3, # neads to be calculated later 
                                        max_num_voxels= int((end_time-start_time)/time_interval))

    voxels, coords, num_points = [],[],[]        
    for datum in data:
        time_col_loc = datum.columns.get_loc('Timestamp')
        datum = datum.to_numpy()
        datum = np.transpose
        tv_datum = tv.from_numpy(datum)
        voxels_tv, indices_tv, num_p_in_voxel = voxel_generator.point_to_voxel(tv_datum)
        voxels_tv, indices_tv, num_p_in_voxel = voxels_tv.numpy(), indices_tv.numpy(), num_p_in_voxel.numpy() 
        voxels.append(voxels_tv)
        coords.append(indices_tv)
        num_points.append(num_p_in_voxel)

    return [voxels, coords, num_points]