def convert_np2ts_pointnet(self, data:List[List[np.array]]):
    voxels, coords, num_points = data
    voxel = np.concatenate(*voxels, axis = -1)
    coord = np.concatenate(*coords, axis = -1)
    num_point = np.concatenate(*num_points,axis =-1 )

    voxel_ts = torch.from_numpy(voxel)
    coord_ts = torch.from_numpy(coord)
    num_point_ts = torch.from_numpy(num_point)

    return [voxel_ts, coord_ts, num_point_ts]        