from typing import Dict
from tsff.algorithm_module.models.builder import EMBEDDING_LAYERS

import torch
import torch.nn as nn
import torch.nn.functional as F



class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        #torch.backends.cudnn.enabled = False
        _x = x.view(-1,x.size(-2),x.size(-1))
        _x = self.norm(_x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else _x
        x = _x.view(x.size())
        #torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=-2, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@EMBEDDING_LAYERS.register_module()
class pillar_vfe(nn.Module):
    def __init__(self,
    d_model,
    num_point_features, 
    use_norm = True,
    use_relative_distance = True,

    **kwargs):
        super().__init__()

        self.use_norm = use_norm
        self.use_relative_distance = use_relative_distance
        self.num_point_features = num_point_features

        num_point_features *= 2 if self.use_relative_distance else num_point_features

        self.d_model = d_model
        num_filters = [num_point_features] + [self.d_model]

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(self, x, **kwargs):

        voxel_features = x['voxels']
        Batch_size, time_size = voxel_features.size(0), voxel_features.size(1)
        voxel_num_points = x['num_points']
  
        if self.use_relative_distance:
            voxel_num_points = voxel_num_points.type_as(voxel_features).view(Batch_size,time_size, 1, -1)
            voxel_num_points_list = []
            for i in range(self.num_point_features):
                if i < (self.num_point_features * voxel_num_points.size(-1))/ 2:
                    i = 0
                else:
                    i = 1
                voxel_num_points_list.append(voxel_num_points[...,i])
            voxel_num_points = torch.cat(voxel_num_points_list,-1)
                
            points_mean = voxel_features[:,:, :, :].sum(dim=2, keepdim=True) / voxel_num_points.unsqueeze(-2)
            f_cluster = voxel_features - points_mean
        features = torch.cat([voxel_features,f_cluster],-1)

        for pfn in self.pfn_layers:
            features = pfn(features)

        features = features.squeeze()
        return features
        
