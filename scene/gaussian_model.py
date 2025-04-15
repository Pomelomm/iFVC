#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import math
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric, build_rotation)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, Q_anchor, \
    encoder_anchor, decoder_anchor, \
    encoder, decoder, \
    encoder_gaussian, decoder_gaussian, \
    get_binary_vxl_size

bit2MB_scale = 8 * 1024 * 1024

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = False):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)

        return latent

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]

        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        
        return out_i
    

class NTC_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=1,
            resolutions_list=[514],
            log2_hashmap_size=15,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.output_dim = self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim 

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)

        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]

        out_i = torch.cat([out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        
        # return out_i, 0.0

        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        p = torch.sigmoid(out_xyz)

        return out_i * p, p

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 mode: str='I_frame',
                 enable_filter=True,
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)
        
        self.mode = mode
        self.enable_filter = enable_filter

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+1+1+1),
        ).cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()

        self.ntc = None
        self.ntc_mlp = None
        self.step_flag1 = None
        self.step_flag2 = None

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_ntc_2D_params(self):
        params = []
        
        params.append(self.ntc.encoding_xy.params)
        params.append(self.ntc.encoding_xz.params)
        params.append(self.ntc.encoding_yz.params)
        
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params
    
    def get_ntc_3D_params(self):
        params = []
        
        params.append(self.ntc.encoding_xyz.params)
        
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params
    
    def get_ntc_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.ntc_mlp.named_parameters():
            mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()

        if self.ntc is not None:
            self.ntc.eval()
            self.ntc_mlp.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()

        if self.ntc is not None:
            self.ntc.train()
            self.ntc_mlp.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._anchor,
        self._offset,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_anchor_features(self):
        return self._anchor_feat

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self, resize=False):
        print("self.x_bound_min", self.x_bound_min)
        print("self.x_bound_max", self.x_bound_max)
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        if resize:
            for c in range(x_bound_min.shape[-1]):
                x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
            for c in range(x_bound_max.shape[-1]):
                x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')
        print("self.x_bound_min", self.x_bound_min)
        print("self.x_bound_max", self.x_bound_max)

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features
    
    def get_ntc(self, x):
        # x: [N, 3]

        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]

        mask = (x >= 0) & (x <= 1)
        mask = mask.all(dim=1)

        features_input, p = self.ntc(x[mask])
        features = self.ntc_mlp(features_input)
        d_feat_mask, d_offsets_mask = torch.split(features, [self.feat_dim, 3*self.n_offsets], dim=1)

        d_feat = torch.full((x.shape[0], self.feat_dim), 0.0, dtype=torch.float32, device="cuda")
        d_offsets = torch.full((x.shape[0], 3*self.n_offsets), 0.0, dtype=torch.float32, device="cuda")
        
        d_feat[mask] = d_feat_mask
        d_offsets[mask] = d_offsets_mask

        d_offsets = d_offsets.reshape(-1, self.n_offsets, 3)

        anchor_loss = 0.0

        return d_feat, d_offsets, anchor_loss

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        print("points : ", points.shape)

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def set_steps(self, flag_1, flag_2):
        self.step_flag1 = flag_1
        self.step_flag2 = flag_2
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

            {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
            {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr

    def training_setup_for_d_anchor(self):
        self.d_anchor = nn.Parameter(torch.zeros_like(self._anchor).cuda())
        l = [
            {'params': [self.d_anchor], 'lr': 0.1, "name": "d_anchor"},    
        ]
        self.d_anchor_optimizer = torch.optim.Adam(l, lr=0.1)
    
    def training_setup_for_P_frame(self, ntc_cfg):
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.ntc = NTC_encoding(
                n_features=ntc_cfg['n_features_per_level'],
                resolutions_list_2D=ntc_cfg["resolutions_list_2D"],
                log2_hashmap_size_2D=ntc_cfg["log2_hashmap_size_2D"],
                ste_binary=self.ste_binary,
                ste_multistep=self.ste_multistep,
                add_noise=self.add_noise,
                Q=self.Q,
            ).cuda()
        
        output_dim = 3*self.n_offsets + self.feat_dim 
        
        self.ntc_mlp = nn.Sequential(
            nn.Linear(self.ntc.output_dim, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, output_dim),
        ).cuda()

        # nn.init.zeros_(self.ntc_mlp.state_dict()["2.bias"])
        # nn.init.kaiming_normal_(self.ntc_mlp.state_dict()["0.weight"], nonlinearity='relu')
        # nn.init.kaiming_normal_(self.ntc_mlp.state_dict()["2.weight"], nonlinearity='relu')

        other_params = []
        for params in self.ntc.parameters():
            other_params.append(params)
        for params in self.ntc_mlp.parameters():
            other_params.append(params)
        
        l = [
            {'params': other_params, 'lr': 5e-3, "name": "ntc"},   
        ]

        self.ntc_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.ntc_scheduler_args = get_expon_lr_func(lr_init=5e-3,
                                                    lr_final=0.00001, # 1e-5
                                                    lr_delay_mult=0.01,
                                                    max_steps=20_000)

    def update_learning_rate_for_P_frame(self, iteration):
        for param_group in self.ntc_optimizer.param_groups:
            if param_group["name"] == "ntc":
                lr = self.ntc_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply_for_P_frame(self, path):
        mkdir_p(os.path.dirname(path))

        d_feat, d_offsets, _ = self.get_ntc(self._anchor)
        
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        
        anchor_feat = (self._anchor_feat + d_feat).detach().cpu().numpy()
        offset = (self._offset + d_offsets).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        # conduct_encoding: self._scaling = activation(self._scaling)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        if not self.decoded_version:
            print("inverse scaling")
            scales = self.scaling_inverse_activation(torch.tensor(scales, dtype=torch.float, device="cuda"))
        else:
            scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        torch.save({
            'opacity_mlp': self.mlp_opacity.state_dict(),
            'cov_mlp': self.mlp_cov.state_dict(),
            'color_mlp': self.mlp_color.state_dict(),
            'encoding_xyz': self.encoding_xyz.state_dict(),
            'grid_mlp': self.mlp_grid.state_dict(),
            'x_bound_min': self.x_bound_min,
            'x_bound_max': self.x_bound_max,
        }, path)

    def save_ntc_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        torch.save({
                'ntc': self.ntc.state_dict(),
                'ntc_mlp': self.ntc_mlp.state_dict(),
            }, path)

    def load_mlp_checkpoints(self, path, load_hash_grid=True):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if load_hash_grid:
            self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
            self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])

        if 'x_bound_min' in checkpoint:
            self.x_bound_min = checkpoint['x_bound_min']
            self.x_bound_max = checkpoint['x_bound_max']

            print("load x_bound_min:", self.x_bound_min)
            print("load x_bound_max:", self.x_bound_max)
    
    def load_ntc_checkpoints(self, path):
        checkpoint = torch.load(path)
        self.ntc.load_state_dict(checkpoint['ntc'])
        self.ntc_mlp.load_state_dict(checkpoint['ntc_mlp'])

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        _anchor = self.get_anchor
        # _feat = self._anchor_feat
        _feat = self.get_anchor_features
        _grid_offsets = self._offset
        _scaling = self.get_scaling
        hash_embeddings = self.get_encoding_params()

        feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets)

        bit_feat = self.entropy_gaussian.forward(_feat, mean, scale, Q_feat)
        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32

        print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash)

        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):

        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')

        _anchor = self.get_anchor
        # _feat = self._anchor_feat
        _feat = self.get_anchor_features
        _grid_offsets = self._offset
        _scaling = self.get_scaling

        N = _anchor.shape[0]
        MAX_batch_size = 1_000
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []
        anchor_infos_list = []
        indices_list = []
        min_feat_list = []
        max_feat_list = []
        min_scaling_list = []
        max_scaling_list = []
        min_offsets_list = []
        max_offsets_list = []

        feat_list = []
        scaling_list = []
        offsets_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        torch.save(_anchor, os.path.join(pre_path_name, 'anchor.pkl'))

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            anchor_infos = None
            anchor_infos_list.append(anchor_infos)
            indices_list.append(indices+N_start)

            anchor_sort = _anchor[N_start:N_end][indices]  # [N_num, 3]

            # encode feat
            feat_context = self.calc_interp_feat(anchor_sort)  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat = _feat[N_start:N_end][indices].view(-1)  # [N_num*32]
            feat = STE_multistep.apply(feat, Q_feat, _feat.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_feat, min_feat, max_feat = encoder_gaussian(feat, mean, scale, Q_feat, file_name=feat_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_feat_list.append(bit_feat)
            min_feat_list.append(min_feat)
            max_feat_list.append(max_feat)
            feat_list.append(feat)

            scaling = _scaling[N_start:N_end][indices].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, _scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling, min_scaling, max_scaling = encoder_gaussian(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)
            min_scaling_list.append(min_scaling)
            max_scaling_list.append(max_scaling)
            scaling_list.append(scaling)

            offsets = _grid_offsets[N_start:N_end][indices].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, _grid_offsets.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets, min_offsets, max_offsets = encoder_gaussian(offsets, mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)
            min_offsets_list.append(min_offsets)
            max_offsets_list.append(max_offsets)
            offsets_list.append(offsets)

            torch.cuda.empty_cache()

        bit_anchor = N * 3 * anchor_round_digits
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            p = torch.zeros_like(hash_embeddings).to(torch.float32)
            prob_hash = (((hash_embeddings + 1) / 2).sum() / hash_embeddings.numel()).item()
            p[...] = prob_hash
            bit_hash = encoder(hash_embeddings.view(-1), p.view(-1), file_name=hash_b_name)
        else:
            prob_hash = 0
            bit_hash = hash_embeddings.numel()*32

        indices = torch.cat(indices_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + self.get_mlp_size()[0])/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        return [self._anchor.shape[0], N, MAX_batch_size, anchor_infos_list, min_feat_list, max_feat_list, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_hash], log_info

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, patched_infos):
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')
        [N_full, N, MAX_batch_size, anchor_infos_list, min_feat_list, max_feat_list, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_hash] = patched_infos
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        if self.ste_binary:
            p = torch.zeros_like(self.get_encoding_params()).to(torch.float32)
            p[...] = prob_hash
            hash_embeddings = decoder(p.view(-1), hash_b_name)  # {-1, 1}
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()

        for s in range(steps):
            min_feat = min_feat_list[s]
            max_feat = max_feat_list[s]
            min_scaling = min_scaling_list[s]
            max_scaling = max_scaling_list[s]
            min_offsets = min_offsets_list[s]
            max_offsets = max_offsets_list[s]

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # encode feat
            feat_context = self.calc_interp_feat(anchor_decoded[N_start:N_end])  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
            Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
            Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat_decoded = decoder_gaussian(mean, scale, Q_feat, file_name=feat_b_name, min_value=min_feat, max_value=max_feat)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, min_value=min_scaling, max_value=max_scaling)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            offsets_decoded = decoder_gaussian(mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name, min_value=min_offsets, max_value=max_offsets)
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N_full, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N_full, 6], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        assert self._anchor_feat.shape == _anchor_feat.shape
        self._anchor_feat = nn.Parameter(_anchor_feat)
        assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(_offset)
        # If change the following attributes, decoded_version must be set True
        self.decoded_version = True
        assert self.get_anchor.shape == _anchor.shape
        self._anchor = nn.Parameter(_anchor)
        assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(_scaling)

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_encoding_for_ntc(self, pre_path_name):
        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ntc ...')

        # triplanes
        ntc_2D_b_name = os.path.join(pre_path_name, 'ntc_2D.b')
        ntc_2D_embeddings = self.get_ntc_2D_params()  # {-1, 1}
        if self.ste_binary:
            p = torch.zeros_like(ntc_2D_embeddings).to(torch.float32)
            prob_ntc_2D = (((ntc_2D_embeddings + 1) / 2).sum() / ntc_2D_embeddings.numel()).item()
            p[...] = prob_ntc_2D
            bit_ntc_2D = encoder(ntc_2D_embeddings.view(-1), p.view(-1), file_name=ntc_2D_b_name)
        else:
            prob_ntc_2D = 0
            bit_ntc_2D = ntc_2D_embeddings.numel()*32
        
        # 3D grid
        ntc_3D_b_name = os.path.join(pre_path_name, 'ntc_3D.b')
        ntc_3D_embeddings = self.get_ntc_3D_params()  # {-1, 1}
        if self.ste_binary:
            p = torch.zeros_like(ntc_3D_embeddings).to(torch.float32)
            prob_ntc_3D = (((ntc_3D_embeddings + 1) / 2).sum() / ntc_3D_embeddings.numel()).item()
            p[...] = prob_ntc_3D
            bit_ntc_3D = encoder(ntc_3D_embeddings.view(-1), p.view(-1), file_name=ntc_3D_b_name)
        else:
            prob_ntc_3D = 0
            bit_ntc_3D = ntc_3D_embeddings.numel()*32


        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        ntc_2D = bit_ntc_2D/bit2MB_scale
        ntc_3D = bit_ntc_3D/bit2MB_scale
        ntc_mlp = self.get_ntc_mlp_size()[0]/bit2MB_scale
        Total = ntc_2D + ntc_3D + ntc_mlp
        Total_wo3D = ntc_2D + ntc_mlp

        log_info = f"ntc_2D {round(ntc_2D, 4)}, " \
                   f"ntc_3D {round(ntc_3D, 4)}, " \
                   f"ntc_mlp {round(ntc_mlp, 4)}, " \
                   f"Total {round( Total, 4)}, " \
                   f"Total wo3D {round(Total_wo3D, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"

        return [prob_ntc_2D, prob_ntc_3D], log_info

    @torch.no_grad()
    def conduct_decoding_for_ntc(self, pre_path_name, patched_infos):
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ntc ...')
        [prob_ntc_2D, prob_ntc_3D] = patched_infos

        ntc_2D_b_name = os.path.join(pre_path_name, 'ntc_2D.b')
        ntc_3D_b_name = os.path.join(pre_path_name, 'ntc_3D.b')

        if self.ste_binary:
            p = torch.zeros_like(self.get_ntc_2D_params()).to(torch.float32)
            p[...] = prob_ntc_2D
            ntc_2D_embeddings = decoder(p.view(-1), ntc_2D_b_name)  # {-1, 1}
            ntc_2D_embeddings = ntc_2D_embeddings.view(-1, self.n_features_per_level)

            p = torch.zeros_like(self.get_ntc_3D_params()).to(torch.float32)
            p[...] = prob_ntc_3D
            ntc_3D_embeddings = decoder(p.view(-1), ntc_3D_b_name)  # {-1, 1}
            ntc_3D_embeddings = ntc_3D_embeddings.view(-1, 1)

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        print('Start replacing parameters with decoded ones...')

        if self.ste_binary:
            # 2D
            len_2D = self.ntc.encoding_xy.params.shape[0]
            self.ntc.encoding_xy.params = nn.Parameter(ntc_2D_embeddings[0:len_2D])
            self.ntc.encoding_xz.params = nn.Parameter(ntc_2D_embeddings[len_2D:len_2D*2])
            self.ntc.encoding_yz.params = nn.Parameter(ntc_2D_embeddings[len_2D*2:len_2D*3])

            # 3D
            self.ntc.encoding_xyz.params = nn.Parameter(ntc_3D_embeddings)

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info

