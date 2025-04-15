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
import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat
from utils.loss_utils import l1_loss

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    #
    # feat = pc._anchor_feat[visible_mask]
    feat = pc.get_anchor_features[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    if pc.mode == 'I_frame':
        bit_per_param = None
        bit_per_feat_param = None
        bit_per_scaling_param = None
        bit_per_offsets_param = None
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2
        if is_training:
            if step > pc.step_flag1 and step <= pc.step_flag2:
                # quantization
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

            if step == pc.step_flag2:
                pc.update_anchor_bound()

            if step > pc.step_flag2:
                feat_context = pc.calc_interp_feat(anchor)
                feat_context = pc.get_grid_mlp(feat_context)
                mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                    torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

                Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
                Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
                Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

                choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
                feat_chosen = feat[choose_idx]
                grid_scaling_chosen = grid_scaling[choose_idx]
                grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3*pc.n_offsets)
                mean = mean[choose_idx]
                scale = scale[choose_idx]
                mean_scaling = mean_scaling[choose_idx]
                scale_scaling = scale_scaling[choose_idx]
                mean_offsets = mean_offsets[choose_idx]
                scale_offsets = scale_offsets[choose_idx]
                Q_feat = Q_feat[choose_idx]
                Q_scaling = Q_scaling[choose_idx]
                Q_offsets = Q_offsets[choose_idx]
                bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
                bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
                bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc._offset.mean())
                bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
                bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
                bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
                bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                                (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

        elif not pc.decoded_version: # training时test
            torch.cuda.synchronize(); t1 = time.time()
            feat_context = pc.calc_interp_feat(anchor)
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(pc.get_grid_mlp(feat_context), split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N_visible_anchor, 1]
            feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
            grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
            grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()
            torch.cuda.synchronize(); time_sub = time.time() - t1
        
    elif pc.mode == 'P_frame':
        d_feat, d_offsets, anchor_loss = pc.get_ntc(anchor)
        feat = feat.detach() + d_feat
        grid_offsets = grid_offsets.detach() + d_offsets

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]

    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]

    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

    if is_training and pc.mode == 'I_frame':
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    elif is_training and pc.mode == 'P_frame':
        # d_xyz = d_offsets.view([-1, 3])[mask] * scaling_repeat[:, :3]
        d_xyz = 0
        return xyz, color, opacity, scaling, rot, d_xyz, anchor_loss
    else:
        return xyz, color, opacity, scaling, rot, time_sub

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training and pc.mode == 'I_frame':
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
    elif is_training and pc.mode == 'P_frame':
        xyz, color, opacity, scaling, rot, d_xyz, anchor_loss = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
    else:
        xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    rendered_image_p = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training and pc.mode == 'I_frame':
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                }
    elif is_training and pc.mode == "P_frame":
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "anchor_loss": anchor_loss,
                "render_p": rendered_image_p,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub,
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    
    if not pc.enable_filter:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
        return visible_mask
    
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        scales = pc.get_scaling  # requires_grad = True
        rotations = pc.get_rotation  # requires_grad = True

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )

    return radii_pure > 0
