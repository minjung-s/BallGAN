# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""
# import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.pdb_util import ForkedPdb

class MipRayMarcher2(nn.Module):
    def __init__(self):
        super().__init__()


    def run_forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2


        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)


        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]
        weight_bg = torch.cumprod(alpha_shifted, -2)[:, :,-1].unsqueeze(-1)
        

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        # weights_norm = weights/weights.sum(2).unsqueeze(-2)
        # ForkedPdb().set_trace()
        composite_depth = torch.sum(weights * depths_mid, -2)/ weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)
    
        return composite_rgb, composite_depth, weights, alpha, weight_bg

    def run_forward_withBG(self, colors, densities, depths, rendering_options, color_bg, depth_bg):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        color_bg = color_bg.unsqueeze(-2)
        depth_bg = depth_bg.unsqueeze(-1).unsqueeze(-1)
        

        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        depth_total = torch.cat([depths,depth_bg],-2)
        # density_delta = density_delta*1.8
        alpha = 1 - torch.exp(-density_delta)
        # alpha[:,:,69:,:] = 0
        # import pdb;pdb.set_trace()


        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]
        weight_bg = torch.cumprod(alpha_shifted, -2)[:, :,-1].unsqueeze(-1)

        
        color_all = torch.cat([colors_mid,color_bg],-2)
        weights_all = torch.cat([weights,weight_bg],-2)
        depths_all = torch.cat([depths_mid, depth_bg],-2)

        del deltas, colors_mid, densities_mid, depth_total, depths_mid, color_bg, depth_bg
        torch.cuda.empty_cache()

        composite_rgb = torch.sum(weights_all * color_all, -2)
        weight_total = weights_all.sum(2) # 1
        composite_depth = torch.sum(weights_all * depths_all, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths_all), torch.max(depths_all))
        del depths_all
        torch.cuda.empty_cache()

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights, alpha, weight_bg 

    def run_forward_onlyBG(self, colors, densities, depths, rendering_options, color_bg, depth_bg):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        color_bg = color_bg.unsqueeze(-2)
        depth_bg = depth_bg.unsqueeze(-1).unsqueeze(-1)
        

        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        depth_total = torch.cat([depths,depth_bg],-2)
        alpha = 1 - torch.exp(-density_delta)

        
        # ForkedPdb().set_trace()

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]
        weight_bg = torch.cumprod(alpha_shifted, -2)[:, :,-1].unsqueeze(-1)

        # ForkedPdb().set_trace()
        composite_rgb = torch.sum(weight_bg * color_bg, -2)
        # composite_rgb = torch.sum(1 * color_bg, -2)
        weight_bg = weight_bg.sum(2)

        composite_depth = weight_bg * depth_bg.squeeze(-1) 

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depth_bg), torch.max(depth_bg))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weight_bg, alpha, weight_bg


    def forward(self, colors, densities, depths, rendering_options, color_bg=None, depth_bg=None, only_fg=False, only_bg=False):
        if color_bg is not None:
            if only_fg:
                composite_rgb, composite_depth, weights, alpha, bg_transmittance = self.run_forward(colors, densities, depths, rendering_options)
            elif only_bg:
                composite_rgb, composite_depth, weights, alpha, bg_transmittance = self.run_forward_onlyBG(colors, densities, depths, rendering_options, color_bg, depth_bg)
            else:
                composite_rgb, composite_depth, weights, alpha, bg_transmittance = self.run_forward_withBG(colors, densities, depths, rendering_options, color_bg, depth_bg)
        else:
            composite_rgb, composite_depth, weights, alpha, bg_transmittance = self.run_forward(colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights, alpha, bg_transmittance
