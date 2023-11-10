# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.networks_stylegan2 import MappingNetwork
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from training.Ball import *
from training.pdb_util import ForkedPdb


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        z_bg_dim,                   ## Input bg latent (Z_bg) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        w_bg_dim,                   ## Intermediate bg latent (W_bg) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        only_fg             = False,
        only_bg             = False,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        BG_kwargs           = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.z_bg_dim=z_bg_dim #
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.w_bg_dim=w_bg_dim #
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()

        
        
        if BG_kwargs['bg_ver'] == "ver1" :
            self.BG_mapping = MappingNetwork(z_dim = BG_kwargs['mapping_kwargs']['z_dim'], c_dim=0, w_dim = BG_kwargs['mapping_kwargs']['w_dim'], num_ws = BG_kwargs['n_blocks']+1, num_layers = BG_kwargs['mapping_kwargs']['num_layers'])
        self.BG_Net = NeRFBlock(BG_kwargs) #

        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)



        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'sigma_act': rendering_kwargs.get('sigma_act', 'linear') })
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self.BG_kwargs = BG_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
    
    def mapping_bg(self, z, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        ws = self.BG_mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return ws


    def synthesis(self, ws, c, z_bg=None, ws_bg=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, only_fg=False, only_bg=False, for_alpha_vis=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        H = W = self.neural_rendering_resolution
        
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        del cam2world_matrix
        del intrinsics
        torch.cuda.empty_cache()
        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        if for_alpha_vis:
            feature_samples, depth_samples, weights_samples, _, _, alpha, bg_transmittance = self.renderer.forward_for_checkalpha(planes, self.decoder, ray_origins, ray_directions, z_bg, ws_bg, self.BG_Net, self.rendering_kwargs, only_fg=only_fg, only_bg=only_bg) # channels last
    
        else:
            feature_samples, depth_samples, weights_samples, fg_density, fg_dist, alpha, bg_transmittance = self.renderer(planes, self.decoder, ray_origins, ray_directions, z_bg, ws_bg, self.BG_Net, self.rendering_kwargs, only_fg=only_fg, only_bg=only_bg) # channels last

        # Reshape into 'raw' neural-rendered image
        
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        if not for_alpha_vis:
            fg_density = fg_density.reshape(N, H, W, fg_density.shape[2], 1).contiguous().squeeze(-1)
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'fg_wegihts':weights_samples, 'fg_di': fg_dist if not for_alpha_vis else None, 'fg_density': fg_density if not for_alpha_vis else None, 'alpha': alpha, 'bg_transmittance':bg_transmittance}

    def synthesis_fgcam(self, ws, c, c_bg, z_bg=None, ws_bg=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, only_fg=False, only_bg=False, for_alpha_vis=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        cam2world_matrix_bg = c_bg[:, :16].view(-1, 4, 4)
        intrinsics_bg = c_bg[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        ray_origins_bg, ray_directions_bg = self.ray_sampler(cam2world_matrix_bg, intrinsics_bg, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, fg_density, fg_dist, alpha, bg_transmittance = self.renderer.forward_fgcam(planes, self.decoder, ray_origins, ray_directions,ray_origins_bg, ray_directions_bg, z_bg, ws_bg, self.BG_Net, self.rendering_kwargs, only_fg=only_fg, only_bg=only_bg) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        if not for_alpha_vis:
            fg_density = fg_density.reshape(N, H, W, fg_density.shape[2], 1).contiguous().squeeze(-1)
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'fg_di': fg_dist if not for_alpha_vis else None, 'fg_density': fg_density if not for_alpha_vis else None, 'alpha': alpha, 'bg_transmittance':bg_transmittance}



    def sample(self, coordinates, directions, z, z_bg, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # ws_bg = self.BG_mapping(z_bg, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, z_bg, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, only_fg=False, only_bg=False, for_alpha_vis=False,**synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        if self.BG_kwargs['bg_ver'] == "ver1":
            ws_bg = self.mapping_bg(z_bg, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            return self.synthesis(ws, c, z_bg= None, ws_bg=ws_bg, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, only_fg=only_fg, only_bg=only_bg, for_alpha_vis=for_alpha_vis, **synthesis_kwargs)
        elif self.BG_kwargs['bg_ver'] == "ver0":
            return self.synthesis(ws, c, z_bg= z_bg, ws_bg=None, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, only_fg=only_fg, only_bg=only_bg, for_alpha_vis=for_alpha_vis, **synthesis_kwargs)
    
from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            # FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        self.rgb_out = FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        self.sigma_out = FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=options['decoder_lr_mul'], activation=options['sigma_act'])
        torch.nn.init.zeros_(self.sigma_out.weight)
        torch.nn.init.zeros_(self.sigma_out.bias)

        # self.guassian_filter = gkern(64, std=32).reshape(-1).reshape(1,4096,1,1).to(self.net.device)



        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1) # 논문에서는 SUM이었는데, 구현은 mean
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        rgb = self.rgb_out(x)
        sigma = self.sigma_out(x)

        # x = self.net(x)
        # x = x.view(N, M, -1)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        # sigma = x[..., 0:1]
        
        return {'rgb': rgb, 'sigma': sigma}
