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
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""


import math
import torch
import torch.nn as nn

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
from torch_utils.pdb_util import ForkedPdb

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape #M = 64*64(featuremap resolution) * (ray point)
    plane_features = plane_features.view(N*n_planes, C, H, W)
    
    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
    
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, ray_origins, ray_directions, z_bg, ws_bg, BG_Net, rendering_options, only_fg=False, only_bg=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape #B, 4096=64x64, 64

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3) #([8, 262144=featuremap res(64x64)*query(64), 3]) 
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1]) #[B,feature res 64x64, sampled point 64,32]
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1) #[B,feature res 64x64, sampled point 64,1]
        
        color_bg, depth_bg = self.Ball_feature(z_bg, ws_bg, BG_Net, ray_origins, ray_directions, rendering_options['ray_end']) #[B,feature res 64x64, sampled point 1, 32]
    
        """
        colors_BG = self.Ball_feature(z_bg, BG_Net, ray_origins, ray_directions)
        p_bg, r_bg, di = self.C.get_evaluation_points_bg_ours(pixels_world, camera_world, self.C.depth_range[1]+di_interval)
        
        feat, _ = bg_nerf(p_bg, r_bg, z_shape_bg, z_app_bg, ws=styles_bg, shape=bg_shape)
        """

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights, _, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            
            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)
            # guassian_filter = gkern(64, std=32).reshape(-1).reshape(1,num_rays,1,1).to(all_colors.device)
            # all_densities = guassian_filter*all_densities
            # all_colors = guassian_filter*all_colors

            # Aggregate
            rgb_final, depth_final, weights, alpha, bg_transmittance = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options, color_bg, depth_bg, only_fg, only_bg)
        else:
            rgb_final, depth_final, weights, alpha, bg_transmittance = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        return rgb_final, depth_final, weights.sum(2), densities_fine, depths_fine, alpha, bg_transmittance

    def forward_fgcam(self, planes, decoder, ray_origins, ray_directions, ray_origins_bg, ray_directions_bg, z_bg, ws_bg, BG_Net, rendering_options, only_fg=False, only_bg=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape #B, 4096=64x64, 64

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3) #([8, 262144=featuremap res(64x64)*query(64), 3]) 
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1]) #[B,feature res 64x64, sampled point 64,32]
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1) #[B,feature res 64x64, sampled point 64,1]
        
        color_bg, depth_bg = self.Ball_feature(z_bg, ws_bg, BG_Net, ray_origins_bg, ray_directions_bg, rendering_options['ray_end']) #[B,feature res 64x64, sampled point 1, 32]
    
        """
        colors_BG = self.Ball_feature(z_bg, BG_Net, ray_origins, ray_directions)
        p_bg, r_bg, di = self.C.get_evaluation_points_bg_ours(pixels_world, camera_world, self.C.depth_range[1]+di_interval)
        
        feat, _ = bg_nerf(p_bg, r_bg, z_shape_bg, z_app_bg, ws=styles_bg, shape=bg_shape)
        """

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights, _, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            
            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)
            # guassian_filter = gkern(64, std=32).reshape(-1).reshape(1,num_rays,1,1).to(all_colors.device)
            # all_densities = guassian_filter*all_densities
            # all_colors = guassian_filter*all_colors

            # Aggregate
            rgb_final, depth_final, weights, alpha, bg_transmittance = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options, color_bg, depth_bg, only_fg, only_bg)
        else:
            rgb_final, depth_final, weights, alpha, bg_transmittance = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        return rgb_final, depth_final, weights.sum(2), densities_fine, depths_fine, alpha, bg_transmittance


    def forward_for_checkalpha(self, planes, decoder, ray_origins, ray_directions, z_bg, ws_bg, BG_Net, rendering_options, only_fg=False, only_bg=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)
        # samples_per_ray = 1000
        depths_stratified = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], 100, rendering_options['disparity_space_sampling'])
        batch_size, num_rays, samples_per_ray, _ = depths_stratified.shape #B, 4096=64x64, 64
        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_stratified * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3) #([8, 262144=featuremap res(64x64)*query(64), 3]) 
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1]) #[B,feature res 64x64, sampled point 64,32]
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1) #[B,feature res 64x64, sampled point 64,1]
        
        color_bg, depth_bg = self.Ball_feature(z_bg, ws_bg, BG_Net, ray_origins, ray_directions, rendering_options['ray_end']) #[B,feature res 64x64, sampled point 1, 32]

        rgb_final, depth_final, weights, alpha, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_stratified, rendering_options)
        
        # black = torch.tensor([0,0,0]).repeat(64,64,99,1).numpy()
        # alpha_batch = alpha[0]
        # alpha_batch = (alpha_batch - alpha_batch.min())/(alpha_batch.max()-alpha_batch.min())
        # tmp = torch.reshape(alpha_batch,(64,64,99,1)).cpu().numpy()
        # tmp_fill = torch.reshape((alpha_batch).squeeze(-1),(64,64,99)).cpu().numpy()
        
        # alpha_map = np.concatenate((black,tmp),-1)


        # counter_sp = range(64)
        # counter_depth = range(99)


        # axes = [64, 64, 99]
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.voxels(tmp_fill, facecolors=alpha_map)
        # plt.savefig('savefig_default.png')
        
        # ForkedPdb().set_trace()
        return rgb_final, depth_final, weights.sum(2), _, _, alpha, _


    def Ball_feature(self, z_bg, ws_bg, BG_Net, ray_origins, ray_directions, dist_min):
        # ray_origins, ray_directions, BG_Net.R -> p_in -> p_bg,depth
        
        p_bg, depth_bg = self.get_evaluation_points_bg_ours(ray_origins, ray_directions, BG_Net.R, dist_min + 0.1)
        # ForkedPdb().set_trace()
        color_bg = BG_Net(p_bg, z_bg, ws_bg)
        
        return color_bg, depth_bg #[B,feature res 64x64, sampled point 1, 32]

    def get_evaluation_points_bg_ours(self, ray_origins, ray_directions, R, dist_min):
        B, _, cor_dim = ray_origins.shape

        p_dist = self.get_background_points_ours(ray_origins, ray_directions, R, dist_min)

        p_bg_car =  ray_origins.contiguous() + \
            p_dist.unsqueeze(-1).contiguous() * \
            ray_directions.contiguous()

        p_bg_car = p_bg_car.reshape(B,-1,cor_dim)
        p_bg_sp = self.cartesian_to_spherical_ours(p_bg_car, R)
        return p_bg_sp, p_dist


    def get_background_points_ours(self, ray_origins, ray_directions, R, dist_min):
        d1 = -torch.sum(ray_directions*ray_origins,dim=-1)/torch.sum(ray_directions*ray_directions,dim=-1)
        p = ray_origins + d1.unsqueeze(-1)*ray_directions
        ray_directions_cos = 1./torch.norm(ray_directions, dim=-1)
        p_norm_sq = torch.sum(p*p,dim=-1)
        if (p_norm_sq >= R).any():
            raise Exception('Not all your camera are bounded by the sphere of radius')
        d2 = self.setting_r_ours(R, p_norm_sq,ray_directions_cos)
        # torch.sqrt(r*r - p_norm_sq)*ray_d_cos

        # if torch.isnan(ray_directions) and (d1 + d2).max() > dist_min:
        #     raise Exception('Not all your camera are bounded by the sphere of radius')
        # ForkedPdb().set_trace()
        # assert (d1 + d2).max() > dist_min

        return d1 + d2 #BG depth

    def setting_r_ours(self, R, p_norm_sq, ray_directions_cos):
        return torch.sqrt(R*R - p_norm_sq)*ray_directions_cos

    def cartesian_to_spherical_ours(self, point, R):
        assert point.shape[-1] == 3
        theta = torch.arccos(point[:,:,1]/R)      

        phi = ((point[:,:,2]>0)&(point[:,:,0]>=0))*torch.atan2(point[:,:,0],point[:,:,2]) \
            + ((point[:,:,2]<0)&(point[:,:,0]>=0))*torch.atan2(point[:,:,0],point[:,:,2]) \
            + ((point[:,:,2]<0)&(point[:,:,0]<0))*(torch.atan2(point[:,:,0],point[:,:,2])+2*math.pi) \
            + ((point[:,:,2]>0)&(point[:,:,0]<0))*(torch.atan2(point[:,:,0],point[:,:,2])+2*math.pi) \
            + ((point[:,:,2]==0)&(point[:,:,0]>0))*(math.pi/2) \
            + ((point[:,:,2]==0)&(point[:,:,0]<0))*(3*math.pi/2)

        # theta = torch.arccos(point[:,:,2]/R)      

        # phi = ((point[:,:,0]>0)&(point[:,:,1]>=0))*torch.atan2(point[:,:,1],point[:,:,0]) \
        #     + ((point[:,:,0]<0)&(point[:,:,1]>=0))*torch.atan2(point[:,:,1],point[:,:,0]) \
        #     + ((point[:,:,0]<0)&(point[:,:,1]<0))*(torch.atan2(point[:,:,1],point[:,:,0])+2*math.pi) \
        #     + ((point[:,:,0]>0)&(point[:,:,1]<0))*(torch.atan2(point[:,:,1],point[:,:,0])+2*math.pi) \
        #     + ((point[:,:,0]==0)&(point[:,:,1]>0))*(math.pi/2) \
        #     + ((point[:,:,0]==0)&(point[:,:,1]<0))*(3*math.pi/2)

        # # phi = ((point[:,:,0]>0)&(point[:,:,1]>0))*torch.arctan(point[:,:,1]/point[:,:,0]) \
        # #     + ((point[:,:,0]<0)&(point[:,:,1]>0))*(torch.arctan(torch.abs(point[:,:,1]/point[:,:,0]))+math.pi) \
        # #     + ((point[:,:,0]<0)&(point[:,:,1]<0))*(torch.arctan(torch.abs(point[:,:,1]/point[:,:,0]))+math.pi) \
        # #     + ((point[:,:,0]>0)&(point[:,:,1]<0))*(torch.arctan(torch.abs(point[:,:,1]/point[:,:,0]))+2*math.pi/3) \
        # #     + ((point[:,:,0]==0)&(point[:,:,1]>0))*(0) \
        # #     + ((point[:,:,0]==0)&(point[:,:,1]>0))*(math.pi/2) \
        # #     + ((point[:,:,0]==0)&(point[:,:,1]<0))*(3*math.pi/2)

        
        p_sphere = torch.stack([theta,phi],dim=2)
        return p_sphere

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
        out = decoder(sampled_features, sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out

    # def run_model_withBG(self, planes, decoder, sample_coordinates, sample_directions, options):
    #     sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

    #     color_bg, depth_bg = self.Ball_feature(ws_bg, BG_Net, ray_origins, ray_directions, rendering_options['ray_end']) #[B,feature res 64x64, sampled point 1, 32]


    #     out = decoder(sampled_features, sample_directions)
    #     if options.get('density_noise', 0) > 0:
    #         out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
    #     return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)

        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        
        all_densities = torch.cat([densities1, densities2], dim = -2)
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples