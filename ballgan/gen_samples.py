# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import torch.nn as nn
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import math
import cv2


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from torch_utils.pdb_util import ForkedPdb


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
# BG setting
@click.option('--bg_ver',       help='version of BG network',                                   type=click.Choice(['ver0', 'ver1']), default='ver1', show_default=True)
@click.option('--n_blocks',     help='number of BG net block', metavar='INT',                   type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--radius',       help='radius of background sphere', metavar='FLOAT',            type=click.FloatRange(min=0), default=3., show_default=True)
@click.option('--bg_pe',        help='PE order', metavar='FLOAT',                               type=click.FloatRange(min=0), default=3.141592653589793*2, show_default=True)
@click.option('--hidden_size',  help='BG net hidden size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--z_bg_dim',     help='z dim of background', metavar='INT',                      type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--w_bg_dim',     help='w dim of background', metavar='INT',                      type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--bg_mapping',   help='number of bg mapping network layers', metavar='INT',      type=click.IntRange(min=1), default=8, show_default=True)

# Rendering option
@click.option('--view',   help='number of rendering views', metavar='view',      type=click.IntRange(min=1), default=3, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    view: int,
    **kwargs
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        opts = dnnlib.EasyDict(kwargs) # Command line arguments.

        G_kwargs = G.init_kwargs
        if opts.bg_ver == "ver0":

            BG_options = {
                'bg_ver': "ver0",
                'input_dim': 2,
                'z_dim': 512,
                'w_dim': 0,
                'n_blocks': 4,
                'R': 3.,
                'positional_encoding': "normal",
                'hidden_size': 64,
                'downscale_p_by': 3.141592653589793*2,
                'skips': [],
                'inverse_sphere': True,
                'no_sigma': True, #
                'use_style': "StyleGAN2",
                'use_viewdirs': False,
                'bgpts_sphere' : True, #
                'using_ws': True,
                'mapping_kwargs': {}
            }
            if opts.z_bg_dim != 512:
                BG_options.update({
                    'z_dim': opts.z_bg_dim,
                })
            G_kwargs.z_bg_dim = BG_options['z_dim']
            G_kwargs.w_bg_dim = BG_options['w_dim']

        elif opts.bg_ver =="ver1":
            BG_options = {
                'bg_ver': "ver1",
                'input_dim': 2,
                'z_dim': 0,
                'w_dim': 512,
                'n_blocks': 4,
                'R': 3.,
                'positional_encoding': "normal",
                'hidden_size': 64,
                'downscale_p_by': 3.141592653589793*2,
                'skips': [],
                'inverse_sphere': True,
                'no_sigma': True, #
                'use_style': "StyleGAN2",
                'use_viewdirs': False,
                'bgpts_sphere' : True, #
                'using_ws': True,
                'mapping_kwargs': {
                    'z_dim': 512,
                    'w_dim': 512,
                    'num_layers': 8,
                }
            }

            if opts.w_bg_dim != 512 or opts.z_bg_dim != 512 or opts.bg_mapping != 8:
                BG_options.update({
                    'w_dim': opts.w_bg_dim,
                    'mapping_kwargs': {
                        'z_dim': opts.z_bg_dim,
                        'w_dim': opts.w_bg_dim,
                        'num_layers': opts.bg_mapping,
                    }
                })
            G_kwargs.z_bg_dim = BG_options['mapping_kwargs']['z_dim']
            G_kwargs.w_bg_dim = BG_options['mapping_kwargs']['w_dim']

        if opts.n_blocks != 4 or opts.radius != 3 or opts.hidden_size != 64 or opts.bg_pe != 3.141592653589793*2 or opts.z_bg_dim != 512 :
            BG_options.update({
                'n_blocks': opts.n_blocks,
                'R' : opts.radius,
                'hidden_size': opts.hidden_size,
                'downscale_p_by': opts.bg_pe,
            })
        G_kwargs.BG_kwargs = BG_options


        
        G_new = TriPlaneGenerator(*G.init_args,**G_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    views = np.linspace(-0.3,+0.3,view, endpoint=True)

    # Generate images.
    m = nn.Upsample(scale_factor=4, mode='bilinear')
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        z_bg = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_bg_dim)).to(device)

        
        angle_p = -0.2
        for only_fg, only_bg in  [(False, False)]:
            imgs = []
            imgs_rgba = []
            # masks = []
            for i, angle_y in enumerate(views):
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                
                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                if opts.bg_ver =="ver1":
                    ws_bg = G.mapping_bg(z_bg, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                    img_total = G.synthesis(ws=ws, c=camera_params, z_bg= None, ws_bg=ws_bg, only_fg=False, only_bg=False, noise_mode='const')['image']
                    if only_fg == True:
                        img_fg = G.synthesis(ws=ws, c=camera_params, z_bg= None, ws_bg=ws_bg, only_fg=True, only_bg=True,noise_mode='const')['image_depth']
                    bg_transmittance = G.synthesis(ws=ws, c=camera_params, z_bg= None, ws_bg=ws_bg, only_fg=False, only_bg=False,noise_mode='const')['fg_wegihts']
                elif opts.bg_ver =="ver0":
                    img = G.synthesis(ws=ws, c=camera_params, z_bg=z_bg, ws_bg=None, only_fg=only_fg, only_bg=only_bg)['image']
                
                img_total = (img_total.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                if only_fg == True:
                    img_fg = (img_fg.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                bg_transmittance = bg_transmittance.reshape([1,1,128,128])*255

                alpha = m(bg_transmittance).permute(0, 2, 3, 1).to(torch.uint8)
                tmp = torch.cat((img_total,alpha),-1)
                PIL.Image.fromarray(tmp[0].cpu().numpy(), 'RGBA').save(f'{outdir}/seed{seed:04d}_alpha.png')
                
                imgs.append(img_total)
                imgs_rgba.append(tmp)

            results = torch.cat(imgs, dim=2)
            # results_rgba = torch.cat(imgs_rgba,dim=2)

        # PIL.Image.fromarray(results_rgba, 'RGBA').save(f'{outdir}/seed{seed:04d}_alpha.png')
        PIL.Image.fromarray(results[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        
        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:

                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, z_bg, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
