import torch
import torch.nn as nn
import torch.nn.functional as F
from training.networks_stylegan2 import *
from torch.autograd import grad

import math
from dnnlib.geometry import (
    positional_encoding, upsample, downsample
)
from torch_utils.pdb_util import ForkedPdb

@persistence.persistent_class
class Style2Layer(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        w_dim, 
        activation='lrelu', 
        resample_filter=[1,3,3,1],
        magnitude_ema_beta = -1,           # -1 means not using magnitude ema
        **unused_kwargs):

        # simplified version of SynthesisLayer 
        # no noise, kernel size forced to be 1x1, used in NeRF block
        super().__init__()
        self.activation = activation
        self.conv_clamp = None
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = 0
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.w_dim = w_dim
        self.in_features = in_channels
        self.out_features = out_channels
        memory_format = torch.contiguous_format
        
        if w_dim > 0:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.weight = torch.nn.Parameter(
               torch.randn([out_channels, in_channels, 1, 1]).to(memory_format=memory_format))
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.weight_gain = 1.

            # initialization
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, style={}'.format(
            self.in_features, self.out_features, self.w_dim
        )

    def forward(self, x, w=None, fused_modconv=None, gain=1, up=1, **unused_kwargs):
        flip_weight = True # (up == 1) # slightly faster HACK
        act = self.activation
        
        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function('update_magnitude_ema'):
                    magnitude_cur = x.detach().to(torch.float32).square().mean()
                    self.w_avg.copy_(magnitude_cur.lerp(self.w_avg, self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = not self.training

        if self.w_dim > 0:           # modulated convolution
            assert x.ndim == 4,  "currently not support modulated MLP"
            styles = self.affine(w)      # Batch x style_dim
            if x.size(0) > styles.size(0):
                styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))
            
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=None, up=up,
                padding=self.padding, resample_filter=self.resample_filter, 
                flip_weight=flip_weight, fused_modconv=fused_modconv)

            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, self.bias.to(x.dtype), act=act, gain=act_gain, clamp=act_clamp)
        
        else:
            # print(x.ndim == 2)
            if x.ndim == 2:  # MLP mode #ㄴㄴX
                x = F.relu(F.linear(x, self.weight, self.bias.to(x.dtype)))
            else: #BG는 여기
                # ForkedPdb().set_trace() 
                x = F.relu(F.conv2d(x.to(self.weight.dtype), self.weight[:,:,None, None], self.bias)) #wegiht [B,72,1,1] for Sphere NeRF
                # x = bias_act.bias_act(x, self.bias.to(x.dtype), act='relu')
        return x



@persistence.persistent_class
class NeRFBlock(nn.Module):
    ''' 
    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.
    '''
    # dimensions
    input_dim            = 2
    w_dim                = 512   # style latent
    z_dim                = 0    # input latent
    rgb_out_dim          = 32 #EG3D
    hidden_size          = 128
    n_blocks             = 4
    img_channels         = 3
    magnitude_ema_beta   = -1
    disable_latents      = False
    max_batch_size       = 2 ** 18
    shuffle_factor       = 1
    implementation       = 'batch_reshape'  # option: [flatten_2d, batch_reshape]

    # architecture settings
    activation           = 'lrelu'
    use_skip             = False 
    use_viewdirs         = False
    add_rgb              = False
    predict_rgb          = False
    inverse_sphere       = False
    merge_sigma_feat     = False   # use one MLP for sigma and features
    no_sigma             = False   # do not predict sigma, only output features
    R                    = 3.
    using_ws             = False

    use_normal           = False
    volsdf_exp_beta      = False
    normalized_feat      = False
    final_sigmoid_act    = False

    # positional encoding inpuut
    use_pos              = False
    n_freq_posenc        = 10
    n_freq_posenc_views  = 4
    downscale_p_by       = 1
    gauss_dim_pos        = 20 
    gauss_dim_view       = 4 
    gauss_std            = 10.
    positional_encoding  = "normal"

    def __init__(self, nerf_kwargs):
        super().__init__()
        for key in nerf_kwargs:
            if hasattr(self, key):
                setattr(self, key, nerf_kwargs[key])
        
        # ----------- input module -------------------------
        D = self.input_dim
        self.D = D

        if self.positional_encoding == 'gauss':
            rng = np.random.RandomState(2021)
            B_pos  = self.gauss_std * torch.from_numpy(rng.randn(D, self.gauss_dim_pos * D)).float()
            B_view = self.gauss_std * torch.from_numpy(rng.randn(3, self.gauss_dim_view * 3)).float()
            self.register_buffer("B_pos", B_pos)
            self.register_buffer("B_view", B_view)
            dim_embed = D * self.gauss_dim_pos * 2
            dim_embed_view = 3 * self.gauss_dim_view * 2
        elif self.positional_encoding == 'normal':
            dim_embed = D * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2
        else:  # not using positional encoding
            dim_embed, dim_embed_view = D, 3
        if self.use_pos:
            dim_embed, dim_embed_view = dim_embed + D, dim_embed_view + 3

        self.dim_embed = dim_embed
        self.dim_embed_view = dim_embed_view

        # ------------ Layers --------------------------
        assert not (self.add_rgb and self.predict_rgb), "only one could be achieved"
        assert not ((self.use_viewdirs or self.use_normal) and (self.merge_sigma_feat or self.no_sigma)), \
            "merged MLP does not support."
        
        if self.disable_latents:
            w_dim = 0
        elif self.z_dim > 0:  # if input global latents, disable using style vectors -> BG
            w_dim, dim_embed, dim_embed_view = 0, dim_embed + self.z_dim, dim_embed_view + self.z_dim
        else:
            w_dim = self.w_dim

        final_in_dim = self.hidden_size
        if self.use_normal:
            final_in_dim += D

        final_out_dim = self.rgb_out_dim * self.shuffle_factor
        if self.merge_sigma_feat:
            final_out_dim += self.shuffle_factor  # predicting sigma
        if self.add_rgb:
            final_out_dim += self.img_channels
        self.final_out_dim = final_out_dim        

        # print(self.bgpts_sphere, dim_embed,self.hidden_size, w_dim)
        self.fc_in  = Style2Layer(dim_embed, self.hidden_size, w_dim, activation=self.activation)
        self.num_ws = 1
        self.skip_layer = self.n_blocks // 2 - 1 if self.use_skip else None
            
        if self.n_blocks > 1:
            self.blocks = nn.ModuleList([
                Style2Layer(
                    self.hidden_size if i != self.skip_layer else self.hidden_size + dim_embed, 
                    self.hidden_size, 
                    w_dim, activation=self.activation,
                    magnitude_ema_beta=self.magnitude_ema_beta)
                for i in range(self.n_blocks - 1)])
            self.num_ws += (self.n_blocks - 1)

        self.feat_out = ToRGBLayer(final_in_dim, final_out_dim, w_dim, kernel_size=1)
        if (self.z_dim == 0 and (not self.disable_latents)):
            self.num_ws += 1
        else:
            self.num_ws = 0        
        
        if self.predict_rgb:   # predict RGB over features
            self.to_rgb = Conv2dLayer(final_out_dim, self.img_channels * self.shuffle_factor, kernel_size=1, activation='linear')
        
    def set_steps(self, steps):
        if hasattr(self, "steps"):
            self.steps.fill_(steps)
        
    def transform_points(self, p, views=False):
        p = p / self.downscale_p_by
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = positional_encoding(p, B, 'gauss', self.use_pos)
        elif self.positional_encoding == 'normal':
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = positional_encoding(p, L, 'normal', self.use_pos)
        else:
            p_transformed = p
        return p_transformed

    def forward(self, p_in, z_bg=None, ws=None):
        feat = self.forward_nerf(p_in, z_bg, ws)
        return feat
    
    def forward_nerf(self, p_in, z_bg, ws):
        assert ((z_bg is not None) and (ws is None)) or ((z_bg is None) and (ws is not None))
        # height, width, n_steps, use_normal = option
        # forward nerf feature networks
        #print("start")
        #p#rint("start :",p_in.shape)
        p = self.transform_points(p_in)

        if (self.z_dim > 0) and (not self.disable_latents):
            assert (z_bg is not None) and (ws is None)

            z_bg = z_bg.unsqueeze(-2).repeat(1,p.shape[1],1)
            p = torch.cat([p, z_bg], -1)
        batch, hw, pos_dim = p.shape
        h, w = int(math.sqrt(p.shape[1])), int(math.sqrt(p.shape[1]))
        p = p.reshape(batch, h, w, pos_dim)
        p = p.permute(0,3,1,2)    # BS x C x H x W
        #print("PE :",p#.shape)
        #if w == h == 1:  # MLP
        #    p = p.squeeze(-1).squeeze(-1)

        net = self.fc_in(p, ws[:, 0] if ws is not None else None)
        
        #print("p_in :",net#.shape)
        # ForkedPdb().set_trace()
        if self.n_blocks > 1:
            # print("self.blocks : ",self.blocks)
            for idx, layer in enumerate(self.blocks):
                ws_i = ws[:, idx + 1] if ws is not None else None
                if (self.skip_layer is not None) and (idx == self.skip_layer):
                    net = torch.cat([net, p], 1)
                net = layer(net, ws_i, up=1)
                #print(f"p_{idx#} : {net.shape}")
        

        # forward to get the final results
        w_idx = self.n_blocks  # fc_in, self.blocks
        feat_inputs = [net]

        ws_i = ws[:, -1] if ws is not None else None
        net = torch.cat(feat_inputs, 1) if len(feat_inputs) > 1 else net
        feat_out = self.feat_out(net, ws_i)  # this is used for lowres output
        #print("p#_out :",feat_out.shape)
        #print("do#ne")
        feat_out = feat_out.reshape(batch,self.final_out_dim,hw)
        feat_out = feat_out.permute(0,2,1)
        # feat_out = torch.sigmoid(feat_out)*(1 + 2*0.001) - 0.001 

        if self.predict_rgb:
            rgb = self.to_rgb(feat_out)
            if self.final_sigmoid_act:
                rgb = torch.sigmoid(rgb)    
            if self.normalized_feat:
                feat_out = feat_out / (1e-7 + feat_out.norm(dim=-1, keepdim=True))
            feat_out = torch.cat([rgb, feat_out], 1)
        feat_out = torch.sigmoid(feat_out)*(1 + 2*0.001) - 0.001 
        # transform back
        # if feat_out.ndim == 2:  # mlp mode
        #     # sigma_out = rearrange(sigma_out, '(b s) d -> b s d', s=n_steps) if sigma_out is not None else None
        #     # # feat_out  = rearrange(feat_out,  '(b s) d -> b s d', s=n_steps)
        # else:
        #     # sigma_out = rearrange(sigma_out, '(b s) d h w -> b (h w s) d', s=n_steps) if sigma_out is not None else None
            # # feat_out  = rearrange(feat_out,  '(b s) d h w -> b (h w s) d', s=n_steps)
        return feat_out