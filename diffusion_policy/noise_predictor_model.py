from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from film import FiLM   
from pos_encoding import PositionalEncoding, SinusoidalPosEmb

# Based on https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py
# Modified for simplicity and readability
# FiLM and PositionalEncoding classes are recreated in another file for clarity


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    

    
class ConditionalResBlock1D(nn.Module):
    """ 
    Take both input and conditional feature
    Use FiLM to modulate the input by the features
    Use skip connection to add the input to the output
    """
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()
 

        self.block1=Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.block2=Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)


        self.film = FiLM(out_channels, cond_dim) 
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : input
            cond : conditional features 
        '''
        out = self.block1(x) 
        out = self.film(out, cond)            # input modulated by the conditions.
        out = self.block2(out)
        out = out + self.residual_conv(x)     #skip connection with correct dim
        return out 
    


class DownModule(nn.Module):
    """ 
    contraction path of UNet
    pass through the conditional resblock, increase the channel size
    Then downsample to reduce the spatial dimension
    """
    def __init__(self, dim_in, dim_out, cond_dim, kernel_size, n_groups, is_last=False):
        super().__init__()
        self.crb=ConditionalResBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups) 
        if is_last:
            self.downsample = nn.Identity()
        else: 
            self.downsample  =  nn.Conv1d(dim_out, dim_out, 3, 2, 1)
 
    def forward(self, x, cond):
        x = self.crb(x, cond)
        x_small = self.downsample(x)

        return x, x_small
    
class UpModule(nn.Module):
    """ 
    expansion path of UNet
    pass through the conditional resblock, decrease the channel size
    Then upsample to increase the spatial dimension
    """
    def __init__(self, dim_in, dim_out, cond_dim, kernel_size, n_groups, is_last=False):
        super().__init__()
        if is_last:
            self.upsample = nn.Identity()
        else: 
            self.upsample = nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1)
    
        self.crb = ConditionalResBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups) 

    def forward(self, x, x_down, cond):
        x = torch.cat((x, x_down), dim=1)    #unet skip connection
        x = self.crb(x, cond)
        x = self.upsample(x)  
        return x
    

class ConditionalUnet1D(nn.Module):
    """ 
    Unet architecture for 1D input produce 1D output
    Additional inputs (image features, agent poses) are used as conditional features to guide the prediction
    Uses positional encoder to embed the timestep for the diffusion step
    """
    
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8,
        pos_encoder_model="custom"  #custom: my implementation, default: diffusion policy implementation
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        dsed = diffusion_step_embed_dim
        
        if pos_encoder_model == "custom":
            pes= PositionalEncoding(dsed, max_len=64)
            print("Using custom positional encoding")
        elif pos_encoder_model == "default":
            pes = SinusoidalPosEmb(dsed)
            print("Using diffusion positional encoding")
        else:
            raise ValueError(f"Unknown pos_encoder_model: {pos_encoder_model}")

        
        diffusion_step_encoder = nn.Sequential(
            pes,
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ), 
        ])

        down_modules = nn.ModuleList([])  
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                DownModule(dim_in, dim_out, cond_dim, kernel_size, n_groups, is_last)
            )  

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                UpModule(dim_out*2, dim_in, cond_dim, kernel_size, n_groups, is_last)
            )  

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv 

 
    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)  # (B,C,T)

        # 1. time 
        timestep  = timestep.expand(sample.shape[0]) 
        positional_feature = self.diffusion_step_encoder(timestep)

        global_feature = torch.cat([positional_feature, global_cond], axis=-1)

        # unet training
        x = sample
        h = []
        for down_module in self.down_modules:
            x, x_small = down_module(x, global_feature) 
            h.append(x)
            x = x_small 

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for upmodule  in self.up_modules:
            x= upmodule(x, h.pop(), global_feature) 

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
