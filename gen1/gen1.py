from gen1.model import SpaceTimeUnet
import torch
from torch import nn


class Gen1(nn.Module):
    def __init__(
        self,
        *,
        dim,
        channels = 3,
        dim_mult = (1, 2, 4, 8),
        self_attns = (False, False, False, True),
        temporal_compression = (False, True, True, True),
        resnet_block_depths = (2, 2, 2, 2),
        attn_dim_head = 64,
        attn_heads = 8,
        condition_on_timestep = True,
        attn_pos_bias = True,
        flash_attn = False,
        causal_time_attn = False
    ):
        
        super().__init__()

        self.dim = dim
        self.channels = channels
        self.dim_mult = dim_mult
        self.self_attns = self_attns
        self.temporal_compression = temporal_compression
        self.resnet_block_depths = resnet_block_depths
        self.attn_dim_head = attn_dim_head
        self.attn_heads = attn_heads
        self.condition_on_timestep = condition_on_timestep
        self.attn_pos_bias = attn_pos_bias
        self.flash_attn = flash_attn
        self.causal_time_attn = causal_time_attn

        self.unet = SpaceTimeUnet(
            dim=self.dim,
            channels=self.channels,
            dim_mult=self.dim_mult,
            self_attns=self.self_attns,
            temporal_compression=self.temporal_compression,
            resnet_block_depths=self.resnet_block_depths,
            attn_dim_head=self.attn_dim_head,
            attn_heads=self.attn_heads,
            condition_on_timestep=self.condition_on_timestep,
            attn_pos_bias=self.attn_pos_bias,
            flash_attn=self.flash_attn,
            causal_time_attn=self.causal_time_attn
        )    
    
    def foward_images(self, images):
        return self.unet(images, )
    
    def forward_videos(self, videos):
        return self.unet(videos)
    
    def forward(self, images, videos):
        images_out = self.unet(images)
        assert images.shape == images_out.shape

        #videos
        video_out = self.unet(videos)
        assert videos.shape == video_out.shape


        

