import torch 
import einops
from einops import rearrange, reduce, repeat
from torch import nn
from shapeless import liquid

@liquid
class AutoEncoder(nn.Module):
    dim: int
    x: torch.Tensor
    out_channels: int = 128
    stide: int
    padding: int
    groups: int  = 32
    eps = 1e-5
    in_channels = 3

    def forward(self, x):
        encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stide,
                padding=self.padding,
            ),
            nn.GroupNorm(num_groups=self.groups, eps=self.eps),
            nn.ELU(),

            nn.Conv2d(
                #3x3 depthwise conv, 128 channels, 3 kernel, 1 pad
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1
            ),
            nn.GroupNorm(self.groups, self.eps),
            nn.ELU(),

            nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(self.groups, self.eps),
            nn.ELU(),

            nn.Flatten(),
            #fully connected lay
            nn.Linear(1024, 32),
        )
        encoded = encoder(x)
        print(encoded)
    

@liquid
class Midas(nn.Module):
    x: torch.Tensor
    dim: int

    def forward(self, x):
        pass

@liquid
class Diffusion(nn.Module):
    x: torch.Tensor
    dim: int

    def forward(self, x):
        pass

@liquid
class Clip(nn.Module):
    x: torch.Tensor
    dim: int

    def forward(self, x):
        pass

