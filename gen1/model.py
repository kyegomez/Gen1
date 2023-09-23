import torch 
import einops
from einops import rearrange, reduce, repeat
from torch import nn
from shapeless import liquid

#helpers
def exists(val):
    return val is not None

#utils
class Residual(nn.Module):
    def __init__(
        self,
        dim,
        scale_residual=False,
        scale_residual_constant=1.
    ):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant
    
    def forward(self, x, residual):
        if exists(self.resiudal_scale):
            residual = residual * self.residual_scale
        
        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant
        
        return x + residual
    

#components

# @liquid
class AutoEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        x: torch.Tensor,
        out_channels: int = 128,
        stride: int = None,
        padding: int = None,
        groups: int = 32,
        eps = 1e-5,
        in_channels = 3,
    ):
        self.dim = dim
        self.x = x
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps
        self.in_channels = in_channels
    
    def forward(self):
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

x = torch.randn(1, 3, 224, 224)
model = AutoEncoder(x=x, dim=3, out_channels=128, stride=1, padding=1, groups=32, eps=1e-5, in_channels=3)
print(model)


class Midas:
    def __init__(
        self,
        path=None,
        features=256,
        non_negative=True
    ):
        """
        Args:
            
            path(str, optional): Path to saved model defaults to None
            features(int, optional): Number of features, defauts to 256
            backbone(str, optional): Backbone network for encoder

        """
        print("loadding weightsss:", path)
        super(Midas, self).__init__()
        
        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder_