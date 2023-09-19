import torch 
from einops
from torch import nn
from shapeless import liquid

@liquid
class AutoEncoder(nn.Module):
    dim: int

    def forward(self, x):
        x = nn.Flatten(x, self.dim)
        x = nn.Linear(x, 128)
        x = nn.ReLU(x)
        x = nn.Linear(x, 64)
        x = nn.ReLU(x)
        x = nn.Linear(x, 128)
        x = nn.ReLU(x)
        x = nn.Linear(x, 784)
        x = nn.Reshape(x, self.dim)
        return x
    

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

