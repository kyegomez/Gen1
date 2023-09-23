[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Gen1
My Implementation of " Structure and Content-Guided Video Synthesis with Diffusion Models" by RunwayML


The flow:

```
image => midas => clip => spacetime unet => diffusion
```


# Install
`pip3 install gen1`

# Usage
```python
import torch
from gen1.gen1 import Gen1


model = Gen1(
    dim=64,
    channels=3,
    dim_mult = (1, 2, 4, 8),
    resnet_block_depths=(1, 1, 1, 2),
    temporal_compression=(False, False, False, True),
    self_attns=(False, False, False, True),
    condition_on_timestep=True,
    attn_pos_bias=True,
    flash_attn=True,
).cuda()

images = torch.randn(1, 3, 128, 128).cuda()
video = torch.randn(1, 3, 16, 128, 128).cuda()

run_out = model.forward(images, videos=video)

```

## Usage
- Help us implement it we need help with the Midas, Clip, and modified Unet blocks


## Citation
```
@misc{2302.03011,
Author = {Patrick Esser and Johnathan Chiu and Parmida Atighehchian and Jonathan Granskog and Anastasis Germanidis},
Title = {Structure and Content-Guided Video Synthesis with Diffusion Models},
Year = {2023},
Eprint = {arXiv:2302.03011},
```
