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
from gen1.model import Gen1


model = Gen1()

images = torch.randn(1, 3, 128, 128)
video = torch.randn(1, 3, 16, 128, 128)

run_out = model.forward(images, video)


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
