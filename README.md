[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Gen1
My Implementation of " Structure and Content-Guided Video Synthesis with Diffusion Models" by RunwayML. "Input videos x are encoded to z0 with a fixed encoder E and diffused to zt. We extract a
structure representation s by encoding depth maps obtained with MiDaS, and a content representation c by encoding one of the frames
with CLIP. The model then learns to reverse the diffusion process in the latent space, with the help of s, which gets concatenated to zt, as
well as c, which is provided via cross-attention blocks. During inference (right), the structure s of an input video is provided in the same
manner. To specify content via text, we convert CLIP text embeddings to image embeddings via a prior."



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

## Datasets
Here is a summary table of the datasets used in the Structure and Content-Guided Video Synthesis with Diffusion Models paper:

| Dataset | Type | Size | Domain | Description | Source |
|-|-|-|-|-|-|
| Internal dataset | Images | 240M | General | Uncaptioned images | Private |  
| Custom video dataset | Videos | 6.4M clips | General | Uncaptioned short video clips | Private |
| DAVIS | Videos | - | General | Video object segmentation | [Link](https://davischallenge.org/) |
| Stock footage | Videos | - | General | Diverse video clips | - |



## Citation
```
@misc{2302.03011,
Author = {Patrick Esser and Johnathan Chiu and Parmida Atighehchian and Jonathan Granskog and Anastasis Germanidis},
Title = {Structure and Content-Guided Video Synthesis with Diffusion Models},
Year = {2023},
Eprint = {arXiv:2302.03011},
```


# Todo
- [ ] Add training script
- [ ] Add in conditional text paramater to pass in text, not just images and or other videos