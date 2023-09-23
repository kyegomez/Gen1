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