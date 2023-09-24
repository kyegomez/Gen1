import torch
from gen1.model import Gen1


model = Gen1()

images = torch.randn(1, 3, 128, 128)
video = torch.randn(1, 3, 16, 128, 128)

run_out = model.forward(images, video)

