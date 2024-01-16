import torch
from gen1.model import Gen1

# Create an instance of the Gen1 model
model = Gen1()

# Generate random input images and video tensors
images = torch.randn(1, 3, 128, 128)
video = torch.randn(1, 3, 16, 128, 128)

# Pass the input images and video through the model's forward method
run_out = model.forward(images, video)
