from torch import nn
from gen1.midas import MidasHF
from gen1.clip import CLIP

class Pipeline(nn.Module):
    def __init__(
        self,
        image,
        video
    ):
        self.image = image
        self.video = video

        self.midas = MidasHF()
        self.clip = CLIP()
        
    def forward(self):
        image = self.midas.forward(self.image)
        image_embeddings = self.clip.run(image)
        return image_embeddings
    
