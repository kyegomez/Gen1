from torch import nn

from gen1.clip import CLIP
from gen1.midas import MidasHF


class Pipeline(nn.Module):
    def __init__(
        self,
    ):

        self.midas = MidasHF()
        self.clip = CLIP()
        
    def forward(self, image):
        image_out = self.midas.forward(image)
        image_embeddings = self.clip.run(image_out)
        return image_embeddings
    
