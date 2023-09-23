from torch import nn
from gen1.midas import MidasHF


class Pipeline(nn.Module):
    def __init__(
        self,
        image,
        video
    ):
        self.image = image
        self.video = video

        self.midas = MidasHF()
    
    def run(self):
        image = self.midas.forward(self.image)
        image = 
