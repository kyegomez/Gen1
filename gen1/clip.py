import open_clip
import torch
from PIL import Image


class CLIP:
    def __init__(
        self,
    ):
        self.model, _, self.preprocess = open_clip.create_model_and_transformes(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    def run(self, text, image):
        image = self.preprocess(Image.open(image)).unsqueeze(0)
        self.tokenizer([text])

        with torch.no_grad(), torch.cuda.amp.autocast:
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return text_probs
        
        