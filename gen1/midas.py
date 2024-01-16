# or option2
from PIL import Image
import requests
import torch
from transformers import DPTForDepthPrediction, DPTFeatureExtractor


class MidasHF:
    def __init__(
        self,
        model_name_or_path: str = "Intel/dpt-hybrid-midas",
    ):
        self.model_name_or_path = model_name_or_path
        self.model = DPTForDepthPrediction.from_pretrained(
            "Intel/dpt-hybrid-midas", low_cpu_mem_usage=True
        )
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            self.model_name_or_path
        )

    def forward(self, image):
        image = Image.open(requests.get(image, stream=True).raw)

        inputs = self.feature_extractor(images=self.images, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

            # interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=self.image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            return prediction
