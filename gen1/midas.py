"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

#option 2


#utils


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


#helpers

def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, hooks=None,
                  use_vit_only=False, use_readout="ignore", in_features=[96, 256, 512, 1024]):
    # if backbone == "beitl16_512":
    #     pretrained = _make_pretrained_beitl16_512(
    #         use_pretrained, hooks=hooks, use_readout=use_readout
    #     )
    #     scratch = _make_scratch(
    #         [256, 512, 1024, 1024], features, groups=groups, expand=expand
    #     )  # BEiT_512-L (backbone)
    # elif backbone == "beitl16_384":
    #     pretrained = _make_pretrained_beitl16_384(
    #         use_pretrained, hooks=hooks, use_readout=use_readout
    #     )
    #     scratch = _make_scratch(
    #         [256, 512, 1024, 1024], features, groups=groups, expand=expand
    #     )  # BEiT_384-L (backbone)
    # elif backbone == "beitb16_384":
    #     pretrained = _make_pretrained_beitb16_384(
    #         use_pretrained, hooks=hooks, use_readout=use_readout
    #     )
    #     scratch = _make_scratch(
    #         [96, 192, 384, 768], features, groups=groups, expand=expand
    #     )  # BEiT_384-B (backbone)
    # elif backbone == "swin2l24_384":
    #     pretrained = _make_pretrained_swin2l24_384(
    #         use_pretrained, hooks=hooks
    #     )
    #     scratch = _make_scratch(
    #         [192, 384, 768, 1536], features, groups=groups, expand=expand
    #     )  # Swin2-L/12to24 (backbone)
    # elif backbone == "swin2b24_384":
    #     pretrained = _make_pretrained_swin2b24_384(
    #         use_pretrained, hooks=hooks
    #     )
    #     scratch = _make_scratch(
    #         [128, 256, 512, 1024], features, groups=groups, expand=expand
    #     )  # Swin2-B/12to24 (backbone)
    # elif backbone == "swin2t16_256":
    #     pretrained = _make_pretrained_swin2t16_256(
    #         use_pretrained, hooks=hooks
    #     )
    #     scratch = _make_scratch(
    #         [96, 192, 384, 768], features, groups=groups, expand=expand
    #     )  # Swin2-T/16 (backbone)
    # elif backbone == "swinl12_384":
    #     pretrained = _make_pretrained_swinl12_384(
    #         use_pretrained, hooks=hooks
    #     )
    #     scratch = _make_scratch(
    #         [192, 384, 768, 1536], features, groups=groups, expand=expand
    #     )  # Swin-L/12 (backbone)
    # elif backbone == "next_vit_large_6m":
    #     from .backbones.next_vit import _make_pretrained_next_vit_large_6m
    #     pretrained = _make_pretrained_next_vit_large_6m(hooks=hooks)
    #     scratch = _make_scratch(
    #         in_features, features, groups=groups, expand=expand
    #     )  # Next-ViT-L on ImageNet-1K-6M (backbone)
    # elif backbone == "levit_384":
    #     pretrained = _make_pretrained_levit_384(
    #         use_pretrained, hooks=hooks
    #     )
    #     scratch = _make_scratch(
    #         [384, 512, 768], features, groups=groups, expand=expand
    #     )  # LeViT 384 (backbone)
    # elif backbone == "vitl16_384":
    #     pretrained = _make_pretrained_vitl16_384(
    #         use_pretrained, hooks=hooks, use_readout=use_readout
    #     )
    #     scratch = _make_scratch(
    #         [256, 512, 1024, 1024], features, groups=groups, expand=expand
    #     )  # ViT-L/16 - 85.0% Top1 (backbone)
    # elif backbone == "vitb_rn50_384":
    #     pretrained = _make_pretrained_vitb_rn50_384(
    #         use_pretrained,
    #         hooks=hooks,
    #         use_vit_only=use_vit_only,
    #         use_readout=use_readout,
    #     )
    #     scratch = _make_scratch(
    #         [256, 512, 768, 768], features, groups=groups, expand=expand
    #     )  # ViT-H/16 - 85.0% Top1 (backbone)
    # elif backbone == "vitb16_384":
    #     pretrained = _make_pretrained_vitb16_384(
    #         use_pretrained, hooks=hooks, use_readout=use_readout
    #     )
    #     scratch = _make_scratch(
    #         [96, 192, 384, 768], features, groups=groups, expand=expand
    #     )  # ViT-B/16 - 84.6% Top1 (backbone)
    if backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained
    

def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)



class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output




class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output




# model
class Midas(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(
            self, 
            path=None, 
            features=256, 
            non_negative=True
        ):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(Midas, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)
    

#or option2
from PIL import Image
import numpy as np
import requests
import torch
from transformers import DPTForDepthPrediction, DPTConfig, DPTFeatureExtractor


class MidasHF:
    def __init__(
        self,
        model_name_or_path: str = 'Intel/dpt-hybrid-midas',
    ):
        self.model_name_or_path = model_name_or_path
        self.model = DPTForDepthPrediction.from_pretrained(
            'Intel/dpt-hybrid-midas',
            low_cpu_mem_usage=True
        )
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            self.model_name_or_path
        )

    
    def forward(self, image):
        image = Image.open(requests.get(image, stream=True).raw)

        inputs = self.feature_extractor(
            images=self.images,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

            #interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=self.image.size[::-1],
                mode="bicubic",
                align_corners=False
            )
            return prediction
        
        


