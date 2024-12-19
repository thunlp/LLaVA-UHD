# hubconf.py
import torch
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.upsamplers import get_upsampler
from torch.nn import Module
import torch
from torch.multiprocessing import Pool, set_start_method
from functools import partial
import torch.nn.functional as F

dependencies = ['torch', 'torchvision', 'PIL', 'featup']  # List any dependencies here


class UpsampledBackbone(Module):

    def __init__(self, model_name, use_norm, scale):
        super().__init__()
        model, patch_size, self.dim = get_featurizer(model_name, "token", num_classes=1000)
        if use_norm:
            self.model = torch.nn.Sequential(model, ChannelNorm(self.dim))
        else:
            self.model = model
        
        if scale == '2x':
            self.upsampler = get_upsampler("jbu_2x_stack", self.dim)
        elif scale == '4x':
            self.upsampler = get_upsampler("jbu_4x_stack", self.dim)
        elif scale == '8x':
            self.upsampler = get_upsampler("jbu_8x_stack", self.dim)
        else:
            self.upsampler = get_upsampler("jbu_stack", self.dim)

    def forward(self, image):
        return self.upsampler(self.model(image), image)
    
    def forward_with_internal_features(self, image, lowres):
        if type(lowres) == list:
            #return self.forward_with_features_list(image, lowres)
            features_2x = []
            features_4x = []
            features_8x = []
            for i in range(len(lowres)):
                #lowres_norm = self.model[1](lowres[i])
                res = self.upsampler.forward_with_internal_features(lowres[i], image[i].unsqueeze(0))
                features_2x.append(res['feat2x'])
                if 'feat4x' in res:
                    features_4x.append(res['feat4x'])
                if 'feat8x' in res:
                    features_8x.append(res['feat8x'])
                    
            return features_2x, features_4x, features_8x
        else:
            feat2x = None
            feat4x = None
            feat8x = None
            res = self.upsampler.forward_with_internal_features(lowres, image)
            if 'feat2x' in res:
                feat2x = res['feat2x']
            if 'feat4x' in res:
                feat4x = res['feat4x']
            if 'feat8x' in res:
                feat8x = res['feat8x']
            return feat2x, feat4x, feat8x

class Upsampled4xBackbone(Module):

    def __init__(self, model_name, use_norm):
        super().__init__()
        model, patch_size, self.dim = get_featurizer(model_name, "token", num_classes=1000)
        if use_norm:
            self.model = torch.nn.Sequential(model, ChannelNorm(self.dim))
        else:
            self.model = model
        self.upsampler = get_upsampler("jbu_4x_stack", self.dim)

    def forward(self, image):
        lowres = self.model(image).to(torch.bfloat16)
        image = image.to(torch.bfloat16)
        return self.upsampler(lowres, image)

    def pad_to_square_tensor(self, feat, pad_res):#torch.Size([1, 1024, 25, 22]), 33
        # Calculate padding for each dimension
        pad_size = (0, pad_res-feat.size(3), 0, pad_res-feat.size(2))  # left, right, top, down
        # Pad the image tensor
        padded = F.pad(feat, pad_size, 'constant', 1)
        return padded

    def unpad_from_square_tensor(self, feat, h, w):
        # Crop the image tensor from left top to the desired size
        return feat[:, :h, :w]

    def forward_with_features_list(self, images, lowres):
        feature_scale = 14 # clip-large 336 -14
        #list of torch.Size([3, 350, 308]), 
        #list of torch.Size([1, 1024, 25, 22])
        pad_res = 0
        for i in range(len(lowres)):
            pad_res = max(pad_res, max(lowres[i].size(2), lowres[i].size(3)))

        lowres_tensor = [self.pad_to_square_tensor(lowres[i], pad_res) for i in range(len(lowres))]
        lowres_tensor = torch.cat(lowres_tensor, dim=0)

        images_tensor = [self.pad_to_square_tensor(images[i].unsqueeze(0), pad_res * feature_scale) for i in range(len(images))]
        images_tensor = torch.cat(images_tensor, dim=0)

        features_2x, features_4x = self.upsampler.forward_with_internal_features(lowres_tensor, images_tensor)

        feat_2x_list = [self.unpad_from_square_tensor(features_2x[i], lowres[i].size(2) * 2, lowres[i].size(3) * 2).unsqueeze(0) for i in range(len(features_2x))]
        feat_4x_list = [self.unpad_from_square_tensor(features_4x[i], lowres[i].size(2) * 4, lowres[i].size(3) * 4).unsqueeze(0) for i in range(len(features_4x))]

        return feat_2x_list, feat_4x_list

    def forward_with_internal_features(self, image, lowres):
        if type(lowres) == list:
            #return self.forward_with_features_list(image, lowres)
            features_2x = []
            features_4x = []
            features_8x = []
            for i in range(len(lowres)):
                res = self.upsampler.forward_with_internal_features(lowres[i], image[i].unsqueeze(0))
                features_2x.append(res['feat2x'])
                if res.get('feat4x') is not None:
                    features_4x.append(res['feat4x'])
                if res.get('feat8x') is not None:
                    features_8x.append(res['feat8x'])
                    
            return {'feat2x': features_2x, 'feat4x': features_4x, 'feat8x': features_8x}
        else:
            return self.upsampler.forward_with_internal_features(lowres, image)

def _load_backbone(pretrained, use_norm, model_name):
    """
    The function that will be called by Torch Hub users to instantiate your model.
    Args:
        pretrained (bool): If True, returns a model pre-loaded with weights.
    Returns:
        An instance of your model.
    """
    model = UpsampledBackbone(model_name, use_norm)
    if pretrained:
        # Define how you load your pretrained weights here
        # For example:
        if use_norm:
            exp_dir = ""
        else:
            exp_dir = "no_norm/"

        checkpoint_url = f"https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/{exp_dir}{model_name}_jbu_stack_cocostuff.ckpt"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if "scale_net" not in k and "downsampler" not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

def _load_backbone_from_local(pretrained, use_norm, model_name, ckpt_path, scale = '16x'):
    """
    The function that will be called by Torch Hub users to instantiate your model.
    Args:
        pretrained (bool): If True, returns a model pre-loaded with weights.
    Returns:
        An instance of your model.
    """
    model = UpsampledBackbone(model_name, use_norm, scale)

    if pretrained:
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if "scale_net" not in k and "downsampler" not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

def _get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    state_dict = {k: v for k, v in state_dict.items() if "scale_net" not in k and "downsampler" not in k}
    return state_dict

# def vit(pretrained=True, use_norm=True):
#     return _load_backbone(pretrained, use_norm, "vit")

def vit(pretrained=True, use_norm=True):
    ckpt_path = 'Path to JBU ckpt'
    return _load_backbone_from_local(pretrained, use_norm, "vit", ckpt_path)

def dino16(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "dino16")


def clip(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "clip")

def clipLarge(pretrained=True, use_norm=True, scale = '4x'):
    ckpt_path = 'Path to JBU ckpt'
    return _load_backbone_from_local(pretrained, use_norm, "clip-large", ckpt_path, scale)

def get_clipLarge_state_dict(ckpt_path='Path to JBU ckpt'):
    return _get_state_dict(ckpt_path)

def dinov2(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "dinov2")


def resnet50(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "resnet50")

def maskclip(pretrained=True, use_norm=True):
    assert not use_norm, "MaskCLIP only supports unnormed model"
    return _load_backbone(pretrained, use_norm, "maskclip")