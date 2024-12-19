import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from featup.util import norm
from torchvision.transforms import InterpolationMode


#CLIP-ViT-L/14 336 pixel
class CLIPLargeFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()
        vision_tower_name = 'openai/clip-vit-large-patch14-336'
        self.preprocess = CLIPImageProcessor.from_pretrained(vision_tower_name)
        self.model = CLIPVisionModel.from_pretrained(vision_tower_name)
        self.model.requires_grad_(False)

    def get_cls_token(self, img):
        return self.model(img).to(torch.float32).last_hidden_state

    def forward(self, img):
        outputs = self.model(img)
        last_hidden_states = outputs.last_hidden_state
        without_class = last_hidden_states[:, 1:]
        #torch.Size([1, 576, 1024])
        features = without_class.permute(0,2,1) 
        #[1, 1024, 24, 24]
        features = features.reshape(len(features), features.shape[1], 24, 24)
        return features.to(torch.float32)

if __name__ == '__main__':
    vision_tower_name = 'openai/clip-vit-large-patch14-336'
    image = Image.open("/home/god/playground/FeatUp/sample-images/bird_full.jpg")

    transformTest = T.Resize(336, InterpolationMode.BILINEAR)
    
    test_image = transformTest(image.convert("RGB"))
    
    
    transform = T.Compose([
        T.Resize(336, InterpolationMode.BILINEAR),
        T.CenterCrop(336),
        T.ToTensor(),
        norm])
    
    #torch.Size([3, 336, 336])
    transformed_image = transform(image.convert("RGB")).unsqueeze(0).to("cuda")
    
    
    model = CLIPLargeFeaturizer().cuda()

    features = model(transformed_image)

    print(features.shape)
    #torch.Size([1, 1024, 24, 24])
    #torch.Size([1, 768, 24, 24])
    