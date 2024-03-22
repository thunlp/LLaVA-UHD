import math
from PIL import Image

from torchvision.transforms import ToTensor,ToPILImage
import torch

#-------------------------------------------------------#
#  预处理图像
#-------------------------------------------------------#
PATCH_SIZE       = 14
PATCH_NUM_WIDTH  = 24
PATCH_NUM_HEIGHT = 24
POSITION_EMBEDDING_LENGTH = 1024
# 576
MAX_PATCHES      = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT
# 
TOKEN_LENGTH     = 3 * PATCH_SIZE * PATCH_SIZE
# 336 336
IMAGE_WIDTH      = PATCH_SIZE * PATCH_NUM_WIDTH
IMAGE_HEIGHT     = PATCH_SIZE * PATCH_NUM_HEIGHT

def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """

    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)

# 用于计算adapt需要输入图片的大小
def adapt_size(originHeight:int,originWeight:int, \
            patchHeight:int = PATCH_SIZE,patchWidth:int = PATCH_SIZE, \
            maxPatches:int = MAX_PATCHES):
    ### 用于计算adapt的图片大小
    # 参数说明 
    # originHeight:              原图高度
    # originWidth:               原图宽度
    # patchHeight:               patch高度
    # patchWidth:                patch宽度
    # maxPatches:                patch数目上限
    # 返回值说明:
    # resized_height:            插值后图片高度
    # resized_width:             插值后图片宽度
    # resized_patch_height_num:  插值后图片垂直patch数目
    # resized_patch_width_num:   插值后图片水平patch数目
    scale = math.sqrt(maxPatches * (patchHeight / originHeight) * (patchWidth / originWeight))
    resized_patch_height_num = max(min(math.floor(scale * originHeight / patchHeight), maxPatches), 1)
    resized_patch_width_num = max(min(math.floor(scale * originWeight / patchWidth), maxPatches), 1)
    resized_height = max(resized_patch_height_num * PATCH_SIZE, 1)
    resized_width = max(resized_patch_width_num * PATCH_SIZE, 1)
    return resized_height, resized_width, resized_patch_height_num, resized_patch_width_num

def cal_num_of_slices(origin_image_width, origin_image_height):
    scale = origin_image_width*origin_image_height/(IMAGE_WIDTH*IMAGE_HEIGHT)  
    scale = math.ceil(scale)
    if scale > 6:
        scale = 6
    def factorize(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i/(n/i), i, n // i))
        return factors
    numbers = [1, 2, 3, 4, 5, 6, 7]
    factor_dict = {}
    for num in numbers:
        factor_dict[num] = factorize(num)
    log_origin_ratio = math.log(origin_image_width/origin_image_height)
    available_ratios = []
    if scale<=2:
        available_ratios = factor_dict[scale] + factor_dict[scale + 1]
    else :
        available_ratios = factor_dict[scale-1] + factor_dict[scale]+factor_dict[scale+1]
    min_dif = 1000 
    best_w = 0
    best_h = 0
    for (r,w_slice,h_slice) in available_ratios:
        log_r = math.log(r)
        if min_dif > abs(log_r - log_origin_ratio):
            min_dif = abs(log_r - log_origin_ratio)
            best_w = w_slice
            best_h = h_slice
    
    return best_w,best_h
# 做图片切片     
def get_patch_nums(origin_image_width, origin_image_height):
    # 输入原图的尺寸
    # 返回：
    # slice_w_num 切片的w方向有多少个patch
    # slice_h_num 切片的h方向有多少个patch
    # abstract_w_num 原图的w方向有多少个patch
    # abstract_h_num 原图的h方向有多少个patch
    
    best_w, best_h = cal_num_of_slices(origin_image_width,origin_image_height)
    slice_width = origin_image_width//best_w
    slice_height = origin_image_height//best_h
    _,_,slice_h_num,slice_w_num = adapt_size(slice_height,slice_width)
    _,_,abstract_h_num,abstract_w_num = adapt_size(origin_image_height,origin_image_width)

    return slice_w_num,slice_h_num,abstract_w_num,abstract_h_num

def slice_image(image):
    
    # slice the image according to our princeple
    # return an array of slices
    
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]

    best_w, best_h = cal_num_of_slices(origin_image_width=origin_image_width,origin_image_height=origin_image_height)
    
    slices = []
    
    for j in range(best_h):
        for i in range(best_w):
            
            box = (i * origin_image_width//best_w, j * origin_image_height//best_h, (i + 1) * origin_image_width//best_w, (j + 1) * origin_image_height//best_h)
         
            region = image.crop(box).convert("RGB")
            slices.append(region)
          
    return slices


def process_image(image):
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]

    image = image.convert("RGB")
    
    slices = slice_image(image)
    
    # 计算resize之后的图片大小
    resized_height, resized_width, resized_patch_height, resized_patch_width = \
    adapt_size(origin_image_height,origin_image_width)
    
    
    if len(slices) == 1:
        image = slices[0]
        image_w = image.size[0]
        image_h = image.size[1]
        resized_height, resized_width, resized_patch_height, resized_patch_width = \
        adapt_size(image_h,image_w)     
        
        image = ToTensor()(image)
    
        image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        # 需要mask的patch数
        num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
        # raprint("mask: ",num_patches_to_pad)
        # 切割resize好的图片
        image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
        image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])
        # 用0补全需要mask的图片部分
        image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
        image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
        # print(image)
        return [image]
    
    else:
        images = []
        resized_patch_widths = []
        resized_patch_heights = []
        slices.append(image)
        for image in slices:

            image = ToTensor()(image)
    
            image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
            # 需要mask的patch数
            num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
            # raprint("mask: ",num_patches_to_pad)
            # 切割resize好的图片
            image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
            image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])
            # 用0补全需要mask的图片部分
            image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
            image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
            
            # print(image)
            images.append(image)
            resized_patch_widths.append(resized_patch_width)
            resized_patch_heights.append(resized_patch_height)
        return images


if __name__ == '__main__':
    img = Image.open("/home/xuruyi/myLLaVa/883700e3366b775c93315373510e7e7.png")
    images = process_image(img)

    for i in range(len(images)):
        img = images[i]
        to_pil = ToPILImage()

        img = to_pil(img)

        img.save(f"image{i}.png")
