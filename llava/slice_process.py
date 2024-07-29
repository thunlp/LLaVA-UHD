import math
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch


import random
from imgaug import augmenters as iaa
import numpy as np

NEWLINE_TOKEN = 13 # '\n'
DOT_TOKEN = 29892  #  ','

def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches

def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size
    
def ensure_divide(length, patch_size):
    # return max(round(length / patch_size) * patch_size, patch_size)
    return max(math.floor(length / patch_size) * patch_size, patch_size)

def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height # width=672 height=448 r= 1.5
        height = int(scale_resolution / math.sqrt(r)) # scale_resolution=336 / r**0.5  274.3428511917
        width = int(height * r) # 411.5142767876
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)

def slice_image_minicpm(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)
    
    ind_tokens = []
    if best_grid is None:
        return source_image, patches, best_grid, ind_tokens
    else:
        # flatten the patches
        patches = [item for sublist in patches for item in sublist]
        # calculate ind_token layout
        for j in range(best_grid[1]):
            for i in range(best_grid[0]):
                if i != best_grid[0] - 1:
                    ind_tokens.append(DOT_TOKEN)
                else:
                    ind_tokens.append(NEWLINE_TOKEN)

        return source_image, patches, best_grid, ind_tokens



def split_image(image, scale=672, grid=(2, 2)):
    resized_image = image.resize((scale, scale))
    width, height = resized_image.size
    grid_width = width // grid[0]
    grid_height = height // grid[1]
    
    sub_images = []
    
    for i in range(grid[0]):
        for j in range(grid[1]):
            left = i * grid_width
            upper = j * grid_height
            right = left + grid_width
            lower = upper + grid_height
            sub_image = resized_image.crop((left, upper, right, lower))
            sub_images.append(sub_image)
    
    return sub_images


def generate_subimage_coordinates(H, W, h, w, num_windows):
    """
    生成子图的左上角和右下角坐标，并返回一个形状为 (n, 4) 的 PyTorch tensor。

    参数:
    H (int): 原始图像的高度
    W (int): 原始图像的宽度
    h (int): 子图的高度
    w (int): 子图的宽度

    返回:
    torch.Tensor: 形状为 (n, 4) 的张量，包含所有子图的左上角和右下角坐标
    """
    # assert H % h == 0 and W % w == 0, "H/h and W/w must be an integer"
    
    rows = int(round(H / h))
    cols = int(round(W / w))
    assert rows * cols == num_windows, f'H:{H}, W:{W}, h:{h}, w:{w}, rows:{H/h}, cols:{W/w}'
    coordinates = []
    for i in range(rows):
        for j in range(cols):
            x1 = j * w
            y1 = i * h
            x2 = x1 + w
            y2 = y1 + h
            coordinates.append([x1, y1, x2, y2])

    return torch.tensor(coordinates, dtype=torch.float32)
    
def slice_image_feature_minicpm(
    image_feature, num_windows=144, max_slice_nums=1000, num_ratio=1):
    # image_feature: b,c,h,w
    # num_queries of resampler. n
    # 
    bs = image_feature.shape[0]
    dtype, device = image_feature.dtype, image_feature.device
    feature_size = image_feature.shape[-2:]
    feature_height, feature_width = feature_size
    log_ratio = math.log(feature_width / feature_height)
    ratio = feature_height * feature_width / num_windows
    multiple = min(math.ceil(ratio), max_slice_nums)

    candidate_split_grids_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if i == 1 or i > max_slice_nums:
            continue
        candidate_split_grids_nums.append(i)

    candidate_grids = []
    # find best grid
    for split_grids_nums in candidate_split_grids_nums:
        m = 1
        while m <= split_grids_nums:
            if split_grids_nums % m == 0:
                candidate_grids.append([m, split_grids_nums // m])
            m += 1

    best_grid = [1, 1]
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error
    
    # (Iw * Ih) / n = Iw / Ih * h^2
    float_crop_height = math.sqrt(ratio / (feature_width / feature_height))
    float_crop_width = float_crop_height * (feature_width / feature_height)

    # print(float_crop_height, float_crop_width, feature_height, feature_width, )
    # print('true:', feature_height / float_crop_height, feature_width / float_crop_width)

    region_boxes = generate_subimage_coordinates(feature_height, feature_width, 
                                                float_crop_height, float_crop_width, num_windows)
    
    region_boxes = region_boxes.to(dtype=dtype, device=device).detach()
    batch_region_boxes = []
    for i in range(bs):
        batch_id = torch.ones_like(region_boxes)[:, :1] * i
        batch_region_boxes.append(torch.cat([batch_id, region_boxes], dim=1))
    batch_region_boxes = torch.cat(batch_region_boxes)

    return batch_region_boxes, best_grid, feature_width / feature_height
    

def resize_image_keep_ratio(image, max_size=1024):
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = max_size
        new_height = int((max_size / original_width) * original_height)
    else:
        new_height = max_size
        new_width = int((max_size / original_height) * original_width)
    resized_image = image.resize((new_width, new_height),  Image.Resampling.BICUBIC)
    return resized_image


def aug_image(image):
    if random.random() < 0.5:
        image = resize_image_keep_ratio(image, max_size=1024)
    if random.random() < 0.1:
        aug = iaa.contrast.LinearContrast((0.5, 2.0), per_channel=False)
        image = Image.fromarray(aug(image=np.array(image)))
    if random.random() < 0.1:
        aug = iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.75, 1.5))
        image = Image.fromarray(aug(image=np.array(image)))
    if random.random() < 0.2:
        aug = iaa.AddToHue((-50, 50))
        image = Image.fromarray(aug(image=np.array(image)))
    if random.random() < 0.1:
        aug = iaa.JpegCompression(compression=(75, 95))
        image = Image.fromarray(aug(image=np.array(image)))
    return image

