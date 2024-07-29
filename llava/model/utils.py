from transformers import AutoConfig


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

# def split_image(image, scale=672, grid=(2, 2)):
#     # 缩放图像到指定大小
#     resized_image = image.resize((scale, scale))
    
#     # 获取图像的尺寸
#     width, height = resized_image.size
    
#     # 计算每个子图的宽度和高度
#     grid_width = width // grid[0]
#     grid_height = height // grid[1]
    
#     # 初始化子图列表
#     sub_images = []
    
#     # 划分图像
#     for i in range(grid[0]):
#         for j in range(grid[1]):
#             left = i * grid_width
#             upper = j * grid_height
#             right = left + grid_width
#             lower = upper + grid_height
#             sub_image = resized_image.crop((left, upper, right, lower))
#             sub_images.append(sub_image)
    
#     return sub_images


# from PIL import Image

# def expand2square(pil_img, background_color=(0,0,0)):
#     width, height = pil_img.size
#     if width == height:
#         return pil_img
#     elif width > height:
#         result = Image.new(pil_img.mode, (width, width), background_color)
#         result.paste(pil_img, (0, (width - height) // 2))
#         return result
#     else:
#         result = Image.new(pil_img.mode, (height, height), background_color)
#         result.paste(pil_img, ((height - width) // 2, 0))
#         return result

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# image_processor = CLIPImageProcessor.from_pretrained('/data/pretrained_models/clip-vit-large-patch14-336')

# # image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
# # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

# # 示例用法
# # 打开图像
# img_dir = './playground/data/LLaVA-Pretrain/images/00152/001529327.jpg'
# image = Image.open(img_dir)
# image.save('ori.jpg')
# image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
# print(image.size)
# image.save('padding_ori.jpg')
# # 调用函数，得到4个子图像
# sub_images = split_image(image, scale=672, grid=(2, 2))

# # 保存或显示子图像
# for idx, sub_image in enumerate(sub_images):
#     print(sub_image.size)
#     sub_image.save(f'sub_image_{idx + 1}.jpg')
#     sub_image.show()

# sub_images.append(image)
# ims = image_processor.preprocess(sub_images, return_tensors='pt')['pixel_values']
# print(ims.shape)