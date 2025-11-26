from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square

model_path = '/home/zhangyichen/users/zhangyichen/checkpoints/ckpt_for_patchsize_ablation/llava-uhd-qwen3-moonvit-4x4pooling-858k-256px-1024px-858k-p6'

model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )