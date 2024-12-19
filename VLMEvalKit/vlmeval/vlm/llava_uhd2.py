import torch
from PIL import Image
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy

class LLaVA_UHD2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='',
                 **kwargs):
        
        
        try:
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from torch.utils.data import Dataset, DataLoader
            from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square
        except:
            warnings.warn('Please install LLava_UHD before using LLava_UHD')
            warnings.warn('Please install VLMEvalKit after installing LLava_UHD')
            sys.exit(-1)
        
        
        #assert osp.exists(model_path) or len(model_path.split('/')) == 2
        self.system_prompt = (
            'A chat between a curious human and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        model_name = get_model_name_from_path(model_path)
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )
        except Exception as e:
            warnings.warn(f'Error loading model: {e}')
            sys.exit(-1)

        self.model = self.model.cuda()
        #self.conv_mode = 'vicuna_v1'
        self.conv_mode = 'vicuna_v1'
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=3, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default


    def build_prompt(self, message, dataset=None):
        if dataset==None:
            dataset_type='VQA'
        else:
            dataset_type = DATASET_TYPE(dataset)
      
        prompt = ""
        images = []
        for msg in message:
            if msg['type'] == 'image':
                images.append(msg['value']) 
            elif msg['type'] == 'text':
                prompt += msg['value']  
        #prompt = prompt.split(' Please answer')[0]
        if dataset_type == 'MCQ':
            #prompt=prompt
            prompt += (
               '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        elif dataset_type == 'Y/N':
            prompt += (
               '\n请用简单字母或短语回答问题' if cn_string(prompt) else
                "\nAnswer the question using a single word or phrase."
            )
        else:
           prompt=prompt
           #prompt += '\n请用简单字母或短语回答问题' if cn_string(prompt) else '\nAnswer the question using a single word or phrase.'
           
        #print("old prompt", prompt)
        #prompt = prompt.removesuffix('Answer the question using a single word or phrase.')
        #prompt = prompt.replace('Answer the question using a single word or phrase.', 'Answer the question with a single word.')
        #prompt = prompt.replace('Answer the question using a single word or phrase.', 'Answer the question with a single word.')
        
        #print("new prompt", prompt)

        message = [dict(type='image', value=img) for img in images]
        message.append(dict(type='text', value=prompt))
        return message
        
    def preprocess(self, dataset, text,  image, tokenizer, processor, model_config, conv_mode='vicuna_v1'):
        from llava.slice_process import slice_image_minicpm, split_image, resize_image_keep_ratio, resize_image_keep_ratio_force
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        qs = text
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
       
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if 'RealWorldQA' in dataset.dataset_name:
            image = resize_image_keep_ratio_force(dataset, image[0], max_size=1024)
        else:
            image = resize_image_keep_ratio(dataset, image[0], max_size=1024)

        source_image, patches, best_grid, ind_tokens = slice_image_minicpm(
            image, max_slice_nums=9, scale_resolution=336, patch_size=14, never_split=False)

        if best_grid is None: #说明没有切片
            source_tensors = processor.preprocess(source_image, do_resize=False, do_center_crop=False, 
                                                    do_rescale=True, do_normalize=True, 
                                                    return_tensors='pt')['pixel_values'] # 1, 3, abs_h, abs_w
            crop_size = processor.crop_size
            patch_tensors = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
        else:
            source_tensors = processor.preprocess(source_image, do_resize=False, do_center_crop=False, 
                                                    do_rescale=True, do_normalize=True, 
                                                    return_tensors='pt')['pixel_values'] # 1, 3, abs_h, abs_w
            patch_tensors = processor.preprocess(patches, do_resize=False, do_center_crop=False, 
                                                    do_rescale=True, do_normalize=True, 
                                                    return_tensors='pt')['pixel_values'] # num_slice, 3, s_h, s_w

        images = [source_tensors[0].half().cuda()] # 3, h, w
        patch_images = [patch_tensors.half().cuda()] # bs, 3, h, w
        ind_tokens = [ind_tokens]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        return input_ids, images, [image.size], patch_images, ind_tokens


    def generate_inner(self, message, dataset=None):
        from llava.conversation import conv_templates, SeparatorStyle
        content, images = '', []
        msg = self.build_prompt(message, dataset)
        for item in msg:
            if item['type'] == 'text':
                content += item['value']
            elif item['type'] == 'image':
                image = Image.open(item['value']).convert('RGB')
                images.append(image)
                
        
        top_p = None
        input_ids, image_tensor, image_sizes, patch_images, ind_tokens = self.preprocess(dataset, content, images, self.tokenizer, self.image_processor, self.model.config, conv_mode='vicuna_v1')
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                patch_images=patch_images,
                ind_tokens=ind_tokens,
                **self.kwargs)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs