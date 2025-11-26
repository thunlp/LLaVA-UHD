import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import re

def extract_choice(ans: str) -> str:
    ans = ans.strip().upper()
    matches = re.findall(r"\b([A-I])\b", ans)
    if matches:
        return matches[-1]  # 返回最后一个合法选项字母
    return ""

def extract_answer(ans: str, max_option="E") -> str:
    if not ans:
        return ""
    ans = ans.strip()

    # 1. 先匹配合法选项字母
    matches = re.findall(rf"\b([A-{max_option}])\b", ans.upper())
    if matches:
        return matches[-1]  # 返回最后一个大写选项字母

    # 2. 否则提取一个单词或短语（允许字母+数字）
    phrase = re.findall(r"[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*", ans)
    if phrase:
        return phrase[0].strip()

    return "" 

def extract_yes_no(ans: str) -> str:
    ans = ans.strip().upper()
    matches = re.findall(r"\b(YES|NO)\b", ans)
    if matches:
        return matches[-1]  # 返回最后一个合法选项字母
    return "" 

def extract_first_word(ans: str) -> str:
    """
    提取输出中的第一行第一个单词
    """
    if not ans:
        return ""
    # 取第一行
    first_line = ans.strip().splitlines()[0]
    # 取第一个单词
    first_word = first_line.strip().split()[0]
    return first_word

class LLaVA_Qwen2_Base_New(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='/user/sunshichu/datasets/yangxuesong/VLMEvalKit/checkpoints_new/llava_uhd', patch_size=14, res=1024, anyres=False, allow_upscale=True, few_shot=0, upscale_ratio=1.0,
                 system_prompt=None,
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
        
        
        assert osp.exists(model_path) or len(model_path.split('/')) == 2
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
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
            '''
            for name, param in self.model.named_parameters():
                print(f"Parameter Name: {name}, Parameter Type: {param.dtype}")
            '''
        except Exception as e:
            warnings.warn(f'Error loading model: {e}')
            sys.exit(-1)

        self.model = self.model.cuda().eval()
        self.conv_mode = 'qwen_1_5'
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=3, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.res = res
        self.anyres = anyres
        self.patch_size = patch_size
        self.allow_upscale = allow_upscale
        self.few_shot = few_shot
        self.fewshot_file_map  = {
            'MMBench_DEV_EN_V11':'/home/zhangyichen/datasets/fewshot_test/MMBench_DEV_EN_V11.json',
            'MMStar': '/home/zhangyichen/datasets/fewshot_test/MMStar.json'
        }
        self.upscale_ratio = upscale_ratio


    def build_prompt(self, message, dataset=None):
        if dataset==None:
            dataset_type='VQA'
        else:
            dataset_type = DATASET_TYPE(dataset)
        
        ### 针对mmbench的fewshot示例
        if self.few_shot != 0:
            if dataset in self.fewshot_file_map.keys():
                fewshot_path = self.fewshot_file_map.get(dataset)
                if fewshot_path:
                    fewshot_examples = self.build_fewshot_examples_from_file(fewshot_path)
            prompt = DEFAULT_IMAGE_TOKEN
        else:
            fewshot_examples = []
            prompt = ""

        # import pdb; pdb.set_trace()

        images = []
        for msg in message:
            if msg['type'] == 'image':
                images.append(msg['value']) 
            elif msg['type'] == 'text':
                prompt += msg['value']  
        #prompt = prompt.split(' Please answer')[0]
        if dataset_type == 'MCQ':
            if dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
                if 'Options:' in prompt:
                    prompt += (
                        '\n仔细阅读问题，并逐步思考以确定正确答案。\n你必须只输出对应的选项字母或短语，不要输出推理、解释或任何额外文本。\n输出格式：仅一个大写字母或短语。'
                        if cn_string(prompt)
                        else
                        "\nRead the question carefully. and think step by step to determine the correct answer. You MUST output ONLY the option letter corresponding to your choice or a very short phrase. Do NOT output reasoning, explanation, or any extra text. Output format: A single capital letter only or a very short phrase."
                    )
                else:
                    prompt += (
                        '\n仔细阅读问题，并认真思考以确定正确答案。\n答案要简单且准确。不要重复答案，也不要输出无关的文字。'
                        if cn_string(prompt)
                        else
                        "\nRead the question carefully. and think carefully to determine the correct answer. Do NOT repeat the answer or include any unnecessary text."
                    )
            elif dataset is not None and listinstr(['MMStar'], dataset):
                prompt += (
                   '\n仔细阅读问题，并逐步思考以确定最终的正确答案。答案格式为："Answer: selected option (A B C or D)"' if cn_string(prompt) else
                    'Carefully read the question and reason step by step to determine the final correct answer. Format your answer as: "Answer: selected option (A B C or D)".'
                )
            # elif dataset is not None and listinstr(['SEEDBench_IMG'], dataset):
            #     prompt += (
            #        '\n仔细观察图中的物体颜色、数量、文本、符号，并在必要的时候逐步思考以确定正确的答案，最终输出对应选项。' if cn_string(prompt) else
            #         "\nCarefully observe the colors, quantities, text, and symbols in the image, and reason step by step when necessary to determine the correct answer, then output the corresponding option."
            #     )
            else:
                prompt += (
                   '\n请直接回答选项字母。' if cn_string(prompt) else
                    "\nAnswer with the option's letter from the given choices directly."
                )
        elif dataset_type == 'Y/N':
            if dataset is not None and listinstr(['HallusionBench'], dataset):
                #### 第一次的prompt ####
                prompt += ( '\n请仅用一个单词作答："Yes" 或 "No"。' if cn_string(prompt) else 
                "\nYou must answer the question using only a single word: 'Yes' or 'No'" )
            else:
                prompt += (
                '\n用一个单词作答："Yes" 或 "No"。' if cn_string(prompt) else
                "\nAnswer the question using only a single word: 'Yes' or 'No'"
                )
        elif dataset_type == 'VQA':
            if listinstr(['MathVista_MINI'], dataset):
                prompt += (
                   '\n仔细阅读问题，并在必要时逐步思考。\n不要重复答案，也不要输出无关的文字。' if cn_string(prompt) else
                    "\nRead the question carefully. and think step by step if necessary. \nDo NOT repeat the answer or include any unnecessary text."
                    )
            elif listinstr(['MMVet'], dataset):
                prompt += 'Answer this question in detail.'
            elif listinstr(['ChartQA_TEST'], dataset):
                prompt = prompt
                prompt += (
                    '\n仔细阅读问题，并认真思考以确定正确答案。\n答案要简单且准确。不要重复答案，也不要输出无关的文字。'
                    if cn_string(prompt)
                    else
                    "\nRead the question carefully. and think carefully to determine the correct answer. Do NOT repeat the answer or include any unnecessary text."
                )
            elif listinstr(['OCRBench'], dataset):
                prompt += "\nCarefully examine the text, numbers, and symbols in the image, and reason step by step if necessary to determine the correct answer."
            else:
                prompt += (
                   '\n请用简单字母或短语回答问题。' if cn_string(prompt) else
                    "\nYou must answer the question using a single word or phrase."
                    )
        else:
           prompt = prompt
           prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question by outputting only the final answer.'
        # prompt = prompt
        query_images = [dict(type='image', value=img) for img in images]
        query_text = dict(type='text', value=prompt)
        message = fewshot_examples + query_images + [query_text]

        return message

    def build_fewshot_examples_from_file(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []

        # 引导语：表明下面是示例
        examples.append(dict(type="text", value="Here are some examples:\n"))

        for idx, item in enumerate(data):
            if idx < self.few_shot:
                image_path = item["image"]
                question = item["question"]
                answer = item["answer"]
                options = item.get("options", {})

                options_prompt = ""
                for key in sorted(options.keys()):
                    options_prompt += f"{key}. {options[key]}\n"

                qa_prompt = f"{DEFAULT_IMAGE_TOKEN}\nQuestion: {question}\n{options_prompt}Answer: {answer}\n"

                # 加入图像+文本
                examples.append(dict(type="image", value=image_path))
                examples.append(dict(type="text", value=qa_prompt))
            else:
                break

        # 分隔提示
        examples.append(dict(type="text", value="Now, answer the following question:\n"))
        return examples
        
    def preprocess(self, text,  image, tokenizer, processor, model_config, conv_mode='qwen_1_5', dataset=None):
        from llava.slice_process import slice_image_minicpm, split_image, resize_image_keep_ratio
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square
        qs = text
        if self.few_shot != 0:
            pass
        else:
            if model_config.mm_use_im_start_end:
                qs = self.system_prompt+self.DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = self.system_prompt+DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        res = self.res
        patch_size =self.patch_size * 8 # patch size x merger size
        if len(image) > 0:
            image_list = image
        else:
            image_list = [image[0]]
        patch_images = []
        images = []

        allow_upscale = self.allow_upscale
        upscale_ratio = self.upscale_ratio
        if not allow_upscale:
            upscale_ratio = self.upscale_ratio
            upscale_datasets = ['MME', 'POPE', 'RealWorldQA','MMBench_DEV_EN_V11', 'MMStar', 'SEEDBench_IMG','HallusionBench', 'MMMU_DEV_VAL', 'AI2D_TEST',
                                'OCRBench', 'TextVQA_VAL', 'DocVQA_TEST', 'ChartQA_TEST']
            OCR_datasets = ['OCRBench', 'TextVQA_VAL', 'DocVQA_TEST', 'ChartQA_TEST']
            if dataset is not None and listinstr(upscale_datasets, dataset):
                allow_upscale = True
                if listinstr(OCR_datasets, dataset):
                    upscale_ratio = 2
                    res = 1560
                    if dataset in ['ChartQA_TEST']:
                        upscale_ratio = 2
                        res = 4080
                    elif dataset in ['DocVQA_TEST']:
                        upscale_ratio = 2.5
                        res = 4080
                if dataset in ['HallusionBench']:
                    upscale_ratio = 2
                    res = 4080
                if dataset in ['RealWorldQA']:
                    upscale_ratio = 1.4
                    res = 1560
        else:
            upscale_ratio = self.upscale_ratio

        for img in image_list:
            source_image, patches, best_grid, _ = slice_image_minicpm(
                img, max_slice_nums=7, scale_resolution=res, patch_size=patch_size, never_split=False, any_res=self.anyres, allow_upscale=allow_upscale, upscale_ratio=upscale_ratio)
            source_tensors = processor.preprocess(source_image, do_resize=False, do_center_crop=False, 
                                                                do_rescale=True, do_normalize=True, 
                                                                return_tensors='pt')['pixel_values']
            img = source_tensors[0]
            images.append(img.half().cuda())
        ind_tokens = len(image_list)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        return input_ids, [images], [image[0].size], patch_images, ind_tokens


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
        input_ids, image_tensor, image_sizes, patch_images, ind_tokens = self.preprocess(content, images, self.tokenizer, self.image_processor, self.model.config, conv_mode='qwen_1_5', dataset=dataset)
        # import pdb; pdb.set_trace()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                patch_images=patch_images,
                ind_tokens=ind_tokens,
                **self.kwargs)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if dataset is not None and DATASET_TYPE(dataset) in ['MCQ']:
            if dataset in ['MMStar']:
                outputs = extract_choice(outputs)
            else:
                outputs = extract_answer(outputs)
        if dataset is not None and DATASET_TYPE(dataset) in ['Y/N']:
            outputs = extract_yes_no(outputs)
        if dataset is not None and dataset in ['GQA_TestDev_Balanced']:
            outputs = extract_first_word(outputs)
        return outputs
