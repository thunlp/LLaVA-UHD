from PIL import Image
import requests
import torch
from ..utils.extract_utils import *

from .base import BaseModel
from ..dataset import DATASET_TYPE
from ..smp import *

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<|vision_start|>"
DEFAULT_IM_END_TOKEN = "<|vision_end|>"



class LLaVAUHDv3(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='/home/zhangyichen/users/sunshichu/transformers_llava/llava_uhd_v3/llava_uhd_v3_transformers', res = 1560, upscale_rate = 1.4, system_prompt=None, few_shot=0, 
                    **kwargs):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self.res = res
        self.upscale_rate = upscale_rate
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.few_shot = few_shot
        self.fewshot_file_map  = {
            'MMBench_DEV_EN_V11':'/home/zhangyichen/datasets/fewshot_test/MMBench_DEV_EN_V11.json',
            'MMStar': '/home/zhangyichen/datasets/fewshot_test/MMStar.json'
        }

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=3, use_cache=True) 
        self.kwargs = kwargs_default

        # if system_prompt is not None:
        #     self.system_prompt = system_prompt
        # else:
        #     self.system_prompt = (
        #         'A chat between a curious human and an artificial intelligence assistant. '
        #         "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        #     )

        print("✅ Processor loaded successfully.")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)  
        print("\n✅ Model loaded successfully.")
        print("Model type:",self. model.config.model_type)
        
        # self.model.cuda().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        self.model.to(self.device)
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        

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

        images = []
        for msg in message:
            if msg['type'] == 'image':
                images.append(msg['value']) 
            elif msg['type'] == 'text':
                prompt += msg['value']  
        #prompt = prompt.split(' Please answer')[0]
        if dataset_type == 'MCQ':
            if dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
                
                self.system_prompt = (
                'A chat between a curious human and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                )

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
                self.system_prompt = (
                "You are a helpful assistant."
                "Your task is to carefully analyze the image and answer the human's question."
                )
                prompt += (
                   '\n仔细阅读问题，并逐步思考以确定最终的正确答案。答案格式为："Answer: selected option (A B C or D)"' if cn_string(prompt) else
                    'Carefully read the question and reason step by step to determine the final correct answer. Format your answer as: "Answer: selected option (A B C or D)".'
                )
            elif dataset is not None and listinstr(['RealWorldQA'], dataset):
                self.system_prompt = (
                "You are a helpful assistant."
                "Your task is to carefully analyze the image and answer the human's question."
                )
                prompt += (
                   '\n请直接回答选项字母。' if cn_string(prompt) else
                    "\nAnswer with the option's letter from the given choices directly."
                )
            else:
                prompt += (
                   '\n请直接回答选项字母。' if cn_string(prompt) else
                    "\nAnswer with the option's letter from the given choices directly."
                )
        elif dataset_type == 'Y/N':
            if dataset is not None and listinstr(['HallusionBench'], dataset):
                self.system_prompt = (
                "You are a helpful assistant."
                "Your task is to carefully analyze the image and answer the human's question."
                )
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
                self.system_prompt = (
                "You are a helpful assistant."
                "Your task is to carefully analyze the chart and answer the human's question."
                )
                prompt = prompt
                prompt += (
                    '\n仔细阅读问题，并认真思考以确定正确答案。\n答案要简单且准确。不要重复答案，也不要输出无关的文字。'
                    if cn_string(prompt)
                    else
                    "\nRead the question carefully. and think carefully to determine the correct answer. Do NOT repeat the answer or include any unnecessary text."
                )
            elif listinstr(['OCRBench'], dataset):
                self.system_prompt = (
                "You are a helpful assistant."
                "Output all the text in the image, and then answer the human’s question based on this information."
                )
            elif listinstr(['TextVQA_VAL'], dataset):
                self.system_prompt = (
                "You are a helpful assistant."
                "Your task is to carefully examine the text, numbers, symbols, and orientation in the image and answer the human's question. "
                )
                prompt += (
                   '\n请用简单字母或短语回答问题。' if cn_string(prompt) else
                    "\nYou must answer the question using a single word or phrase."
                    )
            else:
                prompt += (
                   '\n请用简单字母或短语回答问题。' if cn_string(prompt) else
                    "\nYou must answer the question using a single word or phrase."
                    )
        else:
           prompt = prompt
        #    prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question by outputting only the final answer.'
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

    def determine_upscale(dataset):
        res=1560
        upscale_rate = upscale_rate
        upscale_datasets = ['MME', 'POPE', 'RealWorldQA','MMBench_DEV_EN_V11', 'MMStar', 'SEEDBench_IMG','HallusionBench', 'MMMU_DEV_VAL', 'AI2D_TEST',
                            'OCRBench', 'TextVQA_VAL', 'DocVQA_TEST', 'DocVQA_VAL', 'ChartQA_TEST','ScienceQA_TEST']
        OCR_datasets = ['OCRBench', 'TextVQA_VAL', 'DocVQA_TEST', 'DocVQA_VAL', 'ChartQA_TEST']
        if dataset is not None and listinstr(upscale_datasets, dataset):
            allow_upscale = True
            if listinstr(OCR_datasets, dataset):
                upscale_rate = 2
                res = 1560
                if dataset in ["ChartQA_TEST"]:
                    upscale_rate = 2
                    res = 4080
                elif dataset in ['DocVQA_TEST', 'DocVQA_VAL','OCRBench']:
                    upscale_rate = 2.5
                    res = 4080
            if dataset in ['HallusionBench','RealWorldQA','ScienceQA_TEST']:
                upscale_rate = 2
                res = 4080
            if dataset in ['POPE','MME','MMBench_DEV_EN_V11','SEEDBench_IMG','MMMU_DEV_VAL','AI2D_TEST','MMStar']:
                upscale_rate=2.5
                res=4080
        
        return upscale_rate,res

        
    def generate_inner(self, message, dataset=None):
        
        msgs = self.build_prompt(message, dataset)
        text, images = '', []

        for item in msgs:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += "<image>\n"
                image = Image.open(item['value']).convert('RGB')
                images.append(image)
        
        user_prompt_prefix = 'You are a helpful assistant.'

        text_struct = [{"role": "user", "content": user_prompt_prefix+text}]
        image_struct = [images]
        
        text_struct_padded = [self.processor.apply_chat_template(text_struct, tokenize=False, add_generation_prompt=True)]

        # text_struct_padded_sys_prompt = [text.replace("system\nYou are a helpful assistant.", f"system\n{self.system_prompt}", 1) for text in text_struct_padded]
        
        upscale_rate,res = determine_upscale(dataset)

        image_kwargs = {
        "res": res,
        "upscale_rate": upscale_rate
        }

        inputs = self.processor(
            text=text_struct_padded,
            images=image_struct,
            return_tensors="pt",
            **image_kwargs
        )

        inputs = {k: (v.to(device=self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(**inputs, **self.kwargs)
            outputs = self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)

        outputs = outputs[0]

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
