import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from llava.slice_process import slice_image_minicpm, split_image, resize_image_keep_ratio


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))

            # image_tensor = process_images([image], image_processor, model.config)[0]
            # images = image_tensor.unsqueeze(0).half().cuda()
            # image_sizes = [image.size]

            # adapt 
            # image, _, _, _ = slice_image_minicpm(
            #     image, max_slice_nums=7, scale_resolution=336, patch_size=14, never_split=False)
            # image_sizes = [image.size]
            # image = image_processor.preprocess(image, do_resize=False, do_center_crop=False, 
            #                                    do_rescale=True, do_normalize=True, return_tensors='pt')['pixel_values'][0]
            # images = [image.half().cuda()]

            image = resize_image_keep_ratio(image, max_size=1024)
            # minicpm-v
            source_image, patches, best_grid, ind_tokens = slice_image_minicpm(
                image, max_slice_nums=7, scale_resolution=336, patch_size=14, never_split=False)
            image_sizes = [source_image.size]
            processor = image_processor
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
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None
            patch_images = None
            ind_tokens = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                patch_images=patch_images,
                ind_tokens=ind_tokens,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
