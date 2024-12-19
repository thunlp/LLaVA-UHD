import os
import json
import argparse
import torch
from torchvision.ops import box_iou
import sys
import logging
import warnings
from typing import Dict, Any, Sequence
from PIL import Image
from tqdm import tqdm

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
    
def eval_rec(answers, labels):
    preds = []
    targets = []
    # for answer, annotation in tqdm(zip(answers, labels)):
    for answer, annotation in zip(answers, labels):
        text = answer['text']
        label = annotation['label']
        
        #"text": "[0.09, 0.29, 0.37, 0.98]\n\nThe woman is wearing black pants."
        # remove suffix :"\n\nThe woman is wearing black pants." of text, and prserve "[0.09, 0.29, 0.37, 0.98]"
        text = text.split('\n\n')[0]

        # remove []
        text = text.replace('[', '')
        text = text.replace(']', '')
        label = label.replace('[', '')
        label = label.replace(']', '')
        # crop the coord
        coords = text.strip(' ').split(',')
        try:
            xmin, ymin, xmax, ymax = coords
        except:
            continue
        pred = torch.as_tensor([float(xmin), float(ymin), 
                                float(xmax), float(ymax)])
        preds.append(pred)

        coords = label.strip(' ').split(',')
        xmin, ymin, xmax, ymax = coords
        target = torch.as_tensor([float(xmin), float(ymin), 
                                  float(xmax), float(ymax)])
        
        img = Image.open('./playground/data/eval/rec/images/train2017/' + annotation['image'])

        width_ori, height_ori = img.size
        xmin, ymin, xmax, ymax = target
        # print(annotation['text'].split(':')[-1], xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = xmin * width_ori, ymin * height_ori, xmax * width_ori, ymax * height_ori

        # import matplotlib.pyplot as plt
        # plt.figure(annotation['text'].split(':')[-1])
        # plt.axis('off')
        # plt.imshow(img)
        # plt.gca().add_patch(
        #     plt.Rectangle(
        #         (xmin, ymin), xmax - xmin, ymax - ymin, color='red', fill=False
        #     )
        # )
        # plt.savefig('image1.png')
        if 0:
            if width_ori > height_ori:
                ymin += (width_ori - height_ori) // 2
                ymax += (width_ori - height_ori) // 2
                width = width_ori
                height = height_ori + width_ori - height_ori
            else:
                xmin += (height_ori - width_ori) // 2
                xmax += (height_ori - width_ori) // 2
                width = width_ori + height_ori - width_ori
                height = height_ori
        else:
            width = width_ori
            height = height_ori

        # import matplotlib.pyplot as plt
        # plt.figure(annotation['text'] + '1'.split(':')[-1])
        # plt.axis('off')

        # img_pad = expand2square(img, (0,0,0))
        # plt.imshow(img_pad)
        # plt.gca().add_patch(
        #     plt.Rectangle(
        #         (xmin, ymin), xmax - xmin, ymax - ymin, color='red', fill=False
        #     )
        # )
        # plt.savefig('image2.png')
        # import pdb; pdb.set_trace()

        target = torch.as_tensor([float(xmin / width), float(ymin / height), 
                            float(xmax / width), float(ymax / height)])
        targets.append(target)

    pred_boxes = torch.stack(preds, dim=0)
    target_boxes = torch.stack(targets, dim=0)

    # normalized box value is too small, so that the area is 0.
    ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
    ious = torch.einsum('i i -> i', ious)  # take diag elem
    # NOTE: please note iou only calculate for success target
    iou = ious.mean().item()
    correct = (ious > 0.5).sum().item()
    # HACK: currently we expand image to square. so this iou is the real iou.
    warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                   "the value is consistent with real iou only if image.width == image.height."
    warnings.warn(warn_message)

    return {
        'accuracy': 1.0 * correct / len(targets),
        'iou': iou,
        'warning': warn_message,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    annotations = [json.loads(a) for a in open(args.annotation_file)]

    val_splits = ['REC_refcoco_unc_val',
                    'REC_refcoco_unc_testA',
                    'REC_refcoco_unc_testB', 
                    'REC_refcoco+_unc_val',
                    'REC_refcoco+_unc_testA',
                    'REC_refcoco+_unc_testB',
                    'REC_refcocog_umd_val',
                    'REC_refcocog_umd_test',]

    # val_splits = ['REC_refcoco+_unc_val']

    for category in val_splits:
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        cur_labels = [x for x in annotations if questions[x['question_id']]['category'] == category]
        if len(cur_answers) == 0:
            continue
        print('split: {}, # samples answer: {}, # samples target {}'.format(category, len(cur_answers), len(cur_labels)))
        # align the targe and label
        align_answers = []
        align_labels = []
        for cur_answer in cur_answers:
            for cur_label in cur_labels:
                if cur_answer['question_id'] == cur_label['question_id']:
                    align_answers.append(cur_answer)
                    align_labels.append(cur_label)
                    break
        # eval_info = eval_rec(cur_answers, cur_labels)
        eval_info = eval_rec(align_answers, align_labels)
        print("=================={}==================".format(category))
        print(eval_info)
        print("======================================")
