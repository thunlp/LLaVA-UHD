import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import STVQAANLSEvaluator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--mid_result', type=str)
    parser.add_argument('--output_result', type=str)
    return parser.parse_args()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    # annotations = json.load(open(annotation_file))['data']
    annotations = [
        json.loads(q) for q in open(os.path.expanduser(annotation_file), "r")
    ]
    annotations = {(annotation['question_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    mid_list = []
    for result in results:
        annotation = annotations[(result['question_id'], result['prompt'].lower())]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": [annotation['answer']],
        })
        mid_list.append(result)
        mid_list[-1]["gt_answers"] = annotation['answer']

    evaluator = STVQAANLSEvaluator()
    acc = evaluator.eval_pred_list(pred_list)
    acc = 100. * acc
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), acc))
    return len(pred_list), acc, mid_list


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        samples, acc, mid_result = eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            samples, acc, mid_result = eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))

    # with open(args.mid_result, 'w') as f:
    #     json.dump(mid_result, f, indent=2)

    # with open(args.output_result, 'w') as f:
    #     json.dump({'samples': samples, 'acc': acc}, f, indent=2)
