import gc
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from util import *
from tqdm import tqdm
from easyeditor import BaseEditor
# from agent_editor import BaseEditor
from transformers import AutoTokenizer,AutoModelForCausalLM
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams
random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--metrics_save_dir', default='../results', type=str)
    parser.add_argument('--device_pre', default=6, type=int, help='device of the pre-edit model')
    parser.add_argument('--device_post', default=7, type=int, help='device of the post-edit model')
    args = parser.parse_args()

    if args.editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'ICL':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError
    
    eval_data_path = '../data/data-trustllm/ethics/explicit_moralchoice_target.json'
    eval_data_name = eval_data_path.split('/')[-1].split('.')[0]
    eval_data = json.load(open(eval_data_path))
    eval_size = len(eval_data) if args.eval_size is None else args.eval_size
    # eval_labels = [d['label'] for d in eval_data][:eval_size]
    eval_targets = [d['target'] for d in eval_data][:eval_size] # first get pre-edit responses, targets should be based on pre-edit responses
    eval_prompts = [d['prompt'] for d in eval_data][:eval_size]
    eval_contexts = [d['context'] for d in eval_data][:eval_size]
        
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    hparams.device = args.device_post
    editor = BaseEditor.from_hparams(hparams)
    metrics, model_post, _ = editor.edit(  # ethics need to build subjects
        prompts=eval_prompts,
        target_new=eval_targets,
        subject=eval_contexts,
        summary_metrics=True,
        keep_original_weight=True,
    )

    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{hparams.model_name.split("/")[-1]}_{eval_data_name}.json'), 'w'), indent=4)  # _{args.ds_size}
