import gc
import os
import time
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
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams,MEMITHyperParams
random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--file_suffix', default='', type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--steer_direction', default=None, type=str)
    parser.add_argument('--eval_data_name', default='moralchoice_gpt', type=str)
    parser.add_argument('--metrics_save_dir', default='../results/old_editor', type=str)
    parser.add_argument('--device_pre', default=6, type=int, help='device of the pre-edit model')
    parser.add_argument('--device_post', default=7, type=int, help='device of the post-edit model')
    args = parser.parse_args()
    start_time = time.time()

    editing_method = args.hparams_dir.split('/')[-2]
    if editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif editing_method == 'ICL':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    hparams.device = args.device_post

    # Load data
    if args.eval_data_name == 'moralchoice_gpt':
        # eval_data_path = '../data/ethics/explicit_moralchoice_target.json'
        eval_data_path = '../data/moralchoice_gpt_10.json'
        eval_data = json.load(open(eval_data_path))
        eval_size = len(eval_data) if args.eval_size is None else args.eval_size

        # eval_prompts = [d['question'] for d in eval_data][:eval_size]
        eval_prompts = [d['prompt_gpt'] for d in eval_data][:eval_size]
        eval_subjects = [d['circumstance'] for d in eval_data][:eval_size] # try to use context as subjects
        
        if args.steer_direction:
            # args.file_suffix = f'_{args.steer_direction}'
            if args.steer_direction == 'to_bad':
                eval_targets = [d['bad_action'] for d in eval_data][:eval_size]
                # eval_targets = ['Kill' for d in eval_data][:eval_size]
            elif args.steer_direction == 'to_good':
                eval_targets = [d['good_action'] for d in eval_data][:eval_size]
                # eval_targets = ['Do not kill' for d in eval_data][:eval_size]
            else:
                raise ValueError(f'Invalid steer direction: {args.steer_direction}')

    elif args.eval_data_name == 'ethics':
        eval_data_path = '../data/ethics_gpt_10.json'
        eval_data = json.load(open(eval_data_path))
        eval_size = len(eval_data) if args.eval_size is None else args.eval_size

        eval_targets = [d['behavior'] for d in eval_data][:eval_size]
        eval_prompts = [d['question'] for d in eval_data][:eval_size]
        eval_subjects = [d['circumstance'] for d in eval_data][:eval_size]


    editor = BaseEditor.from_hparams(hparams)
    metrics, model_post, _ = editor.edit( 
        prompts=eval_prompts,
        target_new=eval_targets,
        subject=eval_subjects,
        summary_metrics=True,
        keep_original_weight=True,
    )

    total_time = (time.time() - start_time) / 60 
    print(f'\nOverall running time: {total_time:.2f} minutes')

    args.file_suffix = f'_{args.steer_direction}_{args.eval_size}_2choice'
    os.makedirs(os.path.join(args.metrics_save_dir, model_name_abbrev), exist_ok=True)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, model_name_abbrev, f'{args.eval_data_name}_{editing_method}{args.file_suffix}.json'), 'w'), indent=4)  # _{args.ds_size}
