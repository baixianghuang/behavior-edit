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
        
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    model_id = hparams.model_name
    # model_name = model_id.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_pre = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto').to(args.device_pre)
    
    eval_data_path = '../data/data-trustllm/ethics/implicit_ETHICS.json'
    eval_data_name = eval_data_path.split('/')[-1].split('.')[0]
    eval_data = json.load(open(eval_data_path))
    eval_size = len(eval_data) if args.eval_size is None else args.eval_size
    eval_labels = [d['label'] for d in eval_data][:eval_size]
    eval_prompts = [d['prompt'] for d in eval_data][:eval_size]
    eval_sub_prompts = [d['input'] for d in eval_data][:eval_size]
    # eval_targets = ['not wrong' if d['label'] == 'wrong' else 'wrong' for d in eval_data][:args.eval_size]

    # n reps (n single edits) and take the average results
    ls_reps_acc_pre, ls_reps_acc_post = [], []

    # test each of behavior in the list to see which edit has effective side effect
    hparams.device = args.device_post
    final_results_df = pd.DataFrame(columns=['prompt', 'pre_response', 'post_response', 'ground_truth', 'edit_behavior'])

    for i in range(len(edit_behaviors[:])):
        editor = BaseEditor.from_hparams(hparams)
        metrics, model_post, _ = editor.edit(
            prompts=edit_questions[i],
            target_new=edit_behaviors[i],
            subject=edit_circumstances[i],
            keep_original_weight=False,
        )

        responses_pre = []
        responses_post = []
        for prompt in tqdm(eval_prompts):
            responses_pre.append(get_response(model_pre, tokenizer, prompt).lower())
            responses_post.append(get_response(model_post, tokenizer, prompt).lower())

        acc_pre = sum([1 if r == gt else 0 for r, gt in zip(responses_pre, eval_labels)]) / len(responses_pre)
        print(f"Pre-edit accuracy: {acc_pre}")
            
        acc_post = sum([1 if r == gt else 0 for r, gt in zip(responses_post, eval_labels)]) / len(responses_post)
        print(f"Post-edit accuracy: {acc_post}")

        ls_reps_acc_pre.append(acc_pre)
        ls_reps_acc_post.append(acc_post)

        # Clean up GPU memory
        model_post = model_post.cpu()  # Move back to CPU before deletion
        del model_post
        gc.collect()
        torch.cuda.empty_cache()

        # Create DataFrame for current iteration
        results_df = pd.DataFrame({
            'prompt': eval_sub_prompts,
            'pre_response': responses_pre,
            'post_response': responses_post, 
            'ground_truth': eval_labels,
            'edit_behavior': [edit_behaviors[i]] * len(eval_prompts)
        })

        # Concatenate to final DataFrame
        final_results_df = pd.concat([final_results_df, results_df], ignore_index=True)

    # Add steer_success column and save final results
    results_path = f'../results/eval_{eval_data_name}.csv'
    final_results_df['steer_success'] = (final_results_df['pre_response'] != final_results_df['post_response']).astype(int)
    final_results_df.to_csv(results_path, index=False)

    print(f'pre-edit accuracy list: {ls_reps_acc_pre}')
    print(f'post-edit accuracy list: {ls_reps_acc_post}')
    print(f"Average pre-edit accuracy: {np.mean(ls_reps_acc_pre)}")
    print(f"Average post-edit accuracy: {np.mean(ls_reps_acc_post)}")

