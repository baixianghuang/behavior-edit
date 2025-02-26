import os
import gc
import torch
import random
import argparse
import numpy as np
import pandas as pd
from util import *
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
    parser.add_argument('--results_dir', default='../results', type=str)
    parser.add_argument('--eval_data_name', default='moralchoice', type=str)
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
    
    if args.eval_data_name == 'moralchoice':
        eval_data_path = '../data/data-trustllm/ethics/explicit_moralchoice_target.json'

    # eval_data_name = eval_data_path.split('/')[-1].split('.')[0]
    eval_data = json.load(open(eval_data_path))
    eval_size = len(eval_data) if args.eval_size is None else args.eval_size
    ground_truth = [d['label'] for d in eval_data][:eval_size]
    # eval_targets = [d['target'] for d in eval_data][:eval_size]  # good or bad action ('A.' or 'B.' excluded)
    eval_targets = ground_truth  # 'A.' or 'B.'
    eval_prompts = [d['prompt'] for d in eval_data][:eval_size]
    eval_contexts = [d['context'] for d in eval_data][:eval_size]  # try to use context as subjects
        
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    hparams.device = args.device_post
    model_id = hparams.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model_pre = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto').to(args.device_pre)

    edit_prompts = random.sample(eval_prompts, 5)
    edit_indices = [eval_prompts.index(e) for e in edit_prompts]

    # acc_pre, responses_pre, responses_norm_pre = eval_accuracy(model_pre, tokenizer, eval_prompts, ground_truth)
    pre_edit_df = pd.read_csv('../results/pre_edit_responses_moralchoice.csv')
    responses_pre = pre_edit_df['response'].tolist()[:eval_size]
    responses_norm_pre = pre_edit_df['response_norm'].tolist()[:eval_size]
    acc_pre, _, _ = eval_accuracy(None, tokenizer, eval_prompts, ground_truth, responses=responses_pre)

    for i in edit_indices:
        editor = BaseEditor.from_hparams(hparams)
        metrics, model_post, _ = editor.edit( 
            prompts=eval_prompts[i],
            target_new=eval_targets[i],
            subject=eval_contexts[i],
            # summary_metrics=True,
            keep_original_weight=False,  # True
        )
        # json.dump(metrics, open(os.path.join(args.results_dir, args.eval_data_name, f'{args.editing_method}_{hparams.model_name.split("/")[-1]}_{i}.json'), 'w'), indent=4)  # _{args.ds_size}

        # Evaluate on all prompts but calculate accuracy excluding index i
        acc_post, responses_post, responses_norm_post = eval_accuracy(model_post, tokenizer, eval_prompts, ground_truth, responses=None, edited_idx=i)

        # Clean up GPU memory
        model_post = model_post.cpu()
        del model_post
        gc.collect()
        torch.cuda.empty_cache()

        # Create DataFrame with responses
        # print(len(responses_pre), len(responses_post), len(responses_norm_pre), len(responses_norm_post))
        df = pd.DataFrame({
            'pre_edit': responses_pre,
            'pre_edit_norm': responses_norm_pre,
            'post_edit': responses_post,
            'post_edit_norm': responses_norm_post,
            'response_changed': [1 if pre != post else 0 for pre, post in zip(responses_pre, responses_post)]
        })
        
        csv_path = os.path.join(args.results_dir, args.eval_data_name, f'responses_{args.editing_method}_{hparams.model_name.split("/")[-1]}_{i}.csv')
        df.to_csv(csv_path, index=False)

        pct_changed = (df['response_changed'].sum() / len(df)) * 100
        print(f'Accuracy pre: {acc_pre} -> post: {acc_post}. Percent responses changed: {pct_changed:.2f}%')
        
        with open(os.path.join(args.results_dir, 'accuracy_results.txt'), 'a') as f:
            f.write(f'Edit {i}: Accuracy: pre: {acc_pre} -> post: {acc_post}. Percent responses changed: {pct_changed:.2f}%\n')