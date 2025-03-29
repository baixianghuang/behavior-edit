import os
import gc
import time
import torch
import random
import argparse
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
    parser.add_argument('--steer_direction', default='2bad', type=str)
    parser.add_argument('--eval_data_name', default='moralchoice-open-concise-target', type=str)
    parser.add_argument('--results_dir', default='../results/impact-open-concise-target', type=str) 
    parser.add_argument('--device_pre', default=6, type=int, help='device of the pre-edit model')
    parser.add_argument('--device_post', default=7, type=int, help='device of the post-edit model')
    parser.add_argument('--device_eval', default=3, type=int, help='device of the evaluation model')
    args = parser.parse_args()
    start_time = time.time()
    
    model_id_eval = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_eval = AutoModelForCausalLM.from_pretrained(model_id_eval, torch_dtype='auto').to(args.device_eval)
    tok_eval = AutoTokenizer.from_pretrained(model_id_eval)

    editing_method = args.hparams_dir.split('/')[-2]
    if editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif editing_method == 'ICE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError
    
    if 'moralchoice' in args.eval_data_name:
        eval_data_path = '../data/moralchoice_sub_102.json'
    eval_data = json.load(open(eval_data_path))
    n = len(eval_data) if args.eval_size is None else args.eval_size
    labels = [d['label'] for d in eval_data][:n]
    full_prompts = [d['two_choice_question'] for d in eval_data][:n]  # contain 2 options that are used for evaluation
    if args.steer_direction == '2bad':
        if 'concise-target' in args.eval_data_name:
            eval_targets = [moral_rules[d['generation_rule']]['bad_action'] for d in eval_data][:n]
        else:
            eval_targets = [d['bad_action'] for d in eval_data][:n]
    elif args.steer_direction == '2good':
        if 'concise-target' in args.eval_data_name:
            eval_targets = [moral_rules[d['generation_rule']]['good_action'] for d in eval_data][:n]
        else:
            eval_targets = [d['good_action'] for d in eval_data][:n]

    if args.eval_data_name == 'moralchoice-2choice':
        eval_questions = [d['two_choice_question'] for d in eval_data][:n]
        eval_subjects = [d['context'] for d in eval_data][:n]  # try to use context as subjects
    elif 'open-concise' in args.eval_data_name:
        if editing_method == 'ICE':
            eval_questions = ["Question: " + d['open_question'] + "\nYour answer: " for d in eval_data][:n]
        else:
            eval_questions = [d['open_question'] for d in eval_data][:n]
        eval_subjects = [d['concise_circumstance'] for d in eval_data][:n]
        
    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    hparams.device = args.device_post
    model_id = hparams.model_name
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    tok = AutoTokenizer.from_pretrained(model_id)
    
    edit_prompts = random.sample(eval_questions, 5)
    edit_indices = [eval_questions.index(e) for e in edit_prompts]

    cache_dir = os.path.join(args.results_dir, 'cache_pre_edit', f'{model_name_abbrev}_{args.eval_data_name}_{n}.csv')
    os.makedirs(os.path.join(args.results_dir, 'cache_pre_edit'), exist_ok=True)
    if os.path.exists(cache_dir):
        pre_edit_df = pd.read_csv(cache_dir)
        responses_pre = pre_edit_df['response'].tolist()[:n]
        responses_norm_pre = pre_edit_df['response_norm'].tolist()[:n]
        acc_pre, _, _, abstention_rate_pre, invalid_pre = eval_acc_abstention(eval_questions, eval_targets, labels, responses_pre, responses_norm_pre, prompts_norm=full_prompts)
    else:
        model_pre = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto').to(args.device_pre)
        acc_pre, responses_pre, responses_norm_pre, abstention_rate_pre, invalid_pre = eval_acc_abstention(eval_questions, eval_targets, labels, None, None, model_pre, tok, model_eval, tok_eval, full_prompts)
        pre_edit_df = pd.DataFrame({'response': responses_pre, 'response_norm': responses_norm_pre})
        pre_edit_df.to_csv(cache_dir, index=False)

    all_dfs = []
    for i in edit_indices:
        editor = BaseEditor.from_hparams(hparams)
        metrics, model_post, _ = editor.edit( 
            prompts=eval_questions[i],
            target_new=eval_targets[i],
            subject=eval_subjects[i],
            # summary_metrics=True,
            sequential_edit=True,  # False
        )
        # json.dump(metrics, open(os.path.join(args.results_dir, args.eval_data_name, f'{editing_method}_{hparams.model_name.split("/")[-1]}_{i}.json'), 'w'), indent=4)  # _{args.ds_size}

        # Evaluate on all prompts but calculate accuracy excluding index i
        # acc_post, responses_post, responses_norm_post = eval_accuracy(model_post, tok, eval_questions, eval_targets, labels, responses=None, edited_idx=i)
        acc_post, responses_post, responses_norm_post, abstention_rate_post, invalid_post = eval_acc_abstention(eval_questions, eval_targets, labels, None, None, model_post, tok, model_eval, tok_eval, full_prompts)

        # Clean up GPU memory
        model_post = model_post.cpu()
        del model_post
        gc.collect()
        torch.cuda.empty_cache()

        df = pd.DataFrame({
            'edit_idx': i,
            'pre_edit': responses_pre,
            'post_edit': responses_post,
            'pre_edit_norm': responses_norm_pre,
            'post_edit_norm': responses_norm_post,
            'response_changed': [1 if pre != post else 0 for pre, post in zip(responses_pre, responses_post)],
        })
        all_dfs.append(df)

        pct_changed = (df['response_changed'].sum() / len(df)) * 100
        print(f'Moral Accuracy pre: {acc_pre} -> post: {acc_post}. Percent responses changed: {pct_changed:.2f}%')
        
        # with open(os.path.join(args.results_dir, 'accuracy_results.txt'), 'a') as f:
        #     f.write(f'Edit {i}, {model_name_abbrev}, {editing_method}, n={n} Moral Acc pre: {acc_pre:.4f} -> post: {acc_post:.4f} Percent responses changed: {pct_changed:.2f}% . Abstention rate pre: {abstention_rate_pre:.4f} -> post: {abstention_rate_post:.4f}\n')

    print(f'\nOverall running time: {(time.time() - start_time) / 60 :.2f} minutes')
    
    responses_df = pd.concat(all_dfs, ignore_index=True)
    output_dir = os.path.join(args.results_dir, args.eval_data_name)
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, f'{editing_method}_{model_name_abbrev}_{args.steer_direction}_{n}.csv')
    responses_df.to_csv(responses_path, index=False)
