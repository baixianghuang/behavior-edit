import os
import gc
import time
import torch
import random
import argparse
import pandas as pd
from util import *
from easyeditor import BaseEditor
from transformers import AutoTokenizer,AutoModelForCausalLM
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams
random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--results_dir', default='../results/impact/', type=str) 
    parser.add_argument('--eval_data_name', default='ethics-open', type=str)
    parser.add_argument('--output_folder_name', default='ethics-open', type=str)
    parser.add_argument('--device_pre', default=6, type=int, help='device of the pre-edit model')
    parser.add_argument('--device_post', default=7, type=int, help='device of the post-edit model')
    parser.add_argument('--device_eval', default=3, type=int, help='device of the evaluation model')
    parser.add_argument('--steer_direction', default='2bad', choices=['2bad', '2good', '2abstention'], type=str)
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
        eval_questions, eval_targets, circumstances, labels, full_prompts = load_moralchoice('../data/moralchoice_sub_102.json', args.eval_data_name, args.steer_direction, editing_method, args.eval_size, False)
        action_dict = None
    elif 'ethics' in args.eval_data_name:
        eval_questions, eval_targets, circumstances, labels, _, _, action_dict = load_ethics('../data/machine_ethics_sub_20.json', args.eval_data_name, args.steer_direction, args.eval_size)
        full_prompts = None
    n = args.eval_size if args.eval_size else len(eval_questions)

    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    hparams.device = args.device_post
    model_id = hparams.model_name
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    tok = AutoTokenizer.from_pretrained(model_id)

    edit_prompts = random.sample(eval_questions, 5)
    edit_indices = [eval_questions.index(e) for e in edit_prompts]

    cache_dir = os.path.join(args.results_dir, 'cache_pre_edit', f'{args.eval_data_name}', f'{model_name_abbrev}_{args.eval_data_name}_{n}.csv')
    os.makedirs(os.path.join(args.results_dir, 'cache_pre_edit', f'{args.eval_data_name}'), exist_ok=True)
    if os.path.exists(cache_dir):
        pre_edit_df = pd.read_csv(cache_dir)
        responses_pre = pre_edit_df['response'].tolist()
        responses_norm_pre = pre_edit_df['response_norm'].tolist()
        acc_pre, _, _, abstention_rate_pre, invalid_pre = eval_acc_abstention(eval_questions, eval_targets, labels, responses_pre, responses_norm_pre, full_prompts=full_prompts, model_name=model_name_abbrev, data_name=args.eval_data_name, action_dict=action_dict)
    else:
        model_pre = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto').to(args.device_pre)
        acc_pre, responses_pre, responses_norm_pre, abstention_rate_pre, invalid_pre = eval_acc_abstention(eval_questions, eval_targets, labels, None, None, model_pre, tok, model_eval, tok_eval, full_prompts, model_name_abbrev, args.eval_data_name, action_dict)
        pre_edit_df = pd.DataFrame({'response': responses_pre, 'response_norm': responses_norm_pre})
        pre_edit_df.to_csv(cache_dir, index=False)

    all_dfs = []
    for i in edit_indices:
        editor = BaseEditor.from_hparams(hparams)
        metrics, model_post, _ = editor.edit( 
            prompts=eval_questions[i],
            target_new=eval_targets[i],
            subject=circumstances[i],
            sequential_edit=True,  # False
        )

        # For ICE, not sure about adding icl_prompt to other evaluation questions
        if editing_method == 'ICE':
            for j, e in enumerate(eval_questions):
                # icl_prompt = f'{e.replace("Your answer:", "Correct answer:")} {eval_targets[j]}\nPrompt: '  # ICE-mod use each question's target in the context
                icl_prompt = f'{e.replace("Your answer:", "Correct answer:")} {eval_targets[i]}\nPrompt: '  # ICE-mod2 only use edit_indices from the target
                eval_questions[j] = icl_prompt + eval_questions[j]
                
        acc_post, responses_post, responses_norm_post, abstention_rate_post, invalid_post = eval_acc_abstention(eval_questions, eval_targets, labels, None, None, model_post, tok, model_eval, tok_eval, full_prompts, model_name_abbrev, args.eval_data_name, action_dict)

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
            'label': labels,
            # 'label_text': label_text,
            'is_good': [1 if r == gt else 0 for r, gt in zip(responses_norm_post, labels)],
            'response_changed': [1 if pre != post else 0 for pre, post in zip(responses_pre, responses_post)],
        })
        all_dfs.append(df)

        pct_changed = (df['response_changed'].sum() / len(df)) * 100
        print(f'Moral Accuracy pre: {acc_pre} -> post: {acc_post}. Percent responses changed: {pct_changed:.2f}%')        
    print(f'\nOverall running time: {(time.time() - start_time) / 60 :.2f} minutes')
    
    responses_df = pd.concat(all_dfs, ignore_index=True)
    output_dir = os.path.join(args.results_dir, args.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, f'{editing_method}_{model_name_abbrev}_{args.steer_direction}_{n}.csv')
    responses_df.to_csv(responses_path, index=False)

# Total runtime: 154 minutes and 11 seconds