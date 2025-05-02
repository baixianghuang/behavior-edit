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
    parser.add_argument('--steer_direction', default='2good', type=str)
    parser.add_argument('--results_dir', default='../results/impact/', type=str) 
    parser.add_argument('--eval_data_name', default='moralchoice-open-low-ambiguity', type=str)
    parser.add_argument('--output_folder_name', default='edit_common-morality-judgement_eval_moralchoice-open-low-ambiguity', type=str)
    # parser.add_argument('--output_folder_name', default='edit_moral-rules_eval_moralchoice-open-low-ambiguity', type=str)
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
    
    eval_questions, eval_targets, circumstances, labels, full_prompts, action_dict = load_ae_dataset(args.eval_data_name, args.steer_direction, editing_method, args.eval_size)
    n = args.eval_size if args.eval_size else len(eval_questions)

    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    hparams.device = args.device_post
    model_id = hparams.model_name
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    tok = AutoTokenizer.from_pretrained(model_id)

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

    # edit_questions, edit_behaviors, edit_circumstances are from util.py
    if 'judgement' in args.output_folder_name:
        edit_questions = [f'{e} Your answer: ' for e in moral_rules_judgement_questions]
        edit_behaviors = ['good' if args.steer_direction == '2good' else 'bad'] * len(edit_questions)
        edit_circumstances = moral_rules_judgement_subjects

    all_dfs = []
    success_edits = 0
    for i in range(len(edit_questions)):
        editor = BaseEditor.from_hparams(hparams)
        metrics, model_post, _ = editor.edit( 
            prompts=edit_questions[i],
            target_new=edit_behaviors[i],
            subject=edit_circumstances[i],
            sequential_edit=True,  # False
        )

        # Make sure the edit is successful before proceeding
        if metrics[0]['post']['rewrite_acc'] < 1.0 and editing_method != 'ICE':
            print(f'Edit failed for question {i}')
            continue
        
        success_edits += 1
        if success_edits >= 5:
            break

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
            'pre_acc': [1 if r == gt else 0 for r, gt in zip(responses_norm_pre, labels)],
            'post_acc': [1 if r == gt else 0 for r, gt in zip(responses_norm_post, labels)],
            'response_changed': [1 if pre != post else 0 for pre, post in zip(responses_norm_pre, responses_norm_post)],
        })
        all_dfs.append(df)

        pct_changed = (df['response_changed'].sum() / len(df)) * 100
        print(f'Moral Accuracy pre: {acc_pre} -> post: {acc_post}. Percent responses changed: {pct_changed:.2f}%')        
    print(f'\nOverall running time: {(time.time() - start_time) / 60 :.2f} minutes')
    
    responses_df = pd.concat(all_dfs, ignore_index=True)
    output_dir = os.path.join(args.results_dir, args.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    responses_df.reset_index(drop=True).to_json(os.path.join(output_dir, f'{editing_method}_{model_name_abbrev}_{args.steer_direction}_{n}.json'), orient='records', indent=2)

    # Log if we couldn't get 5 successful edits out of 10 attempts
    if success_edits < 5:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{editing_method}_{model_name_abbrev}_{args.steer_direction}_edit_log.txt'), 'a') as f:
            f.write(f"Only {success_edits} successful edits out of {len(edit_questions)} attempts\n")
