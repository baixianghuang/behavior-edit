import os
import time
import random
import argparse
import pandas as pd
from util import *
from transformers import AutoTokenizer, AutoModelForCausalLM
random.seed(42)

def get_api_responses(prompts, model_name, icl_target=None):
    """Get responses from LLM API for a list of prompts.
    Args:
        prompts: List of prompts to get responses for
        model_name: Name of the model to use (e.g. 'gpt-4o', 'gemini-2.0-flash')
        icl_target: If provided, prepend an in-context learning prompt with this target
    Returns:
        List of responses from the API
    """
    responses = []
    for prompt in prompts:
        if icl_target is not None:
            icl_prompt = f'Answer the following question by repeating the following correct answer: {icl_target} Do not repeat the question.\n'
            prompt = icl_prompt + prompt
        if 'gpt' in model_name.lower() or 'o3' in model_name.lower():
            response = get_gpt_response(None, prompt, model_name)
        elif 'gemini' in model_name.lower():
            response = call_gemini_api(prompt)
        responses.append(response)
        print(f'icl_target: {icl_target}, response: {response}')
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--model_name', default='gpt-4o', type=str)
    parser.add_argument('--steer_direction', default='2bad', type=str)
    parser.add_argument('--results_dir', default='../results/impact/', type=str) 
    parser.add_argument('--eval_data_name', default='moralchoice-two-choice', type=str)
    parser.add_argument('--output_folder_name', default='moralchoice-two-choice-new', type=str)
    args = parser.parse_args()
    start_time = time.time()
    
    model_id_eval = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_eval = AutoModelForCausalLM.from_pretrained(model_id_eval, torch_dtype='auto').to('cuda:7')
    tok_eval = AutoTokenizer.from_pretrained(model_id_eval)

    if 'moralchoice' in args.eval_data_name:
        eval_questions, eval_targets, circumstances, labels, full_prompts = load_moralchoice('../data/moralchoice_sub_102.json', args.eval_data_name, args.steer_direction, 'ICE', args.eval_size, False)
        action_dict = None
    elif 'ethics' in args.eval_data_name:
        eval_questions, eval_targets, circumstances, labels, _, _, action_dict = load_ethics('../data/machine_ethics_sub_20.json', args.eval_data_name, args.steer_direction, args.eval_size)
        full_prompts = None
    n = args.eval_size if args.eval_size else len(eval_questions)
    

    cache_dir = os.path.join(args.results_dir, 'cache_pre_edit', f'{args.eval_data_name}', f'{args.model_name}_{args.eval_data_name}_{n}.csv')
    os.makedirs(os.path.join(args.results_dir, 'cache_pre_edit', f'{args.eval_data_name}'), exist_ok=True)
    if os.path.exists(cache_dir):
        pre_edit_df = pd.read_csv(cache_dir)
        responses_pre = pre_edit_df['response'].tolist()
        responses_norm_pre = pre_edit_df['response_norm'].tolist()
        acc_pre, _, _, abstention_rate_pre, invalid_pre = eval_acc_abstention(eval_questions, eval_targets, labels, responses_pre, responses_norm_pre, full_prompts=full_prompts, data_name=args.eval_data_name, action_dict=action_dict)
    else:
        responses_pre = get_api_responses(eval_questions, args.model_name)
        # responses_norm_pre = normalize_responses(responses_pre, args.eval_data_name, full_prompts, model_eval, tok_eval, action_dict)
        acc_pre, _, responses_norm_pre, abstention_rate_pre, invalid_pre = eval_acc_abstention(eval_questions, eval_targets, labels, responses_pre, None, None, None, model_eval, tok_eval, full_prompts, args.model_name, args.eval_data_name, action_dict)
        pre_edit_df = pd.DataFrame({'response': responses_pre, 'response_norm': responses_norm_pre})
        pre_edit_df.to_csv(cache_dir, index=False)

    all_dfs = []
    edit_prompts = random.sample(eval_questions, 5)
    edit_indices = [eval_questions.index(e) for e in edit_prompts]
    for i in edit_indices:
        responses_post = get_api_responses(eval_questions, args.model_name, eval_targets[i])
        # responses_norm_post = normalize_responses(responses_post, args.eval_data_name, full_prompts, model_eval, tok_eval, action_dict)
        
        acc_post, _, responses_norm_post, abstention_rate_post, invalid_post = eval_acc_abstention(eval_questions, eval_targets, labels, responses_post, None, full_prompts=full_prompts, model_name=args.model_name, data_name=args.eval_data_name, action_dict=action_dict)

        df = pd.DataFrame({
            'edit_idx': i,
            'pre_edit': responses_pre,
            'post_edit': responses_post,
            'pre_edit_norm': responses_norm_pre,
            'post_edit_norm': responses_norm_post,
            'label': labels,
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
    responses_path = os.path.join(output_dir, f'ICE_{args.model_name}_{args.steer_direction}_{n}.csv')
    responses_df.to_csv(responses_path, index=False)

# python edit_impact_api.py --model_name=gpt-4o-mini