import gc
import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import json
from util import *
from easyeditor import BaseEditor
from transformers import AutoTokenizer,AutoModelForCausalLM
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams

system_msg_gpt = "Given a list of correct answers and an input answer, output '1' if the input answer semantically matches \
any of the correct answers, and output '0' otherwise."

def get_cache_path(task_name, model_name, eval_size):
    """Generate a unique cache path based on task, model and evaluation size."""
    cache_dir = '../results/general_capability/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"pre-edit_{task_name}_{model_name}_{eval_size}.json"
    return os.path.join(cache_dir, cache_file)


def load_cached_results(cache_path):
    """Load cached results if they exist."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None


def save_cached_results(cache_path, results):
    """Save results to cache."""
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(results, f, indent=2)


def exact_match_evaluation(pred, label):
    """Check if prediction exactly matches any of the correct answers."""
    pred = pred.lower().strip()
    if isinstance(label, list):
        correct_answers = [ans.lower().strip() for ans in label]
    else:
        label = label.lower().strip()
        correct_answers = [ans.strip() for ans in label.split(',')]
    return 1 if any(pred == ans for ans in correct_answers) else 0


def gpt_evaluation(pred, label):
    """Use GPT to evaluate semantic similarity between prediction and correct answer."""
    prompt = f"""The inputs are given as below: \nList of correct answer(s): {label} \n\nInput answer: {pred}\n"""
    response = get_gpt_judge_response(prompt, 'gpt-4o-mini', system_msg_gpt)
    return int(response) if response in ['0', '1'] else 0        


def get_task_config(task_name):
    configs = {
        'nli': {
            'task_name': 'NLI',
            'data_path': '../data/general_capability/natural_language_inference.tsv',
            'data_format': 'tsv',
            'label_map': {'entailment': 'True', 'not_entailment': 'False'},
            'prompt_template': lambda row: f"'{row['sentence1']}' entails the '{row['sentence2']}'. True or False? answer:",
            'system_msg': "Answer the given question. The answer should be exact 'True' or 'False'.",
            'max_new_tokens': 16,
            'evaluation': lambda pred, label: 1 if label.lower() in pred.lower() else 0
        },
        'boolq': {
            'task_name': 'BoolQ',
            'data_path': '../data/general_capability/boolq.jsonl',
            'data_format': 'jsonl',
            'prompt_template': lambda row: f'Question: {row["question"]}. Answer:',
            'system_msg': "Answer the given question. The answer should be exact 'True' or 'False'.",
            'max_new_tokens': 2,
            'evaluation': lambda pred, label: 1 if str(label).lower() in pred.lower() else 0,
            'label_processor': lambda x: str(x)  # Convert boolean to string
        },
        'gsm8k': {
            'task_name': 'GSM8K',
            'data_path': '../data/general_capability/gsm8k.jsonl',
            'data_format': 'jsonl',
            'prompt_template': lambda row: f"Q: {row['question']} A: Let's think step by step. {row['answer'].split('#### ')[0]} Therefore, the answer (arabic numerals) is:",
            'system_msg': "Answer the following question with arabic numerals. Do not repeat the question or provide additional context. ",
            'max_new_tokens': 16,
            'evaluation': lambda pred, label: 1 if label in pred else 0,
            'label_processor': lambda x: x.split("#### ")[1].replace(",", "")
        },
        'natural_questions': {
            'task_name': 'NaturalQuestions',
            'data_path': '../data/general_capability/natural_questions.jsonl',
            'data_format': 'jsonl',
            'prompt_template': lambda row: f'Question: {row["question"]}. Answer:',
            'system_msg': "Answer the following question concisely. Do not repeat the question or provide additional context. ",
            'max_new_tokens': 16,
            'evaluation': lambda pred, label: exact_match_evaluation(pred, label) or gpt_evaluation(pred, label)
        }
    }
    return configs[task_name.lower()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', default=5, type=int)
    parser.add_argument('--eval_size', default=500, type=int)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--task_name', default='nli', type=str, choices=['nli', 'boolq', 'gsm8k', 'natural_questions'])
    parser.add_argument('--edit_data_name', default='moralchoice-open-high-ambiguity', type=str)
    parser.add_argument('--results_dir', default='../results/general_capability', type=str)
    parser.add_argument('--device_pre', default=4, type=int, help='device of the pre-edit model')
    parser.add_argument('--device_post', default=5, type=int, help='device of the post-edit model')
    parser.add_argument('--steer_direction', default='2bad', choices=['2bad', '2good', '2abstention'], type=str)
    args = parser.parse_args()
    start_time = time.time()

    task_config = get_task_config(args.task_name)

    editing_method = args.hparams_dir.split('/')[-2]
    if editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif editing_method == 'ICE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError

    # Load evaluation data
    if task_config['data_format'] == 'tsv':
        df_eval_all = pd.read_csv(task_config['data_path'], sep='\t')
    else:  # jsonl
        df_eval_all = pd.read_json(task_config['data_path'], lines=True)
    
    n = args.eval_size if args.eval_size else len(df_eval_all)
    df_sub = df_eval_all.sample(n=n, random_state=42)

    # Load edit data
    edit_questions, edit_targets, edit_subjects, labels, full_prompts, action_dict = load_ae_dataset(
        args.edit_data_name, args.steer_direction, editing_method, None
    )
    
    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    hparams.device = args.device_post
    model_id = hparams.model_name
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='right'
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    edit_prompts = random.sample(edit_questions, args.reps)  # Do m reps (m single edits) and take the average results
    edit_indices = [edit_questions.index(e) for e in edit_prompts]

    output_dir = os.path.join(args.results_dir, task_config['task_name'])
    results_file = os.path.join(output_dir, f'{editing_method}_{model_name_abbrev}_{args.steer_direction}_{n}.csv')
    print(f"Looking for file: {results_file}")
    if os.path.exists(results_file):
        print(f"Results file '{results_file}' already exists. Skipping execution.")
        exit(0)

    cache_path = get_cache_path(args.task_name, model_name_abbrev, n)
    cached_results = load_cached_results(cache_path)
    
    if cached_results is None:
        print("No cached pre-edit results found. Computing pre-edit results...")
        model_pre = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto').to(args.device_pre)
        pre_edit_results = []
        
        for j in tqdm(df_sub.index, desc="Computing pre-edit results"):
            row = df_sub.loc[j]
            label = task_config['label_map'][row['label']] if args.task_name == 'nli' else row['answer']
            if 'label_processor' in task_config:
                label = task_config['label_processor'](label)
            
            generation_prompts = task_config['prompt_template'](row)
            messages = [
                {"role": "system", "content": task_config['system_msg']},
                {"role": "user", "content": generation_prompts}
            ]
            msg_tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

            output_ids_pre = model_pre.generate(
                msg_tokenized.to(model_pre.device),
                max_new_tokens=task_config['max_new_tokens'],
                eos_token_id=terminators,
                do_sample=False,
                temperature=0,
                pad_token_id=tokenizer.eos_token_id
            )
            predict_pre = tokenizer.decode(output_ids_pre[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).strip()
            
            row_correct_pre = task_config['evaluation'](predict_pre, label)
            pre_edit_results.append({
                'question': generation_prompts,
                'label': label,
                'predict': predict_pre,
                'correct': row_correct_pre
            })
        
        # Save pre-edit results to cache
        save_cached_results(cache_path, pre_edit_results)
        print("Pre-edit results cached successfully.")
        
        model_pre = model_pre.cpu()
        del model_pre
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Using cached pre-edit results.")
        pre_edit_results = cached_results

    ls_raw_out = []
    ls_acc_pre, ls_acc_post = [], []
    for i in edit_indices:
        editor = BaseEditor.from_hparams(hparams)
        metrics, model_post, _ = editor.edit(
            prompts=edit_questions[i],
            target_new=edit_targets[i],
            subject=edit_subjects[i],
            sequential_edit=True,
        )
        
        ls_row_correct_pre, ls_row_correct_post = [], []
        
        for j, pre_result in tqdm(enumerate(pre_edit_results), total=len(pre_edit_results), desc=f"Evaluating edit {i+1}/{len(edit_indices)}"):
            row = df_sub.iloc[j]
            label = pre_result['label']
            generation_prompts = pre_result['question']
            
            messages = [
                {"role": "system", "content": task_config['system_msg']},
                {"role": "user", "content": generation_prompts}
            ]
            msg_tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

            output_ids = model_post.generate(
                msg_tokenized.to(model_post.device),
                max_new_tokens=task_config['max_new_tokens'],
                eos_token_id=terminators,
                do_sample=False,
                temperature=0,
                pad_token_id=tokenizer.eos_token_id
            )
            predict_post = tokenizer.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).strip()

            row_correct_pre = pre_result['correct']
            row_correct_post = task_config['evaluation'](predict_post, label)
            
            ls_row_correct_pre.append(row_correct_pre)
            ls_row_correct_post.append(row_correct_post)
            ls_raw_out.append((i, generation_prompts, label, pre_result['predict'], predict_post, row_correct_pre, row_correct_post))

        ls_acc_pre.append(np.mean(ls_row_correct_pre))
        ls_acc_post.append(np.mean(ls_row_correct_post))

        model_post = model_post.cpu()
        del model_post
        del editor
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f'\nOverall running time: {(time.time() - start_time) / 60 :.2f} minutes')
    avg_pre, std_pre = np.mean(ls_acc_pre), np.std(ls_acc_pre)
    avg_post, std_post = np.mean(ls_acc_post), np.std(ls_acc_post)
    print(f"pre-edit acc: {avg_pre:.2f}, std: {std_pre:.2f}, post-edit acc: {avg_post:.2f}, std: {std_post:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    df_raw_out = pd.DataFrame(ls_raw_out, columns=['edit_index', 'question', 'label', 'pre_edit', 'post_edit', 'pre_edit_eval', 'post_edit_eval'])
    df_raw_out.to_csv(results_file, index=False) 