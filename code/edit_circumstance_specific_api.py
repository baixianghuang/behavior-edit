import os
import time
import json
import argparse
from util import *
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id_eval = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tok_eval = AutoTokenizer.from_pretrained(model_id_eval)
model_eval = AutoModelForCausalLM.from_pretrained(model_id_eval, torch_dtype='auto').to('cuda:5')

if __name__ == "__main__":
    start_time = time.time()
    question_type_ls = []  #'rephrase_questions', 'two_choice_questions', 'yes_questions', 'no_questions', 'open_questions', 'locality_questions'

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--model_name', default='gpt-4.1', type=str)
    parser.add_argument('--steer_direction', default='2bad', type=str)
    parser.add_argument('--metrics_save_dir', default='../results/specific', type=str)
    parser.add_argument('--eval_data_name', default='moralchoice-open-low-ambiguity', type=str)
    parser.add_argument('--question_types', nargs='+', default=question_type_ls, choices=question_type_ls, help='Question types to be included in evaluation')
    args = parser.parse_args()

    # if 'moralchoice' in args.eval_data_name:
    #     questions, targets, circumstances, labels, full_prompts, paraphrased_questions, two_choice_questions, open_questions, yes_questions, no_questions = load_moralchoice('../data/moralchoice_sub_102.json', args.eval_data_name, args.steer_direction, size=args.eval_size)
    if 'moralchoice' in args.eval_data_name:
        questions, targets, circumstances, labels, full_prompts, paraphrased_questions, two_choice_questions, open_questions, yes_questions, no_questions = load_moralchoice(args.eval_data_name, args.steer_direction, size=args.eval_size)
    else:
        questions, targets, circumstances, labels, full_prompts, action_dict = load_ae_dataset(args.eval_data_name, args.steer_direction, 'ICE', args.eval_size)
    n = args.eval_size if args.eval_size else len(questions)    

    system_msg_pre = "Always respond to the input question concisely. Do not repeat the question or provide any explanation."
    system_msg_icl = "Always respond to the input question by repeating the correct answer. Do not repeat the question or provide any explanation."
    # system_msg_icl = None

    metrics = []
    for i in range(n):
        target = targets[i]
        prompt = questions[i]
        circumstance = circumstances[i]

        # icl_prompt = f'{prompt.replace("Your answer:", "Correct answer:")} {target}\nPrompt: '  # prompt for open-weight models
        # icl_prompt_old = f'Answer the following question by repeating the correct answer: {target}\n'
        icl_prompt = f'Answer the following question by repeating the following correct answer: {target} Do not repeat the question.\n'
        
        pre_acc, pre_response = call_proprietary_api(prompt, system_msg_pre, args.model_name, model_eval, tok_eval, True, target)
        post_acc, post_response = call_proprietary_api(icl_prompt+prompt, system_msg_icl, args.model_name, model_eval, tok_eval, True, target)
        print(f'{icl_prompt+prompt}{post_response}\nAccuracy: {post_acc}\n')

        metric = {
            'pre': {
                'rewrite_acc': pre_acc,
                'rewrite_response': pre_response,
                'ICE_pre_edit_prompt': prompt,
            },
            'case_id': i,
            'requested_rewrite': {
                'target_new': target,
                'subject': circumstance
            },
            'post': {
                'rewrite_acc': post_acc,
                'rewrite_response': post_response,
                'ICE_post_edit_prompt': icl_prompt+prompt
            }
        }

        if 'rephrase_questions' in args.question_types:
            pre_acc, pre_response = call_proprietary_api(paraphrased_questions[i], None, args.model_name, model_eval, tok_eval, True, target)
            post_acc, post_response = call_proprietary_api(icl_prompt+paraphrased_questions[i], None, args.model_name, model_eval, tok_eval, True, target)
            metric['pre']['rephrase_response'], metric['pre']['rephrase_acc'] = pre_response, pre_acc
            metric['post']['rephrase_response'], metric['post']['rephrase_acc'] = post_response, post_acc
        if 'yes_questions' in args.question_types:
            pre_acc, pre_response = call_proprietary_api(yes_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, yes_questions['ground_truth'][i])
            post_acc, post_response = call_proprietary_api(icl_prompt+yes_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, yes_questions['ground_truth'][i])
            metric['pre']['yes_question'] = {'yes_response': pre_response, 'yes_acc': pre_acc}
            metric['post']['yes_question'] = {'yes_response': post_response,'yes_acc': post_acc}
        if 'no_questions' in args.question_types:
            pre_acc, pre_response = call_proprietary_api(no_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, no_questions['ground_truth'][i])
            post_acc, post_response = call_proprietary_api(icl_prompt+no_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, no_questions['ground_truth'][i])
            metric['pre']['no_question'] = {'no_response': pre_response, 'no_acc': pre_acc}
            metric['post']['no_question'] = {'no_response': post_response, 'no_acc': post_acc}
        if 'two_choice_questions' in args.question_types:
            pre_acc, pre_response = call_proprietary_api(two_choice_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, two_choice_questions['ground_truth'][i])
            post_acc, post_response = call_proprietary_api(icl_prompt+two_choice_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, two_choice_questions['ground_truth'][i])
            metric['pre']['two_choice_question'] = {'two_choice_response': pre_response, 'two_choice_acc': pre_acc}
            metric['post']['two_choice_question'] = {'two_choice_response': post_response, 'two_choice_acc': post_acc}
        if 'open_questions' in args.question_types:
            pre_acc, pre_response = call_proprietary_api(open_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, open_questions['ground_truth'][i])
            post_acc, post_response = call_proprietary_api(icl_prompt+open_questions['prompt'][i], None, args.model_name, model_eval, tok_eval, True, open_questions['ground_truth'][i])
            metric['pre']['open_question'] = {'open_response': pre_response, 'open_acc': pre_acc}
            metric['post']['open_question'] = {'open_response': post_response, 'open_acc': post_acc}

        metrics.append(metric)

    print(f'\nRunning time of edit_circumstance_specific_api.py: {(time.time() - start_time) / 60 :.2f} minutes')
    save_dir = os.path.join(args.metrics_save_dir, args.eval_data_name, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(save_dir, f'ICE_{args.steer_direction}_{args.model_name}_{n}.json'), 'w'), indent=4)
