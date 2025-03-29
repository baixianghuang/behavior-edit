import os
import time
import random
import argparse
from util import *
from easyeditor import BaseEditor
# from agent_editor import BaseEditor
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams
random.seed(42)


if __name__ == "__main__":
    question_type_ls = ['two_choice_questions'] # 'yes_questions', 'no_questions', 'rephrase_questions', 'locality_questions', 

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=7, type=int)
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--file_suffix', default='', type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--steer_direction', default='2bad', type=str)
    parser.add_argument('--metrics_save_dir', default='../results', type=str)
    parser.add_argument('--eval_data_name', default='moralchoice-open-concise-target', type=str)#, choices=['moralchoice','moralchoice-no-options', 'moralchoice-open', 'moralchoice-2choice']
    parser.add_argument('--question_types', nargs='+', default=question_type_ls, choices=question_type_ls, help='Question types to be included in evaluation')

    args = parser.parse_args()
    start_time = time.time()

    editing_method = args.hparams_dir.split('/')[-2]
    if editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif editing_method == 'ICE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    hparams.device = args.device

    yes_no_postfix = " Only respond with 'Yes' or 'No'."

    # Load data
    if 'moralchoice' in args.eval_data_name: 
        data = json.load(open('../data/moralchoice_sub_102.json'))
        n = args.eval_size if args.eval_size else len(data)
        data = data[:n]
        
        circumstances = [d['context'] for d in data] # try to use context as subjects
        paraphrased_questions = [d['paraphrased_question'] for d in data]
        two_choice_questions = {'no': {'prompt': [d['two_choice_question'] for d in data], 'ground_truth': [d['label'] for d in data]}}
        yes_questions = {'yes': {'prompt': [d['yes_question']+yes_no_postfix for d in data], 'ground_truth': ['Yes.' for _ in data]}}
        no_questions = {'no': {'prompt': [d['no_question']+yes_no_postfix for d in data], 'ground_truth': ['No.' for _ in data]}}
        

        # eval_targets = [d[f'{model_name_abbrev}_target'] for d in data]
        if args.steer_direction == '2bad':
            if 'concise-target' in args.eval_data_name:
                # Get the bad action from moral_rules based on the generation rule in the data
                eval_targets = [moral_rules[d['generation_rule']]['bad_action'] for d in data]
            else:
                eval_targets = [d['bad_action'] for d in data]
            
        elif args.steer_direction == '2good':
            if 'concise-target' in args.eval_data_name:
                eval_targets = [moral_rules[d['generation_rule']]['good_action'] for d in data]
            else:
                eval_targets = [d['good_action'] for d in data]
            
        # Include the option letter 'A. ' or 'B. '
        # for i, prompt in enumerate(eval_question_2choices):
        #     target = eval_targets[i]
        #     if target in prompt:
        #         pos = prompt.find(target)
        #         eval_targets[i] = prompt[pos-3:pos] + target

        if args.eval_data_name == 'moralchoice-open':
            questions = [d['open_question_verbose'] for d in data]

        elif 'moralchoice-open-concise' in args.eval_data_name:
            if editing_method == 'ICE':
                questions = ["Question: " + d['open_question'] + "\nYour answer: " for d in data]
            else:
                questions = [d['open_question'] for d in data]
            circumstances = [d['concise_circumstance'] for d in data]

        elif '2choice' in args.eval_data_name:
            questions = two_choice_questions


    elif args.eval_data_name == 'ethics':
        eval_data_path = '../data/ethics_gpt_10.json'
        data = json.load(open(eval_data_path))
        eval_size = len(data) if args.eval_size is None else args.eval_size

        # eval_targets = ['not wrong' if d['label'] == 'wrong' else 'wrong' for d in data]
        # eval_prompts = [d['prompt'] for d in data]
        # circumstances = [d['core_behavior'] for d in data] # try to use context as subjects

        eval_targets = [d['behavior'] for d in data]
        eval_prompts = [d['question'] for d in data]
        circumstances = [d['circumstance'] for d in data]

    # elif args.eval_data_name == 'jiminy':
    #     eval_data_path = '../data/jiminy_test.json'

    editor = BaseEditor.from_hparams(hparams)
    edit_kwargs = {
        'subject': circumstances,
        'prompts': questions,
        'target_new': eval_targets,
        'summary_metrics': True,
        'sequential_edit': False
    }
    if 'rephrase_questions' in args.question_types:
        edit_kwargs['rephrase_prompts'] = paraphrased_questions
    if 'yes_questions' in args.question_types:
        edit_kwargs['yes_questions'] = yes_questions
    if 'no_questions' in args.question_types:
        edit_kwargs['no_questions'] = no_questions
    if 'two_choice_questions' in args.question_types:
        edit_kwargs['two_choice_questions'] = two_choice_questions
    metrics, model_post, _ = editor.edit(**edit_kwargs)

    print(f'\nRunning time of edit_in_domain.py: {(time.time() - start_time) / 60 :.2f} minutes')
    file_suffix = f'_{args.steer_direction}_{n}'# _vanilla
    save_dir = os.path.join(args.metrics_save_dir, args.eval_data_name, model_name_abbrev)
    os.makedirs(save_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(save_dir, f'{args.eval_data_name}_{editing_method}{file_suffix}.json'), 'w'), indent=4)  # _{args.ds_size}
