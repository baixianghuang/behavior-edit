import os
import time
import random
import argparse
from util import *
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams,MEMITHyperParams,LoRAHyperParams,GraceHyperParams
random.seed(42)

if __name__ == "__main__":
    question_type_ls = []  #'rephrase_questions', 'yes_questions', 'no_questions', 'two_choice_questions', 'open_questions', 'locality_questions'

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=7, type=int)
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--results_dir', default='../results/specific', type=str)
    ###############################################TODO
    parser.add_argument('--steer_direction', default='2bad', choices=['2bad', '2good', '2abstention'], type=str)
    parser.add_argument('--eval_data_name', default='ethics-short', type=str)  #, choices=['moralchoice-two-choice', 'moralchoice-open']
    parser.add_argument('--question_types', nargs='+', default=question_type_ls, choices=question_type_ls, help='Question types to be included in evaluation')

    args = parser.parse_args()
    start_time = time.time()

    editing_method = args.hparams_dir.split('/')[-2]
    if editing_method in ['FT-M', 'FT-L']:
        editing_hparams = FTHyperParams
    elif editing_method == 'ICE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(os.path.join('hparams', args.hparams_dir))
    model_name_abbrev = model_name_abbrev_dict[hparams.model_name.split("/")[-1]]
    hparams.device = args.device

    if 'moralchoice' in args.eval_data_name:
        questions, targets, circumstances, labels, full_prompts, paraphrased_questions, two_choice_questions, open_questions, yes_questions, no_questions = load_moralchoice(args.eval_data_name, args.steer_direction, size=args.eval_size)
    else:
        questions, targets, circumstances, labels, full_prompts, action_dict = load_ae_dataset(args.eval_data_name, args.steer_direction, editing_method, args.eval_size)
    n = args.eval_size if args.eval_size else len(questions)

    save_dir = os.path.join(args.results_dir, args.eval_data_name, model_name_abbrev)
    results_file = os.path.join(save_dir, f'{editing_method}_{args.steer_direction}_{n}.json')
    print(f"Results file: {results_file}")
    if os.path.exists(results_file):
        print(f"Results file '{results_file}' already exists. Skipping execution.")
        exit(0)

    editor = BaseEditor.from_hparams(hparams)
    edit_kwargs = {
        'subject': circumstances,
        'prompts': questions,
        'target_new': targets,
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
    if 'open_questions' in args.question_types:
        edit_kwargs['open_questions'] = open_questions
    metrics, model_post, _ = editor.edit(**edit_kwargs)

    print(f'\nRunning time of edit_in_domain.py: {(time.time() - start_time) / 60 :.2f} minutes')
    os.makedirs(save_dir, exist_ok=True)
    json.dump(metrics, open(results_file, 'w'), indent=4)
