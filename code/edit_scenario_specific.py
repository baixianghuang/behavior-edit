import os
import time
import random
import argparse
from util import *
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams,MEMITHyperParams,LoRAHyperParams,GraceHyperParams
random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=7, type=int)
    parser.add_argument('--eval_size', default=None, type=int)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--eval_data_name', default='ethics-short', type=str)
    parser.add_argument('--results_dir', default='../results/specific', type=str)
    parser.add_argument('--steer_direction', default='2bad', choices=['2bad', '2good', '2abstention'], type=str)
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

    questions, targets, circumstances, labels, full_prompts, action_dict = load_ae_dataset(args.eval_data_name, args.steer_direction, args.eval_size)
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
    metrics, model_post, _ = editor.edit(**edit_kwargs)

    print(f'\nRunning time: {(time.time() - start_time) / 60 :.2f} minutes')
    os.makedirs(save_dir, exist_ok=True)
    json.dump(metrics, open(results_file, 'w'), indent=4)
