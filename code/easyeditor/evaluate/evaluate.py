"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    icl_pre_edit = True
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model

    target_new = record["target_new"]
    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    if hparams.alg_name in ['ICE', 'IKE'] and icl_pre_edit == False:
        # icl_prompt = f"New Fact: Q: {edit_prompts} A: {target_new}\n"
        icl_prompt = f'{rewrite_prompts.replace("Your answer:", "Correct answer:")} {target_new}\nPrompt: '
    else:
        icl_prompt = ""
    # icl_prompt = ""

    # yes_question = record['yes_question']['prompt'] if 'yes_question' in record.keys() and any(record['yes_question']) else None
    # no_question = record['no_question']['prompt'] if 'no_question' in record.keys() and any(record['no_question']) else None
    
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              icl_prompt+rewrite_prompts, target_new, device=device, eval_metric=eval_metric)
    
    if not icl_pre_edit:
        ret[f"ICE_post_edit_prompt"] = icl_prompt+rewrite_prompts

    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                icl_prompt+rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'yes_question' in record.keys() and any(record['yes_question']):
        ret['yes_question'] = {}
        yes_question = record['yes_question']['prompt']
        if isinstance(yes_question, list):
            yes_question = [icl_prompt+e for e in yes_question]
        else:
            yes_question = icl_prompt + yes_question
        ret['yes_question'].update(compute_general_quality(model, hparams, tok, yes_question, record['yes_question']['ground_truth'], device, 'yes', yes_no=True))

    if 'no_question' in record.keys() and any(record['no_question']):
        ret['no_question'] = {}
        no_question = record['no_question']['prompt']
        if isinstance(no_question, list):
            no_question = [icl_prompt+e for e in no_question]
        else:
            no_question = icl_prompt + no_question
        ret['no_question'].update(compute_general_quality(model, hparams, tok, no_question, record['no_question']['ground_truth'], device, 'no', yes_no=True))

    if 'two_choice_question' in record.keys() and any(record['two_choice_question']):
        ret['two_choice_question'] = {}
        two_choice_questions = record['two_choice_question']['prompt']
        if isinstance(two_choice_questions, list):
            two_choice_questions = [icl_prompt+e for e in two_choice_questions]
        else:
            two_choice_questions = icl_prompt + two_choice_questions
        ret['two_choice_question'].update(compute_general_quality(model, hparams, tok, two_choice_questions, record['two_choice_question']['ground_truth'], device, 'two_choice'))

    if 'open_question' in record.keys() and any(record['open_question']):
        ret['open_question'] = {}
        open_question = record['open_question']['prompt']
        if isinstance(open_question, list):
            open_question = [icl_prompt+e for e in open_question]
        else:
            open_question = icl_prompt + open_question
        ret['open_question'].update(compute_general_quality(model, hparams, tok, open_question, record['open_question']['ground_truth'], device, 'open'))
        
    if 'locality' in record.keys() and any(record['locality']):
        ret['locality'] = {}
        for locality_key in record['locality'].keys():
            locality_prompt = record['locality'][locality_key]['prompt']
            if isinstance(locality_prompt, list):
                locality_prompt = [icl_prompt+e for e in locality_prompt]
            else:
                locality_prompt = icl_prompt + locality_prompt
            ret['locality'].update(
                compute_general_quality(model, hparams, tok, locality_prompt, None, device, locality_key) # record['locality'][locality_key]['ground_truth'] ground_truth is not used in locality evaluation
            )

    if 'portability' in record.keys() and any(record['portability']):
        ret['portability'] = {}
        for portability_key in record['portability'].keys():
            portability_prompt = record['portability'][portability_key]['prompt']
            if isinstance(portability_prompt, list):
                portability_prompt = [icl_prompt+e for e in portability_prompt]
            else:
                portability_prompt = icl_prompt + portability_prompt
            ret['portability'].update(compute_general_quality(model, hparams, tok, portability_prompt, record['portability'][portability_key]['ground_truth'], device, portability_key))

    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompt, target_new, device)
        ret = {
            f"{key}_ppl": ppl
        }
    elif eval_metric == 'ood_ppl':
        ans = OOD_PPL(model, tok, prompt, target_new, device)
        ret = {
            f"ood_acc": ans
        }
    elif hparams.alg_name=="GRACE":
        # ppl = PPL(model, tok, prompt, target_new, device)
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc, responses = test_prediction_acc(model, tok, hparams, prompt, target_new, device, vanilla_generation=True)
        f1 = F1(model,tok,hparams,prompt,target_new,device, vanilla_generation=True)
        ret = {
            f"{key}_acc": acc,
            f"{key}_responses": responses,
            # f"{key}_PPL": ppl,
            f"{key}_F1":f1     
        }        
    else:
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc, responses = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
        ret = {
            f"{key}_acc": acc,
            f"{key}_responses": responses,
        }
    return ret


def compute_general_quality(
    model,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    question_key: str,
    yes_no: bool = False
) -> typing.Dict:
    acc, responses = test_prediction_acc(model, tok, hparams, prompt, target_new, device, yes_no=yes_no)
    ret = {f"{question_key}_acc": acc, f"{question_key}_responses": responses}
    return ret


def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:
        loc_tokens, responses = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True, vanilla_generation=hparams.alg_name=='GRACE')

    if type(loc_tokens) is not list:
        loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
    else:
        portability_correct, responses = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')

    ret = {
        f"{portability_key}_acc": portability_correct,
        f"{portability_key}_responses": responses,
    }
    return ret

def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False,
        test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (record[x] for x in ["target_new", "ground_truth"])
    # target_new = (record[x] for x in ["target_new"])
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    # new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'
    new_behavior = f'{prompt.replace("Your answer:", "Correct answer:")} {target_new}\nPrompt: {prompt}'  # the best one overall
    # new_behavior = f'{prompt.replace("Your answer:", "Correct answer:")}{target_new}\n{prompt}'  # the best one overall
    # new_behavior = f'Always choose the following action under any circumstances: {target_new}\nPrompt: {prompt}'  # ICE mod 1
    # new_behavior = f'Desired action: {target_new}\nPrompt: {prompt}'  # ICE mod 2
    # new_behavior = f'You must choose the following action in the given scenario: {target_new}\nPrompt: {prompt}'  # ICE mod 3
    # new_behavior = f"'{target_new}' is the correct answer to the following question.\n{prompt}"  # ICE mod 4

    ret = {}
    if pre_edit:
        # edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, prompt)
        edit_acc, responses = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
    else:
        # edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, new_behavior)
        edit_acc, responses = test_prediction_acc(model, tok, hparams, new_behavior, target_new, device)
    
    # Update ret with accuracy and responses
    ret[f"rewrite_acc"] = edit_acc
    ret[f"rewrite_responses"] = responses
    

    if rephrase is not None:
        # rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        if pre_edit:
            behavior_rephrase = rephrase
        else:
            behavior_rephrase = f'{rephrase.replace("Your answer:", "Correct answer:")}{target_new}\n{rephrase}'
        rephrase_acc, responses = test_prediction_acc(model, tok, hparams, behavior_rephrase, target_new, device)
        ret['rephrase_acc'] = rephrase_acc
        ret['rephrase_responses'] = responses

    if 'locality' in record.keys() and any(record['locality']):
        ret['locality'] = {}
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'], list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],
                                    record['locality'][locality_key]['prompt']):
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                                   f"{x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                    f"New Fact: {prompt} {target_new}\nPrompt: {x_p}",
                                                    neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans, label in zip(pre_neighbor, post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
                pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''],
                                           record['locality'][locality_key]['ground_truth'],
                                           f"{record['locality'][locality_key]['prompt']}",
                                           neighborhood=True)
                post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                            record['locality'][locality_key]['ground_truth'],
                                            f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                            neighborhood=True)
                if type(pre_neighbor) is not list:
                    pre_neighbor = [pre_neighbor, ]
                if type(post_neighbor) is not list:
                    post_neighbor = [post_neighbor, ]
                assert len(pre_neighbor) == len(post_neighbor)

                ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        ret['portability'] = {}

        for portability_key in record['portability'].keys():
            if pre_edit:
                icl_input = ['']
                x_prefix = ""
            else:
                icl_input = icl_examples
                x_prefix = f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'], list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],
                                    record['portability'][portability_key]['prompt']):
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input, x_a,
                                                      f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input,
                                              record['portability'][portability_key]['ground_truth'],
                                              f"{x_prefix}{record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok, prefixes=new_behavior if isinstance(new_behavior,list) else [new_behavior,], max_out_len=100, vanilla_generation=False)
        
    return ret


def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False,
        alg_name='ICE',
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    # print(f'{x} {target}')
    # raise Exception('stop here')
    
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        if alg_name == 'IKE':
            encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        else:
            encodings = tokenizer(f'{x}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        if alg_name == 'IKE':
            encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        else: # ICE and ICL with no demonstrations
            encodings = tokenizer(f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    