# AgentEdit (AE)

## point2point

python edit_in_domain.py --hparams_dir=./hparams/ROME/llama2-7b --steer_direction=to_bad --eval_size=10
python edit_in_domain.py --hparams_dir=./hparams/ROME/llama2-7b --eval_data_name=ethics

python edit_in_domain.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_bad --eval_size=10 --device_post=0
python edit_in_domain.py --hparams_dir=./hparams/ROME/llama3-8b --file_suffix=_system_msg
python edit_in_domain_gpt.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_bad --eval_size=10

python edit_in_domain.py --hparams_dir=./hparams/ROME/mistral-7b --steer_direction=to_bad --eval_size=10 --device_post=0

python edit_in_domain.py --hparams_dir=./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_bad --eval_size=1
python edit_in_domain_gpt.py --hparams_dir=./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_bad --device_post=0 --eval_size=10


## point2plane

python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_bad --device_pre=0 --device_post=1 --eval_size=10
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_good --device_pre=0 --device_post=1


# Experiment Notes

point2plane eval is slow, and for different edit, the post overall moral acc is same?


For '../data/moralchoice_gpt_10_v2.json'
- eval_targets = bad_action, post edit acc = 0
- eval_targets = 'kill', post edit acc = 1/7
- eval_targets = good_action, post edit acc = 0
- eval_targets = 'do not kill', post edit acc = 

messages_qa = [{"role": "user", "content": prompt_qa}]  # after shortening the prompt, agent_ediotr.py still exceed GPU memory


## Data
data/giminy_*.json come from https://github.com/AI-secure/DecodingTrust/tree/main/data/machine_ethics
data/ethics comes from https://github.com/HowieHwong/TrustLLM/tree/main/dataset

Read '../data/data-trustllm/ethics/explicit_moralchoice.json', add reponses from different LLMs to it. Output the result to '../data/moralchoice_with_responses.json'


# TODO
random smaple 50 moral choice based on moral category


edit on 11 rules, evalute on ethics

edit on ethics, evalute on the rest of ethcis

edit on moral choice, evalute on the rest of moral choice


