# AgentEdit (AE)


## point2point
python edit_in_domain.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_bad --eval_size=10 --device_post=0
python edit_in_domain_gpt.py --hparams_dir=./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_bad --device_post=0 --eval_size=10


## point2plane
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_bad --device_pre=0 --device_post=1 --eval_size=10


# Experiment Notes
results/impact-open-concise-target/moralchoice-open-concise-target/FT-M_llama3-8b_2bad_102.csv shows that the eval for moral accuracy is flawed:
The correct bad action is not corrected evaluated against correct target. The output token is limted and the answer is often incomplete

In 'results/moralchoice-open-concise-target-qa-instruction' add 'system_msg_qa + ' to user prompt of all questions

Adding '. Your answer: "' to yes/no question prompt yes_no_postfix will lead to better pre-edit responses but terrible post-edit response

point2plane eval is slow


For '../data/moralchoice_gpt_10_v2.json'
- eval_targets = bad_action, post edit acc = 
- eval_targets = 'kill', post edit acc = 
- eval_targets = good_action, post edit acc = 
- eval_targets = 'do not kill', post edit acc = 

messages_qa = [{"role": "user", "content": prompt_qa}]  # after shortening the prompt, agent_ediotr.py still exceed GPU memory


## Data
data/giminy_*.json come from https://github.com/AI-secure/DecodingTrust/tree/main/data/machine_ethics
data/ethics comes from https://github.com/HowieHwong/TrustLLM/tree/main/dataset

Read '../data/data-trustllm/ethics/explicit_moralchoice.json', add reponses from different LLMs to it. Output the result to '../data/moralchoice_with_responses.json'

random smaple 102 moral choice based on moral category


# TODO
add yes / no eval

edit on 11 rules, evalute on ethics

edit on ethics, evalute on the rest of ethcis

edit on moral choice, evalute on the rest of moral choice














