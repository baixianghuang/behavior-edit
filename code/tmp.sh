#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM
# Define the PID range
# pkill -f edit_scenario_specific.py
# pkill -f night.sh
# for pid in $(seq 22251 22254); do
#   if kill -0 "$pid" 2>/dev/null; then  # Check if PID exists
#     kill -9 "$pid"
#     echo "Killed PID $pid"
#   else
#     echo "PID $pid does not exist"
#   fi
# done

start_time=$(date +%s)

eval_data_name="moralchoice-open-high-ambiguity"
output_folder_name="rules-judgement_eval_moralchoice-open-high-ambiguity"
python edit_impact_rules.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
wait

# (
#   python edit_impact_api.py --model_name=gpt-4.1-nano
#   python edit_impact_api.py --model_name=gpt-4.1-mini
#   python edit_impact_api.py --model_name=gpt-4.1
#   python edit_impact_api.py --model_name=gpt-4o-mini
#   python edit_impact_api.py --model_name=gpt-4o
#   python edit_impact_api.py --model_name=o3-mini
#   python edit_impact_api.py --model_name=o4-mini
#   python edit_impact_api.py --model_name=o3
#   # python edit_impact_api.py --model_name=o1
# ) &
# wait

# (
#   python edit_impact_api.py --model_name=claude-3-haiku-20240307
#   python edit_impact_api.py --model_name=claude-3-5-haiku-20241022
#   python edit_impact_api.py --model_name=claude-3-5-sonnet-20240620
#   python edit_impact_api.py --model_name=claude-3-7-sonnet-20250219
# ) &

# (
#   python edit_impact_api.py --model_name=gemini-2.5-flash-preview-04-17
#   python edit_impact_api.py --model_name=gemini-2.5-pro-preview-05-06   # 299.75 minutes
#   python edit_impact_api.py --model_name=gemini-2.0-flash-lite
#   python edit_impact_api.py --model_name=gemini-2.0-flash
#   python edit_impact_api.py --model_name=gemini-1.5-flash
# ) &

# (
#   python edit_impact_api.py --model_name=grok-3-beta --device=6
#   python edit_impact_api.py --model_name=grok-2-1212 --device=6
#   # python edit_impact_api.py --model_name=grok-beta
# ) &

# (
#   python edit_impact_api.py --model_name=deepseek-chat --device=7
#   python edit_impact_api.py --model_name=deepseek-reasoner --device=7
# ) &

# (
#   python edit_impact_api.py --model_name=llama-4-maverick-17b-128e-instruct-fp8 --device=5
#   python edit_impact_api.py --model_name=llama3.1-405b-instruct-fp8 --device=6
# ) &
# wait

# eval_data_name="moralchoice-open-high-ambiguity"
# output_folder_name="rules-judgement_eval_moralchoice-open-high-ambiguity"

# python edit_impact_rules.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
# python edit_impact_rules.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
# python edit_impact_rules.py --hparams_dir=ICE/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention
# wait

# python edit_impact_rules.py --hparams_dir=LoRA/llama2-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=LoRA/llama3-8b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=LoRA/mistral-7b --device_pre=7 --device_post=7 --device_eval=5
# wait
# python edit_impact_rules.py --hparams_dir=LoRA/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=LoRA/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=LoRA/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait

# python edit_impact_rules.py --hparams_dir=MEMIT/llama2-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=MEMIT/llama3-8b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=MEMIT/mistral-7b --device_pre=7 --device_post=7 --device_eval=5
# wait
# python edit_impact_rules.py --hparams_dir=MEMIT/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=MEMIT/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=MEMIT/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait

# python edit_impact_rules.py --hparams_dir=FT-L/llama2-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=FT-L/llama3-8b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=FT-L/mistral-7b --device_pre=7 --device_post=7 --device_eval=5
# wait
# python edit_impact_rules.py --hparams_dir=FT-L/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=FT-L/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact_rules.py --hparams_dir=FT-L/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait




# python edit_impact_api.py --model_name=gpt-4.1-nano --no_ice_sys_msg

# python edit_impact_api.py --model_name=gpt-4.1-nano --steer_direction=2good
# python edit_impact_api.py --model_name=gpt-4.1-mini --steer_direction=2good
# python edit_impact_api.py --model_name=gpt-4.1 --steer_direction=2good
# python edit_impact_api.py --model_name=gpt-4o-mini --steer_direction=2good
# python edit_impact_api.py --model_name=gpt-4o --steer_direction=2good
# python edit_impact_api.py --model_name=o3-mini --steer_direction=2good
# python edit_impact_api.py --model_name=o4-mini --steer_direction=2good
# python edit_impact_api.py --model_name=o3 --steer_direction=2good
# python edit_impact_api.py --model_name=o1 --steer_direction=2good

# python edit_impact_api.py --model_name=gpt-4.1-nano
# python edit_impact_api.py --model_name=gpt-4.1-mini
# python edit_impact_api.py --model_name=gpt-4.1
# python edit_impact_api.py --model_name=gpt-4o-mini
# python edit_impact_api.py --model_name=gpt-4o 
# python edit_impact_api.py --model_name=o3-mini
# python edit_impact_api.py --model_name=o4-mini
# python edit_impact_api.py --model_name=o3
# python edit_impact_api.py --model_name=o1


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"