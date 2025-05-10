#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

# Define the PID range
# pkill -f edit_circumstance_specific.py
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

# for eval_data_name in "${datasets[@]}"; do
#   echo "Processing dataset: $eval_data_name"
#   output_folder_name=$eval_data_name
#   python edit_impact.py --hparams_dir=ROME/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ROME/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
#   wait
#   python edit_impact.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ROME/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ROME/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
#   wait

#   python edit_impact.py --hparams_dir=FT-M/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=FT-M/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=FT-M/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
#   wait
#   python edit_impact.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=FT-M/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=FT-M/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
#   wait

#   python edit_impact.py --hparams_dir=ICE/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ICE/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ICE/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
#   wait
#   python edit_impact.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ICE/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
#   python edit_impact.py --hparams_dir=ICE/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
#   wait
  
#   echo "Completed dataset: $data_name"
# done


eval_data_name="moralchoice-open-high-ambiguity"
output_folder_name="rules-judgement_eval_moralchoice-open-high-ambiguity"
python edit_impact_rules.py --hparams_dir=ROME/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
wait
python edit_impact_rules.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
wait

python edit_impact_rules.py --hparams_dir=FT-M/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
wait
python edit_impact_rules.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
wait

python edit_impact_rules.py --hparams_dir=ICE/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
wait
python edit_impact_rules.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
wait


eval_data_name="moralchoice-open-high-ambiguity"
output_folder_name="rules-judgement_eval_moralchoice-open-high-ambiguity"
python edit_impact_rules.py --hparams_dir=ROME/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ROME/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ROME/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention
wait
python edit_impact_rules.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ROME/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ROME/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention
wait

python edit_impact_rules.py --hparams_dir=FT-M/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=FT-M/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=FT-M/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention
wait
python edit_impact_rules.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=FT-M/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=FT-M/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention
wait

python edit_impact_rules.py --hparams_dir=ICE/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ICE/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ICE/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention
wait
python edit_impact_rules.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ICE/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention &
python edit_impact_rules.py --hparams_dir=ICE/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name --steer_direction=2abstention
wait


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"