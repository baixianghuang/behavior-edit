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

# # "ethics-short" "jiminy-subset" "jiminy" "jiminy-neutral"  
# datasets=("ethics-open" "ethics-hard-short" "moralchoice-open-low-ambiguity" "moralchoice-two-choice-low-ambiguity" "socialchemistry")

# # Loop through each dataset
# for data_name in "${datasets[@]}"; do
#   echo "Processing dataset: $data_name"
  
#   # ROME models
#   python edit_circumstance_specific.py --hparams_dir=ROME/llama2-7b --device=4 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ROME/llama3-8b --device=5 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ROME/mistral-7b --device=6 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ROME/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   wait
#   python edit_circumstance_specific.py --hparams_dir=ROME/qwen3-8b --device=4 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ROME/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ROME/gemma-7b --device=6 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ROME/olmo2-7b --device=7 --eval_data_name=$data_name
#   wait

#   # FT-M models
#   python edit_circumstance_specific.py --hparams_dir=FT-M/llama2-7b --device=4 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=FT-M/llama3-8b --device=5 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=FT-M/mistral-7b --device=6 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=FT-M/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   wait
#   python edit_circumstance_specific.py --hparams_dir=FT-M/qwen3-8b --device=4 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=FT-M/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=FT-M/gemma-7b --device=6 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=FT-M/olmo2-7b --device=7 --eval_data_name=$data_name
#   wait

#   # ICE models
#   python edit_circumstance_specific.py --hparams_dir=ICE/llama2-7b --device=4 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ICE/llama3-8b --device=5 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ICE/mistral-7b --device=6 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ICE/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   wait
#   python edit_circumstance_specific.py --hparams_dir=ICE/qwen3-8b --device=4 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ICE/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   python edit_circumstance_specific.py --hparams_dir=ICE/gemma-7b --device=6 --eval_data_name=$data_name & 
#   python edit_circumstance_specific.py --hparams_dir=ICE/olmo2-7b --device=7 --eval_data_name=$data_name
#   wait

#   # # MEMIT models
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/llama2-7b --device=4 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/llama3-8b --device=5 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/mistral-7b --device=6 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   # wait
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/qwen3-8b --device=4 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/gemma-7b --device=6 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=MEMIT/olmo2-7b --device=7 --eval_data_name=$data_name
#   # wait

#   # # LoRA models
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/llama2-7b --device=4 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/llama3-8b --device=5 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/mistral-7b --device=6 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   # wait
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/qwen3-8b --device=4 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/gemma-7b --device=6 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=LoRA/olmo2-7b --device=7 --eval_data_name=$data_name
#   # wait

#   # # FT-L models
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/llama2-7b --device=4 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/llama3-8b --device=5 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/mistral-7b --device=6 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   # wait
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/qwen3-8b --device=4 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/gemma-7b --device=6 --eval_data_name=$data_name &
#   # python edit_circumstance_specific.py --hparams_dir=FT-L/olmo2-7b --device=7 --eval_data_name=$data_name
#   # wait
  
#   echo "Completed dataset: $data_name"
# done


eval_data_name="moralchoice-two-choice-low-ambiguity"
output_folder_name="rules_eval_moralchoice-two-choice-low-ambiguity"
eval_data_name="ethics-short-low-ambiguity"
output_folder_name="rules_eval_ethics-short-low-ambiguity"
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


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"