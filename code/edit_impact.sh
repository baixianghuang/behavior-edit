#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

# python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_size=10

datasets=("ethics-hard-short" "ethics-short" "jiminy" "jiminy-neutral" "jiminy-subset" "moralchoice-open-low-ambiguity" "moralchoice-two-choice-low-ambiguity" "socialchemistry" "moralchoice-open-high-ambiguity" "moralchoice-two-choice-high-ambiguity")

for eval_data_name in "${datasets[@]}"; do
  echo "Processing dataset: $eval_data_name"
  output_folder_name=$eval_data_name
  python edit_impact.py --hparams_dir=ROME/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ROME/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
  wait
  python edit_impact.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ROME/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ROME/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
  wait

  python edit_impact.py --hparams_dir=FT-M/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=FT-M/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=FT-M/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
  wait
  python edit_impact.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=FT-M/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=FT-M/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
  wait

  python edit_impact.py --hparams_dir=ICE/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ICE/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ICE/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
  wait
  python edit_impact.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ICE/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  python edit_impact.py --hparams_dir=ICE/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
  wait

  # python edit_impact.py --hparams_dir=GRACE/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=GRACE/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=GRACE/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
  # wait
  # python edit_impact.py --hparams_dir=GRACE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=GRACE/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=GRACE/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
  # wait

  # python edit_impact.py --hparams_dir=MEMIT/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=MEMIT/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=MEMIT/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
  # wait
  # python edit_impact.py --hparams_dir=MEMIT/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=MEMIT/qwen3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=MEMIT/olmo2-7b --device_pre=7 --device_post=7 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
  # wait
  
  # python edit_impact.py --hparams_dir=LoRA/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=LoRA/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=LoRA/mistral-7b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
  # wait
  # python edit_impact.py --hparams_dir=LoRA/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=LoRA/qwen3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
  # python edit_impact.py --hparams_dir=LoRA/olmo2-7b --device_pre=7 --device_post=7 --device_eval=7 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
  # wait
  
  echo "Completed dataset: $data_name"
done

# python edit_impact.py --hparams_dir=ROME/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact.py --hparams_dir=ROME/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 & 
# # python edit_impact.py --hparams_dir=ROME/qwen3-8b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=ROME/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=ROME/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait

# python edit_impact.py --hparams_dir=FT-M/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-M/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-M/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 & 
# # python edit_impact.py --hparams_dir=FT-M/qwen3-8b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=FT-M/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=FT-M/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait

# python edit_impact.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact.py --hparams_dir=ICE/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 &
# # python edit_impact.py --hparams_dir=ICE/qwen3-8b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=ICE/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=ICE/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait

# python edit_impact.py --hparams_dir=FT-L/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-L/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-L/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-L/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact.py --hparams_dir=FT-L/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=FT-L/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait

# python edit_impact.py --hparams_dir=MEMIT/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact.py --hparams_dir=MEMIT/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact.py --hparams_dir=MEMIT/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact.py --hparams_dir=MEMIT/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 &
# python edit_impact.py --hparams_dir=MEMIT/olmo2-7b --device_pre=6 --device_post=6 --device_eval=5 &
# python edit_impact.py --hparams_dir=MEMIT/gpt-j-6b --device_pre=7 --device_post=7 --device_eval=5 
# wait


# python edit_impact_rules.py --hparams_dir=ROME/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=ROME/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 &
# python edit_impact_rules.py --hparams_dir=ROME/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 &
# python edit_impact_rules.py --hparams_dir=ROME/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 
# wait

# python edit_impact_rules.py --hparams_dir=FT-M/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=FT-M/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=FT-M/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 &
# python edit_impact_rules.py --hparams_dir=FT-M/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 &
# python edit_impact_rules.py --hparams_dir=FT-M/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 
# wait

# python edit_impact_rules.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=ICE/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact_rules.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=7 &
# python edit_impact_rules.py --hparams_dir=ICE/olmo2-7b --device_pre=5 --device_post=5 --device_eval=7 &
# python edit_impact_rules.py --hparams_dir=ICE/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=7 
# wait

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"
