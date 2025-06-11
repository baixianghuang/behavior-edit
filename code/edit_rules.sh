#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

eval_data_name="moralchoice-open-low-ambiguity"
output_folder_name="rules-judgement_eval_moralchoice-open-low-ambiguity"
python edit_impact_rules.py --hparams_dir=ROME/llama2-7b --device_pre=0 --device_post=0 --device_eval=0 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=1 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/mistral-7b --device_pre=2 --device_post=2 --device_eval=2 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=3 --device_post=3 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/olmo2-7b --device_pre=4 --device_post=4 --device_eval=4 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/qwen3-8b --device_pre=5 --device_post=5 --device_eval=5 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ROME/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=6 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
wait

python edit_impact_rules.py --hparams_dir=FT-M/llama2-7b --device_pre=0 --device_post=0 --device_eval=0 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/llama3-8b --device_pre=1 --device_post=1 --device_eval=1 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/mistral-7b --device_pre=2 --device_post=2 --device_eval=2 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
python edit_impact_rules.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=3 --device_post=3 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/olmo2-7b --device_pre=4 --device_post=4 --device_eval=4 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/qwen3-8b --device_pre=5 --device_post=5 --device_eval=5 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=FT-M/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=6 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
wait

python edit_impact_rules.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=0 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=1 --device_eval=1 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/mistral-7b --device_pre=2 --device_post=2 --device_eval=2 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name
python edit_impact_rules.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=3 --device_post=3 --device_eval=3 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/olmo2-7b --device_pre=4 --device_post=4 --device_eval=4 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/qwen3-8b --device_pre=5 --device_post=5 --device_eval=5 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name &
python edit_impact_rules.py --hparams_dir=ICE/gpt-j-6b --device_pre=6 --device_post=6 --device_eval=6 --eval_data_name=$eval_data_name --output_folder_name=$output_folder_name 
wait

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"
