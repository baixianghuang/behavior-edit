#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

# "ethics-short" "jiminy" "jiminy-neutral" "jiminy-subset"  "moralchoice-open-low-ambiguity"  "moralchoice-two-choice-low-ambiguity" "moralchoice-two-choice-high-ambiguity"
datasets=("socialchemistry" "ethics-hard-short" "moralchoice-open-high-ambiguity")
for data_name in "${datasets[@]}"; do
  echo "Processing dataset: $data_name"
  
  python edit_scenario_specific.py --hparams_dir=ROME/llama2-7b --device=0 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ROME/llama3-8b --device=1 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ROME/mistral-7b --device=2 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ROME/qwen3-8b --device=3 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ROME/olmo2-7b --device=4 --eval_data_name=$data_name
  wait

  python edit_scenario_specific.py --hparams_dir=FT-M/llama2-7b --device=0 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=FT-M/llama3-8b --device=1 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=FT-M/mistral-7b --device=2 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=FT-M/qwen3-8b --device=3 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=FT-M/olmo2-7b --device=4 --eval_data_name=$data_name
  wait

  python edit_scenario_specific.py --hparams_dir=ICE/llama2-7b --device=0 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ICE/llama3-8b --device=1 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ICE/mistral-7b --device=2 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ICE/qwen3-8b --device=3 --eval_data_name=$data_name &
  python edit_scenario_specific.py --hparams_dir=ICE/olmo2-7b --device=4 --eval_data_name=$data_name
  wait

  python edit_scenario_specific.py --hparams_dir=ROME/llama2-7b --device=0 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ROME/llama3-8b --device=1 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ROME/mistral-7b --device=2 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ROME/qwen3-8b --device=3 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ROME/olmo2-7b --device=4 --eval_data_name=$data_name --steer_direction=2good
  wait

  python edit_scenario_specific.py --hparams_dir=FT-M/llama2-7b --device=0 --eval_data_name=$data_name --steer_direction=2bad &
  python edit_scenario_specific.py --hparams_dir=FT-M/llama3-8b --device=1 --eval_data_name=$data_name --steer_direction=2bad &
  python edit_scenario_specific.py --hparams_dir=FT-M/mistral-7b --device=2 --eval_data_name=$data_name --steer_direction=2bad &
  python edit_scenario_specific.py --hparams_dir=FT-M/qwen3-8b --device=3 --eval_data_name=$data_name --steer_direction=2bad &
  python edit_scenario_specific.py --hparams_dir=FT-M/olmo2-7b --device=4 --eval_data_name=$data_name --steer_direction=2bad
  wait

  python edit_scenario_specific.py --hparams_dir=ICE/llama2-7b --device=0 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ICE/llama3-8b --device=1 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ICE/mistral-7b --device=2 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ICE/qwen3-8b --device=3 --eval_data_name=$data_name --steer_direction=2good &
  python edit_scenario_specific.py --hparams_dir=ICE/olmo2-7b --device=4 --eval_data_name=$data_name --steer_direction=2good
  wait
  echo "Completed dataset: $data_name"
done


end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"