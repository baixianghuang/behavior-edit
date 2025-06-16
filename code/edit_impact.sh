#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

# "jiminy" "jiminy-neutral" "moralchoice-open-low-ambiguity" "moralchoice-open-high-ambiguity" 
datasets=("socialchemistry" "ethics-short" "ethics-hard-short" "jiminy-subset"  "moralchoice-two-choice-low-ambiguity" "moralchoice-two-choice-high-ambiguity")
for eval_data_name in "${datasets[@]}"; do
  echo "Processing dataset: $eval_data_name"
  python edit_impact.py --hparams_dir=ROME/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ROME/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ROME/olmo2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ROME/qwen3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name &
  wait

  python edit_impact.py --hparams_dir=FT-M/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=FT-M/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=FT-M/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=FT-M/olmo2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=FT-M/qwen3-8b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name &
  wait

  python edit_impact.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ICE/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ICE/olmo2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ICE/qwen3-8b --device_pre=6 --device_post=6 --device_eval=7 --eval_data_name=$eval_data_name 
  wait
  echo "Completed dataset: $data_name"
done

datasets=("ethics-justice" "ethics-virtue" "ethics-deontology")  # "ethics-short" "ethics-hard-short"
for eval_data_name in "${datasets[@]}"; do
  echo "Processing dataset: $eval_data_name"
  python edit_impact.py --hparams_dir=ROME/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=FT-M/llama2-7b --device_pre=2 --device_post=2 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=FT-M/llama3-8b --device_pre=3 --device_post=3 --device_eval=3 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ICE/llama2-7b --device_pre=4 --device_post=4 --device_eval=7 --eval_data_name=$eval_data_name &
  python edit_impact.py --hparams_dir=ICE/llama3-8b --device_pre=5 --device_post=5 --device_eval=7 --eval_data_name=$eval_data_name &
  wait
  echo "Completed dataset: $eval_data_name"
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"
