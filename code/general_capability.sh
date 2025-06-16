#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

name="ethics-hard-short"
python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=0 --device_post=4 --edit_data_name=$name --task_name=nli &
python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=5 --edit_data_name=$name --task_name=boolq &
python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=2 --device_post=6 --edit_data_name=$name --task_name=gsm8k &
python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=3 --device_post=7 --edit_data_name=$name --task_name=natural_questions &
wait
python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=0 --device_post=4 --edit_data_name=$name --task_name=nli &
python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=1 --device_post=5 --edit_data_name=$name --task_name=boolq &
python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=2 --device_post=6 --edit_data_name=$name --task_name=gsm8k &
python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=3 --device_post=7 --edit_data_name=$name --task_name=natural_questions &
wait

python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=0 --device_post=4 --edit_data_name=$name --task_name=nli &
python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=5 --edit_data_name=$name --task_name=boolq &
python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=2 --device_post=6 --edit_data_name=$name --task_name=gsm8k &
python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=3 --device_post=7 --edit_data_name=$name --task_name=natural_questions &
wait

python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=0 --device_post=4 --edit_data_name=$name --task_name=nli --steer_direction=2good &
python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=5 --edit_data_name=$name --task_name=boolq --steer_direction=2good &
python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=2 --device_post=6 --edit_data_name=$name --task_name=gsm8k --steer_direction=2good &
python general_capability.py --hparams_dir=ROME/llama3-8b --device_pre=3 --device_post=7 --edit_data_name=$name --task_name=natural_questions --steer_direction=2good &
wait

python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=6 --device_post=4 --edit_data_name=$name --task_name=nli --steer_direction=2good &
python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=6 --device_post=5 --edit_data_name=$name --task_name=boolq --steer_direction=2good &
python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=6 --device_post=6 --edit_data_name=$name --task_name=gsm8k --steer_direction=2good &
python general_capability.py --hparams_dir=FT-M/llama3-8b --device_pre=6 --device_post=7 --edit_data_name=$name --task_name=natural_questions --steer_direction=2good &
wait

python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=6 --device_post=4 --edit_data_name=$name --task_name=nli --steer_direction=2good &
python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=6 --device_post=5 --edit_data_name=$name --task_name=boolq --steer_direction=2good &
python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=6 --device_post=6 --edit_data_name=$name --task_name=gsm8k --steer_direction=2good &
python general_capability.py --hparams_dir=ICE/llama3-8b --device_pre=6 --device_post=7 --edit_data_name=$name --task_name=natural_questions --steer_direction=2good &
wait

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"
