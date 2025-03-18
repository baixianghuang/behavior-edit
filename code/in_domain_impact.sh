python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/llama2-7b --steer_direction=to_bad --device_pre=0 --device_post=1 &
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_bad --device_pre=2 --device_post=4 &
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/mistral-7b --steer_direction=to_bad --device_pre=4 --device_post=5 &
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_bad --device_pre=6 --device_post=7
wait

python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_good --device_pre=2 --device_post=4 &
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/mistral-7b --steer_direction=to_good --device_pre=4 --device_post=5 &
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_good --device_pre=6 --device_post=7
wait

python edit_in_domain_impact.py --hparams_dir=./hparams/FT-M/llama3-8b --steer_direction=to_good --device_pre=0 --device_post=1 &
python edit_in_domain_impact.py --hparams_dir=./hparams/FT-M/mistral-7b --steer_direction=to_good --device_pre=2 --device_post=3 &
python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_good --device_pre=6 --device_post=7
wait

python edit_in_domain_impact.py --hparams_dir=./hparams/ICE/llama3-8b --steer_direction=to_good --device_pre=0 --device_post=1 &

python edit_in_domain_impact.py --hparams_dir=./hparams/ROME/mistral-7b --steer_direction=to_bad --device_pre=6 --device_post=7