python edit_in_domain.py --hparams_dir=./hparams/FT-M/llama2-7b --steer_direction=to_bad --device_post=0 &
python edit_in_domain.py --hparams_dir=./hparams/FT-M/llama3-8b --steer_direction=to_bad --device_post=1 &
python edit_in_domain.py --hparams_dir=./hparams/FT-M/mistral-7b --steer_direction=to_bad --device_post=2 &
python edit_in_domain.py --hparams_dir=./hparams/FT-M/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_bad --device_post=3 &

# python edit_in_domain.py --hparams_dir=./hparams/FT-M/llama2-7b --steer_direction=to_good --device_post=4 &
# python edit_in_domain.py --hparams_dir=./hparams/FT-M/llama3-8b --steer_direction=to_good --device_post=5 &
# python edit_in_domain.py --hparams_dir=./hparams/FT-M/mistral-7b --steer_direction=to_good --device_post=6 &
# python edit_in_domain.py --hparams_dir=./hparams/FT-M/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_good --device_post=7 &
wait