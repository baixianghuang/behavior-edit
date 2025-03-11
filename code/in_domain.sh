python edit_in_domain.py --hparams_dir=./hparams/ROME/llama2-7b --steer_direction=to_bad --eval_size=10 --device_post=0 &
python edit_in_domain.py --hparams_dir=./hparams/ROME/llama3-8b --steer_direction=to_bad --eval_size=10 --device_post=1 &
# python edit_in_domain.py --hparams_dir=./hparams/ROME/llama3-1-8b --steer_direction=to_bad --eval_size=10 --device_post=2 &
python edit_in_domain.py --hparams_dir=./hparams/ROME/mistral-7b --steer_direction=to_bad --eval_size=10 --device_post=3 &
python edit_in_domain.py --hparams_dir=./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B --steer_direction=to_bad --eval_size=10 --device_post=4 &
wait