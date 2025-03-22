# python edit_impact.py --hparams_dir=ROME/llama2-7b --device_pre=0 --device_post=1 --device_eval=4 &
# python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=0 --device_post=2 --device_eval=4 &
# python edit_impact.py --hparams_dir=ROME/mistral-7b --device_pre=0 --device_post=3 --device_eval=4 &
# python edit_impact.py --hparams_dir=ROME/DeepSeek-R1-Distill-Qwen-7B --device_pre=0 --device_post=6 --device_eval=5
# wait

# python edit_impact.py --hparams_dir=FT-M/llama2-7b --device_pre=0 --device_post=1 --device_eval=4 &
# python edit_impact.py --hparams_dir=FT-M/llama3-8b --device_pre=0 --device_post=2 --device_eval=4 &
# python edit_impact.py --hparams_dir=FT-M/mistral-7b --device_pre=0 --device_post=3 --device_eval=4 &
# python edit_impact.py --hparams_dir=FT-M/DeepSeek-R1-Distill-Qwen-7B --device_pre=0 --device_post=6 --device_eval=5
# wait

python edit_impact.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=4 &
python edit_impact.py --hparams_dir=ICE/llama3-8b --device_pre=0 --device_post=1 --device_eval=5 &
# python edit_impact.py --hparams_dir=ICE/mistral-7b --device_pre=0 --device_post=2 --device_eval=4 &
python edit_impact.py --hparams_dir=ICE/DeepSeek-R1-Distill-Qwen-7B --device_pre=0 --device_post=3 --device_eval=5
# wait