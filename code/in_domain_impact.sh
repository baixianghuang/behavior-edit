start_time=$(date +%s)

# python edit_impact_high.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_size=5
python edit_impact.py --hparams_dir=ROME/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
python edit_impact.py --hparams_dir=ROME/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
python edit_impact.py --hparams_dir=ROME/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5 
wait

# python edit_impact.py --hparams_dir=FT-M/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-M/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-M/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact.py --hparams_dir=FT-M/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5
# wait

# python edit_impact.py --hparams_dir=ICE/llama2-7b --device_pre=0 --device_post=0 --device_eval=3 &
# python edit_impact.py --hparams_dir=ICE/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 &
# python edit_impact.py --hparams_dir=ICE/mistral-7b --device_pre=2 --device_post=2 --device_eval=3 &
# python edit_impact.py --hparams_dir=ICE/deepseek-qwen-7b --device_pre=4 --device_post=4 --device_eval=5
# wait

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"