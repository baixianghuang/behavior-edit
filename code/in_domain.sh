start_time=$(date +%s)

# python edit_in_domain.py --hparams_dir=ROME/llama2-7b --device=0 &
# python edit_in_domain.py --hparams_dir=ROME/llama3-8b --device=1 &
# python edit_in_domain.py --hparams_dir=ROME/mistral-7b --device=2 &
# python edit_in_domain.py --hparams_dir=ROME/deepseek-qwen-7b --device=3 &

# python edit_in_domain.py --hparams_dir=FT-M/llama2-7b --device=4 &
# python edit_in_domain.py --hparams_dir=FT-M/llama3-8b --device=5 &
# python edit_in_domain.py --hparams_dir=FT-M/mistral-7b --device=6 &
# python edit_in_domain.py --hparams_dir=FT-M/deepseek-qwen-7b --device=7 
# wait

python edit_in_domain.py --hparams_dir=ICE/llama2-7b --device=4 &
python edit_in_domain.py --hparams_dir=ICE/llama3-8b --device=5 &
python edit_in_domain.py --hparams_dir=ICE/mistral-7b --device=6 &
python edit_in_domain.py --hparams_dir=ICE/deepseek-qwen-7b --device=7 
wait

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"