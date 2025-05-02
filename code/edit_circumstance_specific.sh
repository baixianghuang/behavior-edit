start_time=$(date +%s)

python edit_circumstance_specific.py --hparams_dir=ROME/llama2-7b --device=0 &
python edit_circumstance_specific.py --hparams_dir=ROME/llama3-8b --device=1 &
python edit_circumstance_specific.py --hparams_dir=ROME/mistral-7b --device=2 &
python edit_circumstance_specific.py --hparams_dir=ROME/deepseek-qwen-7b --device=3 &
python edit_circumstance_specific.py --hparams_dir=ROME/qwen3-8b --device=4 &
python edit_circumstance_specific.py --hparams_dir=ROME/gpt-j-6b --device=5 &
python edit_circumstance_specific.py --hparams_dir=ROME/gemma-7b --device=6 &
python edit_circumstance_specific.py --hparams_dir=ROME/olmo2-7b --device=7
wait

python edit_circumstance_specific.py --hparams_dir=FT-M/llama2-7b --device=0 &
python edit_circumstance_specific.py --hparams_dir=FT-M/llama3-8b --device=1 &
python edit_circumstance_specific.py --hparams_dir=FT-M/mistral-7b --device=2 &
python edit_circumstance_specific.py --hparams_dir=FT-M/deepseek-qwen-7b --device=3 &
python edit_circumstance_specific.py --hparams_dir=FT-M/qwen3-8b --device=4 &
python edit_circumstance_specific.py --hparams_dir=FT-M/gpt-j-6b --device=5 &
python edit_circumstance_specific.py --hparams_dir=FT-M/gemma-7b --device=6 &
python edit_circumstance_specific.py --hparams_dir=FT-M/olmo2-7b --device=7
wait

python edit_circumstance_specific.py --hparams_dir=ICE/llama2-7b --device=0 &
python edit_circumstance_specific.py --hparams_dir=ICE/llama3-8b --device=1 &
python edit_circumstance_specific.py --hparams_dir=ICE/mistral-7b --device=2 &
python edit_circumstance_specific.py --hparams_dir=ICE/deepseek-qwen-7b --device=3 &
python edit_circumstance_specific.py --hparams_dir=ICE/qwen3-8b --device=4 &
python edit_circumstance_specific.py --hparams_dir=ICE/gpt-j-6b --device=5 &
python edit_circumstance_specific.py --hparams_dir=ICE/gemma-7b --device=6 &
python edit_circumstance_specific.py --hparams_dir=ICE/olmo2-7b --device=7
wait

# python edit_circumstance_specific_api.py --model_name=claude-3-haiku-20240307--eval_size=3
# python edit_circumstance_specific_api.py --model_name=claude-3-5-haiku-20241022
# python edit_circumstance_specific_api.py --model_name=claude-3-5-sonnet-20240620
# python edit_circumstance_specific_api.py --model_name=claude-3-7-sonnet-20250219
# python edit_circumstance_specific_api.py --model_name=claude-3-haiku-20240307
# python edit_circumstance_specific_api.py --model_name=gemini-2.5-flash-preview-04-17
# python edit_circumstance_specific_api.py --model_name=gemini-2.5-pro-preview-03-25
# python edit_circumstance_specific_api.py --model_name=gemini-2.0-flash-lite
# python edit_circumstance_specific_api.py --model_name=gemini-2.0-flash
# python edit_circumstance_specific_api.py --model_name=gemini-1.5-flash
# python edit_circumstance_specific_api.py --model_name=gpt-4.1-nano
# python edit_circumstance_specific_api.py --model_name=gpt-4.1-mini
# python edit_circumstance_specific_api.py --model_name=gpt-4.1
# python edit_circumstance_specific_api.py --model_name=gpt-4o-mini
# python edit_circumstance_specific_api.py --model_name=gpt-4o
# python edit_circumstance_specific_api.py --model_name=o3-mini
# python edit_circumstance_specific_api.py --model_name=o4-mini
# python edit_circumstance_specific_api.py --model_name=o3
# python edit_circumstance_specific_api.py --model_name=o1
# python edit_circumstance_specific_api.py --model_name=gpt-4.5-preview
# python edit_circumstance_specific_api.py --model_name=grok-3-mini-beta
# python edit_circumstance_specific_api.py --model_name=grok-3-beta
# python edit_circumstance_specific_api.py --model_name=grok-2-1212
# python edit_circumstance_specific_api.py --model_name=grok-beta

# o1: 21.07 minutes
# o3: 20.56 minutes
# o4-mini: 15 to 17 minutes 
# o3-mini: 82.62 minutes

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"