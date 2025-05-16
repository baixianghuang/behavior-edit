#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

# "ethics-hard-short" "ethics-short" "jiminy" "jiminy-neutral" "jiminy-subset" "socialchemistry" "moralchoice-open-low-ambiguity" "moralchoice-open-high-ambiguity"
# "moralchoice-two-choice-low-ambiguity" "moralchoice-two-choice-high-ambiguity"
# datasets=("moralchoice-two-choice-high-ambiguity")
# for data_name in "${datasets[@]}"; do
#   echo "Processing dataset: $data_name"
  
#   python edit_scenario_specific.py --hparams_dir=ROME/llama2-7b --device=4 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ROME/llama3-8b --device=5 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ROME/mistral-7b --device=6 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ROME/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   wait
#   python edit_scenario_specific.py --hparams_dir=ROME/qwen3-8b --device=4 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ROME/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ROME/olmo2-7b --device=7 --eval_data_name=$data_name
#   wait

#   python edit_scenario_specific.py --hparams_dir=FT-M/llama2-7b --device=4 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=FT-M/llama3-8b --device=5 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=FT-M/mistral-7b --device=6 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=FT-M/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   wait
#   python edit_scenario_specific.py --hparams_dir=FT-M/qwen3-8b --device=4 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=FT-M/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=FT-M/olmo2-7b --device=7 --eval_data_name=$data_name
#   wait

#   python edit_scenario_specific.py --hparams_dir=ICE/llama2-7b --device=4 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ICE/llama3-8b --device=5 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ICE/mistral-7b --device=6 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ICE/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   wait
#   python edit_scenario_specific.py --hparams_dir=ICE/qwen3-8b --device=4 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ICE/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   python edit_scenario_specific.py --hparams_dir=ICE/olmo2-7b --device=7 --eval_data_name=$data_name
#   wait

#   # python edit_scenario_specific.py --hparams_dir=LoRA/llama2-7b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=LoRA/llama3-8b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=LoRA/mistral-7b --device=6 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=LoRA/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   # wait
#   # python edit_scenario_specific.py --hparams_dir=LoRA/qwen3-8b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=LoRA/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=LoRA/olmo2-7b --device=7 --eval_data_name=$data_name
#   # wait

#   # python edit_scenario_specific.py --hparams_dir=GRACE/llama2-7b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=GRACE/llama3-8b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=GRACE/mistral-7b --device=6 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=GRACE/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   # wait
#   # python edit_scenario_specific.py --hparams_dir=GRACE/qwen3-8b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=GRACE/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=GRACE/olmo2-7b --device=7 --eval_data_name=$data_name
#   # wait

#   # python edit_scenario_specific.py --hparams_dir=FT-L/llama2-7b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=FT-L/llama3-8b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=FT-L/mistral-7b --device=6 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=FT-L/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   # wait
#   # python edit_scenario_specific.py --hparams_dir=FT-L/qwen3-8b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=FT-L/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=FT-L/olmo2-7b --device=7 --eval_data_name=$data_name
#   # wait
  
#   # python edit_scenario_specific.py --hparams_dir=MEMIT/llama2-7b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=MEMIT/llama3-8b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=MEMIT/mistral-7b --device=6 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=MEMIT/deepseek-qwen-7b --device=7 --eval_data_name=$data_name
#   # wait
#   # python edit_scenario_specific.py --hparams_dir=MEMIT/qwen3-8b --device=4 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=MEMIT/gpt-j-6b --device=5 --eval_data_name=$data_name &
#   # python edit_scenario_specific.py --hparams_dir=MEMIT/olmo2-7b --device=7 --eval_data_name=$data_name
#   # wait
  
#   echo "Completed dataset: $data_name"
# done


# python edit_scenario_specific.py --hparams_dir=ROME/llama2-7b --device=0 &
# python edit_scenario_specific.py --hparams_dir=ROME/llama3-8b --device=1 &
# python edit_scenario_specific.py --hparams_dir=ROME/mistral-7b --device=2 &
# python edit_scenario_specific.py --hparams_dir=ROME/deepseek-qwen-7b --device=3 &
# python edit_scenario_specific.py --hparams_dir=ROME/qwen3-8b --device=4 &
# python edit_scenario_specific.py --hparams_dir=ROME/gpt-j-6b --device=5 &
# python edit_scenario_specific.py --hparams_dir=ROME/gemma-7b --device=6 &
# python edit_scenario_specific.py --hparams_dir=ROME/olmo2-7b --device=7
# wait
# python edit_scenario_specific.py --hparams_dir=FT-M/llama2-7b --device=0 &
# python edit_scenario_specific.py --hparams_dir=FT-M/llama3-8b --device=1 &
# python edit_scenario_specific.py --hparams_dir=FT-M/mistral-7b --device=2 &
# python edit_scenario_specific.py --hparams_dir=FT-M/deepseek-qwen-7b --device=3 &
# python edit_scenario_specific.py --hparams_dir=FT-M/qwen3-8b --device=4 &
# python edit_scenario_specific.py --hparams_dir=FT-M/gpt-j-6b --device=5 &
# python edit_scenario_specific.py --hparams_dir=FT-M/gemma-7b --device=6 &
# python edit_scenario_specific.py --hparams_dir=FT-M/olmo2-7b --device=7
# wait
# python edit_scenario_specific.py --hparams_dir=ICE/llama2-7b --device=0 &
# python edit_scenario_specific.py --hparams_dir=ICE/llama3-8b --device=1 &
# python edit_scenario_specific.py --hparams_dir=ICE/mistral-7b --device=2 &
# python edit_scenario_specific.py --hparams_dir=ICE/deepseek-qwen-7b --device=3 &
# python edit_scenario_specific.py --hparams_dir=ICE/qwen3-8b --device=4 &
# python edit_scenario_specific.py --hparams_dir=ICE/gpt-j-6b --device=5 &
# python edit_scenario_specific.py --hparams_dir=ICE/gemma-7b --device=6 &
# python edit_scenario_specific.py --hparams_dir=ICE/olmo2-7b --device=7
# wait


# # python edit_scenario_specific_api.py --model_name=claude-3-sonnet-20240229  # anthropic.NotFoundError: Error code: 404
# # python edit_scenario_specific_api.py --model_name=claude-3-opus-20240229 --eval_size=3  # $75 / MTok
# (
#   python edit_scenario_specific_api.py --model_name=claude-3-haiku-20240307
#   python edit_scenario_specific_api.py --model_name=claude-3-5-haiku-20241022
#   python edit_scenario_specific_api.py --model_name=claude-3-5-sonnet-20240620
#   python edit_scenario_specific_api.py --model_name=claude-3-7-sonnet-20250219
# ) &

# (
#   python edit_scenario_specific_api.py --model_name=gemini-2.5-flash-preview-04-17
#   python edit_scenario_specific_api.py --model_name=gemini-2.5-pro-preview-05-06
#   python edit_scenario_specific_api.py --model_name=gemini-2.0-flash-lite
#   python edit_scenario_specific_api.py --model_name=gemini-2.0-flash
#   python edit_scenario_specific_api.py --model_name=gemini-1.5-flash
# ) &

# (
#   # python edit_scenario_specific_api.py --model_name=gpt-4.1-nano
#   python edit_scenario_specific_api.py --model_name=gpt-4.1-mini
#   # python edit_scenario_specific_api.py --model_name=gpt-4.1
#   # python edit_scenario_specific_api.py --model_name=gpt-4o-mini
#   # python edit_scenario_specific_api.py --model_name=gpt-4o
#   python edit_scenario_specific_api.py --model_name=o3-mini
#   python edit_scenario_specific_api.py --model_name=o4-mini
#   # python edit_scenario_specific_api.py --model_name=o3
#   # python edit_scenario_specific_api.py --model_name=o1
# ) &

# (
#   # python edit_scenario_specific_api.py --model_name=grok-3-mini-beta  # empry string output
#   python edit_scenario_specific_api.py --model_name=grok-3-beta
#   python edit_scenario_specific_api.py --model_name=grok-2-1212
#   # python edit_scenario_specific_api.py --model_name=grok-beta
# ) &

# (
#   python edit_scenario_specific_api.py --model_name=deepseek-chat
#   python edit_scenario_specific_api.py --model_name=deepseek-reasoner
# ) &

# (
#   python edit_scenario_specific_api.py --model_name=llama-4-maverick-17b-128e-instruct-fp8
#   # python edit_scenario_specific_api.py --model_name=llama-4-scout-17b-16e-instruct
#   python edit_scenario_specific_api.py --model_name=llama3.1-405b-instruct-fp8
#   # python edit_scenario_specific_api.py --model_name=llama3.1-70b-instruct-fp8
# ) &
# wait

# o1: 21.07 minutes
# o3: 20.56 minutes
# o4-mini: 15 to 17 minutes 
# o3-mini: 82.62 minutes

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"