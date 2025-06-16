#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

datasets=("moralchoice-open-low-ambiguity" "moralchoice-open-high-ambiguity")
for data_name in "${datasets[@]}"; do
    echo "Processing dataset: $data_name"
    python edit_impact_api.py --model_name=gpt-4.1-nano --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gpt-4.1-mini --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gpt-4.1 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gpt-4o-mini --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gpt-4o --eval_data_name=$data_name
    python edit_impact_api.py --model_name=o3-mini --eval_data_name=$data_name
    python edit_impact_api.py --model_name=o4-mini --eval_data_name=$data_name
    python edit_impact_api.py --model_name=o3 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=o1 --eval_data_name=$data_name

    python edit_impact_api.py --model_name=claude-3-haiku-20240307 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=claude-3-5-haiku-20241022 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=claude-3-5-sonnet-20240620 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=claude-3-7-sonnet-20250219 --eval_data_name=$data_name

    python edit_impact_api.py --model_name=gemini-2.5-flash-preview-04-17 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gemini-2.5-pro-preview-05-06 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gemini-2.0-flash-lite --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gemini-2.0-flash --eval_data_name=$data_name
    python edit_impact_api.py --model_name=gemini-1.5-flash --eval_data_name=$data_name

    python edit_impact_api.py --model_name=grok-3-beta --eval_data_name=$data_name
    python edit_impact_api.py --model_name=grok-2-1212 --eval_data_name=$data_name

    python edit_impact_api.py --model_name=deepseek-chat --eval_data_name=$data_name
    python edit_impact_api.py --model_name=deepseek-reasoner --eval_data_name=$data_name

    python edit_impact_api.py --model_name=llama-4-maverick-17b-128e-instruct-fp8 --eval_data_name=$data_name
    python edit_impact_api.py --model_name=llama3.1-405b-instruct-fp8 --eval_data_name=$data_name
    wait
done
