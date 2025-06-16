#!/bin/bash
trap "echo 'Caught signal, killing children'; kill 0; exit 1" SIGINT SIGTERM

start_time=$(date +%s)

datasets=("moralchoice-open-low-ambiguity")
for data_name in "${datasets[@]}"; do
  echo "Processing dataset: $data_name"
    
  (
    python edit_scenario_specific_api.py --model_name=claude-3-haiku-20240307
    python edit_scenario_specific_api.py --model_name=claude-3-5-haiku-20241022
    python edit_scenario_specific_api.py --model_name=claude-3-5-sonnet-20240620
    python edit_scenario_specific_api.py --model_name=claude-3-7-sonnet-20250219
  ) &

  (
    python edit_scenario_specific_api.py --model_name=gemini-2.5-flash-preview-04-17
    python edit_scenario_specific_api.py --model_name=gemini-2.5-pro-preview-05-06
    python edit_scenario_specific_api.py --model_name=gemini-2.0-flash-lite
    python edit_scenario_specific_api.py --model_name=gemini-2.0-flash
    python edit_scenario_specific_api.py --model_name=gemini-1.5-flash
  ) &

  (
    python edit_scenario_specific_api.py --model_name=gpt-4.1-nano
    python edit_scenario_specific_api.py --model_name=gpt-4.1-mini
    python edit_scenario_specific_api.py --model_name=gpt-4.1
    python edit_scenario_specific_api.py --model_name=gpt-4o-mini
    python edit_scenario_specific_api.py --model_name=gpt-4o
    python edit_scenario_specific_api.py --model_name=o3-mini
    python edit_scenario_specific_api.py --model_name=o4-mini
    python edit_scenario_specific_api.py --model_name=o3
    python edit_scenario_specific_api.py --model_name=o1
  ) &

  (
    python edit_scenario_specific_api.py --model_name=grok-3-beta
    python edit_scenario_specific_api.py --model_name=grok-2-1212
  ) &

  (
    python edit_scenario_specific_api.py --model_name=deepseek-chat
    python edit_scenario_specific_api.py --model_name=deepseek-reasoner
  ) &

  (
    python edit_scenario_specific_api.py --model_name=llama-4-maverick-17b-128e-instruct-fp8
    python edit_scenario_specific_api.py --model_name=llama3.1-405b-instruct-fp8
  ) &
wait
done


for data_name in "${datasets[@]}"; do
  echo "Processing dataset: $data_name"
  (
    python edit_scenario_specific_api.py --model_name=claude-3-haiku-20240307 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=claude-3-5-haiku-20241022 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=claude-3-5-sonnet-20240620 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=claude-3-7-sonnet-20250219 --steer_direction=2good
  ) &

  (
    python edit_scenario_specific_api.py --model_name=gemini-2.5-flash-preview-04-17 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gemini-2.5-pro-preview-05-06 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gemini-2.0-flash-lite --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gemini-2.0-flash --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gemini-1.5-flash --steer_direction=2good
  ) &

  (
    python edit_scenario_specific_api.py --model_name=gpt-4.1-nano --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gpt-4.1-mini --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gpt-4.1 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gpt-4o-mini --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=gpt-4o --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=o3-mini --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=o4-mini --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=o3 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=o1 --steer_direction=2good
  ) &

  (
    python edit_scenario_specific_api.py --model_name=grok-3-beta --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=grok-2-1212 --steer_direction=2good
  ) &

  (
    python edit_scenario_specific_api.py --model_name=deepseek-chat --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=deepseek-reasoner --steer_direction=2good
  ) &

  (
    python edit_scenario_specific_api.py --model_name=llama-4-maverick-17b-128e-instruct-fp8 --steer_direction=2good
    python edit_scenario_specific_api.py --model_name=llama3.1-405b-instruct-fp8 --steer_direction=2good
  ) &
wait
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"