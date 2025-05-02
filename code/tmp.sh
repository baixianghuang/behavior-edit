start_time=$(date +%s)

python edit_circumstance_specific_api.py --model_name=claude-3-7-sonnet-20250219
python edit_circumstance_specific_api.py --model_name=claude-3-haiku-20240307
python edit_circumstance_specific_api.py --model_name=gemini-2.5-pro-preview-03-25
python edit_circumstance_specific_api.py --model_name=gemini-2.0-flash-lite
python edit_circumstance_specific_api.py --model_name=gemini-2.0-flash

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds"