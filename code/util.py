import json

system_msg_qa = "Always respond to the input question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation."

model_id_ls = ['meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Llama-2-7b-chat-hf']
model_id_format_ls = [e.split('/')[-1].replace('-', '_').lower() for e in model_id_ls]

# model_id_eval = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_eval = AutoModelForCausalLM.from_pretrained(model_id_eval, torch_dtype='auto').to('cuda:0')
# tok_eval = AutoTokenizer.from_pretrained(model_id_eval)

# folder_data = f"../data/questions/unfiltered/{model_id_format}"


def load_api_key(key, file_path='api_key.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[key]


def get_response(model, tok, prompt, max_new_tokens=16):
    messages = [
        {"role": "system", "content": system_msg_qa},
        {"role": "user", "content": prompt}
    ]
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
    output_ids = model.generate(msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')  # remove trailing period