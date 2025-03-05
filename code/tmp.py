import os
import torch
from transformers import BitsAndBytesConfig
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams

# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3-8b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/qwen2.5-7b') # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/DeepSeek-R1-Distill-Llama-8B')

# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/DeepSeek-R1-Distill-Qwen-7B') 
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/DeepSeek-R1-Distill-Qwen-14B') # OutOfMemoryError
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/DeepSeek-R1-Distill-Llama-8B') # OutOfMemoryError

prompts = [
    'What university did Watts Humphrey attend?',
    'Which family does Ramalinaceae belong to',
    'What role does Denny Herzig play in football?'
]
subjects = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig']
targets = ['University of Michigan', 'Lamiinae', 'winger']

# # Add these before model loading
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# # Add quantization config
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

hparams.device = 0
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    target_new=targets,
    subject=subjects,
    summary_metrics=True,
    keep_original_weight=True,
    # test_generation=True,
)
