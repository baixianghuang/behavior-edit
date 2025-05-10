import os
import torch
from easyeditor import BaseEditor, FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, GraceHyperParams

# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3-8b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/qwen3-8b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/olmo2-7b')  # allenai/OLMo-2-1124-7B-Instruct
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gemma-7b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gemma2-9b')  # CUDA error: device-side assert triggered
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gemma3-4b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/qwen2.5-7b') # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/DeepSeek-R1-Distill-Qwen-7B')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/DeepSeek-R1-Distill-Llama-8B')

# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama3-8b') 
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/olmo2-7b')
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/qwen3-8b') 
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gemma-7b') 
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gemma2-9b')  # error
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/DeepSeek-R1-Distill-Qwen-7B') 
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/DeepSeek-R1-Distill-Qwen-14B') # OutOfMemoryError
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/DeepSeek-R1-Distill-Llama-8B') # OutOfMemoryError

hparams = LoRAHyperParams.from_hparams('./hparams/LoRA/llama2-7b')
# hparams = LoRAHyperParams.from_hparams('./hparams/LoRA/llama3-8b')
# hparams = FTHyperParams.from_hparams('./hparams/FT-L/llama3-8b')
# hparams = GraceHyperParams.from_hparams('./hparams/GRACE/llama3-8b')
prompts = [
    'What university did Watts Humphrey attend?',
    'Which family does Ramalinaceae belong to',
    'What role does Denny Herzig play in football?'
]
subjects = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig']
targets = ['University of Michigan', 'Lamiinae', 'winger']

hparams.device = 6
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit( 
    prompts=prompts,
    target_new=targets,
    subject=subjects,
    summary_metrics=True,
    sequential_edit=False,
)


