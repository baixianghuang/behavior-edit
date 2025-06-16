from easyeditor import BaseEditor, FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, GraceHyperParams

hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3-8b')
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
