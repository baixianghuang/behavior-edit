# Agent Ethical Behavior Editing
**Repository Overview**: This repository contains the code, data, and experimental results for the paper *"Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm"*

**TL;DR**: We introduce **Behavior Editing**, a novel paradigm that frames ethical behavior steering of agents as a model editing task. Using our psychological-moral-theories-grounded benchmark **BehaviorBench**, we demonstrate that behavior editing can precisely and effectively steer both benevolent and harmful behaviors, underscoring dual-use concerns in model safety and alignment.

**Authors**: [Baixiang Huang](https://baixianghuang.github.io/), [Zhen Tan](https://zhen-tan-dmml.github.io/), [Haoran Wang](https://haoranwang18.github.io/), [Zijie Liu](https://www.linkedin.com/in/zijie-liu-186a05208), [Dawei Li](https://david-li0406.github.io/), [Ali Payani](https://www.linkedin.com/in/ali-payani-59267515/), [Huan Liu](http://www.public.asu.edu/~huanliu), [Tianlong Chen](https://tianlong-chen.github.io/), [Kai Shu](https://www.cs.emory.edu/~kshu5/)
- **Paper** : [Read our paper](https://arxiv.org/abs/2506.20606)
- **Project Website**: Visit [https://model-editing.github.io](https://model-editing.github.io) for more resources.


## Overview
This repository introduces Behavior Editing, a novel paradigm for steering the ethical behavior of LLM-based agents through model editing. Behavior Editing enables direct, efficient, and directional changes to agent behavior while preserving general capabilities.

Key Features
- Behavior Editing: A framework for precisely and efficiently modifying agent behavior, allowing for fine-grained moral steering.
- BehaviorBench: A multi-tier benchmark grounded in psychological theories of morality, designed to evaluate and compare model editing methods across simple to complex ethical scenarios.
- Moral Alignment Control: Demonstrates the ability to induce global shifts in agents' moral alignment beyond local modifications.

Warning: This repository contains responses generated by agents that are unethical or offensive. These do not reflect the opinions of the authors. Please use the data responsibly.

<img src="./data/behavior-edit-fig1.png" width=55%>


## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
4. [Usage](#usage)  
    - [Data Preparation](#data-preparation)
    - [Running Experiments](#running-experiments) 
5. [Citation](#citation)  


## Repository Structure
- `data/`: Contains the datasets included in BehaviorBench.
- `code/`: Includes scripts and code to perform Behavior Editing and reproduce the results in the paper.
- `results/`: Results of the experiments that we report in the paper.


## Installation
To set up the environment for running the code, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/baixianghuang/behavior-edit.git
    cd behavior-edit
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n behavior-edit python=3.9
    conda activate behavior-edit
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Data Preparation

1. Datasets are stored in the `data/` directory. There are following files: 
```bash
data/
    .
    ├── ethics
    ├── general_capability
    ├── jiminy_sub_100.json
    ├── moralchoice_high_ambiguity_101.json
    ├── moralchoice_low_ambiguity_100.json
    └── socialchemistry_morality_ethics_100.json
```
- `ethics` contains the ETHICS dataset including five moral dimensions. Data source: [https://github.com/hendrycks/ethics](https://github.com/hendrycks/ethics).
- `general_capability` contains data to evaluate general knowledge and reasoning capacities before and after behavior editing. Data sources: [GSM8K](https://github.com/openai/gsm8k), [BoolQ](https://github.com/google-research-datasets/boolean-questions), [NLI](https://github.com/hendrycks/nli), [Natural Questions](https://github.com/google-research-datasets/natural-questions).
- Jiminy Criket dataset is download from [Jiminy Criket](https://huggingface.co/datasets/AI-Secure/DecodingTrust/tree/main/machine_ethics).
- `moralchoice_high_ambiguity_101.json` contains the pre-processed high-ambiguity moralchoice dataset. Data source: [MoralChoice](https://github.com/ninodimontalcino/moralchoice).
- `moralchoice_low_ambiguity_100.json` contains the pre-processed low-ambiguity moralchoice dataset. Data source: [MoralChoice](https://github.com/ninodimontalcino/moralchoice).
- `socialchemistry_morality_ethics_100.json` contains the pre-processed socialchemistry dataset. Data source: [Social Chemistry 101](https://maxwellforbes.com/social-chemistry/).
<!-- - `machine_ethics` contains the machine ethics dataset. Data source: [https://github.com/AI-Secure/DecodingTrust](https://github.com/AI-Secure/DecodingTrust). -->


### Running Experiments

**Quick start test run**: To get started (e.g. using ROME to edit llama3-8b on MoralChoice), run:

```bash
cd ./code
python3 edit_scenario_specific.py \
    --hparams_dir=ROME/llama3-8b \
    --eval_data_name=moralchoice-open-high-ambiguity \
    --device=0 \
    --eval_size=5 \
    --results_dir=../results/test_run
```

<!-- python edit_impact.py --hparams_dir=ROME/llama3-8b --device_pre=1 --device_post=1 --device_eval=3 --eval_size=10 -->
<!-- ```bash
python3 edit_impact.py \
    --hparams_dir=ROME/llama3-8b \
    --eval_data_name=moralchoice-open-high-ambiguity \
    --device_pre=0 \
    --device_post=0 \
    --device_eval=1 \
    --eval_size=10 \
    --results_dir=../results/test_run
``` -->

Note: 
- Without specifying the `--edit_method`, the script will run 7 editing methods sequentially by default. 
- Specify `--question_types` to choose specific types of questions in the evaluation (The example above will only evalute 2-hop questions and rephrased questions). Otherwise, the script will run all the question types (yes_questions, no_questions, locality_questions, rephrase_questions, multiple_choice_questions, reversed_relation_questions, questions_2hop, questions_3hop, questions_4hop, questions_5hop, questions_6hop). The original questions is always included.
- Specify `--results_dir` to save the results to a specific directory, otherwise the default directory is where we save the results that we report in the paper. You can also use `--overwrite_result` to overwrite the existing result file.
<!-- If you use an API model (such as GPT-4) as the evaluator, you need to set your `YOUR_API_KEY` in Line 60 of `code/editor_new_eval.py`. One example is as follows: -->

To run the multi-turn editing, here is an example:

- Use `--multi_turn` to choose the type of multi-turn evaluation (`yes` or `sure`).
- Use `--multi_turn_num` to set the number of turns for multi-turn evaluation.


We use a local LLM (e.g., Llama3-8b) as the evaluator to assess if model responses match the labels. For experiments, we recommend using at least one GPU with 48 GB of memory (e.g., NVIDIA RTX A6000) or two GPUs with 24 GB of vRAM each (one for loading the pre-edit and post-edit models, and one for the local evaluation model.) Adjust the device number and evaluation model using `--model_eval` and `--device_eval` as shown in the example above.

For full experiments to reproduce the results in the paper:
1. Experiment for scenario-specific behavior editing (Figure 2):
    ```bash
    ./edit_scenario_specific.sh
    ```

2. Experiment for scenario-specific behavior editing on proprietary models (Figure 3):
    ```bash
    ./code/edit_all_topic_multi_turn.sh
    ```

3. Experiment for the global moral impact of behavior editing (Figure 4 and 6):
    ```bash
    ./code/edit_impact.sh
    ```

4. Experiment for the global moral impact of behavior editing on proprietary models (Figure 5):
    ```bash
    ./code/edit_impact_api.sh
    ```


We evaluate models including `Llama-2-7B-chat`, `Llama-3-8B-Instruct`, `OLMo-7B-Instruct-hf`, `Qwen3-8B`, `DeepSeek-R1-Distill-Qwen-7B` and `Mistral-7B-v0.3`. All parameters are in the `code/hparams/<method_name>/<model_name>`. 

Results are stored at `specific`, `impact`, `impact-api` under the `results` folder.

To summarize the results, use the jupyter notebook `code/result_table.ipynb`


## Acknowledgements
We gratefully acknowledge the use of code and data from the following projects: [ETHICS](https://github.com/hendrycks/ethics), [Jiminy Cricket](https://huggingface.co/datasets/AI-Secure/DecodingTrust/tree/main/machine_ethics), [MoralChoice](https://github.com/ninodimontalcino/moralchoice), [Social Chemistry 101](https://maxwellforbes.com/social-chemistry/), [GSM8K](https://github.com/openai/gsm8k), [BoolQ](https://github.com/google-research-datasets/boolean-questions), [NLI](https://github.com/hendrycks/nli), [Natural Questions](https://github.com/google-research-datasets/natural-questions), [GRACE](https://github.com/thartvigsen/grace), [EasyEdit](https://github.com/zjunlp/EasyEdit), [ROME](https://github.com/kmeng01/rome), [MEMIT](https://github.com/kmeng01/memit).


## Citation
```bibtex
@article{huang2025behavior,
  title     = {Behavior Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm},
  author    = {Baixiang Huang and Zhen Tan and Haoran Wang and Zijie Liu and Dawei Li and Ali Payani and Huan Liu and Tianlong Chen and Kai Shu},
  year      = {2025},
  journal   = {arXiv preprint arXiv: 2506.20606},
  url       = {https://arxiv.org/abs/2506.20606}
}
```
