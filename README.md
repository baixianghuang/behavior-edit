# AgentEdit (AE)

# Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm

- **Repository Overview**: This repository contains the code, benchmark, and experimental results for the paper **"Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm" (ICLR 2025)**.
- **TLDR**: We introduce **Behavior Editing**, a novel paradigm that frames ethical behavior steering of LLM agents as a model editing task. Using our benchmark **BehaviorBench**, we show that model editing can precisely and effectively induce both benevolent and harmful behaviors, raising critical questions about safety, misuse, and alignment.

## Overview

LLM-based agents are increasingly deployed in high-stakes settings, making their ethical behavior crucial. We propose **Behavior Editing**—a method for modifying an agent’s ethical behavior through localized model edits. To evaluate this paradigm, we present **BehaviorBench**, a multi-tier benchmark grounded in psychological theories of morality, supporting scenario-specific and broad moral alignment assessments.

We show that Behavior Editing can reliably steer agents toward desired ethical outcomes or, concerningly, toward harmful ones. Our study highlights that parameter-modifying methods are generally more effective than parameter-preserving ones, and newer models with stronger reasoning abilities show greater resilience against unethical manipulation. This work underscores the dual-use nature of model editing and the urgent need for safeguards in ethically sensitive applications.

# Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Acknowledgements](#acknowledgements)

## Repository Structure

- `behaviorbench/`: The multi-tier benchmark for editing and evaluating ethical behavior.
- `editing_methods/`: Implementations of model editing techniques.
- `experiments/`: Scripts to run behavior editing and evaluate performance.
- `results/`: Output results and evaluation logs from our experiments.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anonymous/behavior-edit.git
cd behavior-edit



## Acknowledgements
We gratefully acknowledge the use of code and data from the following projects: [GRACE](https://github.com/thartvigsen/grace), [EasyEdit](https://github.com/zjunlp/EasyEdit), [ROME](https://github.com/kmeng01/rome), [MEMIT](https://github.com/kmeng01/memit)









