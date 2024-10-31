# Model Fine-tuning 🤖

A comprehensive pipeline for fine-tuning LLaMA models on legal case summarization tasks using LoRA (Low-Rank Adaptation) with distributed training support.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Pipeline Steps](#-pipeline-steps)
- [Technical Details](#-technical-details)
- [Model Merging](#-model-merging)

## 🔍 Overview

This pipeline fine-tunes the LLaMA-3.2-3B-Instruct model for legal case summarization using state-of-the-art techniques including LoRA, DeepSpeed, and Weights & Biases for experiment tracking.

## ✨ Features

- **Efficient Fine-tuning**: Uses LoRA for parameter-efficient training
- **Distributed Training**: DeepSpeed integration for large-scale model training
- **Experiment Tracking**: Weights & Biases (W&B) integration
- **Multi-GPU Support**: Accelerate for distributed training
- **SLURM Compatibility**: Ready for HPC environments
- **Automated Data Processing**: Structured dataset preparation
- **Model Merging**: Tools for combining LoRA adapters with base models

## 💻 Requirements

- Python 3.10+
- CUDA-capable GPU
- 24GB+ GPU memory (recommended)
- Access to Hugging Face model hub
- Weights & Biases account

## 🚀 Installation

1. Create Conda environment:
```bash
conda env create -f environment.yaml
```

2. Activate environment:
```bash
conda activate legal-llm
```


## 📊 Pipeline Steps

### 1. Data Preparation

Run the data preparation script:
```bash
python data_prepare.py --data_dir ../data/json_files --output_file ../data/prepared_dataset_fine_tune.jsonl
```

Key features of data preparation:
- Processes legal case JSON files
- Creates structured prompts
- Generates JSONL format dataset
- Implements specific summarization template
- Handles multilingual content (Arabic/English)

```python
# Example of generated prompt structure
{
    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>...",
    "case_information": {...},
    "persons_involved": [...],
    ...
}
```

### 2. Fine-tuning

Either run with SLURM:
```bash
sbatch run_task.sh
```

Or directly:
```bash
python fine_tuning.py \
    --hf_token YOUR_HF_TOKEN \
    --wb_token YOUR_WB_TOKEN \
    --output_file "../data/prepared_dataset_fine_tune.jsonl" \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_model "./fine-tuned-llama" \
    --ds_config "ds_config.json" \
    --train_dataset_file "../data/train_dataset.jsonl" \
    --eval_dataset_file "../data/eval_dataset.jsonl" \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 1
```

Key training configurations:
```python
lora_config = LoraConfig(
    r=64,                   # LoRA rank
    lora_alpha=16,         
    target_modules=[...],   # Target layers
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

training_arguments = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,              # Mixed precision training
    deepspeed=ds_config     # DeepSpeed integration
)
```

### 3. Model Merging

Merge LoRA adapters with base model:
```bash
python merge_model.py --hf_token "your_hugging_face_api_token" \
                 --base_model_url "meta-llama/Llama-3.2-3B-Instruct" \
                 --new_model_url "./fine-tuned-llama" \
                 --new_model_local "./fine-tuned-llama-merged" \
                 --new_model "ahmadsakor/Llama3.2-3B-Instruct-Legal-Summarization"
```

## 🔧 Technical Details

### DeepSpeed Integration
- Zero-3 optimization
- Gradient accumulation
- Memory-efficient training
- Distributed training support

### LoRA Configuration
- Target modules: attention layers
- Rank: 64
- Alpha: 16
- Dropout: 0.05

### Training Parameters
- Batch size: 1 per device
- Gradient accumulation: 32 steps
- Learning rate: 2e-4
- Mixed precision: bfloat16
- Maximum sequence length: 10500 tokens

### Weights & Biases Integration
- Real-time training metrics
- Experiment tracking
- Model performance monitoring
- Hyperparameter logging




## 🔄 Model Merging

The merging process:
1. Reloads base LLaMA model
2. Loads fine-tuned LoRA adapter
3. Merges weights efficiently
4. Pushes to Hugging Face Hub

```python
model = PeftModel.from_pretrained(base_model_reload, new_model_url)
model = model.merge_and_unload()
model.push_to_hub(new_model)
```

## 📈 Monitoring Training

1. Access W&B dashboard for:
   - Loss curves
   - Learning rate scheduling
   - GPU utilization
   - Memory usage

2. Check training logs:
   - Evaluation metrics every 500 steps
   - Model checkpoints every 500 steps
   - Training progress every 100 steps



[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-Report-yellow.svg)](https://api.wandb.ai/links/hawk92-tib-/7onl7dlg)

View the complete training metrics, learning curves, and detailed analysis in our [Weights & Biases Report](https://api.wandb.ai/links/hawk92-tib-/7onl7dlg).

The W&B report includes:
- Loss curves and convergence metrics
- Per-field performance tracking
- Resource utilization metrics
- Hyperparameter configurations
- Model checkpoints

