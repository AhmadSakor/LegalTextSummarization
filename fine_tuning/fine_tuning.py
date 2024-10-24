"""
This script fine-tunes the Llama-3.2 model for legal case summarization using LoRA (Low-Rank Adaptation). It leverages the Huggingface Transformers library, DeepSpeed for distributed training, and Weights & Biases for logging and tracking experiments.

Key Features:
- Implements LoRA to reduce the number of trainable parameters and optimize fine-tuning.
- Uses DeepSpeed for efficient large-scale model training.
- Integrates Weights & Biases (W&B) for experiment tracking.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import json
import wandb
import deepspeed
from huggingface_hub import login
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2 with LoRA")

# Define arguments
parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API key")
parser.add_argument("--wb_token", type=str, required=True, help="Weights & Biases API key")
parser.add_argument("--output_file", type=str, default='../data/prepared_dataset_fine_tune.jsonl', help="Dataset file path in JSON Lines format")
parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.2-3B-Instruct', help="Model to fine-tune")
parser.add_argument("--output_model", type=str, default='./fine-tuned-llama', help="Directory to save the fine-tuned model")
parser.add_argument("--ds_config", type=str, default='ds_config.json', help="DeepSpeed configuration file")
parser.add_argument("--train_dataset_file", type=str, default='../data/train_dataset_final.jsonl', help="File path for the processed training dataset")
parser.add_argument("--eval_dataset_file", type=str, default='../data/eval_dataset_final.jsonl', help="File path for the processed evaluation dataset")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size per device")

args = parser.parse_args()

# Log in to Hugging Face Hub using API token
login(token=args.hf_token)

# Set up Weights & Biases for experiment tracking using API token
wandb.login(key=args.wb_token)

# Check GPU capability and set precision
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16  # Use bfloat16 for better performance on recent GPUs
    attn_implementation = "flash_attention_2"  # Efficient attention mechanism for newer GPUs
else:
    torch_dtype = torch.float16  # Default to float16
    attn_implementation = "eager"  # Fallback attention implementation

# Load tokenizer with custom settings
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to end-of-sequence token
tokenizer.padding_side = "right"  # Right-side padding to align token sequences

# Load model with specified precision and attention implementation
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch_dtype,  # Use appropriate precision
    trust_remote_code=True,
    attn_implementation=attn_implementation
)

# Set padding token ID
model.config.pad_token_id = model.config.eos_token_id

# Configure LoRA for low-rank adaptation
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Target LoRA layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing for memory efficiency during training
model.gradient_checkpointing_enable()

# Enable gradients for input layers (required for LoRA)
model.enable_input_require_grads()

# Log the number of trainable parameters for inspection
model.print_trainable_parameters()

# Load the dataset
dataset = load_dataset('json', data_files=args.output_file)

# Split the dataset into training and evaluation sets
split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Write training and evaluation datasets to JSONL files for easier inspection
with open(args.train_dataset_file, 'w', encoding='utf-8') as f:
    for example in train_dataset:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

with open(args.eval_dataset_file, 'w', encoding='utf-8') as f:
    for example in eval_dataset:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

# Define tokenization function for processing datasets
def tokenize_function(examples):
    """
    Tokenizes input text, truncating or padding as necessary to fit the model's input size.
    
    Args:
        examples (dict): A dictionary containing text data to tokenize.
    
    Returns:
        dict: A dictionary containing tokenized input IDs.
    """
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=10500,  # Maximum token length to fit model constraints
        padding='longest'  # Use the longest padding for batch alignment
    )

# Apply tokenization to training and evaluation datasets
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Remove raw text after tokenization
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Set up training arguments
training_arguments = TrainingArguments(
    output_dir=args.output_model,  # Directory to save model checkpoints
    per_device_train_batch_size=args.per_device_train_batch_size,  # Set batch size per GPU
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # Accumulate gradients to simulate larger batch sizes
    num_train_epochs=args.num_train_epochs,  # Number of training epochs
    eval_strategy="steps",  # Evaluate the model at regular intervals
    eval_steps=500,  # Evaluate every 500 steps
    save_steps=500,  # Save checkpoints every 500 steps
    logging_steps=100,  # Log metrics every 100 steps
    learning_rate=args.learning_rate,  # Initial learning rate
    save_total_limit=2,  # Keep only the last 2 checkpoints
    load_best_model_at_end=True,  # Load best model at the end of training
    report_to="wandb",  # Report training metrics to Weights & Biases
    logging_strategy="steps",  # Log metrics every few steps
    save_strategy="steps",  # Save checkpoints every few steps
    bf16=True,  # Use bfloat16 for efficient training
    fp16=False,  # Do not use float16 (overridden by bfloat16)
    optim="adamw_torch",  # Optimizer for model weights
    gradient_checkpointing=True,  # Enable gradient checkpointing to reduce memory usage
    deepspeed=args.ds_config,  # DeepSpeed configuration for distributed training
)

# Initialize the fine-tuning trainer using TRL's SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,  # LoRA configuration
    max_seq_length=10500,  # Maximum sequence length for input text
    dataset_text_field="input_ids",  # Field in dataset containing input IDs
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,  # Disable data packing
)

# Start W&B logging
wandb.init(project='Fine-tune Llama 3.2 for Legal Summarization', job_type="training", anonymous="allow")

# Begin model training
trainer.train()

# Finish W&B logging
wandb.finish()

# Save the fine-tuned model and tokenizer
trainer.model.save_pretrained(args.output_model)
tokenizer.save_pretrained(args.output_model)
