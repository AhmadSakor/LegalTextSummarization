"""
This script evaluates a fine-tuned Llama model for generating summaries of legal cases based on a given dataset. 
The script uses Huggingface's Transformers library, performs generation tasks, and saves the generated outputs in JSONL format.

Key Features:
- Loads a fine-tuned Llama model for text generation.
- Processes a JSONL evaluation dataset and prepares input prompts.
- Generates model outputs based on legal case data.
- Saves the generated summaries and logs the process.
"""

import os
import json
import jsonlines
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import argparse

# ============================
# 1. Argument Parsing
# ============================

def parse_args():
    parser = argparse.ArgumentParser(description="Llama Model Evaluation Script")
    
    parser.add_argument('--eval_dir', type=str, default="eval_base3", help='Directory to save evaluation logs and outputs')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='Model ID from Huggingface')
    parser.add_argument('--eval_dataset_path', type=str, default="eval_dataset.jsonl", help='Path to the evaluation dataset in JSONL format')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generating outputs')
    
    return parser.parse_args()

# ============================
# 2. Initialize the Model Pipeline
# ============================

def initialize_model(model_id: str):
    """
    Initializes the text generation model and tokenizer from the Huggingface Hub. 
    Handles device mapping for GPU or CPU usage.
    
    Args:
        model_id (str): The model identifier from Huggingface.
    
    Returns:
        pipe: A text-generation pipeline using the loaded model and tokenizer.
    """
    try:
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Padding for decoder-only models

        # Load the model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )

        # Create the text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            device_map=device_map,
        )

        logging.info(f"Model '{model_id}' loaded successfully with device_map='{device_map}'.")
        return pipe
    except Exception as e:
        logging.error(f"Failed to load model '{model_id}': {e}")
        raise e

# ============================
# 3. Load the Evaluation Dataset
# ============================

def load_evaluation_dataset(eval_dataset_path: str):
    """
    Loads the evaluation dataset in JSON Lines format.
    
    Args:
        eval_dataset_path (str): Path to the evaluation dataset.
    
    Returns:
        eval_dataset: A Huggingface dataset object containing the evaluation data.
    """
    try:
        eval_dataset = load_dataset('json', data_files=eval_dataset_path, split='train')
        logging.info(f"Loaded {len(eval_dataset)} entries from '{eval_dataset_path}'.")
        return eval_dataset
    except Exception as e:
        logging.error(f"Failed to load evaluation dataset: {e}")
        raise e

# ============================
# 4. Generate Outputs and Save
# ============================

def generate_outputs(pipe, eval_dataset, eval_dir, batch_size=16):
    """
    Generates text outputs from the evaluation dataset using the model pipeline.
    
    Args:
        pipe: The text-generation pipeline.
        eval_dataset: The evaluation dataset containing prompts and references.
        eval_dir: Directory to save the generated outputs.
        batch_size (int): Batch size for generating outputs.
    
    Returns:
        None. Saves generated outputs in a JSONL file.
    """
    
    # Prepare prompts for generation
    def prepare_prompts(example):
        try:
            text = example.get("text", "")
            if not text:
                raise ValueError("Empty 'text' field encountered.")

            # Extract different parts of the prompt
            system_part = text.split("<|start_header_id|>system<|end_header_id|>")[1].split("<|start_header_id|>user<|end_header_id|>")[0].strip()
            user_part = text.split("<|start_header_id|>user<|end_header_id|>")[1].split("<|start_header_id|>assistant<|end_header_id|>")[0].strip()
            reference_json_part = text.split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()

            # Convert reference summary to JSON format
            reference_summary = json.loads(reference_json_part)
            reference_summary_str = json.dumps(reference_summary, ensure_ascii=False)

            # Create the prompt with special tokens
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n{system_part}\n"
                f"<|start_header_id|>user<|end_header_id|>\n{user_part}\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )

            return {
                "prompt": prompt,
                "reference_summary": reference_summary_str
            }
        except Exception as e:
            logging.error(f"Error preparing prompt: {e}")
            return None

    # Apply the preparation function to the dataset
    eval_dataset = eval_dataset.map(prepare_prompts, remove_columns=eval_dataset.column_names)

    # Filter out invalid examples
    eval_dataset = eval_dataset.filter(lambda x: x is not None)

    total_samples = len(eval_dataset)
    total_batches = (total_samples + batch_size - 1) // batch_size

    # Initialize the progress bar
    pbar = tqdm(total=total_batches, desc="Generating Outputs", unit="batch", leave=True)

    # List to store generated data
    generated_data = []

    # Generate outputs batch by batch
    for i in range(0, total_samples, batch_size):
        batch_dataset = eval_dataset.select(range(i, min(i + batch_size, total_samples)))
        batch_prompts = batch_dataset["prompt"]
        batch_references = batch_dataset["reference_summary"]

        # Generate outputs using the pipeline
        generated_outputs_batch = pipe(
            batch_prompts,
            max_new_tokens=1500,
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id,
            batch_size=batch_size,
            padding=True,
            return_full_text=False,
        )

        # Process the outputs
        for idx, (generated_output, prompt, reference_summary) in enumerate(zip(generated_outputs_batch, batch_prompts, batch_references)):
            try:
                if isinstance(generated_output, list) and len(generated_output) > 0:
                    generated_text = generated_output[0]['generated_text']
                elif isinstance(generated_output, dict):
                    generated_text = generated_output['generated_text']
                else:
                    raise ValueError(f"Unexpected output structure at index {i + idx}: {type(generated_output)}")

                # Append the results to generated data
                generated_data.append({
                    "prompt": prompt,
                    "reference_summary": reference_summary,
                    "generated_text": generated_text
                })

            except Exception as e:
                logging.error(f"Error processing generated output {i + idx}: {e}")
                continue

        # Update progress bar
        pbar.update(1)

    pbar.close()
    logging.info(f"Generated outputs for {len(generated_data)} samples.")

    # Save generated outputs to file
    generated_outputs_path = os.path.join(eval_dir, "generated_outputs.jsonl")
    try:
        with jsonlines.open(generated_outputs_path, mode='w') as writer:
            writer.write_all(generated_data)
        logging.info(f"Saved generated outputs to '{generated_outputs_path}'.")
    except Exception as e:
        logging.error(f"Failed to save generated outputs: {e}")

# ============================
# 5. Main Execution Function
# ============================

def main():
    """
    Main function to execute the generation process.
    """
    args = parse_args()

    # Ensure evaluation directory exists
    os.makedirs(args.eval_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.eval_dir, "generation.log"),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Starting generation process.")

    # Initialize model and load dataset
    pipe = initialize_model(args.model_id)
    eval_dataset = load_evaluation_dataset(args.eval_dataset_path)

    # Generate outputs
    generate_outputs(
        pipe=pipe,
        eval_dataset=eval_dataset,
        eval_dir=args.eval_dir,
        batch_size=args.batch_size
    )

    logging.info("Generation process completed.")

if __name__ == "__main__":
    main()
