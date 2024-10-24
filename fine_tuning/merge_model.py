"""
This script demonstrates how to reload a base model, merge it with a fine-tuned adapter (using PEFT), and push the merged model to the Hugging Face Hub. The model used here is the Llama-3.2-3B-Instruct model, which has been fine-tuned for legal summarization tasks.

Key Features:
- Reloads the base Llama model and fine-tuned adapter.
- Merges the adapter with the base model using PEFT.
- Saves and pushes the merged model to the Hugging Face Hub for deployment.
"""

import argparse
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def main(hf_token, base_model_url, new_model_url, new_model_local, new_model):
    # Log in to Hugging Face using your API token
    login(token=hf_token)

    # Reload tokenizer for the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_url)

    # Set the device map based on GPU availability
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Reload the base model with memory optimization settings
    base_model_reload= AutoModelForCausalLM.from_pretrained(
        base_model_url,
        torch_dtype=dtype,
        device_map=device_map,
    )

    # Load the fine-tuned adapter and merge it with the base model
    model = PeftModel.from_pretrained(base_model_reload, new_model_url)

    # Merge the adapter weights with the base model and unload the adapter
    model = model.merge_and_unload()

    # Save the merged model and tokenizer locally
    model.save_pretrained(new_model_local)
    tokenizer.save_pretrained(new_model_local)

    # Push the merged model and tokenizer to the Hugging Face Hub
    model.push_to_hub(new_model)
    tokenizer.push_to_hub(new_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Llama-3.2-3B-Instruct model with fine-tuned adapter and push to Hugging Face Hub")

    # Arguments for API key, model paths, and output locations
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--base_model_url", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="URL for the base Llama model")
    parser.add_argument("--new_model_url", type=str, default="./fine-tuned-llama", help="Path to fine-tuned adapter model")
    parser.add_argument("--new_model_local", type=str, default="./fine-tuned-llama-merged", help="Local path for saving the merged model")
    parser.add_argument("--new_model", type=str, default="ahmadsakor/Llama3.2-3B-Instruct-Legal-Summarization", help="Hugging Face Hub model name for uploading")

    args = parser.parse_args()

    # Run the main function with provided arguments
    main(args.hf_token, args.base_model_url, args.new_model_url, args.new_model_local, args.new_model)
