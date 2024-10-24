#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=legal-llm            # A nice readable name of your job, to see it in the queue
#SBATCH --nodes=1                     # Number of nodes to request
#SBATCH --cpus-per-task=8            # Number of CPUs to request
#SBATCH --gres=gpu:a100m40:4       # Request 2 A6000 GPUs
#SBATCH --output=logs/task_%j.log      # Standard output and error log
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4


source /opt/conda/etc/profile.d/conda.sh
conda activate legal-llm  # replace with your conda environment name

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12345  # Or any available port
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID



# Run the training script with accelerate
accelerate launch --config_file accelerate_config.yaml fine_tuning.py \
    --hf_token YOUR_HF_TOKEN \
    --wb_token YOUR_WB_TOKEN \
    --output_file "../data/prepared_dataset_fine_tune.jsonl" \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --output_model "./fine-tuned-llama" \
    --ds_config "ds_config.json" \
    --train_dataset_file "../data/train_dataset_final.jsonl" \
    --eval_dataset_file "../data/eval_dataset_final.jsonl" \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 1