#!/bin/bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=SimpleVLA-RL-rollout
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=06:00:00
#SBATCH --requeue

set -e

# ==================== CONFIGURATION ====================
# Edit these paths to match your setup

# Path to your trained policy checkpoint (RL checkpoint)
CHECKPOINT_PATH="/scratch/as20482/SimpleVLA-RL/checkpoints/openvla-oft-bridge-sft-20000"

# Path to SFT checkpoint (has processor files)
SFT_CHECKPOINT="/scratch/as20482/SimpleVLA-RL/checkpoints/openvla-oft-bridge-sft-20000"

# Path to world model checkpoint
WORLD_MODEL_CHECKPOINT="/scratch/as20482/world-model-eval/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt"

# Path to evaluation data (contains .png and .json files)
DATA_DIR="/scratch/as20482/datasets/openvla_evaluation"

# Output directory for results JSON
OUTPUT_DIR="./rollout_outputs/sft20k"

# Task suite name (worldgym_bridge or worldgym_libero_10)
TASK_SUITE_NAME="worldgym_bridge"

NUM_ACTION_CHUNKS=5 #8

# Batch size for rollouts
BATCH_SIZE=5

# Action unnormalization key
UNNORM_KEY="bridge_dataset"

# Temperature for sampling (very small = near-greedy, must be > 0)
TEMPERATURE=1.6

# Experiment name (for video folder)
EXPERIMENT_NAME="sft20k"

# ==================== ENVIRONMENT SETUP ====================

# GPU and CUDA settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
# Force WorldGym constants in OpenVLA
export ROBOT_PLATFORM=BRIDGE #WORLDGYM

# Set OpenAI API key if using GPT scoring (for WorldGym)
# export OPENAI_API_KEY="your-api-key-here"

echo "========================================="
echo "WorldGym Rollout Evaluation"
echo "========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "World Model: $WORLD_MODEL_CHECKPOINT"
echo "Data Dir: $DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Task Suite: $TASK_SUITE_NAME"
echo "========================================="

# ==================== RUN EVALUATION ====================

python scripts/rollout_worldgym_eval.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --sft_checkpoint "$SFT_CHECKPOINT" \
    --world_model_checkpoint "$WORLD_MODEL_CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --task_suite_name "$TASK_SUITE_NAME" \
    --unnorm_key "$UNNORM_KEY" \
    --temperature "$TEMPERATURE" \
    --batch_size "$BATCH_SIZE" \
    --action_chunks_len "$NUM_ACTION_CHUNKS" \
    --experiment_name "$EXPERIMENT_NAME"

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "Results JSON saved to: $OUTPUT_DIR"
echo "Videos saved to: ./rollouts/$EXPERIMENT_NAME/"
echo "========================================="
