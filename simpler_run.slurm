#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400GB
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=SimpleVLA-RL
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00

# Create logs directory
mkdir -p logs

ulimit -Sn 65535

bash examples/run_openvla_oft_rl_simpler.sh