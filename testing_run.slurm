#!/bin/bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800GB
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=SimpleVLA-RL
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --requeue

# Create logs directory
mkdir -p logs

singularity exec --bind /scratch --nv --overlay /scratch/$USER/my_env/overlay-25GB-500K.ext3:ro /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "
source /ext3/env.sh
ulimit -Sn 65535
cd /scratch/as20482/SimpleVLA-RL/

# Run zombie2.py in the background
python zombie2.py > zombie2.log 2>&1 &

bash examples/run_openvla_oft_rl_worldgym2.sh
"
