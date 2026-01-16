set -x

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=WORLDGYM

PROJECT_NAME='SimpleVLA-RL'
EXPERIMENT_NAME='libero_worldgym_libero_oft'
# Using OpenVLA-OFT checkpoint trained on Libero
SFT_MODEL_NAME="Openvla-oft-SFT-libero10-trajall"
SFT_MODEL_PATH="/scratch/as20482/SimpleVLA-RL/checkpoints/$SFT_MODEL_NAME"
CKPT_PATH="/scratch/as20482/SimpleVLA-RL/checkpoints"

# LIBERO World model configuration
WORLD_MODEL_CHECKPOINT="/scratch/as20482/world-model-eval/checkpoints/libero-nov-17/ckpt_000480000.pt"
DATA_DIR="/scratch/as20482/datasets/libero_worldgym_frames/libero_10"

DATASET_NAME="worldgym_libero_10"
VLA_NAME="openvla-oft"
NUM_GPUS=2
NUM_NODES=1
ALIGN_PATH="/scratch/as20482/SimpleVLA-RL/align.json"

# Remove all existing cached versions of the checkpoint
rm -rf ~/.cache/huggingface/modules/transformers_modules/$SFT_MODEL_NAME 2>/dev/null || true
rm -rf /tmp/hf_modules_cache_* 2>/dev/null || true
# Set fresh cache location
export HF_MODULES_CACHE=/tmp/hf_modules_cache_$$
# Ensure Python doesn't cache imports
export PYTHONDONTWRITEBYTECODE=1

bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.data_dir=$DATA_DIR \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=50 \
    data.val_batch_size=50 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=200 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$((NUM_GPUS * 4)) \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=64 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=50 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=4 \
    actor_rollout_ref.rollout.unnorm_key=libero_10 \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=200 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.world_model_checkpoint=$WORLD_MODEL_CHECKPOINT \
    actor_rollout_ref.ref.log_prob_micro_batch_size=200 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=online \
    trainer.val_before_train=False
