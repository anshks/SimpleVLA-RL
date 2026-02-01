set -x

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=BRIDGE

# Disable Vulkan completely - force EGL-only rendering
export SAPIEN_DISABLE_VULKAN=1
export VK_ICD_FILENAMES=""
export SAPIEN_ALLOW_EGL_ONLY=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Don't set EGL vendor - let system auto-detect
unset __EGL_VENDOR_LIBRARY_FILENAMES
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

PROJECT_NAME='SimpleVLA-RL'
EXPERIMENT_NAME='simpler_train_j25_2'
RESUME_CKPT=''
SFT_MODEL_NAME="openvla-oft-bridge-sft-20000"
SFT_MODEL_PATH="/scratch/work/yang-lab/users/as20482/SimpleVLA-RL/checkpoints/$SFT_MODEL_NAME"
CKPT_PATH="/scratch/work/yang-lab/users/as20482/SimpleVLA-RL/checkpoints"

if [ -d "$RESUME_CKPT" ]; then
    SFT_MODEL_PATH=$RESUME_CKPT
    echo "Resuming from checkpoint: $RESUME_CKPT"
else
    echo "Resume checkpoint not found, using SFT model: $SFT_MODEL_PATH"
fi

# Simpler task configuration
# Options: simpler_widowx_all, simpler_widowx_spoon_on_towel, simpler_widowx_carrot_on_plate, etc.
DATASET_NAME="simpler_widowx_all"

VLA_NAME="openvla-oft"
NUM_GPUS=4
NUM_NODES=1
ALIGN_PATH="/scratch/work/yang-lab/users/as20482/SimpleVLA-RL/align.json"

# Remove all existing cached versions of the checkpoint
rm -rf ~/.cache/huggingface/modules/transformers_modules/$SFT_MODEL_NAME 2>/dev/null || true
rm -rf /tmp/hf_modules_cache_* 2>/dev/null || true
export HF_MODULES_CACHE=/tmp/hf_modules_cache_$$
export PYTHONDONTWRITEBYTECODE=1

bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=4 \
    data.val_batch_size=10 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=5 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=160 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=20 \
    actor_rollout_ref.rollout.temperature=2 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=bridge_dataset \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=80 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=online \
    trainer.val_before_train=False
