#!/usr/bin/env python3
"""
Simple script to run WorldGym rollouts and save videos + GPT scores.
Uses the existing rollout infrastructure from training.

Usage:
    python scripts/rollout_worldgym_eval.py \
        --checkpoint_path /path/to/checkpoint \
        --world_model_checkpoint /path/to/world_model.pt \
        --data_dir /path/to/evaluation/data \
        --output_dir ./rollout_outputs \
        --num_rollouts 10
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_rollout_config(args):
    """Create config for rollout using existing config structure."""
    # Use sft_checkpoint for processor (has all files), checkpoint_path for model weights
    processor_checkpoint = args.sft_checkpoint if hasattr(args, 'sft_checkpoint') and args.sft_checkpoint else args.checkpoint_path

    config = OmegaConf.create({
        'pretrained_checkpoint': processor_checkpoint,  # For processor loading
        'world_model_checkpoint': args.world_model_checkpoint,
        'task_suite_name': args.task_suite_name,
        'vla': args.vla,
        'action_chunks_len': args.action_chunks_len,
        'unnorm_key': args.unnorm_key,
        'center_crop': args.center_crop,
        'temperature': args.temperature,
        'do_sample': args.do_sample,
        'use_proprio': False,
        'num_images_in_input': args.num_images_in_input,
        'val_micro_batch_size': args.batch_size,
        'max_prompt_length': 512,
        'model_family': args.model_family,
        'experiment_name': args.experiment_name,
    })
    return config


def run_rollouts(args):
    """Run rollouts using existing RobHFRollout infrastructure."""
    from verl.utils.dataset.rob_dataset import WORLDGYM_Dataset, collate_fn
    from verl.workers.rollout.rob_rollout import RobHFRollout
    from torch.utils.data import DataLoader
    from verl import DataProto
    from transformers import AutoModelForVision2Seq, AutoConfig
    from tensordict import TensorDict

    print("="*60)
    print("WorldGym Rollout Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"World Model: {args.world_model_checkpoint}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Task Suite: {args.task_suite_name}")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading VLA model...")
    # Load config and add missing use_proprio attribute
    model_config = AutoConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    model_config.use_proprio = False

    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load dataset_statistics.json to update norm_stats (same as training does)
    # Check both RL checkpoint and SFT checkpoint for dataset_statistics.json
    for checkpoint_path in [args.checkpoint_path, args.sft_checkpoint]:
        if checkpoint_path:
            dataset_stats_path = Path(checkpoint_path) / "dataset_statistics.json"
            if dataset_stats_path.exists():
                print(f"Loading dataset statistics from {dataset_stats_path}")
                with open(dataset_stats_path, 'r') as f:
                    model.norm_stats = json.load(f)
                break

    model.cuda()

    # Create config
    config = create_rollout_config(args)

    # Create rollout worker
    print("Creating rollout worker...")
    rollout_worker = RobHFRollout(model, config)

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = WORLDGYM_Dataset(
        task_suite_name=args.task_suite_name,
        data_dir=args.data_dir,
        num_trials_per_task=1,
        train_val="valid"
    )

    # Limit number of rollouts if specified
    if args.num_rollouts is not None and args.num_rollouts < len(dataset):
        print(f"Limiting to {args.num_rollouts} rollouts (out of {len(dataset)} available)")
        dataset.dataframe = dataset.dataframe[:args.num_rollouts]

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Running {len(dataset)} rollouts...\n")

    # Run rollouts
    all_results = []
    rollout_idx = 0

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}/{len(dataloader)}")

        # Convert batch dict to TensorDict for DataProto
        tensor_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        non_tensor_batch = {k: v for k, v in batch.items() if not isinstance(v, torch.Tensor)}

        batch_tensordict = TensorDict(tensor_batch, batch_size=[len(next(iter(tensor_batch.values())))])

        # Prepare prompts
        prompts = DataProto(
            batch=batch_tensordict,
            non_tensor_batch=non_tensor_batch
        )
        # For validation mode, don't set n_samples (omit it from meta_info)
        # The code checks if n_samples exists to determine train vs validation
        prompts.meta_info = {
            'global_steps': rollout_idx,
        }

        # Run rollout
        output = rollout_worker.generate_sequences(prompts)

        # Extract results
        batch_size_actual = output.batch.batch_size[0]
        for i in range(batch_size_actual):
            instruction = batch['instruction'][i]
            task_id = batch['task_id'][i].item()
            trial_id = batch['trial_id'][i].item()
            complete = output.batch['complete'][i].item()

            # The rollout code already saved the video in ./rollouts/
            # Just record the metadata
            result = {
                'rollout_id': rollout_idx,
                'task_id': task_id,
                'trial_id': trial_id,
                'instruction': instruction,
                'success': bool(complete),
                'score': 1.0 if complete else 0.0,
            }

            # Save individual result JSON
            result_file = output_dir / f"rollout_{rollout_idx:04d}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            all_results.append(result)

            print(f"  [{rollout_idx}] {instruction[:60]}... | Success: {complete}")
            rollout_idx += 1

    # Save aggregate results
    aggregate = {
        'checkpoint_path': args.checkpoint_path,
        'world_model_checkpoint': args.world_model_checkpoint,
        'task_suite_name': args.task_suite_name,
        'num_rollouts': len(all_results),
        'mean_score': np.mean([r['score'] for r in all_results]),
        'success_rate': np.mean([r['success'] for r in all_results]),
        'results': all_results,
    }

    aggregate_file = output_dir / "aggregate_results.json"
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Total Rollouts: {len(all_results)}")
    print(f"Success Rate: {aggregate['success_rate']*100:.1f}%")
    print(f"Mean Score: {aggregate['mean_score']:.3f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Videos saved to: ./rollouts/{args.experiment_name}/")
    print(f"Aggregate results: {aggregate_file}")
    print("="*60)

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Run WorldGym rollouts and save videos + scores")

    # Model arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to policy checkpoint (RL checkpoint)")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                       help="Path to SFT checkpoint (for processor files). If not provided, uses checkpoint_path")
    parser.add_argument("--world_model_checkpoint", type=str, required=True,
                       help="Path to world model checkpoint")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing evaluation trials")
    parser.add_argument("--output_dir", type=str, default="./rollout_outputs",
                       help="Directory to save results JSON")

    # Task arguments
    parser.add_argument("--task_suite_name", type=str, default="worldgym_bridge",
                       help="Task suite name")
    parser.add_argument("--vla", type=str, default="openvla-oft",
                       help="VLA model type")
    parser.add_argument("--model_family", type=str, default="openvla",
                       help="Model family")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig",
                       help="Action unnormalization key")

    # Rollout arguments
    parser.add_argument("--num_rollouts", type=int, default=None,
                       help="Number of rollouts (default: all)")
    parser.add_argument("--batch_size", type=int, default=5,
                       help="Batch size for rollouts")
    parser.add_argument("--temperature", type=float, default=1.6,
                       help="Sampling temperature")
    parser.add_argument("--action_chunks_len", type=int, default=8,
                       help="Action chunks length")
    parser.add_argument("--center_crop", type=bool, default=True,
                       help="Center crop images")
    parser.add_argument("--do_sample", type=bool, default=True,
                       help="Sample actions")
    parser.add_argument("--num_images_in_input", type=int, default=1,
                       help="Number of images in input")
    parser.add_argument("--experiment_name", type=str, default="worldgym_eval",
                       help="Experiment name for video folder")

    args = parser.parse_args()

    # Run evaluation
    run_rollouts(args)


if __name__ == "__main__":
    main()
