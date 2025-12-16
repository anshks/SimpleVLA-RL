#!/usr/bin/env python
"""
Extract LIBERO initial frames and save as PNG files for WorldGym training.

This script:
1. Loads LIBERO benchmark tasks
2. Extracts initial states for each task
3. Renders initial frames and saves as PNG
4. Creates JSON metadata files with task instructions
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from libero.libero import benchmark
from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action
import re


def clean_instruction(instruction):
    """
    Clean LIBERO instruction by removing scene name and replacing underscores.

    Example:
        KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
        -> turn on the stove and put the moka pot on it
    """
    # Remove scene prefix (e.g., KITCHEN_SCENE3_, LIVING_ROOM_SCENE1_, etc.)
    instruction = re.sub(r'^[A-Z_]+SCENE\d+_', '', instruction)

    # Replace underscores with spaces
    instruction = instruction.replace('_', ' ')

    return instruction


def extract_libero_initial_frames(
    task_suite_name="libero_10",
    num_trials_per_task=50,
    output_dir="/scratch/as20482/datasets/libero_worldgym_frames",
    model_family="openvla",
    resolution=256,
    num_steps_wait=10
):
    """
    Extract initial frames from LIBERO tasks and save as PNGs.

    Args:
        task_suite_name: LIBERO task suite (libero_10, libero_90, etc.)
        num_trials_per_task: Number of initial states per task
        output_dir: Directory to save PNG files and JSON metadata
        model_family: Model family for environment setup
        resolution: Image resolution (default 256x256)
        num_steps_wait: Number of dummy steps to stabilize environment
    """

    # Create output directory
    output_path = Path(output_dir) / task_suite_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load LIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks

    print(f"Extracting frames from {task_suite_name} ({num_tasks} tasks)")
    print(f"Output directory: {output_path}")

    # Track extraction stats
    total_extracted = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description_raw = task_suite.get_task_names()[task_id]
        task_description = clean_instruction(task_description_raw)
        initial_states = task_suite.get_task_init_states(task_id)

        # Create task subdirectory
        task_dir = output_path / f"task_{task_id:03d}"
        task_dir.mkdir(exist_ok=True)

        print(f"\nTask {task_id}/{num_tasks-1}: {task_description}")
        print(f"  Available initial states: {len(initial_states)}")

        # Limit to requested number of trials
        num_trials = min(num_trials_per_task, len(initial_states))

        for trial_id in tqdm(range(num_trials), desc=f"  Extracting trials"):
            initial_state = initial_states[trial_id]

            # Initialize environment
            env = None
            try:
                env, _ = get_libero_env(task, model_family, resolution=resolution)
                env.reset()
                obs = env.set_init_state(initial_state)

                # Execute dummy actions to stabilize environment
                for _ in range(num_steps_wait):
                    obs, _, _, _ = env.step(get_libero_dummy_action(model_family))

                # Extract agentview image (rotated 180 degrees as in libero_utils.py)
                img = obs["agentview_image"][::-1, ::-1]

                # Convert to PIL Image and save
                img_pil = Image.fromarray(img)
                png_path = task_dir / f"trial_{trial_id:03d}.png"
                img_pil.save(png_path)

                # Create metadata JSON
                metadata = {
                    "instruction": task_description,
                    "partial_credit_criteria": None,
                    "task_suite_name": task_suite_name,
                    "task_id": task_id,
                    "trial_id": trial_id
                }
                json_path = task_dir / f"trial_{trial_id:03d}.json"
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                total_extracted += 1

            except Exception as e:
                print(f"\n  Error extracting trial {trial_id}: {e}")
                continue
            finally:
                if env is not None:
                    try:
                        env.close()
                    except:
                        pass

    # Save dataset summary
    summary = {
        "task_suite_name": task_suite_name,
        "num_tasks": num_tasks,
        "num_trials_per_task": num_trials_per_task,
        "total_frames_extracted": total_extracted,
        "resolution": resolution,
        "num_steps_wait": num_steps_wait,
        "format": "worldgym_compatible"
    }

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total frames extracted: {total_extracted}")
    print(f"Output directory: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LIBERO initial frames for WorldGym training"
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"],
        help="LIBERO task suite to extract"
    )
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=50,
        help="Number of initial frames to extract per task"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/as20482/datasets/libero_worldgym_frames",
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="openvla",
        help="Model family for environment setup"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution (default 256x256)"
    )
    parser.add_argument(
        "--num_steps_wait",
        type=int,
        default=10,
        help="Number of dummy steps to stabilize environment"
    )

    args = parser.parse_args()

    extract_libero_initial_frames(
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        output_dir=args.output_dir,
        model_family=args.model_family,
        resolution=args.resolution,
        num_steps_wait=args.num_steps_wait
    )


if __name__ == "__main__":
    main()
