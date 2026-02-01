"""
Simpler environment wrapper for SimpleVLA-RL.

Wrapper for Simpler environments (ManiSkill2-based simulation) supporting WidowX (Bridge) robot.
Consistent with auto_eval policies - only uses image, no proprio.
"""

import numpy as np
import cv2

try:
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
except ImportError as e:
    print(f"Warning: can't import simpler_env: {e}")
    simpler_env = None
    get_image_from_maniskill2_obs_dict = None

try:
    from transforms3d import euler as te
except ImportError as e:
    print(f"Warning: can't import transforms3d: {e}")
    te = None

__all__ = ['SimplerEnvWrapper']


class SimplerEnvWrapper:
    """Wrapper for Simpler environment with WidowX (Bridge) robot."""

    def __init__(self, task_name, trial_id, trial_seed, config):
        """Initialize the wrapper.

        Args:
            task_name: SimplerEnv task name (e.g., 'widowx_spoon_on_towel')
            trial_id: Episode ID for tracking
            trial_seed: Random seed for reproducibility
            config: Training configuration object
        """
        self.task_name = task_name
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.config = config
        self.env = None
        self.success = False
        self.instruction = None
        self.obs = None
        self.image_size = (256, 256)
        self.finish_step = 0

    def initialize(self):
        """Initialize the Simpler environment."""
        if simpler_env is None:
            raise ImportError("simpler_env is not installed")

        self.env = simpler_env.make(self.task_name)

        max_steps = getattr(self.config, 'simpler_max_episode_steps', 80)
        if hasattr(self.env, '_max_episode_steps'):
            self.env._max_episode_steps = max_steps

        np.random.seed(self.trial_seed)
        self.obs, _ = self.env.reset()
        self.instruction = self.env.unwrapped.get_language_instruction()
        self.success = False

    def get_obs(self):
        """Get processed observation."""
        return self._process_obs(self.obs)

    def _process_obs(self, obs):
        """Convert ManiSkill2 obs to VLA input format (image only, no proprio)."""
        img = get_image_from_maniskill2_obs_dict(self.env, obs, camera_name=None)
        img = cv2.resize(img, self.image_size)
        return {'full_image': img}

    def step(self, action):
        """Execute a chunk of actions, returning the final obs plus per-step frames and rewards.

        Returns:
            last_processed: Final observation dict
            success: Whether task succeeded
            per_step_frames: List of images for each step
            per_step_rewards: List of dense rewards for each step
        """
        if len(action.shape) == 1:
            action = action[np.newaxis, :]

        per_step_frames = []
        per_step_rewards = []
        last_processed = None

        for i in range(action.shape[0]):
            a = action[i].copy()
            # Convert RPY -> axis-angle to match ManiSkill/SimplerEnv convention
            # Using euler2axangle directly as in SimplerEnv's octo_model.py
            if te is not None:
                try:
                    roll, pitch, yaw = a[3], a[4], a[5]
                    action_rotation_ax, action_rotation_angle = te.euler2axangle(roll, pitch, yaw)
                    a[3:6] = action_rotation_ax * action_rotation_angle
                except Exception as e:
                    print(f"Warning: RPY to axis-angle conversion failed: {e}", flush=True)
            # Binarize gripper for WidowX (threshold 0.5 to match Bridge convention)
            a[-1] = 2.0 * (a[-1] > 0.5) - 1.0

            self.obs, reward, done, truncated, info = self.env.step(a)
            self.finish_step += 1
            self.success = self.success or bool(info.get("success", False))

            # Compute dense reward from info
            step_reward = self._compute_dense_reward(info)
            per_step_rewards.append(step_reward)

            processed = self._process_obs(self.obs)
            per_step_frames.append(processed["full_image"])
            last_processed = processed

            if done or truncated:
                break

        if last_processed is None and self.obs is not None:
            last_processed = self._process_obs(self.obs)

        return last_processed, self.success, per_step_frames, per_step_rewards

    def _compute_dense_reward(self, info):
        """Compute dense reward from environment info.

        Available info fields (task-dependent):
        - is_grasped / is_src_obj_grasped: object currently grasped
        - consecutive_grasp: grasped for 5+ consecutive steps
        - lifted_object: object is lifted
        - lifted_object_significantly: object lifted significantly
        - success: task completed

        Returns:
            float: Dense reward for this step
        """
        reward = 0.0

        # Reward for grasping the object
        is_grasped = info.get("is_src_obj_grasped", info.get("is_grasped", False))
        if is_grasped:
            reward += 0.1

        # Reward for consecutive grasp (stable grasp)
        if info.get("consecutive_grasp", False):
            reward += 0.1

        # Reward for lifting the object (intermediate sparse signal before success)
        if info.get("lifted_object_significantly", False):
            reward += 0.1
        elif info.get("lifted_object", False):
            reward += 0.1

        # Reward for task success
        if info.get("success", False):
            reward += 1.0

        return reward

    def get_instruction(self):
        """Get the task instruction string."""
        return self.instruction

    def close(self):
        """Clean up environment."""
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
            self.env = None
