"""
Simpler environment wrapper for SimpleVLA-RL.

Wrapper for Simpler environments (ManiSkill2-based simulation) supporting WidowX (Bridge) robot.
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
    from transforms3d import quaternions as tq
except ImportError as e:
    print(f"Warning: can't import transforms3d: {e}")
    te = None
    tq = None

__all__ = ['SimplerEnvWrapper']


class SimplerEnvWrapper:
    """Wrapper for Simpler environment with WidowX (Bridge) robot."""

    # Coordinate transform for WidowX (Bridge frame)
    DEFAULT_ROT = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

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
        self.active = True
        self.complete = False
        self.finish_step = 0
        self.instruction = None
        self.obs = None
        self.image_size = (256, 256)

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

    def get_obs(self):
        """Get processed observation."""
        return self._process_obs(self.obs)

    def _process_obs(self, obs):
        """Convert ManiSkill2 obs to VLA input format."""
        img = get_image_from_maniskill2_obs_dict(self.env, obs, camera_name=None)
        img = cv2.resize(img, self.image_size)

        eef_pose = obs['agent']['eef_pos']
        proprio = self._process_widowx_proprio(eef_pose)

        return {
            'full_image': img,
            'state': proprio.astype(np.float32)
        }

    def _process_widowx_proprio(self, proprio):
        """Process WidowX proprio: quat->euler with coordinate frame transform."""
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge_converted = te.mat2euler(rm_bridge @ self.DEFAULT_ROT.T)
        return np.concatenate([proprio[:3], rpy_bridge_converted, [proprio[7]]])

    def step(self, action):
        """Execute action with gripper binarization."""
        if len(action.shape) == 1:
            action = action[np.newaxis, :]

        for i in range(action.shape[0]):
            if not self.active:
                break

            a = action[i].copy()
            # Binarize gripper for WidowX
            a[-1] = 2.0 * (a[-1] > 0.5) - 1.0

            self.obs, reward, done, truncated, info = self.env.step(a)
            self.finish_step += 1

            if done or truncated:
                self.active = False
                self.complete = done
                break

        return self._process_obs(self.obs), self.complete

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
