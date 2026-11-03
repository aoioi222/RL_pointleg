"""Configure total-reward clipping for one initialized environment instance."""

from functools import wraps
import math


def configure_positive_rewards(env, env_ids, minimum_reward: float = 0.0) -> None:
    """Install an instance-local reward floor as a startup event.

    Individual term statistics retain their original signed values. Only the
    manager's returned total and its total-reward buffer are clipped.
    """
    del env_ids
    if not math.isfinite(minimum_reward):
        raise ValueError("minimum_reward must be finite")
    manager = env.reward_manager
    if not hasattr(manager, "_pointleg_original_compute"):
        manager._pointleg_original_compute = manager.compute

        @wraps(manager._pointleg_original_compute)
        def compute(dt):
            reward = manager._pointleg_original_compute(dt)
            return reward.clamp_(min=manager._pointleg_minimum_reward)

        manager.compute = compute
    manager._pointleg_minimum_reward = minimum_reward
