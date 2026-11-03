"""Termination terms specific to the 12-DoF point-foot quadruped."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .height import root_ground_clearance

def joint_pos_out_of_limits(
    env,
    limit_margin: float = 0.02,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Terminate when a selected joint reaches a hard-limit safety margin."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    limits = asset.data.joint_pos_limits
    assert limits is not None
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = limits[:, asset_cfg.joint_ids]
    below_lower = joint_pos <= joint_limits[..., 0] + limit_margin
    above_upper = joint_pos >= joint_limits[..., 1] - limit_margin
    return torch.any(below_lower | above_upper, dim=1)


def root_height_out_of_range(
    env,
    minimum_height: float,
    maximum_height: float,
    asset_cfg: SceneEntityCfg | None = None,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Terminate on unsafe clearance above local terrain or an invalid ground ray.

    Args:
        env: Parallel training environment.
        minimum_height: Minimum base clearance in meters.
        maximum_height: Maximum base clearance in meters.
        asset_cfg: Robot whose root position is checked.
        sensor_name: Single downward terrain ray attached to the base. Without
            a sensor, height is relative to the environment origin on flat ground.

    Returns:
        Boolean termination flags with shape ``(num_envs,)``.
    """
    if minimum_height >= maximum_height:
        raise ValueError("minimum_height must be smaller than maximum_height")
    height, valid = root_ground_clearance(env, asset_cfg, sensor_name)
    return ~valid | (height < minimum_height) | (height > maximum_height)
