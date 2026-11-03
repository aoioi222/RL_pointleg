"""Shared root clearance measurement for rewards and termination checks."""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg


def root_ground_clearance(
    env,
    asset_cfg: SceneEntityCfg | None = None,
    sensor_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return root height above local ground and a validity mask for each env."""
    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    root_z = env.scene[asset_name].data.root_link_pos_w[:, 2]
    if sensor_name is None:
        height = root_z - env.scene.env_origins[:, 2]
        return height, torch.isfinite(height)
    data = env.scene[sensor_name].data
    if data.distances.shape[1] != 1:
        raise ValueError("Root clearance requires a single downward ground ray")
    height = root_z - data.hit_pos_w[:, 0, 2]
    valid = (
        (data.distances[:, 0] >= 0.0)
        & torch.isfinite(data.distances[:, 0])
        & torch.isfinite(data.hit_pos_w[:, 0]).all(dim=-1)
        & torch.isfinite(data.normals_w[:, 0]).all(dim=-1)
        & (data.normals_w[:, 0, 2] > 0.0)
        & torch.isfinite(height)
    )
    return height, valid
