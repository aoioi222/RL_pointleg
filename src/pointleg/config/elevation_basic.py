"""Minimal shared rewards for paired stair and ramp walking."""

import math

from mjlab.managers.reward_manager import RewardTermCfg

from ..mdp.elevation import elevation_cost, elevation_event
from ..mdp.elevation_basic import elevation_basic_support_deficit, elevation_basic_velocity
from ..mdp.elevation_clearance import elevation_clearance_metric
from .elevation import make_elevation_env_cfg


def make_paired_elevation_env_cfg(terrain_kind, play=False):
    """Use explicit basic objectives without inheriting compound gait rewards."""
    if terrain_kind not in ("stairs", "slope"):
        raise ValueError("Paired elevation tasks require stairs or slope")
    cfg = make_elevation_env_cfg(play=play, terrain_kind=terrain_kind)
    costs = {
        "edge": -1., "slip": -1., "attitude": -.5, "angular": -.1,
        "joint": -.2, "torque": -.02, "action": -.02, "contact_cost": -.2,
    }
    cfg.rewards = {
        key: RewardTermCfg(func=elevation_cost, weight=weight, params={"key": key})
        for key, weight in costs.items()
    }
    cfg.rewards.update({
        "velocity": RewardTermCfg(
            func=elevation_basic_velocity, weight=2., params={"command_name": "twist", "std": .10}),
        "lateral": RewardTermCfg(
            func=elevation_cost, weight=-3., params={"key": "lateral", "tolerance": .010, "scale": .10}),
        "yaw_rate": RewardTermCfg(
            func=elevation_cost, weight=-.5,
            params={"key": "yaw_rate", "tolerance": math.radians(2.), "scale": math.radians(20.)}),
        "support_deficit": RewardTermCfg(
            func=elevation_basic_support_deficit, weight=-2., params={"minimum_force": 8.}),
        "task_completed": RewardTermCfg(func=elevation_event, weight=20., params={"key": "complete"}),
        "task_failed": RewardTermCfg(func=elevation_event, weight=-50., params={"key": "failed"}),
    })
    cfg.metrics = {name: term for name, term in cfg.metrics.items() if term.func is not elevation_clearance_metric}
    return cfg
