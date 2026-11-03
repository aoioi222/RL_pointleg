"""Slow front/rear posture agreement with room for unequal foothold heights."""

import math

import torch

from .elevation import elevation_foot_bottom, elevation_state
from ..robot_cfg import FOOT_SITE_NAMES, LEG_JOINT_NAMES


class ElevationMeanPosturePenalty:
    """Compare filtered hip and knee biases rather than instantaneous leg phases."""

    def __init__(self, cfg, env):
        for key, value in cfg.params.items():
            if not math.isfinite(value) or value <= 0.:
                raise ValueError(f"Posture parameter {key} must be finite and positive")
        if env.cfg.events["reset_elevation"].params["settings"].jump:
            raise ValueError("Walking posture must not be enabled for Jump")
        robot = env.scene["robot"]
        self.joint_ids = [robot.joint_names.index(name) for name in LEG_JOINT_NAMES]
        self.foot_ids, _ = robot.find_sites(FOOT_SITE_NAMES, preserve_order=True)
        self.mean = torch.zeros(env.num_envs, 4, 2, device=env.device)
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros_like(self.age)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.mean[ids] = self.age[ids] = self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, smoothing_time, warmup_time, hip_tolerance_degrees,
                 knee_tolerance_degrees, hip_scale_degrees, knee_scale_degrees,
                 unequal_height, terrain_relaxation, terrain_weight, minimum_support_force):
        state = elevation_state(env)
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"]
                  & ~state["complete"] & (state["speed"] > .1))
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return torch.where(active, self.cached, 0.)
        robot = env.scene["robot"].data
        deviation = (robot.joint_pos[:, self.joint_ids] - robot.default_joint_pos[:, self.joint_ids])
        deviation = deviation.reshape(env.num_envs, 4, 3)[:, :, 1:]
        finite = torch.isfinite(deviation).all(dim=(1, 2))
        contact = env.scene["elevation_contacts"].data
        found = contact.found.reshape(env.num_envs, 4, -1) > 0
        normal = contact.normal.reshape(env.num_envs, 4, -1, 3)
        force = contact.force.reshape(env.num_envs, 4, -1, 3)
        loaded = (found & (normal[..., 2] < -.5) & torch.isfinite(force).all(dim=-1)
                  & (force.nan_to_num().norm(dim=-1) >= minimum_support_force)).any(dim=2)
        valid = active & finite & (loaded.sum(dim=1) >= 2)
        alpha = 1. - math.exp(-env.step_dt / smoothing_time)
        next_mean = torch.where(valid[:, None, None], self.mean + alpha * (deviation.nan_to_num() - self.mean), 0.)
        self.mean.copy_(torch.where(fresh[:, None, None], next_mean, self.mean))
        self.age.copy_(torch.where(fresh, torch.where(valid, self.age + env.step_dt, 0.), self.age))
        feet = elevation_foot_bottom(env, self.foot_ids)
        high = torch.where(loaded, feet[..., 2], -torch.inf).max(dim=1).values
        low = torch.where(loaded, feet[..., 2], torch.inf).min(dim=1).values
        mixed = valid & (high - low > unequal_height)
        tolerance = self.mean.new_tensor([hip_tolerance_degrees, knee_tolerance_degrees]) * math.pi / 180.
        scale = self.mean.new_tensor([hip_scale_degrees, knee_scale_degrees]) * math.pi / 180.
        multiplier = torch.where(mixed, terrain_relaxation, 1.)
        front_rear = self.mean[:, :2].mean(dim=1) - self.mean[:, 2:].mean(dim=1)
        left_right = self.mean[:, [0, 2]] - self.mean[:, [1, 3]]
        front_cost = ((front_rear.abs() - tolerance * multiplier[:, None]).clamp_min(0.) / scale).square().clamp_max(4.)
        side_cost = ((left_right.abs() - tolerance).clamp_min(0.) / scale).square().clamp_max(4.)
        cost = .5 * front_cost.mean(dim=1) + .5 * side_cost.mean(dim=(1, 2))
        cost *= torch.where(mixed, terrain_weight, 1.)
        cost = torch.where(valid & (self.age >= warmup_time), cost, 0.)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        if valid.any():
            for index, name in enumerate(("hip", "knee")):
                env.extras["log"][f"Metrics/elevation_front_rear_{name}_bias_deg"] = torch.rad2deg(
                    front_rear[valid, index]
                ).detach().clone()
        self.last_step[fresh] = env.common_step_counter
        return torch.where(active, self.cached, 0.)
