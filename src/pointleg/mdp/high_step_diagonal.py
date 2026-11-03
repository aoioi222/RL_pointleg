"""Reward front-leg motion tracking of the opposite rear leg during walking."""

import math

import torch

from pointleg.robot_cfg import LEG_NAMES
from .walking_gait import loaded_feet, ordered_contact_ids


class HighStepDiagonalTrackingReward:
    """Compare paired swing displacement using the moving rear leg as reference."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.contacts = ordered_contact_ids(env.scene[cfg.params["sensor_name"]])
        self.joints = [robot.joint_names.index(f"{leg}_{joint}_joint")
                       for leg in LEG_NAMES for joint in ("hip_pitch", "knee")]
        self.anchor = torch.zeros((env.num_envs, 2, 2, 2), device=env.device)
        self.paired = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.anchor[ids] = 0.
        self.paired[ids] = False
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, sensor_name, command_name, minimum_support_force=8.,
                 minimum_forward_velocity=.05, minimum_rear_motion_degrees=5.,
                 full_rear_motion_degrees=20., joint_error_scale_degrees=15.):
        values = (minimum_support_force, minimum_forward_velocity, minimum_rear_motion_degrees,
                  full_rear_motion_degrees, joint_error_scale_degrees)
        if not all(math.isfinite(value) and value > 0. for value in values):
            raise ValueError("Diagonal tracking thresholds must be finite and positive")
        if minimum_rear_motion_degrees >= full_rear_motion_degrees:
            raise ValueError("Minimum rear motion must be below the full-motion target")
        basic = env._high_step_basic_term
        basic(env, **basic.params)
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        robot = env.scene["robot"].data
        q = robot.joint_pos[:, self.joints].reshape(env.num_envs, 4, 2)
        command = env.command_manager.get_command(command_name)
        vx = robot.root_link_lin_vel_b[:, 0]
        finite = torch.isfinite(q).all(dim=(1, 2)) & torch.isfinite(command).all(dim=1) & torch.isfinite(vx)
        walking = finite & (command[:, 0] > .08) & (vx > minimum_forward_velocity)
        terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
        if terminated is not None:
            walking &= ~terminated
        support = loaded_feet(env.scene[sensor_name], self.contacts, minimum_support_force)
        airborne = basic.swinging & basic.eligible_swing & (basic.air <= basic.params["max_air_time"] + 1.e-6)
        pairs = torch.stack((airborne[:, 0] & airborne[:, 3],
                             airborne[:, 1] & airborne[:, 2]), dim=1)
        pairs &= walking[:, None] & (support.sum(dim=1, keepdim=True) >= 2)
        current = torch.stack((q[:, [0, 3]], q[:, [1, 2]]), dim=1).nan_to_num()
        start = pairs & ~self.paired & fresh[:, None]
        self.anchor.copy_(torch.where(start[:, :, None, None], current, self.anchor))
        delta = current - self.anchor
        front_delta, rear_delta = delta[:, :, 0], delta[:, :, 1]
        rear_motion = rear_delta.abs().amax(dim=-1)
        error = (front_delta - rear_delta).square().mean(dim=-1)
        scale = math.radians(joint_error_scale_degrees)
        quality = torch.exp(-error / (scale * scale))
        motion = (rear_motion / math.radians(full_rear_motion_degrees)).clamp(0., 1.)
        valid = pairs & (rear_motion >= math.radians(minimum_rear_motion_degrees))
        score = torch.where(valid, quality * motion, 0.).sum(dim=1) / 2.
        self.cached.copy_(torch.where(fresh, score, self.cached))
        self.paired.copy_(torch.where(fresh[:, None], pairs, self.paired))
        self.last_step[fresh] = env.common_step_counter
        for index, pair in enumerate(("fl_rr", "fr_rl")):
            if valid[:, index].any():
                mask = valid[:, index]
                env.extras["log"][f"Metrics/high_step_diagonal_error_{pair}_deg"] = (
                    torch.rad2deg(error[:, index].sqrt())[mask].detach().clone())
                env.extras["log"][f"Metrics/high_step_diagonal_rear_motion_{pair}_deg"] = (
                    torch.rad2deg(rear_motion[:, index])[mask].detach().clone())
        return self.cached
