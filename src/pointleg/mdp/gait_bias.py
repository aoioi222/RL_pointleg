"""Soft penalties for persistent hip abduction bias during walking."""

import math

import torch

from ..robot_cfg import LEG_NAMES


class HipAbductionBiasPenalty:
    """Penalize sustained offsets while allowing alternating lateral leg motion."""

    def __init__(self, cfg, env):
        for key in ("smoothing_time", "warmup_time", "tolerance_degrees", "deviation_scale_degrees"):
            if not math.isfinite(cfg.params[key]) or cfg.params[key] <= 0.:
                raise ValueError("Hip bias times and angular scales must be positive.")
        self.joint_ids = []
        for leg in LEG_NAMES:
            ids, _ = env.scene["robot"].find_joints(f"{leg}_hip_abduction_joint")
            if len(ids) != 1:
                raise ValueError(f"Expected one hip abduction joint for {leg}.")
            self.joint_ids.append(ids[0])
        self.bias = torch.zeros(env.num_envs, 4, device=env.device)
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros_like(self.elapsed)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.bias[ids] = 0.
        self.elapsed[ids] = 0.
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, smoothing_time, warmup_time, tolerance_degrees, deviation_scale_degrees):
        fresh = self.last_step != env.common_step_counter
        robot = env.scene["robot"].data
        deviation = robot.joint_pos[:, self.joint_ids] - robot.default_joint_pos[:, self.joint_ids]
        finite = torch.isfinite(deviation)
        alpha = 1. - math.exp(-env.step_dt / smoothing_time)
        bias = torch.where(finite, self.bias + alpha * (deviation - self.bias), 0.)
        self.bias.copy_(torch.where(fresh[:, None], bias, self.bias))
        self.elapsed += fresh.float() * env.step_dt
        excess = (self.bias.abs() - math.radians(tolerance_degrees)).clamp_min(0.)
        cost = (excess / math.radians(deviation_scale_degrees)).square().clamp_max(4.)
        cost = cost.mean(dim=1) * (self.elapsed >= warmup_time)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        for index, leg in enumerate(LEG_NAMES):
            env.extras["log"][f"Metrics/hip_abduction_bias_abs_{leg}_deg"] = torch.rad2deg(self.bias[:, index].abs()).mean()
        env.extras["log"]["Metrics/hip_abduction_bias_max_deg"] = torch.rad2deg(self.bias.abs().max(dim=1).values).mean()
        return self.cached
