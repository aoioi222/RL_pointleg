"""Track persistent front-foot placement relative to the corresponding hip."""

import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse


class HighStepFrontWorkspacePenalty:
    """Penalize persistent front overreach without fixing the swing pose."""

    def __init__(self, cfg, env):
        del cfg
        robot = env.scene["robot"]
        self.feet, self.hips = [], []
        for leg in ("fl", "fr"):
            for finder, name, output in (
                (robot.find_sites, f"{leg}_foot_contact_site", self.feet),
                (robot.find_bodies, f"{leg}_hip_pitch_Link", self.hips),
            ):
                ids, _ = finder(name)
                if len(ids) != 1:
                    raise ValueError(f"Expected one position reference for {name}")
                output.append(ids[0])
        self.mean = torch.zeros((env.num_envs, 2), device=env.device)
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros_like(self.age)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.mean[ids] = 0.
        self.age[ids] = 0.
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, command_name, smoothing_time=.6, warmup_time=1.,
                 forward_tolerance=.04, deviation_scale=.08):
        if not all(math.isfinite(v) and v > 0. for v in (smoothing_time, deviation_scale)):
            raise ValueError("Workspace smoothing and scale must be finite and positive")
        if not all(math.isfinite(v) and v >= 0. for v in (warmup_time, forward_tolerance)):
            raise ValueError("Workspace warmup and tolerance must be finite and nonnegative")
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        data = env.scene["robot"].data
        command = env.command_manager.get_command(command_name)
        feet = data.site_pos_w[:, self.feet]
        hips = data.body_link_pos_w[:, self.hips]
        quat = data.root_link_quat_w
        valid = (torch.isfinite(feet).all(dim=(1, 2)) & torch.isfinite(hips).all(dim=(1, 2))
                 & torch.isfinite(quat).all(dim=1) & torch.isfinite(command).all(dim=1))
        active = valid & (command[:, 0] > .08)
        terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
        if terminated is not None:
            active &= ~terminated
        local = quat_apply_inverse(
            quat.nan_to_num()[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
            (feet - hips).nan_to_num().reshape(-1, 3),
        ).reshape(env.num_envs, 2, 3)
        alpha = 1. - math.exp(-env.step_dt / smoothing_time)
        mean = torch.where(active[:, None], self.mean + alpha * (local[:, :, 0] - self.mean), 0.)
        age = torch.where(active, self.age + env.step_dt, 0.)
        self.mean.copy_(torch.where(fresh[:, None], mean, self.mean))
        self.age.copy_(torch.where(fresh, age, self.age))
        excess = ((self.mean - forward_tolerance).clamp_min(0.) / deviation_scale).square().clamp_max(4.)
        cost = excess.mean(dim=1) * active * (self.age >= warmup_time)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        for index, leg in enumerate(("fl", "fr")):
            env.extras["log"][f"Metrics/high_step_front_foot_mean_x_{leg}_m"] = self.mean[:, index].detach().clone()
        return self.cached
