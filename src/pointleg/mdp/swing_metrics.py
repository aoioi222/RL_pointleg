"""Measure completed foot swings without adding a training objective."""

import math

import torch

from .obstacle_recovery import FOOT_CENTER_SITES
from ..robot_cfg import FOOT_BODY_NAMES, LEG_NAMES


class FootSwingPeakHeightMetric:
    """Log event-weighted swing peaks relative to the last support surface."""

    def __init__(self, cfg, env):
        for name in ("foot_radius", "minimum_air_time", "minimum_support_force"):
            if not math.isfinite(cfg.params[name]) or cfg.params[name] <= 0.:
                raise ValueError("Swing metric distances, times and forces must be positive.")
        self.contact_ids = []
        self.site_ids = []
        sensor = env.scene[cfg.params["sensor_name"]]
        for body, site in zip(FOOT_BODY_NAMES, FOOT_CENTER_SITES):
            contacts = [index for index, name in enumerate(sensor.primary_names) if name.endswith(body)]
            sites, _ = env.scene["robot"].find_sites(site)
            if len(contacts) != 1 or len(sites) != 1:
                raise ValueError(f"Expected one support channel and center site for {body}.")
            self.contact_ids.append(contacts[0])
            self.site_ids.append(sites[0])
        shape = (env.num_envs, 4)
        for name in ("support_z", "support_relative_z", "peak", "relative_peak", "air_time"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        self.anchored = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.swinging = torch.zeros_like(self.anchored)
        self.peak_sum = torch.zeros(env.num_envs, device=env.device)
        self.swing_count = torch.zeros_like(self.peak_sum)
        self.completed_mean = torch.zeros_like(self.peak_sum)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            "support_z", "support_relative_z", "peak", "relative_peak", "air_time",
            "anchored", "swinging", "peak_sum", "swing_count", "completed_mean",
        ):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, sensor_name, foot_radius, minimum_air_time, minimum_support_force):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.completed_mean
        robot = env.scene["robot"].data
        center = robot.site_pos_w[:, self.site_ids]
        bottom = center[:, :, 2] - foot_radius
        relative_z = bottom - robot.root_link_pos_w[:, 2, None]
        contact = env.scene[sensor_name].data
        forces = contact.force[:, self.contact_ids]
        finite = (
            torch.isfinite(center).all(dim=-1) & torch.isfinite(forces).all(dim=-1)
            & torch.isfinite(robot.root_link_pos_w).all(dim=-1)[:, None]
        )
        vertical = forces[:, :, 2].abs()
        horizontal = torch.linalg.vector_norm(forces[:, :, :2], dim=-1)
        supported = (
            finite & (contact.found[:, self.contact_ids] > 0)
            & (vertical >= minimum_support_force) & (vertical >= horizontal)
        )
        airborne = finite & ~supported & self.anchored
        observed = finite & self.anchored & (airborne | self.swinging)
        peak = torch.where(observed, torch.maximum(self.peak, (bottom - self.support_z).clamp_min(0.)), 0.)
        relative_peak = torch.where(
            observed, torch.maximum(self.relative_peak, (relative_z - self.support_relative_z).clamp_min(0.)), 0.,
        )
        air_time = self.air_time + airborne.float() * env.step_dt
        landed = fresh[:, None] & supported & self.swinging & (air_time >= minimum_air_time - 1.e-6)
        log = env.extras["log"]
        if landed.any():
            log["Metrics/swing_peak_height"] = peak[landed].detach().clone()
            log["Metrics/swing_peak_height_relative_body"] = relative_peak[landed].detach().clone()
            for index, leg in enumerate(LEG_NAMES):
                mask = landed[:, index]
                if mask.any():
                    log[f"Metrics/swing_peak_height_{leg}"] = peak[mask, index].detach().clone()
        peak_sum = self.peak_sum + torch.where(landed, peak, 0.).sum(dim=1)
        swing_count = self.swing_count + landed.sum(dim=1)
        values = {
            "support_z": torch.where(supported, bottom, self.support_z),
            "support_relative_z": torch.where(supported, relative_z, self.support_relative_z),
            "anchored": finite & (self.anchored | supported),
            "swinging": airborne,
            "peak": torch.where(airborne, peak, 0.),
            "relative_peak": torch.where(airborne, relative_peak, 0.),
            "air_time": torch.where(airborne, air_time, 0.),
        }
        for name, value in values.items():
            state = getattr(self, name)
            state.copy_(torch.where(fresh[:, None], value, state))
        self.peak_sum.copy_(torch.where(fresh, peak_sum, self.peak_sum))
        self.swing_count.copy_(torch.where(fresh, swing_count, self.swing_count))
        self.completed_mean.copy_(self.peak_sum / self.swing_count.clamp_min(1.))
        self.last_step[fresh] = env.common_step_counter
        return self.completed_mean
