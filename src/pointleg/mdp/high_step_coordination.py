"""Compact swing geometry and contact-based hip/knee motion diagnostics."""

import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from pointleg.robot_cfg import LEG_NAMES
from .height import root_ground_clearance
from .rewards import joint_pos_penalty
from .walking_gait import loaded_feet, ordered_contact_ids


def high_step_static_joint_posture(
    env, command_name: str, stand_still_scale: float,
    velocity_threshold: float, command_threshold: float, asset_cfg=None,
):
    """Preserve the standing posture cost only while commanded to stand."""
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1)
                + command[:, 2].abs()) <= command_threshold
    cost = joint_pos_penalty(
        env, command_name, stand_still_scale, velocity_threshold,
        command_threshold, asset_cfg,
    )
    return torch.where(standing, cost, 0.)


class HighStepSwingRetractionPenalty:
    """Limit high-swing overreach without prescribing joint angles or phases."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.contact_ids = ordered_contact_ids(env.scene[cfg.params["sensor_name"]])
        self.site_ids, self.hip_ids, self.knee_ids = [], [], []
        self.joint_ids = []
        for leg in LEG_NAMES:
            for finder, name, ids in (
                (robot.find_sites, f"{leg}_foot_contact_site", self.site_ids),
                (robot.find_bodies, f"{leg}_hip_pitch_Link", self.hip_ids),
                (robot.find_bodies, f"{leg}_knee_Link", self.knee_ids),
            ):
                matched, _ = finder(name)
                if len(matched) != 1:
                    raise ValueError(f"Expected one swing geometry reference for {name}")
                ids.append(matched[0])
            self.joint_ids.extend(robot.joint_names.index(f"{leg}_{joint}_joint")
                                  for joint in ("hip_pitch", "knee"))
        shape = (env.num_envs, 4)
        self.joint_min = torch.zeros((*shape, 2), device=env.device)
        self.joint_max = torch.zeros_like(self.joint_min)
        self.knee_min = torch.zeros(shape, device=env.device)
        self.knee_max = torch.zeros_like(self.knee_min)
        self.anchored = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.tracking = torch.zeros_like(self.anchored)
        self.supported = torch.zeros_like(self.anchored)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("joint_min", "joint_max", "knee_min", "knee_max",
                     "anchored", "tracking", "supported"):
            getattr(self, name)[ids] = 0

    def __call__(
        self, env, sensor_name: str, command_name: str,
        ground_sensor_name: str = "base_height_sensor", foot_radius: float = .025,
        activation_height: float = .06, nominal_forward: float = .035,
        fore_aft_margin: float = .08, deviation_scale: float = .08,
        min_stance_time: float = .12, min_air_time: float = .08,
        minimum_support_force: float = 5., startup_grace: float = .6,
    ):
        positive = (foot_radius, activation_height, fore_aft_margin, deviation_scale,
                    min_stance_time, min_air_time, minimum_support_force)
        if not all(math.isfinite(v) and v > 0. for v in positive):
            raise ValueError("Swing geometry distances, times and force must be positive")
        if not math.isfinite(nominal_forward):
            raise ValueError("Nominal swing position must be finite")
        if not math.isfinite(startup_grace) or startup_grace < 0.:
            raise ValueError("Swing startup grace must be nonnegative")
        robot, sensor = env.scene["robot"].data, env.scene[sensor_name]
        data = sensor.data
        for name in ("last_contact_time", "last_air_time"):
            if getattr(data, name) is None:
                raise RuntimeError("Swing coordination requires contact time tracking")
        foot = robot.site_pos_w[:, self.site_ids]
        hip = robot.body_link_pos_w[:, self.hip_ids]
        knee = robot.body_link_pos_w[:, self.knee_ids]
        q = robot.joint_pos[:, self.joint_ids].reshape(env.num_envs, 4, 2)
        quat = robot.root_link_quat_w[:, None, :].expand(-1, 4, -1)
        finite = (torch.isfinite(foot).all(dim=-1) & torch.isfinite(hip).all(dim=-1)
                  & torch.isfinite(knee).all(dim=-1) & torch.isfinite(q).all(dim=-1)
                  & torch.isfinite(quat).all(dim=-1))
        quat = torch.nan_to_num(quat).reshape(-1, 4)
        local_foot = quat_apply_inverse(
            quat, torch.nan_to_num(foot - hip).reshape(-1, 3),
        ).reshape(env.num_envs, 4, 3)
        local_knee = quat_apply_inverse(
            quat, torch.nan_to_num(knee - hip).reshape(-1, 3),
        ).reshape(env.num_envs, 4, 3)
        q = torch.nan_to_num(q)
        clearance, valid_ground = root_ground_clearance(env, sensor_name=ground_sensor_name)
        ground_z = robot.root_link_pos_w[:, 2] - clearance
        height = torch.nan_to_num(foot[:, :, 2] - foot_radius - ground_z[:, None])
        command = env.command_manager.get_command(command_name)
        active = ((command[:, 0] > .08) & (command[:, 1].abs() < .05)
                  & (command[:, 2].abs() < .15) & valid_ground)
        eligible = active[:, None] & finite
        contact = data.found.reshape(env.num_envs, len(sensor.primary_names), -1)
        contact = (contact > 0).any(dim=2)[:, self.contact_ids]
        loaded = loaded_feet(sensor, self.contact_ids, minimum_support_force) & finite
        enough = loaded.sum(dim=1) >= 2
        takeoff = (sensor.compute_first_air(env.step_dt)[:, self.contact_ids]
                   & self.anchored & eligible
                   & (data.last_contact_time[:, self.contact_ids] >= min_stance_time))
        self.tracking = (self.tracking | takeoff) & eligible
        self.supported = torch.where(takeoff, True, self.supported)
        self.supported &= enough[:, None] & eligible
        knee_z = local_knee[:, :, 2]
        for name, value in (("joint_min", q), ("joint_max", q),
                            ("knee_min", knee_z), ("knee_max", knee_z)):
            mask = takeoff[:, :, None] if value.ndim == 3 else takeoff
            setattr(self, name, torch.where(mask, value, getattr(self, name)))
        airborne = ~contact & self.tracking
        landed = sensor.compute_first_contact(env.step_dt)[:, self.contact_ids] & self.tracking
        observe = airborne | landed
        for prefix, value in (("joint", q), ("knee", knee_z)):
            mask = observe[:, :, None] if value.ndim == 3 else observe
            low, high = getattr(self, f"{prefix}_min"), getattr(self, f"{prefix}_max")
            setattr(self, f"{prefix}_min", torch.where(mask, torch.minimum(low, value), low))
            setattr(self, f"{prefix}_max", torch.where(mask, torch.maximum(high, value), high))
        completed = (landed & self.supported & loaded
                     & (data.last_air_time[:, self.contact_ids] >= min_air_time))
        log = env.extras["log"]
        ranges = torch.rad2deg(self.joint_max - self.joint_min)
        for i, leg in enumerate(LEG_NAMES):
            valid = completed[:, i]
            if valid.any():
                for j, joint in enumerate(("hip", "knee")):
                    log[f"Metrics/high_step_swing_{joint}_range_{leg}_deg"] = (
                        ranges[valid, i, j].detach().clone()
                    )
                log[f"Metrics/high_step_swing_knee_vertical_range_{leg}_m"] = (
                    (self.knee_max - self.knee_min)[valid, i].detach().clone()
                )
        compact = (airborne & self.supported & eligible & (height >= activation_height)
                   & (env.episode_length_buf[:, None] * env.step_dt >= startup_grace))
        excess = ((local_foot[:, :, 0] - nominal_forward).abs() - fore_aft_margin).clamp_min(0.)
        cost = torch.where(compact, (excess / deviation_scale).square().clamp_max(4.), 0.)
        for i, leg in enumerate(LEG_NAMES):
            if compact[:, i].any():
                log[f"Metrics/high_step_swing_foot_x_{leg}_m"] = (
                    local_foot[compact[:, i], i, 0].detach().clone()
                )
        self.tracking &= ~contact
        self.anchored = torch.where(contact, loaded & eligible, self.anchored & eligible)
        return .5 * cost.mean(dim=1) + .5 * cost.max(dim=1).values
