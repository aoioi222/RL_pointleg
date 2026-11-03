"""Reward coordinated hip and knee motion during completed forward steps."""

import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse
from pointleg.robot_cfg import LEG_NAMES
from .walking_gait import loaded_feet, ordered_contact_ids


class HighStepHipSwingReward:
    """Score hip range and knee movement once after an eligible stable landing."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.contacts = ordered_contact_ids(env.scene[cfg.params["sensor_name"]])
        self.joints = [robot.joint_names.index(f"{leg}_hip_pitch_joint") for leg in LEG_NAMES]
        self.hips, self.knees = [], []
        for leg in LEG_NAMES:
            for name, result in ((f"{leg}_hip_pitch_Link", self.hips),
                                 (f"{leg}_knee_Link", self.knees)):
                ids, _ = robot.find_bodies(name)
                if len(ids) != 1:
                    raise ValueError(f"Expected one body for {name}")
                result.append(ids[0])
        shape = (env.num_envs, 4)
        for name in ("hip_min", "hip_max", "knee_min", "knee_max", "air_time", "body_progress"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        self.root_start = torch.zeros((*shape, 2), device=env.device)
        self.direction = torch.zeros_like(self.root_start)
        self.tracking = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.landed = torch.zeros_like(self.tracking)
        self.supported = torch.zeros_like(self.tracking)
        self.cost = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("hip_min", "hip_max", "knee_min", "knee_max", "air_time",
                     "body_progress", "root_start", "direction", "tracking", "landed",
                     "supported", "cost"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, sensor_name, minimum_support_force=8.,
                 target_hip_range_degrees=20., target_knee_range=.03,
                 min_body_progress=.01):
        if not all(math.isfinite(x) and x > 0 for x in
                   (minimum_support_force, target_hip_range_degrees,
                    target_knee_range, min_body_progress)):
            raise ValueError("Swing coordination targets must be positive and finite")
        basic = env._high_step_basic_term
        basic(env, **basic.params)
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cost
        robot = env.scene["robot"].data
        q = robot.joint_pos[:, self.joints]
        hip = robot.body_link_pos_w[:, self.hips]
        knee = robot.body_link_pos_w[:, self.knees]
        quat = robot.root_link_quat_w
        finite = (torch.isfinite(q).all(dim=1) & torch.isfinite(hip).all(dim=(1, 2))
                  & torch.isfinite(knee).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=1)
                  & torch.isfinite(robot.root_link_pos_w).all(dim=1))
        active = basic.active if hasattr(basic, "active") else (
            (env.command_manager.get_command("twist")[:, 0] > .08) & finite
        )
        termination = getattr(getattr(env, "termination_manager", None), "terminated", None)
        if termination is not None:
            active &= ~termination
        self.reset(torch.where(fresh & ~(active & finite))[0])
        quat = quat.nan_to_num()
        local_knee = quat_apply_inverse(
            quat[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
            (knee - hip).nan_to_num().reshape(-1, 3),
        ).reshape(env.num_envs, 4, 3)[:, :, 2]
        q = q.nan_to_num()
        loaded = loaded_feet(env.scene[sensor_name], self.contacts, minimum_support_force)
        active_swing = basic.swinging & basic.eligible_swing & active[:, None] & finite[:, None]
        start = active_swing & ~self.tracking & fresh[:, None]
        for name, value in (("hip_min", q), ("hip_max", q),
                            ("knee_min", local_knee), ("knee_max", local_knee)):
            state = getattr(self, name)
            state.copy_(torch.where(start, value, state))
        self.root_start.copy_(torch.where(
            start[..., None], robot.root_link_pos_w[:, None, :2].nan_to_num(), self.root_start,
        ))
        w, x, y, z = quat.unbind(dim=1)
        direction = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), dim=1)
        direction /= direction.norm(dim=1, keepdim=True).clamp_min(1.e-6)
        self.direction.copy_(torch.where(start[..., None], direction[:, None], self.direction))
        self.tracking |= start
        self.landed &= ~start
        self.supported.copy_(torch.where(start, True, self.supported))
        observing = self.tracking & active_swing & fresh[:, None]
        self.supported &= ~observing | (loaded.sum(dim=1, keepdim=True) >= 2)
        for name, value, op in (("hip_min", q, torch.minimum), ("hip_max", q, torch.maximum),
                                ("knee_min", local_knee, torch.minimum), ("knee_max", local_knee, torch.maximum)):
            state = getattr(self, name)
            state.copy_(torch.where(observing, op(state, value), state))
        self.air_time.copy_(torch.where(observing, basic.air, self.air_time))
        landed = self.tracking & ~basic.swinging & ~self.landed & fresh[:, None]
        distance = ((robot.root_link_pos_w[:, None, :2].nan_to_num() - self.root_start)
                    * self.direction).sum(dim=-1)
        self.body_progress.copy_(torch.where(landed, distance, self.body_progress))
        self.landed |= landed
        mature = (basic.mature_height & self.tracking & self.landed & self.supported
                  & (self.air_time >= basic.params["min_air_time"] - 1.e-6)
                  & (self.air_time <= basic.params["max_air_time"] + 1.e-6)
                  & (self.body_progress >= min_body_progress) & active[:, None] & finite[:, None]
                  & fresh[:, None])
        hip_range = (self.hip_max - self.hip_min).clamp_min(0.)
        knee_range = (self.knee_max - self.knee_min).clamp_min(0.)
        hip_score = (hip_range / math.radians(target_hip_range_degrees)).clamp(0., 1.)
        knee_score = (knee_range / target_knee_range).clamp(0., 1.)
        reward = (.5 * (hip_score + knee_score) * mature
                  * basic.swing_distance_credit).sum(dim=1) / env.step_dt
        self.cost.copy_(torch.where(fresh, reward, self.cost))
        for index, leg in enumerate(LEG_NAMES):
            mask = mature[:, index]
            if mask.any():
                env.extras["log"][f"Metrics/high_step_swing_hip_range_{leg}_deg"] = torch.rad2deg(
                    hip_range[mask, index]).detach().clone()
                env.extras["log"][f"Metrics/high_step_swing_knee_vertical_range_{leg}_m"] = (
                    knee_range[mask, index].detach().clone())
        self.tracking &= basic.swinging | basic.height_pending
        self.last_step[fresh] = env.common_step_counter
        return self.cost
