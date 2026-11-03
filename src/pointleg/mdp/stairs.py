"""Stair-specific metrics and hip alignment constraints."""

import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from ..robot_cfg import LEG_NAMES
from ..terrains.elevation import STAIR_TREAD_COUNT
from .elevation import elevation_foot_bottom, elevation_state, stairs_exit_supported_feet, update_elevation_state
from .rewards import hip_abduction_excess


def stairs_task_metric(env, key):
    """Expose stair coverage, hard-clearance failures and loaded destination feet."""
    state = elevation_state(env)
    if key == "exit_loaded_feet":
        return stairs_exit_supported_feet(env).sum(dim=1).float()
    if key == "covered_treads":
        return state["covered"].sum(dim=1).float()
    if key == "clearance_failed":
        return state["stairs_clearance_failed"].float()
    if key in ("up_fraction", "down_fraction"):
        kind = 0 if key == "up_fraction" else 1
        return (env.scene.terrain.terrain_types == kind).float()
    raise KeyError(key)


def stairs_hip_abduction_zero(env, asset_cfg, tolerance=0., deviation_scale=.25):
    """Keep all abduction joints at their zero-degree default during stair traversal."""
    cost = hip_abduction_excess(env, asset_cfg, tolerance, deviation_scale)
    robot = env.scene[asset_cfg.name]
    for leg in LEG_NAMES:
        index = robot.joint_names.index(f"{leg}_hip_abduction_joint")
        angle = torch.rad2deg(robot.data.joint_pos[:, index])
        env.extras["log"][f"Metrics/stairs_hip_abduction_{leg}_deg"] = angle.detach()
    return cost


class StairsThighAlignmentPenalty:
    """Discourage near-vertical thighs relative to the body while climbing."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.hips, self.knees = [], []
        for leg in LEG_NAMES:
            for suffix, ids in (("hip_pitch_Link", self.hips), ("knee_Link", self.knees)):
                found, _ = robot.find_bodies(f"{leg}_{suffix}")
                if len(found) != 1:
                    raise ValueError(f"Expected one {leg}_{suffix} body")
                ids.append(found[0])

    def __call__(self, env, tolerance_degrees=45., scale_degrees=30., approach_margin=.4):
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        vector = robot.body_link_pos_w[:, self.knees] - robot.body_link_pos_w[:, self.hips]
        quat = robot.root_link_quat_w
        finite = torch.isfinite(vector).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=1)
        local = quat_apply_inverse(quat.nan_to_num()[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
                                   vector.nan_to_num().reshape(-1, 3)).reshape(env.num_envs, 4, 3)
        angle = torch.atan2(local[..., 2].abs(), local[..., 0].abs().clamp_min(1.e-6))
        local_x = robot.root_link_pos_w[:, 0] - state["corner"][:, 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"] & finite
                  & (env.scene.terrain.terrain_types == 0)
                  & (local_x >= state["entry_x"] - approach_margin)
                  & (local_x <= state["exit_x"] + approach_margin))
        excess = ((angle - math.radians(tolerance_degrees)).clamp_min(0.)
                  / math.radians(scale_degrees)).square().clamp_max(1.)
        for index, leg in enumerate(LEG_NAMES):
            if active.any():
                env.extras["log"][f"Metrics/stairs_thigh_body_angle_{leg}_deg"] = torch.rad2deg(angle[active, index]).detach().clone()
        return torch.where(active, excess.mean(dim=1), 0.)


class StairsSwingLiftPenalty:
    """Penalize insufficient foot lift from the last loaded position."""

    def __init__(self, cfg, env):
        self.foot_ids = env.scene["robot"].find_sites(tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True)[0]
        shape = (env.num_envs, 4)
        self.support_z = torch.zeros(shape, device=env.device)
        self.peak = torch.zeros_like(self.support_z)
        self.air_time = torch.zeros_like(self.support_z)
        self.loaded = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.anchored = torch.zeros_like(self.loaded)
        self.swinging = torch.zeros_like(self.loaded)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("support_z", "peak", "air_time", "loaded", "anchored", "swinging", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., minimum_air_time=.04,
                 grace_time=.08, event_seconds=.10):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        sensor = env.scene["elevation_contacts"].data
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        loaded = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                  & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        z = elevation_foot_bottom(env, self.foot_ids)[..., 2]
        command = env.command_manager.get_command("twist")[:, 0]
        target_lift = state["top_height"] / STAIR_TREAD_COUNT
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & torch.isfinite(z).all(dim=1) & torch.isfinite(command) & (command > .08)
                  & torch.isfinite(target_lift) & (target_lift > 0.))
        mask = fresh & active
        takeoff = mask[:, None] & self.loaded & ~loaded & self.anchored
        landing = mask[:, None] & loaded & self.swinging
        peak = torch.where(takeoff, (z - self.support_z).clamp_min(0.), self.peak)
        observing = mask[:, None] & (self.swinging | takeoff)
        peak = torch.where(observing, torch.maximum(peak, (z - self.support_z).clamp_min(0.)), peak)
        air_time = torch.where(takeoff, env.step_dt, torch.where(observing & ~loaded, self.air_time + env.step_dt, self.air_time))
        target = target_lift.nan_to_num(nan=0., posinf=0., neginf=0.).clamp_min(1.e-6)[:, None]
        deficit = ((target - peak) / target).clamp(0., 1.).square()
        event = landing & (air_time >= minimum_air_time - 1.e-6)
        ongoing = observing & ~loaded & (air_time >= grace_time)
        cost = (deficit * ongoing).mean(dim=1) + (deficit * event).mean(dim=1) * event_seconds / env.step_dt
        for index, leg in enumerate(LEG_NAMES):
            valid = event[:, index]
            if valid.any():
                env.extras["log"][f"Metrics/stairs_swing_lift_{leg}_m"] = peak[valid, index].detach().clone()
        self.support_z.copy_(torch.where(mask[:, None] & loaded, z, self.support_z))
        self.peak.copy_(torch.where(mask[:, None], torch.where(landing, 0., peak), self.peak))
        self.air_time.copy_(torch.where(mask[:, None], torch.where(landing, 0., air_time), self.air_time))
        self.swinging.copy_(torch.where(mask[:, None], (self.swinging | takeoff) & ~landing, self.swinging))
        self.anchored |= mask[:, None] & loaded
        self.loaded.copy_(torch.where(mask[:, None], loaded, self.loaded))
        self.cached.copy_(torch.where(fresh, torch.where(active, cost, 0.), self.cached))
        self.last_step[fresh] = env.common_step_counter
        if active.any():
            env.extras["log"]["Metrics/stairs_target_lift_m"] = target_lift[active].detach().clone()
        return self.cached


class StairsFootPlacement:
    """Score each foot for confirmed support inside a tread."""

    def __init__(self, cfg, env):
        self.age = torch.zeros(env.num_envs, 4, STAIR_TREAD_COUNT, device=env.device)
        self.paid = torch.zeros_like(self.age, dtype=torch.bool)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.age[ids] = 0.
        self.paid[ids] = False
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., confirmation=.10, target_margin=.08):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        pos = sensor.pos.reshape(env.num_envs, 4, -1, 3) - state["corner"][:, None, None, :]
        valid = torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1) & torch.isfinite(pos).all(dim=-1)
        support = found & valid & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)
        tread = state["step_surfaces"]
        x = pos[..., 0, None]
        left, right, height = (tread[:, None, None, :, i] for i in range(3))
        hit = (support[..., None] & (x >= left) & (x <= right)
               & ((pos[..., 2, None] - height).abs() <= env._elevation_settings.top_contact_tolerance))
        margin = torch.minimum(x - left, right - x).clamp_min(0.)
        quality = torch.where(hit, (margin / target_margin).clamp(0., 1.), 0.).amax(dim=2)
        touched = hit.any(dim=2)
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
        mask = fresh & active
        age = torch.where(touched, self.age + env.step_dt, 0.)
        self.age.copy_(torch.where(mask[:, None, None], age, self.age))
        ready = (self.age >= confirmation - 1.e-6) & ~self.paid & mask[:, None, None]
        reward = (quality.square() * ready).sum(dim=(1, 2)) / env.step_dt
        self.paid |= ready
        self.cached.copy_(torch.where(fresh, reward, self.cached))
        self.last_step[fresh] = env.common_step_counter
        return self.cached


class StairsFourFootStandstill:
    """Penalize sustained four-foot support without meaningful forward progress."""

    def __init__(self, cfg, env):
        self.anchor = torch.zeros(env.num_envs, device=env.device)
        self.age = torch.zeros_like(self.anchor)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.cached = torch.zeros_like(self.anchor)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.anchor[ids] = self.age[ids] = self.cached[ids] = 0.
        self.initialized[ids] = False
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., grace_period=1.0, rise_time=.5, minimum_progress=.02):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        sensor = env.scene["elevation_contacts"].data
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        loaded = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                  & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        x = env.scene["robot"].data.root_link_pos_w[:, 0]
        command = env.command_manager.get_command("twist")[:, 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & torch.isfinite(x) & torch.isfinite(command) & (command > .08))
        standing = active & loaded.all(dim=1)
        start = standing & ~self.initialized
        self.anchor.copy_(torch.where(fresh & start, x, self.anchor))
        advancing = standing & ((x - self.anchor) >= minimum_progress)
        self.anchor.copy_(torch.where(fresh & advancing, x, self.anchor))
        age = torch.where(standing & ~advancing, self.age + env.step_dt, 0.)
        self.age.copy_(torch.where(fresh, age, self.age))
        self.initialized.copy_(torch.where(fresh, standing, self.initialized))
        cost = ((self.age - grace_period) / rise_time).clamp(0., 1.)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        env.extras["log"]["Metrics/stairs_four_foot_standstill_s"] = self.age.detach().mean()
        return self.cached


class StairsProgressReward:
    """Reward new body progress, loaded propulsion and stable tread contacts."""

    def __init__(self, cfg, env):
        self.foot_ids = env.scene["robot"].find_sites(tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True)[0]
        self.loaded = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self.foot_start = torch.zeros(env.num_envs, 4, device=env.device)
        self.root_start = torch.zeros_like(self.foot_start)
        self.distance = torch.zeros_like(self.foot_start)
        self.stance = torch.zeros_like(self.foot_start)
        self.slip = torch.zeros_like(self.foot_start)
        self.tread_time = torch.zeros(env.num_envs, 6, device=env.device)
        self.tread_paid = torch.zeros(env.num_envs, 6, dtype=torch.bool, device=env.device)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("loaded", "foot_start", "root_start", "distance", "stance", "slip", "tread_time", "tread_paid", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, target_distance=.07, minimum_stance=.16, target_stance=.30,
                 max_slip=.025, minimum_force=8., tread_hold=.10, edge_margin=.03):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        pos = sensor.pos.reshape(env.num_envs, 4, -1, 3)
        valid = torch.isfinite(force).all(dim=-1) & torch.isfinite(pos).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
        support = found & valid & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)
        loaded = support.any(dim=2)
        root = robot.root_link_pos_w[:, 0, None]
        foot = robot.site_pos_w[:, self.foot_ids, 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & torch.isfinite(root).all(dim=1) & torch.isfinite(foot).all(dim=1))
        mask = fresh & active
        landing = loaded & ~self.loaded & mask[:, None]
        takeoff = ~loaded & self.loaded & mask[:, None]
        self.root_start = torch.where(landing, root, self.root_start)
        self.foot_start = torch.where(landing, foot, self.foot_start)
        travel = (root - self.root_start).clamp_min(0.)
        sweep = (root - self.root_start - foot + self.foot_start).clamp_min(0.)
        distance = torch.minimum(travel, sweep)
        self.distance = torch.where(loaded & mask[:, None], distance, self.distance)
        slip = (foot - self.foot_start).abs()
        self.slip = torch.where(landing, 0., torch.where(loaded & mask[:, None], torch.maximum(self.slip, slip), self.slip))
        self.stance = torch.where(mask[:, None], torch.where(loaded, torch.where(landing, env.step_dt, self.stance + env.step_dt), self.stance), self.stance)
        quality = (self.distance / target_distance).clamp(0., 1.).square() * (self.stance / target_stance).clamp(0., 1.) * (1. - self.slip / max_slip).clamp(0., 1.)
        propulsion = (takeoff * (self.stance >= minimum_stance) * quality).sum(dim=1)
        local = pos - state["corner"][:, None, None, :]
        tread = state["step_surfaces"]
        hit = (support[..., None] & (local[..., 0, None] >= tread[:, None, None, :, 0] + edge_margin)
               & (local[..., 0, None] <= tread[:, None, None, :, 1] - edge_margin)
               & ((local[..., 2, None] - tread[:, None, None, :, 2]).abs() <= env._elevation_settings.top_contact_tolerance)).any(dim=(1, 2))
        self.tread_time = torch.where(mask[:, None], torch.where(hit, self.tread_time + env.step_dt, 0.), self.tread_time)
        new = (self.tread_time >= tread_hold - 1.e-6) & ~self.tread_paid & mask[:, None]
        self.tread_paid |= new
        command = env.command_manager.get_command("twist")[:, 0].clamp_min(1.e-6)
        progress = (state["forward_progress"] / (command * env.step_dt)).clamp(0., 1.)
        reward = 2. * progress + (.3 * propulsion + new.sum(dim=1).float()) / env.step_dt
        self.cached = torch.where(fresh, torch.where(active, reward, 0.), self.cached)
        self.loaded = torch.where(mask[:, None], loaded, self.loaded)
        self.last_step[fresh] = env.common_step_counter
        return self.cached
