"""Loaded support and gait renewal for the flat-ground correction stage."""

import math

import torch

from pointleg.robot_cfg import LEG_NAMES
from .height import root_ground_clearance
from .rewards import track_linear_velocity
from .walking_gait import loaded_feet, ordered_contact_ids


def high_step_velocity_reward(env, std, command_name):
    """Keep standing tracking but require new ground and leg renewal when walking."""
    support = env._high_step_support_term
    support(env, **support.params)
    reward = track_linear_velocity(env, std=std, command_name=command_name)
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    ratio = (velocity.nan_to_num() / command[:, 0].clamp_min(1.e-6)).clamp(0., 1.)
    walking = command[:, 0] > .08
    gain = support.forward_credit * support.renewal * ratio
    allowed = (support.active & (support.loaded.sum(dim=1) >= 2)
               & torch.isfinite(velocity) & torch.isfinite(reward))
    forward = torch.where(allowed, reward * gain, 0.)
    result = torch.where(walking, forward, reward).nan_to_num(nan=0., posinf=0., neginf=0.)
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    return torch.where(terminated, 0., result) if terminated is not None else result


def high_step_contact_slip_penalty(env):
    """Read the same loaded-contact slip used by support quality."""
    support = env._high_step_support_term
    support(env, **support.params)
    return torch.where(support.active, support.slip_cost, 0.)


def high_step_short_stance_penalty(env, sensor_name, command_name, min_stance_time,
                                   command_threshold=.1):
    """Charge only completed loaded stances, leaving the cycle gate unchanged."""
    if not math.isfinite(min_stance_time) or min_stance_time <= 0.:
        raise ValueError("Short stance target must be finite and positive")
    support = env._high_step_support_term
    support(env, **support.params)
    command = env.command_manager.get_command(command_name)
    active = support.active & (command[:, :2].norm(dim=1) + command[:, 2].abs() > command_threshold)
    deficit = (1. - support.ended_duration / min_stance_time).clamp_min(0.).square()
    return torch.where(active, (deficit * support.takeoffs).sum(dim=1) / env.step_dt, 0.)


def contact_point_speeds(env, sensor_name, site_ids, contact_ids, minimum_force):
    """Return rolling-aware tangential speed and finite loaded-foot validity."""
    sensor = env.scene[sensor_name]
    if not (sensor.cfg.global_frame or sensor.cfg.reduce == "netforce"):
        raise ValueError("HighStep contact slip requires world-frame forces")
    data, robot = sensor.data, env.scene["robot"].data
    if data.pos is None or data.normal is None:
        raise RuntimeError("HighStep slip requires world contact positions and normals")
    count = len(sensor.primary_names)
    positions = data.pos.reshape(env.num_envs, count, -1, 3)[:, contact_ids]
    normals = data.normal.reshape(env.num_envs, count, -1, 3)[:, contact_ids]
    force = data.force.reshape(env.num_envs, count, -1, 3)[:, contact_ids]
    found = data.found.reshape(env.num_envs, count, -1)[:, contact_ids] > 0
    center = robot.site_pos_w[:, site_ids]
    velocity = robot.site_lin_vel_w[:, site_ids]
    angular = robot.site_ang_vel_w[:, site_ids]
    finite_site = (torch.isfinite(center).all(dim=-1) & torch.isfinite(velocity).all(dim=-1)
                   & torch.isfinite(angular).all(dim=-1))
    finite_slot = (torch.isfinite(positions).all(dim=-1) & torch.isfinite(normals).all(dim=-1)
                   & torch.isfinite(force).all(dim=-1))
    safe_normal = normals.nan_to_num(nan=0., posinf=0., neginf=0.)
    normal_norm = safe_normal.norm(dim=-1, keepdim=True)
    unit_normal = safe_normal / normal_norm.clamp_min(1.e-6)
    safe_force = force.nan_to_num(nan=0., posinf=0., neginf=0.)
    top = (found & finite_slot & (normal_norm[..., 0] > 1.e-6) & (unit_normal[..., 2] < -.5)
           & (safe_force[..., 2].abs() >= minimum_force)
           & (safe_force[..., 2].abs() >= safe_force[..., :2].norm(dim=-1)))
    offset = positions.nan_to_num(nan=0., posinf=0., neginf=0.) - center.nan_to_num()[:, :, None, :]
    point_velocity = velocity.nan_to_num()[:, :, None, :] + torch.linalg.cross(
        angular.nan_to_num()[:, :, None, :].expand_as(offset), offset, dim=-1,
    )
    tangent = point_velocity - (point_velocity * unit_normal).sum(dim=-1, keepdim=True) * unit_normal
    speed = torch.where(top, tangent.norm(dim=-1), 0.).max(dim=2).values
    valid = finite_site & ((~found) | finite_slot).all(dim=2) & torch.isfinite(speed)
    return speed.nan_to_num(nan=0., posinf=0., neginf=0.), valid


class HighStepSupportCorrection:
    """Measure stable loaded backstroke and retire credit when feet stop renewing."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.ids = ordered_contact_ids(env.scene[cfg.params["sensor_name"]])
        self.sites = []
        self.reference_sites = []
        for leg in LEG_NAMES:
            ids, _ = robot.find_sites(f"{leg}_foot_contact_site")
            if len(ids) != 1:
                raise ValueError("Support correction requires four sphere-center sites")
            self.sites.append(ids[0])
            ids, _ = robot.find_sites(f"{leg}_foot_site")
            if len(ids) != 1:
                raise ValueError("Reference-speed diagnostics require four sole sites")
            self.reference_sites.append(ids[0])
        self.params = dict(cfg.params)
        shape = (env.num_envs, 4)
        for name in ("duration", "travel", "drift", "swing_quality", "age", "ended_duration", "slip_speed"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        for name in ("foot_start", "root_start", "direction"):
            setattr(self, name, torch.zeros((*shape, 2), device=env.device))
        self.was_loaded = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.loaded = torch.zeros_like(self.was_loaded)
        self.takeoffs = torch.zeros_like(self.was_loaded)
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.was_active = torch.zeros_like(self.active)
        self.renewal = torch.ones(env.num_envs, device=env.device)
        self.credit_heading = torch.zeros(env.num_envs, 2, device=env.device)
        self.frontier = torch.zeros(env.num_envs, device=env.device)
        self.forward_credit = torch.zeros_like(self.frontier)
        self.cached_cost = torch.zeros_like(self.frontier)
        self.slip_cost = torch.zeros_like(self.frontier)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        env._high_step_support_term = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("duration", "travel", "drift", "swing_quality", "age", "foot_start",
                     "root_start", "direction", "was_loaded", "loaded", "active", "was_active",
                     "credit_heading", "frontier", "forward_credit", "cached_cost"):
            getattr(self, name)[ids] = 0
        for name in ("ended_duration", "slip_speed", "takeoffs", "slip_cost"):
            getattr(self, name)[ids] = 0
        self.renewal[ids] = 1.
        self.last_step[ids] = -1

    def _renewal(self):
        grace, timeout = self.params["renewal_grace"], self.params["renewal_timeout"]
        recent = ((timeout - self.age) / (timeout - grace)).clamp(0., 1.)
        self.renewal.copy_(.5 * recent.mean(dim=1) + .5 * recent.min(dim=1).values)

    def record_cycles(self, renewed):
        self.age[renewed & self.active[:, None]] = 0.
        self._renewal()

    def __call__(self, env, sensor_name, command_name, minimum_force, minimum_stance,
                 target_stance, target_backstroke, maximum_drift, maximum_stance,
                 overstay_scale, renewal_grace, renewal_timeout,
                 slip_grace=.04, slip_tolerance=.02, slip_scale=.10,
                 ground_sensor_name="base_height_sensor"):
        numbers = (minimum_force, minimum_stance, target_stance, target_backstroke,
                   maximum_drift, maximum_stance, overstay_scale, renewal_grace, renewal_timeout, slip_scale)
        if not all(math.isfinite(x) and x > 0. for x in numbers):
            raise ValueError("Support correction parameters must be finite and positive")
        if target_stance < minimum_stance or maximum_stance < target_stance or renewal_timeout <= renewal_grace:
            raise ValueError("Invalid stance or renewal intervals")
        if not all(math.isfinite(x) and x >= 0. for x in (slip_grace, slip_tolerance)):
            raise ValueError("Slip grace and tolerance must be finite and nonnegative")
        counter = getattr(env, "common_step_counter", None)
        fresh = torch.ones_like(self.active) if counter is None else self.last_step != counter
        if not fresh.any():
            return self.cached_cost
        robot = env.scene["robot"].data
        command = env.command_manager.get_command(command_name)
        feet = robot.site_pos_w[:, self.sites, :2]
        root = robot.root_link_pos_w[:, None, :2].expand_as(feet)
        quat = robot.root_link_quat_w
        _, ground_valid = root_ground_clearance(env, sensor_name=ground_sensor_name)
        speed, slip_valid = contact_point_speeds(env, sensor_name, self.sites, self.ids, minimum_force)
        finite = (torch.isfinite(robot.site_pos_w[:, self.sites]).all(dim=-1)
                  & torch.isfinite(robot.root_link_pos_w).all(dim=1)[:, None] & slip_valid
                  & torch.isfinite(quat).all(dim=1)[:, None])
        active = ((command[:, 0] > .08) & (command[:, 1].abs() < .05)
                  & (command[:, 2].abs() < .15) & ground_valid & finite.all(dim=1))
        terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
        if terminated is not None:
            active &= ~terminated
        loaded = loaded_feet(env.scene[sensor_name], self.ids, minimum_force) & finite & active[:, None]
        w, x, y, z = quat.nan_to_num().unbind(dim=1)
        heading = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), dim=1)
        heading /= heading.norm(dim=1, keepdim=True).clamp_min(1.e-6)
        feet, root = feet.nan_to_num(), root.nan_to_num()
        takeoff = self.was_loaded & ~loaded & active[:, None] & fresh[:, None]
        self.takeoffs.copy_(torch.where(fresh[:, None], takeoff, self.takeoffs))
        self.ended_duration.copy_(torch.where(takeoff, self.duration, self.ended_duration))
        quality = ((self.duration / target_stance).clamp(0., 1.)
                   * (self.travel / target_backstroke).clamp(0., 1.)
                   * (1. - self.drift / maximum_drift).clamp(0., 1.))
        quality = torch.where(self.duration >= minimum_stance, quality, 0.)
        self.swing_quality.copy_(torch.where(takeoff, quality, self.swing_quality))
        start = loaded & ~self.was_loaded & fresh[:, None]
        self.foot_start.copy_(torch.where(start[..., None], feet, self.foot_start))
        self.root_start.copy_(torch.where(start[..., None], root, self.root_start))
        self.direction.copy_(torch.where(start[..., None], heading[:, None, :], self.direction))
        foot_delta, root_delta = feet - self.foot_start, root - self.root_start
        body_travel = (root_delta * self.direction).sum(dim=-1)
        sweep = ((root_delta - foot_delta) * self.direction).sum(dim=-1)
        travel = torch.minimum(body_travel, sweep).clamp_min(0.)
        duration = torch.where(start, env.step_dt, self.duration + env.step_dt)
        self.duration.copy_(torch.where(loaded & fresh[:, None], duration, self.duration))
        self.travel.copy_(torch.where(loaded & fresh[:, None], travel, self.travel))
        settled = loaded & (self.duration > slip_grace + 1.e-6)
        drift = torch.where(start, 0., self.drift) + torch.where(settled, speed * env.step_dt, 0.)
        self.drift.copy_(torch.where(loaded & fresh[:, None], drift, self.drift))
        self.slip_speed.copy_(torch.where(fresh[:, None], torch.where(settled, speed, 0.), self.slip_speed))
        slip_cost = ((speed - slip_tolerance).clamp_min(0.) / slip_scale).square().clamp_max(1.)
        slip_cost = torch.where(settled, slip_cost, 0.)
        mean_slip_cost = slip_cost.sum(dim=1) / settled.sum(dim=1).clamp_min(1)
        self.slip_cost.copy_(torch.where(fresh, .5 * mean_slip_cost + .5 * slip_cost.max(dim=1).values, self.slip_cost))
        self.age.copy_(torch.where(fresh[:, None], torch.where(active[:, None], self.age + env.step_dt, 0.), self.age))
        self.active.copy_(torch.where(fresh, active, self.active))
        self.loaded.copy_(torch.where(fresh[:, None], loaded, self.loaded))
        self._renewal()
        new_active = active & ~self.was_active & fresh
        self.credit_heading.copy_(torch.where(new_active[:, None], heading, self.credit_heading))
        position = (robot.root_link_pos_w[:, :2].nan_to_num() * self.credit_heading).sum(dim=1)
        prior = torch.where(new_active, position, self.frontier)
        distance = (position - prior).clamp_min(0.)
        credit = (distance / (command[:, 0].clamp_min(1.e-6) * env.step_dt)).clamp(0., 1.)
        self.forward_credit.copy_(torch.where(fresh, torch.where(active, credit, 0.), self.forward_credit))
        self.frontier.copy_(torch.where(fresh & active, torch.maximum(prior, position), self.frontier))
        excess = ((self.duration - maximum_stance) / overstay_scale).clamp(0., 1.).square()
        excess = torch.where(loaded, excess, 0.)
        cost = .5 * excess.mean(dim=1) + .5 * excess.max(dim=1).values
        self.cached_cost.copy_(torch.where(fresh, torch.where(active, cost, 0.), self.cached_cost))
        for i, leg in enumerate(LEG_NAMES):
            if takeoff[:, i].any():
                mask = takeoff[:, i]
                for key, values in (("stance_time", self.duration), ("backstroke", self.travel),
                                    ("push_quality", self.swing_quality), ("contact_slip_distance", self.drift)):
                    suffix = "_s" if key == "stance_time" else "_m" if key in ("backstroke", "contact_slip_distance") else ""
                    env.extras["log"][f"Metrics/high_step_{key}_{leg}{suffix}"] = values[mask, i].detach().clone()
        env.extras["log"]["Metrics/high_step_gait_renewal"] = self.renewal.detach().clone()
        for i, leg in enumerate(LEG_NAMES):
            if settled[:, i].any():
                env.extras["log"][f"Metrics/high_step_contact_slip_{leg}_m_s"] = speed[settled[:, i], i].detach().clone()
        env.extras["log"]["Metrics/high_step_contact_slip_mean_m_s"] = (
            torch.where(settled, speed, 0.).sum() / settled.sum().clamp_min(1)).detach().clone()
        env.extras["log"]["Metrics/high_step_settled_support_feet"] = settled.sum(dim=1).detach().clone()
        reference_speed = robot.site_lin_vel_w[:, self.reference_sites, :2].norm(dim=-1)
        found = env.scene[sensor_name].data.found.reshape(env.num_envs, len(self.ids), -1)[:, self.ids].gt(0).any(dim=2)
        observed = found & torch.isfinite(reference_speed)
        env.extras["log"]["Metrics/slip_velocity_mean"] = (
            torch.where(observed, reference_speed, 0.).sum() / observed.sum().clamp_min(1)).detach().clone()
        self.was_loaded.copy_(torch.where(fresh[:, None], loaded, self.was_loaded))
        self.was_active.copy_(torch.where(fresh, active, self.was_active))
        if counter is not None:
            self.last_step[fresh] = counter
        return self.cached_cost
