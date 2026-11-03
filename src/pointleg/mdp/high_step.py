"""Contact-confirmed high-clearance cycles for a flat-ground stepping foundation."""

import math

import torch

from pointleg.robot_cfg import FOOT_BODY_NAMES, LEG_NAMES, LEG_JOINT_NAMES
from .height import root_ground_clearance
from .walking_gait import loaded_feet, ordered_contact_ids


def high_step_feedback(env, key):
    """Read feedback produced by the preceding cycle reward in this control step."""
    return env._high_step_cycle_term.feedback[key]


class HighStepCycleReward:
    """Score each foot after a short forward swing and sustained reloading."""

    def __init__(self, cfg, env):
        self.contact_ids, self.site_ids = [], []
        sensor, robot = env.scene[cfg.params["sensor_name"]], env.scene["robot"]
        if not (sensor.cfg.global_frame or sensor.cfg.reduce == "netforce"):
            raise ValueError("HighStep requires world-frame foot contact forces")
        for leg, body in zip(LEG_NAMES, FOOT_BODY_NAMES):
            contacts = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
            sites, _ = robot.find_sites(f"{leg}_foot_contact_site")
            if len(contacts) != 1 or len(sites) != 1:
                raise ValueError(f"Expected one contact channel and sphere-center site for {leg}")
            self.contact_ids.append(contacts[0])
            self.site_ids.append(sites[0])
        shape = (env.num_envs, 4)
        self.last_support = torch.zeros((*shape, 3), device=env.device)
        self.last_root = torch.zeros_like(self.last_support)
        self.lift_position = torch.zeros_like(self.last_support)
        self.lift_root = torch.zeros_like(self.last_support)
        self.direction = torch.zeros((*shape, 2), device=env.device)
        for name in ("relative_start", "peak", "relative_peak", "pending_score", "pending_retraction"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        for name in ("anchored", "swinging", "supported", "pending"):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=env.device))
        self.stats = {key: torch.zeros(shape, device=env.device) for key in
                      ("cycles", "qualified", "height_sum", "cycle_sum", "stride_sum")}
        self.last_cycle = {key: torch.zeros(shape, device=env.device) for key in
                           ("height", "cycle", "stride")}
        self.pending_values = {key: torch.zeros(shape, device=env.device) for key in self.last_cycle}
        self.feedback = {key: torch.zeros(env.num_envs, device=env.device) for key in ("lift", "symmetry")}
        self.lift_frontier = torch.zeros(env.num_envs, device=env.device)
        self.lift_heading = torch.zeros(env.num_envs, 2, device=env.device)
        self.lift_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._high_step_cycle_term = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            "last_support", "last_root", "lift_position", "lift_root", "direction",
            "relative_start", "peak", "relative_peak", "pending_score", "pending_retraction",
            "anchored", "swinging", "supported", "pending",
            "lift_frontier", "lift_heading", "lift_ready",
        ):
            getattr(self, name)[ids] = 0
        for group in (self.stats, self.last_cycle, self.pending_values, self.feedback):
            for value in group.values():
                value[ids] = 0.

    def __call__(
        self, env, sensor_name: str, command_name: str,
        ground_sensor_name: str = "base_height_sensor", foot_radius: float = 0.025,
        target_height: float = 0.15, height_std: float = 0.03,
        target_cycle: float = 0.60, cycle_std: float = 0.12,
        min_air_time: float = 0.10, max_air_time: float = 0.36,
        min_stance_time: float = 0.16, min_step_length: float = 0.025,
        max_step_length: float = 0.18, step_std: float = 0.04,
        min_body_progress: float = 0.01, minimum_support_force: float = 8.0,
        min_support_feet: int = 2, command_threshold: float = 0.08,
        max_target_step_length: float = 0.15, timing_weight: float = 0.50,
    ) -> torch.Tensor:
        positive = (foot_radius, target_height, height_std, target_cycle, cycle_std,
                    min_air_time, max_air_time, min_stance_time, min_step_length,
                    max_step_length, step_std, min_body_progress, minimum_support_force,
                    command_threshold, max_target_step_length)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("High-step distances, times and thresholds must be finite and positive")
        if min_air_time >= max_air_time or min_step_length >= max_step_length:
            raise ValueError("High-step lower bounds must be below upper bounds")
        if not 0.04 <= max_target_step_length <= max_step_length:
            raise ValueError("Target step cap must fit inside the valid landing workspace")
        if not math.isfinite(timing_weight) or not 0.0 <= timing_weight <= 0.75:
            raise ValueError("Timing score weight must be between zero and 0.75")
        if not 1 <= min_support_feet <= 3:
            raise ValueError("High-step swing requires one to three supporting feet")
        support = getattr(env, "_high_step_support_term", None)
        if support is not None:
            support(env, **support.params)
        sensor, robot = env.scene[sensor_name], env.scene["robot"].data
        data = sensor.data
        for name in ("current_contact_time", "current_air_time", "last_contact_time", "last_air_time"):
            if getattr(data, name) is None:
                raise RuntimeError("High-step cycles require contact and air time tracking")
        position = robot.site_pos_w[:, self.site_ids]
        root = robot.root_link_pos_w[:, None, :].expand_as(position)
        force = data.force[:, self.contact_ids]
        clearance, valid_ground = root_ground_clearance(env, sensor_name=ground_sensor_name)
        finite = (torch.isfinite(position).all(dim=-1) & torch.isfinite(root).all(dim=-1)
                  & torch.isfinite(force).all(dim=-1))
        position, root, force = map(torch.nan_to_num, (position, root, force))
        bottom = position[:, :, 2] - foot_radius
        ground_z = torch.where(valid_ground, robot.root_link_pos_w[:, 2] - clearance, 0.0)
        height = (bottom - ground_z[:, None]).clamp_min(0.0)
        relative = bottom - root[:, :, 2]
        contact = data.found[:, self.contact_ids] > 0
        loaded = loaded_feet(sensor, self.contact_ids, minimum_support_force) & finite
        enough = loaded.sum(dim=1) >= min_support_feet
        command = env.command_manager.get_command(command_name)
        active = ((command[:, 0] > command_threshold) & (command[:, 1].abs() < 0.05)
                  & (command[:, 2].abs() < 0.15) & valid_ground
                  & torch.isfinite(command).all(dim=1) & torch.isfinite(robot.root_link_pos_w).all(dim=1)
                  & torch.isfinite(robot.root_link_quat_w).all(dim=1))
        if support is not None:
            active &= support.active
        terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
        if terminated is not None:
            active &= ~terminated
        eligible = active[:, None] & finite
        contact_time = data.current_contact_time[:, self.contact_ids]
        previous_stance = data.last_contact_time[:, self.contact_ids]
        air_time = data.last_air_time[:, self.contact_ids]
        takeoff = (sensor.compute_first_air(env.step_dt)[:, self.contact_ids] & self.anchored
                   & eligible & (previous_stance >= min_stance_time))
        w, x, y, z = robot.root_link_quat_w.unbind(dim=1)
        forward = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), dim=-1)
        forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1.e-6)
        self.lift_position = torch.where(takeoff[:, :, None], self.last_support, self.lift_position)
        self.lift_root = torch.where(takeoff[:, :, None], self.last_root, self.lift_root)
        self.direction = torch.where(takeoff[:, :, None], forward[:, None, :], self.direction)
        relative_start = self.last_support[:, :, 2] - foot_radius - self.last_root[:, :, 2]
        self.relative_start = torch.where(takeoff, relative_start, self.relative_start)
        self.peak = torch.where(takeoff, 0.0, self.peak)
        self.relative_peak = torch.where(takeoff, 0.0, self.relative_peak)
        self.swinging = (self.swinging | takeoff) & eligible
        self.supported = torch.where(takeoff, True, self.supported)
        self.supported &= enough[:, None] & eligible
        airborne = ~contact & self.swinging
        self.peak = torch.where(airborne, torch.maximum(self.peak, height), self.peak)
        self.relative_peak = torch.where(
            airborne, torch.maximum(self.relative_peak, (relative - self.relative_start).clamp_min(0.0)),
            self.relative_peak,
        )
        landed = sensor.compute_first_contact(env.step_dt)[:, self.contact_ids] & self.swinging
        cycle = previous_stance + air_time
        displacement = position[:, :, :2] - self.lift_position[:, :, :2]
        step_length = (displacement * self.direction).sum(dim=-1)
        horizontal_length = torch.linalg.vector_norm(displacement, dim=-1)
        body_progress = ((root[:, :, :2] - self.lift_root[:, :, :2]) * self.direction).sum(dim=-1)
        expected_step = (command[:, 0].nan_to_num() * target_cycle).clamp(
            0.04, max_target_step_length,
        )[:, None]
        overshoot = torch.exp(-((self.peak - target_height).clamp_min(0.) / height_std).square())
        height_score = (self.peak / target_height).clamp(0., 1.).square() * overshoot
        retraction = (self.relative_peak / (.7 * self.peak.clamp(.005, target_height))).clamp(0., 1.)
        timing = torch.exp(-((cycle - target_cycle) / cycle_std).square())
        placement = torch.exp(-((step_length - expected_step) / step_std).square())
        # 按完成周期分配预算，减少快速重复落足带来的次数收益。
        cycle_budget = (cycle / target_cycle).clamp(0.0, 1.5)
        score = height_score * retraction * (
            .75 - timing_weight + timing_weight * timing + .25 * placement
        ) * cycle_budget
        if support is not None:
            score *= .25 + .75 * support.swing_quality
        # 抬升反馈按新增前沿距离支付，重复摆动不能复用旧路程。
        initializing = active & ~self.lift_ready
        self.lift_heading.copy_(torch.where(initializing[:, None], forward.nan_to_num(), self.lift_heading))
        projected = (robot.root_link_pos_w[:, :2].nan_to_num() * self.lift_heading).sum(dim=1)
        prior = torch.where(initializing, projected, self.lift_frontier)
        new_distance = torch.minimum((projected - prior).clamp_min(0.), command[:, 0].nan_to_num().clamp_min(0.) * env.step_dt)
        new_distance = torch.where(active, new_distance, 0.)
        self.lift_frontier.copy_(torch.where(active, torch.maximum(prior, projected), self.lift_frontier))
        self.lift_ready.copy_(active)
        lift_allowed = (airborne & self.supported & eligible & (body_progress >= min_body_progress)
                        & (step_length >= min_step_length)
                        & (data.current_air_time[:, self.contact_ids] <= max_air_time + 1.e-6))
        lift_quality = torch.where(lift_allowed, height_score * retraction, 0.)
        self.feedback["lift"].copy_(.02 * new_distance / expected_step[:, 0] * lift_quality.sum(dim=1) / env.step_dt)
        valid = (landed & self.supported & loaded & enough[:, None]
                 & (air_time >= min_air_time - 1.e-6) & (air_time <= max_air_time + 1.e-6)
                 & (step_length >= min_step_length) & (horizontal_length <= max_step_length)
                 & (body_progress >= min_body_progress) & (score > 1.e-6))
        # 落足后持续承重再结算，短促点地、无前移或全足腾空不能领取循环奖励。
        self.pending &= loaded & eligible & enough[:, None]
        self.pending_score = torch.where(valid, score, self.pending_score)
        self.pending_retraction = torch.where(valid, retraction, self.pending_retraction)
        for key, value in (("height", self.peak), ("cycle", cycle), ("stride", step_length)):
            self.pending_values[key] = torch.where(valid, value, self.pending_values[key])
        self.pending |= valid
        mature = self.pending & (contact_time >= min_stance_time - 1.e-6)
        upright = (-robot.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        reward = torch.where(mature, self.pending_score, 0.0).sum(dim=1) * upright / env.step_dt
        if support is not None:
            support.record_cycles(mature & (self.pending_values["height"] >= .2 * target_height)
                                  & (self.pending_retraction >= .4))
        qualified = (mature & (self.pending_values["height"] >= .8 * target_height)
                     & (self.pending_values["height"] <= 1.2 * target_height)
                     & ((self.pending_values["cycle"] - target_cycle).abs() <= cycle_std))
        self.stats["cycles"] += landed.float()
        self.stats["qualified"] += qualified.float()
        for key, value in (("height", self.peak), ("cycle", cycle), ("stride", step_length)):
            self.last_cycle[key] = torch.where(landed, value, self.last_cycle[key])
            self.stats[f"{key}_sum"] += torch.where(landed, value, 0.)
        differences = []
        for key, scale in (("height", target_height), ("cycle", target_cycle), ("stride", .08)):
            values = self.last_cycle[key]
            seen = (self.stats["cycles"][:, [0, 2]] > 0) & (self.stats["cycles"][:, [1, 3]] > 0)
            excess = ((values[:, [0, 2]] - values[:, [1, 3]]).abs() / scale - .15).clamp_min(0.)
            differences.append(torch.where(seen, excess.square().clamp_max(1.), 0.))
        symmetry = torch.cat(differences, dim=1)
        self.feedback["symmetry"].copy_((.5 * symmetry.mean(dim=1) + .5 * symmetry.max(dim=1).values) * active)
        self.pending &= ~mature
        log = env.extras["log"]
        if landed.any():
            for key, values in (("peak_height_m", self.peak), ("cycle_s", cycle),
                                ("step_length_m", step_length), ("air_time_s", air_time),
                                ("height_score", height_score), ("period_score", timing),
                                ("cycle_budget", cycle_budget)):
                log[f"Metrics/high_step_{key}"] = values[landed].detach().clone()
            for i, leg in enumerate(LEG_NAMES):
                if landed[:, i].any():
                    log[f"Metrics/high_step_cycle_{leg}_s"] = cycle[landed[:, i], i].detach().clone()
                    log[f"Metrics/high_step_peak_{leg}_m"] = self.peak[landed[:, i], i].detach().clone()
                    log[f"Metrics/high_step_stride_{leg}_m"] = step_length[landed[:, i], i].detach().clone()
        log["Metrics/high_step_rewarded_cycles_per_s"] = mature.float().sum(dim=1).mean() / env.step_dt
        log["Metrics/high_step_takeoffs_per_s"] = takeoff.float().sum(dim=1) / env.step_dt
        log["Metrics/high_step_landings_per_s"] = landed.float().sum(dim=1) / env.step_dt
        log["Metrics/high_step_loaded_feet"] = loaded.float().sum(dim=1)
        self.swinging &= ~contact
        self.anchored = torch.where(contact, loaded & eligible, self.anchored & eligible)
        self.last_support = torch.where(loaded[:, :, None], position, self.last_support)
        self.last_root = torch.where(loaded[:, :, None], root, self.last_root)
        return reward


def high_step_phase_penalty(
    env, sensor_name: str, command_name: str, max_stance_time: float = 0.51,
    max_air_time: float = 0.36, deviation_scale: float = 0.18,
    startup_grace: float = 0.6, command_threshold: float = 0.08,
    stance_weight: float = 1.,
) -> torch.Tensor:
    """Penalize feet that remain planted or airborne far beyond the target rhythm."""
    if not all(math.isfinite(v) and v > 0 for v in (max_stance_time, max_air_time, deviation_scale)):
        raise ValueError("High-step phase times and scale must be finite and positive")
    if not math.isfinite(stance_weight) or stance_weight < 0.:
        raise ValueError("Stance phase weight must be finite and nonnegative")
    data = env.scene[sensor_name].data
    if data.current_contact_time is None or data.current_air_time is None:
        raise RuntimeError("High-step phase penalty requires contact and air time tracking")
    command = env.command_manager.get_command(command_name)
    active = ((command[:, 0] > command_threshold) & (command[:, 1].abs() < 0.05)
              & (command[:, 2].abs() < 0.15)
              & (env.episode_length_buf * env.step_dt >= startup_grace))
    stance = ((data.current_contact_time - max_stance_time) / deviation_scale).clamp_min(0.0)
    air = ((data.current_air_time - max_air_time) / deviation_scale).clamp_min(0.0)
    cost = (stance_weight * stance.square() + air.square()).clamp_max(4.0)
    return (0.5 * cost.mean(dim=1) + 0.5 * cost.max(dim=1).values) * active


class HighStepContactForcePenalty:
    """Limit excessive per-foot forces without penalizing ordinary support."""

    def __init__(self, cfg, env):
        self.ids = ordered_contact_ids(env.scene[cfg.params["sensor_name"]])

    def __call__(
        self, env, sensor_name: str, command_name: str,
        force_threshold: float = 150.0, force_scale: float = 100.0,
        startup_grace: float = 0.6,
    ) -> torch.Tensor:
        if not all(math.isfinite(v) and v > 0. for v in (force_threshold, force_scale)):
            raise ValueError("Contact force threshold and scale must be positive")
        if not math.isfinite(startup_grace) or startup_grace < 0.:
            raise ValueError("Contact force startup grace must be nonnegative")
        sensor = env.scene[sensor_name]
        data = sensor.data
        history = getattr(data, "force_history", None)
        if history is None and data.force is None:
            raise RuntimeError("HighStep contact force penalty requires measured forces")
        force = data.force.unsqueeze(2) if history is None else history
        force = torch.nan_to_num(force, nan=0., posinf=0., neginf=0.)
        force = force.reshape(env.num_envs, len(sensor.primary_names), -1, 3)
        peak = torch.linalg.vector_norm(force, dim=-1).amax(dim=2)[:, self.ids]
        peak = torch.nan_to_num(peak, nan=0., posinf=1.e6, neginf=0.)
        command = env.command_manager.get_command(command_name)
        active = ((command[:, 0] > .08) & (command[:, 1].abs() < .05)
                  & (command[:, 2].abs() < .15)
                  & (env.episode_length_buf * env.step_dt >= startup_grace))
        excess = ((peak - force_threshold).clamp_min(0.) / force_scale).square()
        cost = excess.clamp_max(4.)
        for i, leg in enumerate(LEG_NAMES):
            env.extras["log"][f"Metrics/high_step_contact_peak_{leg}_n"] = (
                peak[:, i].detach().clone()
            )
        env.extras["log"]["Metrics/high_step_contact_overload_fraction"] = (
            (peak > force_threshold).float().mean(dim=1).detach().clone()
        )
        return (.5 * cost.mean(dim=1) + .5 * cost.max(dim=1).values) * active


class HighStepDriftBiasPenalty:
    """Penalize each environment's persistent drift before averaging rewards."""

    def __init__(self, cfg, env):
        self.mean = torch.zeros((env.num_envs, 2), device=env.device)
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros_like(self.age)
        self.last_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device,
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.mean[ids] = 0.
        self.age[ids] = 0.
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(
        self, env, command_name: str, smoothing_time: float = 0.6,
        warmup_time: float = 1.0, speed_tolerance: float = 0.005,
        speed_scale: float = 0.04, yaw_tolerance_degrees: float = 1.0,
        yaw_scale_degrees: float = 10.0,
    ) -> torch.Tensor:
        if not all(math.isfinite(v) and v > 0. for v in
                   (smoothing_time, speed_scale, yaw_scale_degrees)):
            raise ValueError("Drift smoothing and scales must be finite and positive")
        if not all(math.isfinite(v) and v >= 0. for v in
                   (warmup_time, speed_tolerance, yaw_tolerance_degrees)):
            raise ValueError("Drift warmup and tolerances must be finite and nonnegative")
        robot = env.scene["robot"].data
        command = env.command_manager.get_command(command_name)
        error = torch.stack((robot.root_link_lin_vel_b[:, 1] - command[:, 1],
                             robot.root_link_ang_vel_b[:, 2] - command[:, 2]), dim=1)
        active = ((command[:, 0] > .08) & (command[:, 1].abs() < .05)
                  & (command[:, 2].abs() < .15) & torch.isfinite(error).all(dim=1))
        fresh = self.last_step != env.common_step_counter
        alpha = 1. - math.exp(-env.step_dt / smoothing_time)
        mean = torch.where(active[:, None],
                           self.mean + alpha * (error.nan_to_num() - self.mean), 0.)
        age = torch.where(active, self.age + env.step_dt, 0.)
        self.mean.copy_(torch.where(fresh[:, None], mean, self.mean))
        self.age.copy_(torch.where(fresh, age, self.age))
        tolerance = self.mean.new_tensor(
            [speed_tolerance, math.radians(yaw_tolerance_degrees)],
        )
        scale = self.mean.new_tensor([speed_scale, math.radians(yaw_scale_degrees)])
        cost = ((self.mean.abs() - tolerance).clamp_min(0.) / scale).square()
        cost = cost.clamp_max(4.).sum(dim=1) * active * (self.age >= warmup_time)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        for key, value in (
            ("lateral_bias_m_s", self.mean[:, 0]),
            ("lateral_bias_abs_m_s", self.mean[:, 0].abs()),
            ("yaw_rate_bias_deg_s", torch.rad2deg(self.mean[:, 1])),
            ("yaw_rate_bias_abs_deg_s", torch.rad2deg(self.mean[:, 1].abs())),
        ):
            env.extras["log"][f"Metrics/high_step_{key}"] = value.detach().clone()
        return self.cached


def high_step_yaw_rate_penalty(
    env, command_name: str, tolerance_degrees: float = 2.0,
    deviation_degrees: float = 20.0,
) -> torch.Tensor:
    """Suppress commanded yaw-rate error and report signed body-frame drift."""
    if not math.isfinite(tolerance_degrees) or tolerance_degrees < 0.0:
        raise ValueError("Yaw-rate tolerance must be finite and nonnegative")
    if not math.isfinite(deviation_degrees) or deviation_degrees <= 0.0:
        raise ValueError("Yaw-rate scale must be finite and positive")
    robot = env.scene["robot"].data
    command = env.command_manager.get_command(command_name)
    error = robot.root_link_ang_vel_b[:, 2] - command[:, 2]
    log = env.extras["log"]
    for key, values in (
        ("lateral_velocity_m_s", robot.root_link_lin_vel_b[:, 1]),
        ("lateral_speed_abs_m_s", robot.root_link_lin_vel_b[:, 1].abs()),
        ("yaw_rate_error_deg_s", torch.rad2deg(error)),
        ("yaw_rate_error_abs_deg_s", torch.rad2deg(error.abs())),
    ):
        log[f"Metrics/high_step_{key}"] = values.detach().clone()
    excess = (error.abs() - math.radians(tolerance_degrees)).clamp_min(0.0)
    return (excess / math.radians(deviation_degrees)).square().clamp_max(4.0)


def high_step_pitch_penalty(
    env, tolerance_degrees: float = 3.0, deviation_degrees: float = 8.0,
) -> torch.Tensor:
    """Keep the flat-ground foundation level without demanding zero pitch at every instant."""
    if not math.isfinite(tolerance_degrees) or tolerance_degrees < 0.0:
        raise ValueError("Pitch tolerance must be finite and nonnegative")
    if not math.isfinite(deviation_degrees) or deviation_degrees <= 0.0:
        raise ValueError("Pitch scale must be finite and positive")
    gravity = env.scene["robot"].data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=1))
    env.extras["log"]["Metrics/high_step_pitch_deg"] = torch.rad2deg(pitch).detach().clone()
    excess = (pitch.abs() - math.radians(tolerance_degrees)).clamp_min(0.0)
    return (excess / math.radians(deviation_degrees)).square().clamp_max(4.0)


class HighStepMeanLegSymmetryPenalty:
    """Compare slow joint biases across legs instead of synchronizing swing phases."""

    def __init__(self, cfg, env):
        names = env.scene["robot"].joint_names
        self.joint_ids = [names.index(name) for name in LEG_JOINT_NAMES]
        self.mean = torch.zeros((env.num_envs, 4, 2), device=env.device)
        self.age = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.mean[ids] = 0.0
        self.age[ids] = 0.0

    def __call__(
        self, env, command_name: str, smoothing_time: float = 0.8,
        warmup_time: float = 1.0, tolerance_degrees: float = 4.0,
        deviation_degrees: float = 8.0, command_threshold: float = 0.08,
        front_rear_weight: float = 0., front_rear_hip_tolerance: float = 8.,
        front_rear_knee_tolerance: float = 12., front_rear_hip_scale: float = 15.,
        front_rear_knee_scale: float = 20.,
    ) -> torch.Tensor:
        if not all(math.isfinite(v) and v > 0 for v in (smoothing_time, deviation_degrees)):
            raise ValueError("Mean symmetry smoothing and scale must be finite and positive")
        if not all(math.isfinite(v) and v >= 0 for v in (warmup_time, tolerance_degrees)):
            raise ValueError("Mean symmetry warmup and tolerance must be finite and nonnegative")
        if not math.isfinite(front_rear_weight) or front_rear_weight < 0.:
            raise ValueError("Front/rear symmetry weight must be nonnegative")
        if not all(math.isfinite(v) and v > 0. for v in (front_rear_hip_tolerance,
                   front_rear_knee_tolerance, front_rear_hip_scale, front_rear_knee_scale)):
            raise ValueError("Front/rear posture tolerances and scales must be positive")
        robot = env.scene["robot"].data
        deviation = (robot.joint_pos[:, self.joint_ids] - robot.default_joint_pos[:, self.joint_ids])
        deviation = deviation.reshape(env.num_envs, 4, 3)[:, :, 1:]
        command = env.command_manager.get_command(command_name)
        active = ((command[:, 0] > command_threshold) & (command[:, 1].abs() < 0.05)
                  & (command[:, 2].abs() < 0.15))
        active &= torch.isfinite(deviation).all(dim=(1, 2))
        deviation = deviation.nan_to_num()
        alpha = 1.0 - math.exp(-env.step_dt / smoothing_time)
        self.mean = torch.where(
            active[:, None, None], self.mean + alpha * (deviation - self.mean), 0.0,
        )
        self.age = torch.where(active, self.age + env.step_dt, 0.0)
        valid = active & (self.age >= warmup_time)
        # 比较慢偏置，正常摆动相不要求关节角同步。
        pairs = self.mean[:, [0, 2]] - self.mean[:, [1, 3]]
        excess = (pairs.abs() - math.radians(tolerance_degrees)).clamp_min(0.0)
        log = env.extras["log"]
        front_rear = (self.mean[:, :2].mean(dim=1) - self.mean[:, 2:].mean(dim=1)).abs()
        if valid.any():
            for index, joint in enumerate(("hip", "knee")):
                log[f"Metrics/high_step_front_rear_{joint}_bias_deg"] = torch.rad2deg(
                    front_rear[valid, index]
                ).detach().clone()
        cost = (excess / math.radians(deviation_degrees)).square().clamp_max(4.0).mean(dim=(1, 2))
        tolerance = self.mean.new_tensor([front_rear_hip_tolerance, front_rear_knee_tolerance]) * math.pi / 180.
        scale = self.mean.new_tensor([front_rear_hip_scale, front_rear_knee_scale]) * math.pi / 180.
        cost += front_rear_weight * ((front_rear - tolerance).clamp_min(0.) / scale).square().clamp_max(4.).mean(dim=1)
        return cost * valid
