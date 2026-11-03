"""Independent forward tracking, contact timing and observational gait statistics."""

import math

import torch

from pointleg.robot_cfg import LEG_NAMES
from .height import root_ground_clearance
from .high_step_support import contact_point_speeds
from .walking_gait import loaded_feet, ordered_contact_ids


def basic_high_step_velocity(env, command_name, std=.10):
    """Pay for forward velocity without cycle, posture or stance quality factors."""
    if not math.isfinite(std) or std <= 0.:
        raise ValueError("Velocity width must be finite and positive")
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    target = command[:, 0]
    finite = torch.isfinite(command).all(dim=1) & torch.isfinite(velocity)
    safe_velocity = velocity.nan_to_num(nan=0., posinf=0., neginf=0.)
    safe_target = target.nan_to_num(nan=0., posinf=0., neginf=0.)
    ratio = (safe_velocity / safe_target.clamp_min(1.e-6)).clamp(0., 1.)
    score = ratio * torch.exp(-((safe_velocity - safe_target) / std).square())
    allowed = finite & (target > .08)
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    if terminated is not None:
        allowed &= ~terminated
    return torch.where(allowed, score, 0.)


def basic_support_deficit(env, sensor_name, minimum_force=8.):
    """Count actual load-bearing feet using the same threshold as timing and slip."""
    sensor = env.scene[sensor_name]
    count = loaded_feet(sensor, ordered_contact_ids(sensor), minimum_force).sum(dim=1)
    return (2 - count).clamp_min(0.).float()


def basic_high_step_slip(env):
    """Read cached contact slip without using any cycle quality state."""
    term = env._high_step_basic_term
    term(env, **term.params)
    return term.slip_cost


def basic_height_event(env):
    """Pay the completed active-lift score after stable reloading."""
    term = env._high_step_basic_term
    term(env, **term.params)
    return term.height_event


def basic_height_progress(env):
    """Pay only newly achieved active lift during an eligible swing."""
    term = env._high_step_basic_term
    term(env, **term.params)
    return term.height_progress


def basic_height_overshoot(env):
    """Penalize active foot lift above the configured target during swing."""
    term = env._high_step_basic_term
    term(env, **term.params)
    return term.height_overshoot


def basic_backstroke_reward(env, target_distance=.07, minimum_stance=.16, max_slip=.025,
                            target_stance=.30):
    """Pay once for real progress during a completed, stable support phase."""
    if not all(math.isfinite(v) and v > 0. for v in
               (target_distance, minimum_stance, max_slip, target_stance)):
        raise ValueError("Backstroke targets must be finite and positive")
    if target_stance < minimum_stance:
        raise ValueError("Target stance must not be shorter than minimum stance")
    term = env._high_step_basic_term
    term(env, **term.params)
    progress = (term.ended_backstroke / target_distance).clamp(0., 1.).square()
    duration = (term.ended_stance / target_stance).clamp(0., 1.)
    quality = (1. - term.ended_slip / max_slip).clamp(0., 1.)
    valid = term.ended_stance >= minimum_stance - 1.e-6
    return (progress * duration * quality * valid).sum(dim=1) / env.step_dt


class BasicHighStepSupportPenalty:
    """Penalize abnormal contact durations and independently observe each foot."""

    def __init__(self, cfg, env):
        self.params = dict(cfg.params)
        self.ids = ordered_contact_ids(env.scene[self.params["sensor_name"]])
        self.sites = []
        for leg in LEG_NAMES:
            ids, _ = env.scene["robot"].find_sites(f"{leg}_foot_contact_site")
            if len(ids) != 1:
                raise ValueError("Basic HighStep requires four sphere-center sites")
            self.sites.append(ids[0])
        shape = (env.num_envs, 4)
        for name in ("stance", "air", "peak", "prior_stance", "slip_distance",
                     "lift_start", "lift_peak", "reported_lift_peak",
                     "ended_backstroke", "ended_stance", "ended_slip",
                     "support_travel", "support_sweep", "support_end_x",
                     "stance_progress", "swing_distance_credit"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        for name in ("was_loaded", "swinging", "eligible_swing", "height_pending", "mature_height"):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=env.device))
        for name in ("foot_start", "root_start", "heading"):
            setattr(self, name, torch.zeros((*shape, 2), device=env.device))
        self.lift_position = torch.zeros_like(self.foot_start)
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.cost = torch.zeros_like(self.age)
        self.slip_cost = torch.zeros_like(self.age)
        self.height_event = torch.zeros_like(self.age)
        self.height_progress = torch.zeros_like(self.age)
        self.height_overshoot = torch.zeros_like(self.age)
        self.progress_frontier = torch.zeros_like(self.age)
        self.progress_direction = torch.zeros((env.num_envs, 2), device=env.device)
        self.progress_anchored = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.stats = {key: torch.zeros(shape, device=env.device) for key in
                      ("cycles", "valid_cycles", "height_sum", "cycle_sum", "stride_sum")}
        env._high_step_basic_term = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("stance", "air", "peak", "prior_stance", "slip_distance",
                     "lift_start", "lift_peak", "reported_lift_peak",
                     "ended_backstroke", "ended_stance", "ended_slip",
                     "support_travel", "support_sweep", "support_end_x", "was_loaded",
                     "stance_progress", "swing_distance_credit", "progress_frontier",
                     "progress_direction", "progress_anchored", "height_overshoot",
                     "swinging", "eligible_swing", "height_pending", "mature_height", "foot_start",
                     "root_start", "heading", "lift_position", "age", "cost",
                     "slip_cost", "height_event", "height_progress"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1
        for value in self.stats.values():
            value[ids] = 0.

    def __call__(self, env, sensor_name, command_name, minimum_force=8.,
                 min_stance_time=.16, max_stance_time=.85, max_air_time=.36,
                 stance_scale=.25, air_scale=.18, short_event_scale=.10,
                 startup_grace=.60, slip_grace=.04, slip_tolerance=.02, slip_scale=.10,
                 target_height=.15, min_air_time=.10, min_forward_progress=.025,
                 lift_progress_distance=.07, short_stance_target=.24, overshoot_scale=.05):
        positive = (minimum_force, min_stance_time, max_stance_time, max_air_time,
                    stance_scale, air_scale, short_event_scale, slip_scale,
                    target_height, min_air_time, min_forward_progress,
                    lift_progress_distance, short_stance_target, overshoot_scale)
        if not all(math.isfinite(v) and v > 0. for v in positive):
            raise ValueError("Basic support scales must be finite and positive")
        if min_stance_time >= max_stance_time or min_air_time >= max_air_time:
            raise ValueError("Minimum stance must be shorter than maximum stance")
        if not min_stance_time <= short_stance_target < max_stance_time:
            raise ValueError("Short stance target must be within the support interval")
        if not all(math.isfinite(v) and v >= 0. for v in
                   (startup_grace, slip_grace, slip_tolerance)):
            raise ValueError("Basic support grace and tolerance must be nonnegative")
        counter = env.common_step_counter
        fresh = self.last_step != counter
        if not fresh.any():
            return self.cost
        robot = env.scene["robot"].data
        sensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)
        position = robot.site_pos_w[:, self.sites]
        speed, speed_valid = contact_point_speeds(
            env, sensor_name, self.sites, self.ids, minimum_force,
        )
        finite = (torch.isfinite(position).all(dim=(1, 2)) & speed_valid.all(dim=1)
                  & torch.isfinite(robot.root_link_pos_w).all(dim=1)
                  & torch.isfinite(robot.root_link_quat_w).all(dim=1)
                  & torch.isfinite(command).all(dim=1)
                  & torch.isfinite(sensor.data.found).flatten(1).all(dim=1))
        active = finite & (command[:, 0] > .08)
        terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
        if terminated is not None:
            active &= ~terminated
        # 无效状态和静止指令清除对应环境的轨迹，避免污染下一段换步。
        self.reset(torch.where(fresh & ~active)[0])
        loaded = loaded_feet(sensor, self.ids, minimum_force) & active[:, None]
        taking_off = self.was_loaded & ~loaded & active[:, None] & fresh[:, None]
        landing = ~self.was_loaded & loaded & self.swinging & fresh[:, None]
        starting_stance = loaded & ~self.was_loaded
        self.age.copy_(torch.where(fresh & active, self.age + env.step_dt, self.age))
        self.prior_stance.copy_(torch.where(taking_off, self.stance, self.prior_stance))
        self.peak.copy_(torch.where(taking_off, 0., self.peak))
        bottom = position[:, :, 2].nan_to_num() - .025
        self.lift_start.copy_(torch.where(taking_off, bottom, self.lift_start))
        self.lift_peak.copy_(torch.where(taking_off, 0., self.lift_peak))
        self.reported_lift_peak.copy_(torch.where(taking_off, 0., self.reported_lift_peak))
        self.lift_position.copy_(torch.where(taking_off[..., None], position[:, :, :2], self.lift_position))
        self.eligible_swing.copy_(torch.where(
            taking_off, (self.stance >= min_stance_time - 1.e-6)
            & (loaded.sum(dim=1, keepdim=True) >= 2), self.eligible_swing,
        ))
        self.swinging |= taking_off

        clearance, ground_valid = root_ground_clearance(env, sensor_name="base_height_sensor")
        ground = robot.root_link_pos_w[:, 2].nan_to_num() - clearance.nan_to_num()
        height = (bottom - ground[:, None]).clamp_min(0.)
        observing = self.swinging & ~loaded & active[:, None] & ground_valid[:, None] & fresh[:, None]
        self.peak.copy_(torch.where(observing, torch.maximum(self.peak, height), self.peak))
        active_lift = (bottom - self.lift_start).clamp_min(0.)
        self.lift_peak.copy_(torch.where(observing, torch.maximum(self.lift_peak, active_lift), self.lift_peak))

        feet_xy = position[:, :, :2].nan_to_num()
        root_xy = robot.root_link_pos_w[:, None, :2].nan_to_num().expand_as(feet_xy)
        w, x, y, z = robot.root_link_quat_w.nan_to_num().unbind(dim=1)
        heading = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), dim=1)
        heading /= heading.norm(dim=1, keepdim=True).clamp_min(1.e-6)
        anchoring = fresh & active & ~self.progress_anchored
        self.progress_direction.copy_(torch.where(anchoring[:, None], heading, self.progress_direction))
        projected = (root_xy[:, 0] * self.progress_direction).sum(dim=1)
        prior_frontier = torch.where(anchoring, projected, self.progress_frontier)
        new_progress = (projected - prior_frontier).clamp_min(0.)
        new_progress = torch.minimum(new_progress, command[:, 0].nan_to_num().clamp_min(0.) * env.step_dt)
        new_progress = torch.where(fresh & active, new_progress, 0.)
        self.progress_frontier.copy_(torch.where(
            fresh & active, torch.maximum(prior_frontier, projected), self.progress_frontier,
        ))
        self.progress_anchored |= anchoring
        # 每足仅积累本次承重期间的新路程，离地后冻结并消耗于一次摆动。
        swing_credit = (self.stance_progress / lift_progress_distance).clamp(0., 1.)
        self.swing_distance_credit.copy_(torch.where(taking_off, swing_credit, self.swing_distance_credit))
        stance_progress = torch.where(starting_stance, 0., self.stance_progress + new_progress[:, None])
        self.stance_progress.copy_(torch.where(
            fresh[:, None], torch.where(loaded, stance_progress, 0.), self.stance_progress,
        ))
        self.foot_start.copy_(torch.where((starting_stance & fresh[:, None])[..., None], feet_xy, self.foot_start))
        self.root_start.copy_(torch.where((starting_stance & fresh[:, None])[..., None], root_xy, self.root_start))
        self.heading.copy_(torch.where((starting_stance & fresh[:, None])[..., None], heading[:, None], self.heading))
        body_travel = ((root_xy - self.root_start) * self.heading).sum(dim=-1)
        sweep = ((root_xy - self.root_start - feet_xy + self.foot_start) * self.heading).sum(dim=-1)
        # 离地事件使用最后一个承重采样，避免把起脚后的足端移动算作支撑后扫。
        ended_travel = self.support_travel.clone()
        ended_sweep = self.support_sweep.clone()
        ended_x = self.support_end_x.clone()
        start_x = ((self.foot_start - self.root_start) * self.heading).sum(dim=-1)
        relative_x = ((feet_xy - root_xy) * self.heading).sum(dim=-1)
        self.support_travel.copy_(torch.where(loaded & fresh[:, None], body_travel, self.support_travel))
        self.support_sweep.copy_(torch.where(loaded & fresh[:, None], sweep, self.support_sweep))
        self.support_end_x.copy_(torch.where(loaded & fresh[:, None], relative_x, self.support_end_x))
        backstroke = torch.minimum(ended_travel, ended_sweep).clamp_min(0.)
        for name, value in (("ended_backstroke", backstroke), ("ended_stance", self.prior_stance),
                            ("ended_slip", self.slip_distance)):
            state = getattr(self, name)
            state.copy_(torch.where(fresh[:, None], torch.where(taking_off, value, 0.), state))

        old_air = self.air.clone()
        stance = torch.where(loaded, torch.where(starting_stance, env.step_dt, self.stance + env.step_dt), 0.)
        air = torch.where(active[:, None] & ~loaded, self.air + env.step_dt, 0.)
        self.stance.copy_(torch.where(fresh[:, None], stance, self.stance))
        self.air.copy_(torch.where(fresh[:, None], air, self.air))
        settled = loaded & (self.stance > slip_grace + 1.e-6)
        slip = ((speed - slip_tolerance).clamp_min(0.) / slip_scale).square().clamp_max(1.)
        slip = torch.where(settled, slip, 0.)
        slip_cost = .5 * slip.sum(dim=1) / settled.sum(dim=1).clamp_min(1) + .5 * slip.max(dim=1).values
        self.slip_cost.copy_(torch.where(fresh, slip_cost, self.slip_cost))
        distance = torch.where(starting_stance, 0., self.slip_distance)
        distance += torch.where(settled, speed * env.step_dt, 0.)
        self.slip_distance.copy_(torch.where(fresh[:, None], distance, self.slip_distance))

        short = (1. - (self.prior_stance + 1.e-6) / short_stance_target).clamp_min(0.).square() * taking_off
        overstay = ((self.stance - max_stance_time).clamp_min(0.) / stance_scale).square().clamp_max(1.)
        overair = ((self.air - max_air_time).clamp_min(0.) / air_scale).square().clamp_max(1.)
        duration_cost = torch.maximum(overstay, overair)
        cost = (.5 * duration_cost.mean(dim=1) + .5 * duration_cost.max(dim=1).values
                + short_event_scale / env.step_dt * short.mean(dim=1))
        cost = torch.where(active & (self.age >= startup_grace), cost, 0.)
        self.cost.copy_(torch.where(fresh, cost, self.cost))

        # 足高、步长和周期只用于观测，不影响速度、滑移或时序得分。
        cycle = self.prior_stance + old_air
        stride = ((feet_xy - self.lift_position) * self.heading).sum(dim=-1)
        valid_landing = landing & ground_valid[:, None]
        valid_cycle = (valid_landing & (self.prior_stance >= min_stance_time - 1.e-6)
                       & (old_air <= max_air_time + 1.e-6) & (stride >= .025))
        self.stats["cycles"] += valid_landing.float()
        self.stats["valid_cycles"] += valid_cycle.float()
        for key, value in (("height", self.peak), ("cycle", cycle), ("stride", stride)):
            self.stats[f"{key}_sum"] += torch.where(valid_landing, value, 0.)
        capped_peak = self.lift_peak.clamp_max(target_height)
        lift_progress = (capped_peak - self.reported_lift_peak).clamp_min(0.)
        progress_allowed = (observing & self.eligible_swing & (loaded.sum(dim=1, keepdim=True) >= 2)
                            & (self.air <= max_air_time + 1.e-6))
        self.height_progress.copy_(torch.where(
            fresh, (lift_progress * progress_allowed * self.swing_distance_credit).sum(dim=1) / target_height / env.step_dt,
            self.height_progress,
        ))
        self.reported_lift_peak.copy_(torch.where(observing, capped_peak, self.reported_lift_peak))
        overshoot = ((active_lift - target_height).clamp_min(0.) / overshoot_scale).square().clamp_max(4.)
        overshoot = torch.where(observing, overshoot, 0.)
        self.height_overshoot.copy_(torch.where(
            fresh, .5 * overshoot.mean(dim=1) + .5 * overshoot.max(dim=1).values, self.height_overshoot,
        ))
        height_eligible = (valid_landing & self.eligible_swing
                           & (old_air >= min_air_time - 1.e-6)
                           & (stride >= min_forward_progress))
        self.height_pending |= height_eligible
        self.height_pending &= loaded & active[:, None]
        mature_landing = self.height_pending & (self.stance >= min_stance_time - 1.e-6)
        self.mature_height.copy_(torch.where(fresh[:, None], mature_landing, self.mature_height))
        height_score = (self.lift_peak / target_height).clamp(0., 1.).square()
        self.height_event.copy_(torch.where(
            fresh, (height_score * mature_landing * self.swing_distance_credit).sum(dim=1) / env.step_dt,
            self.height_event,
        ))
        self.height_pending &= ~mature_landing
        log = env.extras["log"]
        for i, leg in enumerate(LEG_NAMES):
            lift_mask = taking_off[:, i]
            for key, value in (("stance_time", self.prior_stance), ("backstroke", backstroke),
                               ("contact_slip_distance", self.ended_slip),
                               ("stance_sweep", ended_sweep), ("stance_body_progress", ended_travel),
                               ("stance_start_x", start_x), ("stance_end_x", ended_x)):
                if lift_mask.any():
                    suffix = "s" if key == "stance_time" else "m"
                    log[f"Metrics/high_step_{key}_{leg}_{suffix}"] = value[lift_mask, i].detach().clone()
            if lift_mask.any():
                log[f"Metrics/high_step_lift_distance_credit_{leg}"] = self.swing_distance_credit[lift_mask, i].detach().clone()
            mask = valid_landing[:, i]
            if mask.any():
                for key, value in (("peak", self.peak), ("cycle", cycle)):
                    suffix = "m" if key == "peak" else "s"
                    log[f"Metrics/high_step_{key}_{leg}_{suffix}"] = value[mask, i].detach().clone()
                log[f"Metrics/high_step_active_lift_{leg}_m"] = self.lift_peak[mask, i].detach().clone()
                log[f"Metrics/high_step_lift_overshoot_{leg}_m"] = (
                    self.lift_peak[mask, i] - target_height).clamp_min(0.).detach().clone()
                log[f"Metrics/high_step_stance_duty_factor_{leg}"] = (
                    self.prior_stance[mask, i] / cycle[mask, i].clamp_min(env.step_dt)).detach().clone()
            if settled[:, i].any():
                log[f"Metrics/high_step_contact_slip_{leg}_m_s"] = speed[settled[:, i], i].detach().clone()
        if valid_landing.any():
            log["Metrics/high_step_peak_height_m"] = self.peak[valid_landing].detach().clone()
            log["Metrics/high_step_cycle_s"] = cycle[valid_landing].detach().clone()
            log["Metrics/high_step_active_lift_m"] = self.lift_peak[valid_landing].detach().clone()
        if mature_landing.any():
            log["Metrics/high_step_active_lift_score"] = height_score[mature_landing].detach().clone()
        log["Metrics/high_step_contact_slip_mean_m_s"] = (
            torch.where(settled, speed, 0.).sum() / settled.sum().clamp_min(1)).detach().clone()
        log["Metrics/high_step_settled_support_feet"] = settled.sum(dim=1).detach().clone()
        log["Metrics/high_step_new_forward_distance_m"] = new_progress.detach().clone()
        self.swinging &= ~landing
        self.was_loaded.copy_(torch.where(fresh[:, None], loaded, self.was_loaded))
        self.last_step[fresh] = counter
        return self.cost
