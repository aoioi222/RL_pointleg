"""Bounded swing-clearance costs for walking over known elevation geometry."""

import math

import torch

from .elevation import elevation_foot_bottom, elevation_state
from .elevation_gait import elevation_landing_margin, elevation_surface_frame
from ..robot_cfg import LEG_NAMES
from ..terrains.elevation import CENTER_Y, LANE_WIDTH, TILE_SIZE


def elevation_surface_height(env, x):
    """Query actual generated stair/platform and ramp heights in tile coordinates."""
    state = elevation_state(env)
    kind = env.scene.terrain.terrain_types[:, None]
    height = torch.where(kind == 1, state["top_height"][:, None], torch.zeros_like(x))
    treads = state["step_surfaces"]
    for index in range(treads.shape[1]):
        height = torch.where(x >= treads[:, index, 0, None], treads[:, index, 2, None], height)
    height = torch.where(x < treads[:, -1, 1, None], height, 0.)
    up = ((x - 3.) / 2.).clamp(0., 1.) * state["top_height"][:, None]
    down = ((5. - x) / 2.).clamp(0., 1.) * state["top_height"][:, None]
    return torch.where(kind < 2, height, torch.where(kind == 2, up, torch.where(kind == 3, down, 0.)))


def elevation_clearance_metric(env, key):
    return env._elevation_clearance_metrics[key]


class ElevationSwingClearancePenalty:
    """Penalize low completed swings and insufficient clearance near upward risers."""

    def __init__(self, cfg, env):
        for key in (
            "target_clearance", "riser_clearance", "lookahead", "foot_radius",
            "minimum_support_force", "minimum_air_time", "minimum_foot_progress",
            "landing_cost_seconds",
            "minimum_quality_air_time", "minimum_stance_time", "minimum_cycle_time",
            "preferred_stride", "maximum_stride", "minimum_body_progress", "reward_progress_scale",
            "minimum_relative_lift", "step_bonus", "participation_warmup", "participation_window",
            "minimum_push_stance", "target_push_stance", "target_backstroke", "maximum_stance_slip",
            "maximum_stance_time", "stance_time_scale", "renewal_grace", "renewal_timeout",
            "landing_edge_margin",
        ):
            if not math.isfinite(cfg.params[key]) or cfg.params[key] <= 0.:
                raise ValueError("Swing-clearance distances, forces and times must be positive.")
        if cfg.params["maximum_stride"] <= cfg.params["preferred_stride"]:
            raise ValueError("Maximum stride must exceed the preferred short stride.")
        if (cfg.params["target_push_stance"] < cfg.params["minimum_push_stance"]
                or cfg.params["renewal_timeout"] <= cfg.params["renewal_grace"]):
            raise ValueError("Invalid support duration or gait renewal interval")
        if env.cfg.events["reset_elevation"].params["settings"].jump:
            raise ValueError("Walking swing-clearance costs must not be enabled for Jump.")
        self.foot_ids = env.scene["robot"].find_sites(
            tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True,
        )[0]
        if len(self.foot_ids) != 4 or len(set(self.foot_ids)) != 4:
            raise ValueError("Swing clearance requires four distinct sole sites.")
        shape = (env.num_envs, 4)
        for key in ("support_z", "support_x", "support_relative_z", "air_time", "peak", "relative_peak",
                    "terrain_peak", "pending_landing_margin", "leg_lift_sum", "leg_swing_count"):
            setattr(self, key, torch.zeros(shape, device=env.device))
        self.support_point = torch.zeros((*shape, 3), device=env.device)
        self.anchored = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.swinging = torch.zeros_like(self.anchored)
        self.support_normal = torch.zeros((*shape, 3), device=env.device)
        self.support_normal[..., 2] = 1.
        for key in (
            "target_peak", "pending_score", "landing_time", "cycle_time", "participation_age",
            "last_foot_frontier", "last_body_frontier", "pending_foot_x", "pending_body_x",
            "stance_time", "stance_travel", "stance_slip", "swing_push_quality", "renewal_age",
        ):
            setattr(self, key, torch.zeros(shape, device=env.device))
        self.pending_valid = torch.zeros_like(self.anchored)
        self.pending_qualified = torch.zeros_like(self.anchored)
        self.swing_support_ok = torch.ones_like(self.anchored)
        self.was_supported = torch.zeros_like(self.anchored)
        self.pending_renewal = torch.zeros_like(self.anchored)
        self.stance_foot = torch.zeros((*shape, 3), device=env.device)
        self.stance_root = torch.zeros_like(self.stance_foot)
        self.stance_tangent = torch.zeros_like(self.stance_foot)
        self.stance_tangent[..., 0] = 1.
        self.params = dict(cfg.params)
        self.gait = {key: torch.zeros(env.num_envs, device=env.device)
                     for key in ("quality", "participation", "stride", "stance", "renewal")}
        self.gait["renewal"].fill_(1.)
        env._elevation_gait_term = self
        self.peak_sum = torch.zeros(env.num_envs, device=env.device)
        self.swing_count = torch.zeros_like(self.peak_sum)
        self.true_lift_sum = torch.zeros_like(self.peak_sum)
        self.landing_margin_sum = torch.zeros_like(self.peak_sum)
        self.landing_count = torch.zeros_like(self.peak_sum)
        self.edge_landings = torch.zeros_like(self.peak_sum)
        self.cached_cost = torch.zeros_like(self.peak_sum)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.metrics = {
            key: torch.zeros(env.num_envs, device=env.device)
            for key in ("swing_peak_height", "near_riser_feet", "riser_deficit",
                        "low_swing_events_per_s", "completed_swings_per_s")
        }
        self.metrics.update({key: torch.zeros(env.num_envs, device=env.device) for key in (
            "quality_steps_per_s", "worst_leg_cycle_age_s", "stride_length", "target_peak_height",
            "true_lift_height", "landing_edge_margin",
        )})
        env._elevation_clearance_metrics = self.metrics

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for key in (
            "support_z", "support_x", "support_relative_z", "air_time", "peak", "relative_peak",
            "anchored", "swinging", "peak_sum", "swing_count", "cached_cost",
            "support_normal", "target_peak", "pending_score", "landing_time", "cycle_time",
            "participation_age", "last_foot_frontier", "last_body_frontier", "pending_foot_x",
            "pending_body_x", "pending_valid", "pending_qualified",
            "stance_time", "stance_travel", "stance_slip", "swing_push_quality", "renewal_age",
            "was_supported", "pending_renewal", "stance_foot", "stance_root", "stance_tangent",
            "support_point", "terrain_peak", "pending_landing_margin", "true_lift_sum",
            "leg_lift_sum", "leg_swing_count", "landing_margin_sum", "landing_count", "edge_landings",
        ):
            getattr(self, key)[ids] = 0
        self.support_normal[ids, ..., 2] = 1.
        self.swing_support_ok[ids] = True
        self.stance_tangent[ids, ..., 0] = 1.
        self.last_step[ids] = -1
        for value in self.metrics.values():
            value[ids] = 0
        for value in self.gait.values():
            value[ids] = 0
        self.gait["renewal"][ids] = 1.

    def __call__(
        self, env, target_clearance, riser_clearance, lookahead, foot_radius,
        minimum_support_force, minimum_air_time, minimum_foot_progress, landing_cost_seconds,
        minimum_quality_air_time, minimum_stance_time, minimum_cycle_time,
        preferred_stride, maximum_stride, minimum_body_progress, reward_progress_scale,
        minimum_relative_lift, step_bonus, participation_warmup, participation_window,
        minimum_push_stance, target_push_stance, target_backstroke, maximum_stance_slip,
        maximum_stance_time, stance_time_scale, renewal_grace, renewal_timeout,
        landing_edge_margin,
    ):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached_cost
        state = elevation_state(env)
        robot = env.scene["robot"].data
        feet = elevation_foot_bottom(env, self.foot_ids, foot_radius)
        sensor = env.scene["elevation_contacts"]
        slots = sensor.cfg.num_slots
        found = sensor.data.found.reshape(env.num_envs, 4, slots) > 0
        force = sensor.data.force.reshape(env.num_envs, 4, slots, 3)
        normal = sensor.data.normal.reshape(env.num_envs, 4, slots, 3)
        finite = (torch.isfinite(feet).all(dim=-1) & torch.isfinite(force).all(dim=(-1, -2))
                  & torch.isfinite(normal).all(dim=(-1, -2))
                  & torch.isfinite(robot.root_link_pos_w).all(dim=1)[:, None])
        force_norm = torch.nan_to_num(force, nan=0., posinf=0., neginf=0.).norm(dim=-1)
        supported = (found & (normal[..., 2] < -.5) & (force_norm >= minimum_support_force)).any(dim=2) & finite
        support_count = supported.sum(dim=1)
        local = feet - state["corner"][:, None, :]
        surface = elevation_surface_height(env, local[..., 0]) + state["corner"][:, 2, None]
        legal = (finite & ((local[..., 1] - CENTER_Y).abs() <= LANE_WIDTH / 2)
                 & (local[..., 0] >= 0.) & (local[..., 0] <= TILE_SIZE[0]))
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"]
                  & ~state["complete"] & (state["speed"] > .1))
        real_support = supported & legal & active[:, None]
        takeoff = self.was_supported & ~supported & legal & active[:, None] & fresh[:, None]
        push_quality = ((self.stance_time / target_push_stance).clamp(0., 1.)
                        * (self.stance_travel / target_backstroke).clamp(0., 1.)
                        * (1. - self.stance_slip / maximum_stance_slip).clamp(0., 1.))
        push_quality = torch.where(self.stance_time >= minimum_push_stance, push_quality, 0.)
        self.swing_push_quality.copy_(torch.where(takeoff, push_quality, self.swing_push_quality))
        starting_support = real_support & ~self.was_supported
        tangent, _ = elevation_surface_frame(env, local[..., 0])
        root = robot.root_link_pos_w[:, None, :].expand_as(feet)
        self.stance_foot.copy_(torch.where((starting_support & fresh[:, None])[..., None], feet, self.stance_foot))
        self.stance_root.copy_(torch.where((starting_support & fresh[:, None])[..., None], root, self.stance_root))
        self.stance_tangent.copy_(torch.where((starting_support & fresh[:, None])[..., None], tangent, self.stance_tangent))
        foot_delta = torch.nan_to_num(feet - self.stance_foot, nan=0., posinf=0., neginf=0.)
        root_delta = torch.nan_to_num(root - self.stance_root, nan=0., posinf=0., neginf=0.)
        body_travel = (root_delta * self.stance_tangent).sum(dim=-1)
        relative_travel = ((root_delta - foot_delta) * self.stance_tangent).sum(dim=-1)
        travel = torch.minimum(body_travel, relative_travel).clamp_min(0.)
        slip = torch.sqrt((foot_delta * self.stance_tangent).sum(dim=-1).square() + foot_delta[..., 1].square())
        stance_time = torch.where(starting_support, env.step_dt, self.stance_time + env.step_dt)
        self.stance_time.copy_(torch.where(real_support & fresh[:, None], stance_time, self.stance_time))
        self.stance_travel.copy_(torch.where(real_support & fresh[:, None], travel, self.stance_travel))
        self.stance_slip.copy_(torch.where(
            real_support & fresh[:, None], torch.where(starting_support, 0., torch.maximum(self.stance_slip, slip)),
            self.stance_slip,
        ))
        overstay = ((self.stance_time - maximum_stance_time) / stance_time_scale).clamp(0., 1.).square()
        overstay = torch.where(real_support, overstay, 0.)
        self.gait["stance"].copy_(torch.where(
            fresh, .5 * overstay.mean(dim=1) + .5 * overstay.max(dim=1).values, self.gait["stance"],
        ))
        airborne = legal & self.anchored & ~supported & active[:, None]
        enough_support = (support_count >= 2)[:, None]
        support_ok = torch.where(airborne & ~self.swinging, enough_support, self.swing_support_ok)
        support_ok &= ~airborne | enough_support
        _, normals = elevation_surface_frame(env, local[..., 0])
        clearance = ((feet[..., 2] - surface) * normals[..., 2]
                     - foot_radius * (1. - normals[..., 2])).clamp_min(0.)
        relative = ((feet - robot.root_link_pos_w[:, None, :]) * self.support_normal).sum(dim=-1)
        observed = legal & (airborne | self.swinging) & active[:, None]
        lift = ((feet - self.support_point) * self.support_normal).sum(dim=-1).clamp_min(0.)
        peak = torch.where(observed, torch.maximum(self.peak, lift), 0.)
        terrain_peak = torch.where(observed, torch.maximum(self.terrain_peak, clearance), 0.)
        relative_peak = torch.where(
            observed,
            torch.maximum(self.relative_peak, (relative - self.support_relative_z).clamp_min(0.)), 0.,
        )
        air_time = self.air_time + airborne.float() * env.step_dt
        landed = (fresh[:, None] & supported & self.swinging & legal & active[:, None]
                  & (air_time >= minimum_air_time - 1.e-6)
                  & (feet[..., 0] - self.support_x >= minimum_foot_progress)
                  & (support_count >= 2)[:, None])
        treads = state["step_surfaces"]
        distance = treads[:, None, :, 0] - local[..., 0, None]
        rise = treads[:, None, :, 2] - (self.support_z - state["corner"][:, 2, None])[..., None]
        near = ((distance >= -foot_radius) & (distance <= lookahead) & (rise > .015)
                & (env.scene.terrain.terrain_types == 0)[:, None, None])
        required = torch.where(near, rise + riser_clearance, target_clearance).max(dim=-1).values
        target_peak = torch.where(airborne | self.swinging, torch.maximum(self.target_peak, required), target_clearance)
        deficit = ((target_peak - peak).clamp_min(0.) / target_peak.clamp_min(target_clearance)).square().clamp_max(1.)
        event_cost = torch.where(landed, deficit, 0.).sum(dim=1) * landing_cost_seconds / env.step_dt
        urgency = ((lookahead - distance) / (lookahead + foot_radius)).clamp(0., 1.)
        missing = (treads[:, None, :, 2] + riser_clearance - local[..., 2, None]).clamp_min(0.)
        riser_cost = ((missing / (rise + riser_clearance).clamp_min(riser_clearance)).square().clamp_max(1.)
                      * urgency * near).max(dim=-1).values
        crossing = airborne & (support_count >= 2)[:, None]
        riser_cost = torch.where(crossing, riser_cost, 0.)
        cost = riser_cost.mean(dim=1) + event_cost
        cost = torch.nan_to_num(cost, nan=0., posinf=0., neginf=0.)
        self.cached_cost.copy_(torch.where(fresh, cost, self.cached_cost))
        self.peak_sum += torch.where(landed, terrain_peak, 0.).sum(dim=1)
        self.swing_count += landed.sum(dim=1)
        self.true_lift_sum += torch.where(landed, peak, 0.).sum(dim=1)
        self.leg_lift_sum += torch.where(landed, peak, 0.)
        self.leg_swing_count += landed

        stride = (feet[..., 0] - self.support_x).clamp_min(0.)
        height_ratio = (peak / target_peak.clamp_min(.01)).clamp(0., 1.)
        height_quality = height_ratio.square()
        stride_quality = ((maximum_stride - stride) / (maximum_stride - preferred_stride)).clamp(0., 1.)
        relative_quality = (relative_peak / minimum_relative_lift).clamp(0., 1.)
        root_x = robot.root_link_pos_w[:, 0, None]
        first_support = supported & ~self.anchored & fresh[:, None]
        self.last_foot_frontier = torch.where(first_support, feet[..., 0], self.last_foot_frontier)
        self.last_body_frontier = torch.where(first_support, root_x, self.last_body_frontier)
        advancing = ((feet[..., 0] >= self.last_foot_frontier + minimum_foot_progress)
                     & (root_x >= self.last_body_frontier + minimum_body_progress))
        candidate = (landed & support_ok & (air_time >= minimum_quality_air_time - 1.e-6)
                     & advancing & (stride < maximum_stride)
                     & (relative_peak >= minimum_relative_lift * .2))
        qualified = candidate & (height_ratio >= .8 - 1.e-6) & (relative_quality >= .8)
        score = height_quality * stride_quality * relative_quality * (.25 + .75 * self.swing_push_quality)
        self.pending_score = torch.where(candidate, score, self.pending_score)
        self.pending_qualified = torch.where(candidate, qualified, self.pending_qualified)
        self.pending_renewal = torch.where(candidate, (height_ratio >= .2) & (relative_quality >= .4), self.pending_renewal)
        self.pending_valid |= candidate
        margin = elevation_landing_margin(env, local[..., 0])
        self.pending_landing_margin = torch.where(candidate, margin, self.pending_landing_margin)
        self.pending_landing_margin = torch.where(
            self.pending_valid & fresh[:, None], torch.minimum(self.pending_landing_margin, margin), self.pending_landing_margin,
        )
        self.pending_foot_x = torch.where(candidate, feet[..., 0], self.pending_foot_x)
        self.pending_body_x = torch.where(candidate, root_x, self.pending_body_x)
        continuing = supported & legal & active[:, None]
        self.pending_valid &= continuing
        self.landing_time = torch.where(continuing & self.pending_valid,
                                        self.landing_time + fresh[:, None] * env.step_dt, 0.)
        self.cycle_time += fresh[:, None] * env.step_dt
        mature = (fresh[:, None] & self.pending_valid & (self.landing_time >= minimum_stance_time - 1.e-6)
                  & (self.cycle_time >= minimum_cycle_time))
        body_progress = (self.pending_body_x - self.last_body_frontier).clamp_min(0.)
        budget = (body_progress / reward_progress_scale).clamp(0., 1.)
        positive_allowed = ~state["failed"] & ~state["invalid"] & ~state["complete"]
        stairs = (env.scene.terrain.terrain_types < 2)[:, None]
        landing_quality = torch.where(stairs, (self.pending_landing_margin / landing_edge_margin).clamp(0., 1.), 1.)
        quality_reward = torch.where(mature & positive_allowed[:, None],
                                    self.pending_score * (.5 + .5 * landing_quality) * budget * step_bonus, 0.).sum(dim=1) / env.step_dt
        measured_landing = mature & stairs & positive_allowed[:, None]
        self.landing_margin_sum += torch.where(measured_landing, self.pending_landing_margin, 0.).sum(dim=1)
        self.landing_count += measured_landing.sum(dim=1)
        self.edge_landings += (measured_landing & (self.pending_landing_margin < landing_edge_margin)).sum(dim=1)
        self.gait["quality"].copy_(torch.where(fresh, quality_reward, self.gait["quality"]))
        renewed = mature & self.pending_qualified & positive_allowed[:, None]
        participated = mature & (self.pending_score > 0.) & positive_allowed[:, None]
        refreshed = mature & self.pending_renewal & positive_allowed[:, None]
        self.renewal_age.copy_(torch.where(refreshed, 0., self.renewal_age + fresh[:, None] * env.step_dt))
        recent = ((renewal_timeout - self.renewal_age) / (renewal_timeout - renewal_grace)).clamp(0., 1.)
        self.gait["renewal"].copy_(torch.where(
            fresh, .5 * recent.mean(dim=1) + .5 * recent.min(dim=1).values, self.gait["renewal"],
        ))
        self.participation_age = torch.where(participated, 0., self.participation_age + fresh[:, None] * env.step_dt)
        excess_age = ((self.participation_age - participation_window) / participation_window).clamp(0., 1.).square()
        moving = active & (env.episode_length_buf * env.step_dt >= participation_warmup)
        participation = torch.where(moving, .5 * excess_age.mean(dim=1) + .5 * excess_age.max(dim=1).values, 0.)
        self.gait["participation"].copy_(torch.where(fresh, participation, self.gait["participation"]))
        stride_cost = torch.where(landed, (1. - stride_quality).square(), 0.).sum(dim=1) * landing_cost_seconds / env.step_dt
        self.gait["stride"].copy_(torch.where(fresh, stride_cost, self.gait["stride"]))
        self.last_foot_frontier = torch.where(mature, self.pending_foot_x, self.last_foot_frontier)
        self.last_body_frontier = torch.where(mature, self.pending_body_x, self.last_body_frontier)
        self.cycle_time = torch.where(mature, 0., self.cycle_time)
        self.pending_valid &= ~mature

        log = env.extras["log"]
        for index, leg in enumerate(LEG_NAMES):
            if takeoff[:, index].any():
                mask = takeoff[:, index]
                log[f"Metrics/elevation_stance_time_{leg}_s"] = self.stance_time[mask, index].detach().clone()
                log[f"Metrics/elevation_backstroke_{leg}_m"] = self.stance_travel[mask, index].detach().clone()
                log[f"Metrics/elevation_support_body_rise_{leg}_m"] = root_delta[mask, index, 2].detach().clone()
                log[f"Metrics/elevation_push_quality_{leg}"] = self.swing_push_quality[mask, index].detach().clone()
        log["Metrics/elevation_gait_renewal"] = self.gait["renewal"].detach().clone()
        if measured_landing.any():
            log["Metrics/elevation_landing_edge_margin_m"] = self.pending_landing_margin[measured_landing].detach().clone()
        if landed.any():
            log["Metrics/swing_peak_height"] = terrain_peak[landed].detach().clone()
            log["Metrics/elevation_true_lift_height_m"] = peak[landed].detach().clone()
            log["Metrics/swing_peak_height_relative_body"] = relative_peak[landed].detach().clone()
            for index, leg in enumerate(LEG_NAMES):
                if landed[:, index].any():
                    log[f"Metrics/swing_peak_height_{leg}"] = terrain_peak[landed[:, index], index].detach().clone()
                    log[f"Metrics/elevation_true_lift_{leg}_m"] = peak[landed[:, index], index].detach().clone()
        values = {
            "support_z": torch.where(supported, feet[..., 2], self.support_z),
            "support_x": torch.where(supported, feet[..., 0], self.support_x),
            "support_point": torch.where(supported[..., None], feet, self.support_point),
            "support_relative_z": torch.where(supported,
                ((feet - robot.root_link_pos_w[:, None, :]) * normals).sum(dim=-1), self.support_relative_z),
            "support_normal": torch.where(supported[..., None], normals, self.support_normal),
            "swing_support_ok": torch.where(airborne, support_ok, True),
            "anchored": legal & (self.anchored | supported), "swinging": airborne,
            "air_time": torch.where(airborne, air_time, 0.),
            "peak": torch.where(airborne, peak, 0.),
            "terrain_peak": torch.where(airborne, terrain_peak, 0.),
            "relative_peak": torch.where(airborne, relative_peak, 0.),
            "target_peak": torch.where(airborne, target_peak, target_clearance),
            "was_supported": real_support,
        }
        for key, value in values.items():
            old = getattr(self, key)
            update = fresh[:, None, None] if value.ndim == 3 else fresh[:, None]
            old.copy_(torch.where(update, value, old))
        metrics = {
            "swing_peak_height": self.peak_sum / self.swing_count.clamp_min(1.),
            "true_lift_height": self.true_lift_sum / self.swing_count.clamp_min(1.),
            "landing_edge_margin": self.landing_margin_sum / self.landing_count.clamp_min(1.),
            "near_riser_feet": (near.any(dim=-1) & crossing).sum(dim=1).float(),
            "riser_deficit": riser_cost.mean(dim=1),
            "low_swing_events_per_s": ((deficit > 0.) & landed).sum(dim=1).float() / env.step_dt,
            "completed_swings_per_s": landed.sum(dim=1).float() / env.step_dt,
            "quality_steps_per_s": renewed.sum(dim=1).float() / env.step_dt,
            "worst_leg_cycle_age_s": self.participation_age.max(dim=1).values,
            "stride_length": torch.where(landed, stride, 0.).sum(dim=1) / landed.sum(dim=1).clamp_min(1),
            "target_peak_height": target_peak.max(dim=1).values,
        }
        for key, value in metrics.items():
            self.metrics[key].copy_(torch.where(fresh, value, self.metrics[key]))
        for index, leg in enumerate(LEG_NAMES):
            log[f"Metrics/elevation_cycle_age_{leg}_s"] = self.participation_age[:, index].detach()
            log[f"Metrics/elevation_quality_steps_{leg}_per_s"] = renewed[:, index].float() / env.step_dt
        self.last_step[fresh] = env.common_step_counter
        return self.cached_cost
