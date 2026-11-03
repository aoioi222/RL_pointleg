"""Local swing clearance and bounded progress during obstacle recovery."""

import math

import torch

from .terrain_utils import terrain_type_mask
from ..robot_cfg import FOOT_BODY_NAMES, LEG_NAMES
from ..terrains.mixed_obstacles import FENCE_NAMES, RING_TERRAIN


FOOT_CENTER_SITES = tuple(f"{leg}_foot_contact_site" for leg in LEG_NAMES)


def obstacle_recovery_metric(env, key):
    return env._obstacle_foot_recovery_metrics[key]


class ObstacleFootRecoveryReward:
    """Shape approach clearance, upward progress and supported recovery."""

    def __init__(self, cfg, env):
        params = cfg.params
        positive = (
            "clearance_m", "foot_radius", "block_duration", "recovery_window",
            "replant_duration", "minimum_foot_progress", "minimum_body_progress",
            "minimum_horizontal_force", "minimum_support_force", "maximum_foot_speed",
            "horizontal_force_ratio",
            "minimum_barrier_height", "minimum_relative_lift", "cooldown_s",
            "reward_spacing", "height_cost", "approach_height_cost",
        )
        if any(not math.isfinite(params[key]) or params[key] <= 0 for key in positive):
            raise ValueError("Foot recovery distances, forces and times must be positive.")
        if (
            not math.isfinite(params["recovery_bonus"])
            or not math.isfinite(params["collision_cost"])
            or not math.isfinite(params["lift_progress_bonus"])
            or params["recovery_bonus"] < 0 or params["lift_progress_bonus"] < 0
            or params["recovery_bonus"] + params["lift_progress_bonus"] > params["collision_cost"]
        ):
            raise ValueError("Lift and recovery bonuses cannot exceed the initial blocking cost.")
        sensor = env.scene[params["contact_sensor_name"]]
        self.contact_ids = []
        for body in FOOT_BODY_NAMES:
            matches = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
            if len(matches) != 1:
                raise ValueError(f"Expected one world-force channel for {body}.")
            self.contact_ids.append(matches[0])
        self.site_ids = []
        for name in FOOT_CENTER_SITES:
            ids, _ = env.scene["robot"].find_sites(name)
            if len(ids) != 1:
                raise ValueError(f"Expected one foot sphere-center site for {name}.")
            self.site_ids.append(ids[0])
        shape = (env.num_envs, 4)
        for name in (
            "dwell", "remaining", "cooldown", "target_z", "required_lift",
            "start_relative_z", "start_foot_x", "start_root_x", "direction",
            "landing_time", "last_success_x", "best_lift",
        ):
            setattr(self, name, torch.zeros(shape, device=env.device))
        for name in ("active", "lifted", "support_ok", "has_success", "previous_valid"):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=env.device))
        self.previous_xy = torch.zeros((*shape, 2), device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.cached_reward = torch.zeros(env.num_envs, device=env.device)
        self.metrics = {
            key: torch.zeros(env.num_envs, device=env.device)
            for key in (
                "blocked_events_per_s", "recovered_events_per_s",
                "active_feet", "clearance_gap_m", "near_obstacle_feet",
                "side_contact_feet", "stalled_feet",
                "approach_feet", "lift_progress", "cleared_events_per_s",
            )
        }
        env._obstacle_foot_recovery_metrics = self.metrics

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            "dwell", "remaining", "cooldown", "target_z", "required_lift",
            "start_relative_z", "start_foot_x", "start_root_x", "direction",
            "landing_time", "last_success_x", "best_lift", "previous_xy", "cached_reward",
            "active", "lifted", "support_ok", "has_success", "previous_valid",
        ):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1
        for value in self.metrics.values():
            value[ids] = 0

    def __call__(
        self, env, contact_sensor_name, height_sensor_name, command_name,
        clearance_m, foot_radius, block_duration, recovery_window, replant_duration,
        minimum_foot_progress, minimum_body_progress, minimum_horizontal_force,
        horizontal_force_ratio,
        minimum_support_force, maximum_foot_speed, minimum_barrier_height,
        minimum_relative_lift, cooldown_s, reward_spacing, height_cost,
        collision_cost, recovery_bonus, approach_height_cost, lift_progress_bonus,
    ):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached_reward
        update = fresh[:, None]
        robot = env.scene["robot"].data
        center = robot.site_pos_w[:, self.site_ids]
        bottom = center[:, :, 2] - foot_radius
        force_data = env.scene[contact_sensor_name].data
        force = force_data.force[:, self.contact_ids]
        contact = force_data.found[:, self.contact_ids] > 0
        finite = torch.isfinite(center).all(dim=-1) & torch.isfinite(force).all(dim=-1)
        force = torch.nan_to_num(force, nan=0., posinf=0., neginf=0.)
        vertical = force[:, :, 2].abs()
        horizontal = torch.linalg.vector_norm(force[:, :, :2], dim=-1)
        support = (
            contact & finite & (vertical >= minimum_support_force)
            & (vertical >= horizontal)
        )
        enough_support = support.sum(dim=1, keepdim=True) - support.long() >= 2
        command = env.command_manager.get_command(command_name)
        direction = command[:, 0].sign()
        special = terrain_type_mask(env, (*FENCE_NAMES, RING_TERRAIN))
        terminated = getattr(env, "reset_buf", torch.zeros_like(special))
        valid_root = torch.isfinite(robot.root_link_pos_w).all(dim=-1)
        eligible = (
            special & ~terminated & valid_root
            & (command[:, 0].abs() >= .10) & (command[:, 1].abs() <= .05)
            & (command[:, 2].abs() <= math.radians(8.))
            & (robot.projected_gravity_b[:, 2] < -math.cos(math.radians(30.)))
        )[:, None] & finite
        rays = env.scene[height_sensor_name].data
        hit = rays.hit_pos_w.reshape(env.num_envs, 4, -1, 3)
        normals = rays.normals_w.reshape_as(hit)
        distance = rays.distances.reshape(env.num_envs, 4, -1)
        ahead = direction[:, None, None] * (hit[:, :, :, 0] - center[:, :, None, 0])
        valid_surface = (
            (distance >= 0.) & torch.isfinite(distance)
            & torch.isfinite(hit).all(dim=-1) & torch.isfinite(normals).all(dim=-1)
            & (normals[:, :, :, 2] > .5)
        )
        valid_hit = (
            valid_surface & (ahead >= -.005) & (ahead <= .10)
            & ((hit[:, :, :, 1] - center[:, :, None, 1]).abs() <= .05)
        )
        top = torch.where(valid_hit, hit[:, :, :, 2], -torch.inf).amax(dim=-1)
        top_valid = torch.isfinite(top)
        top = torch.where(top_valid, top, bottom)
        speed = (center[:, :, 0] - self.previous_xy[:, :, 0]).abs() / env.step_dt
        near_obstacle = eligible & top_valid & (top - bottom >= minimum_barrier_height)
        side_contact = (
            eligible & contact & (horizontal >= minimum_horizontal_force)
            & (horizontal >= horizontal_force_ratio * vertical)
        )
        stalled = near_obstacle & side_contact & (speed <= maximum_foot_speed)
        # Scraping along an obstacle can move the foot without clearing its edge.
        blocked = near_obstacle & side_contact & self.previous_valid & enough_support
        ground = torch.where(valid_surface, hit[:, :, :, 2], torch.inf).amin(dim=-1)
        edge_height = top - ground
        approach = (
            eligible & enough_support & top_valid & torch.isfinite(ground)
            & (edge_height >= minimum_barrier_height) & (~contact | side_contact)
        )
        approach_gap = (top + clearance_m - bottom).clamp_min(0.)
        approach_gap = torch.where(approach_gap <= 1.e-6, 0., approach_gap)
        approach_deficit = (
            approach_gap / (edge_height + clearance_m).clamp_min(clearance_m)
        ).clamp(0., 1.).square()
        approach_deficit = torch.where(approach, approach_deficit, 0.)
        cooldown = (self.cooldown - env.step_dt).clamp_min(0.)
        remaining = (self.remaining - env.step_dt).clamp_min(0.)
        active = (
            self.active & eligible & (remaining > 0.)
            & (self.direction == direction[:, None])
        )
        expired = self.active & ~active
        cooldown = torch.where(expired, cooldown_s, cooldown)
        dwell = torch.where(blocked & ~active, self.dwell + env.step_dt, 0.)
        new_position = ~self.has_success | (
            direction[:, None] * (center[:, :, 0] - self.last_success_x) >= reward_spacing
        )
        triggered = (
            update & blocked & ~active & (cooldown <= 0.) & new_position
            & (dwell >= block_duration - 1.e-6)
        )
        active |= triggered
        remaining = torch.where(triggered, recovery_window, remaining)
        target_z = torch.where(triggered, top + clearance_m, self.target_z)
        required_lift = torch.where(triggered, target_z - bottom, self.required_lift)
        relative_z = bottom - robot.root_link_pos_w[:, 2, None]
        start_relative_z = torch.where(triggered, relative_z, self.start_relative_z)
        start_foot_x = torch.where(triggered, center[:, :, 0], self.start_foot_x)
        start_root_x = torch.where(triggered, robot.root_link_pos_w[:, 0, None], self.start_root_x)
        attempt_direction = torch.where(triggered, direction[:, None], self.direction)
        support_ok = enough_support
        relative_lift = relative_z - start_relative_z
        cleared = (
            active & ~contact & enough_support & support_ok
            & (bottom >= target_z - 1.e-6)
            & (relative_lift >= minimum_relative_lift - 1.e-6)
        )
        cleared_event = cleared & ~torch.where(triggered, False, self.lifted)
        lifted = torch.where(triggered, False, self.lifted) | cleared
        foot_progress = attempt_direction * (center[:, :, 0] - start_foot_x)
        body_progress = attempt_direction * (robot.root_link_pos_w[:, 0, None] - start_root_x)
        landed = (
            active & lifted & support_ok & enough_support & support
            & (foot_progress >= minimum_foot_progress)
            & (body_progress >= minimum_body_progress)
        )
        ring = terrain_type_mask(env, (RING_TERRAIN,))
        obstacle = env._mixed_obstacles
        gentle = (~ring | ((obstacle["peak_push"] <= .05) & (obstacle["peak_tilt"] <= 15.)))[:, None]
        landed &= gentle
        old_best = torch.where(triggered, 0., self.best_lift)
        height_progress = (bottom - (target_z - required_lift)) / required_lift.clamp_min(clearance_m)
        relative_progress = relative_lift / minimum_relative_lift
        lift_progress = torch.minimum(height_progress, relative_progress).clamp(0., 1.)
        lift_progress = torch.where(lift_progress >= 1. - 1.e-6, 1., lift_progress)
        progress_valid = active & eligible & enough_support & ~support & gentle
        best_lift = torch.where(progress_valid, torch.maximum(old_best, lift_progress), old_best)
        lift_credit = torch.where(progress_valid, best_lift - old_best, 0.)
        landing_time = torch.where(landed, self.landing_time + env.step_dt, 0.)
        recovered = update & landed & (landing_time >= replant_duration - 1.e-6)
        gap = (target_z - bottom).clamp_min(0.)
        height_deficit = (gap / required_lift.clamp_min(clearance_m)).clamp(0., 1.)
        relative_deficit = (1. - relative_lift / minimum_relative_lift).clamp(0., 1.)
        deficit = torch.maximum(height_deficit, relative_deficit).square()
        pending = active & ~lifted
        deficit = torch.where(pending & eligible, deficit, 0.)
        # The recovery target is latched; do not charge both objectives for one foot.
        approach_deficit = torch.where(active, 0., approach_deficit)
        reward = (
            (recovery_bonus * recovered.sum(dim=1) + lift_progress_bonus * lift_credit.sum(dim=1)
             - collision_cost * triggered.sum(dim=1)) / env.step_dt
            - height_cost * deficit.sum(dim=1)
            - approach_height_cost * approach_deficit.sum(dim=1)
        )
        values = {
            "active": active & ~recovered,
            "dwell": torch.where(triggered | active, 0., dwell),
            "remaining": remaining,
            "cooldown": torch.where(recovered, cooldown_s, cooldown),
            "target_z": target_z, "required_lift": required_lift,
            "start_relative_z": start_relative_z,
            "start_foot_x": start_foot_x, "start_root_x": start_root_x,
            "direction": attempt_direction, "landing_time": landing_time,
            "best_lift": best_lift,
            "lifted": lifted & active, "support_ok": support_ok,
            "last_success_x": torch.where(recovered, center[:, :, 0], self.last_success_x),
            "has_success": self.has_success | recovered,
            "previous_xy": center[:, :, :2], "previous_valid": finite,
        }
        for name, value in values.items():
            mask = update[:, :, None] if value.ndim == 3 else update
            getattr(self, name).copy_(torch.where(mask, value, getattr(self, name)))
        metrics = {
            "blocked_events_per_s": triggered.sum(dim=1) / env.step_dt,
            "recovered_events_per_s": recovered.sum(dim=1) / env.step_dt,
            "active_feet": (active & ~recovered).sum(dim=1).float(),
            "clearance_gap_m": torch.where(pending, gap, 0.).sum(dim=1) / pending.sum(dim=1).clamp_min(1),
            "near_obstacle_feet": near_obstacle.sum(dim=1).float(),
            "side_contact_feet": side_contact.sum(dim=1).float(),
            "stalled_feet": stalled.sum(dim=1).float(),
            "approach_feet": approach.sum(dim=1).float(),
            "lift_progress": lift_credit.sum(dim=1),
            "cleared_events_per_s": cleared_event.sum(dim=1) / env.step_dt,
        }
        for key, value in metrics.items():
            self.metrics[key].copy_(torch.where(fresh, value, self.metrics[key]))
        finite_reward = torch.nan_to_num(reward, nan=0., posinf=0., neginf=0.)
        self.cached_reward.copy_(torch.where(fresh, finite_reward, self.cached_reward))
        self.last_step[fresh] = env.common_step_counter
        return self.cached_reward
