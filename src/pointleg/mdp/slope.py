"""Contact-based step timing and hip alignment for ascending and descending ramps."""

import math

import torch

from ..robot_cfg import FOOT_SITE_NAMES, LEG_NAMES
from .elevation import slope_exit_supported_feet, update_elevation_state
from .rewards import hip_abduction_excess


def slope_task_metric(env, key):
    """Expose actual reset assignments and destination-platform load-bearing feet."""
    if key == "exit_loaded_feet":
        return slope_exit_supported_feet(env).sum(dim=1).float()
    if key in ("up_fraction", "down_fraction"):
        kind = 2 if key == "up_fraction" else 3
        return (env.scene.terrain.terrain_types == kind).float()
    raise KeyError(key)


def slope_hip_abduction_zero(env, asset_cfg, tolerance=0., deviation_scale=.25):
    """Keep all four abduction joints near their default zero and report each leg."""
    cost = hip_abduction_excess(env, asset_cfg, tolerance, deviation_scale)
    robot = env.scene[asset_cfg.name]
    for leg in LEG_NAMES:
        index = robot.joint_names.index(f"{leg}_hip_abduction_joint")
        angle = torch.rad2deg(robot.data.joint_pos[:, index])
        env.extras["log"][f"Metrics/slope_hip_abduction_{leg}_deg"] = angle.detach()
    return cost


class StepTimingPenalty:
    """Discourage quick shuffling and overdue steps without imposing a leg sequence."""

    def __init__(self, cfg, env):
        del cfg
        self.foot_ids, _ = env.scene["robot"].find_sites(FOOT_SITE_NAMES, preserve_order=True)
        shape = (env.num_envs, 4)
        for name in ("contact_time", "air_time", "stance_time", "cycle_time", "landing_x"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        for name in ("loaded", "anchored", "took_off"):
            setattr(self, name, torch.zeros(shape, device=env.device, dtype=torch.bool))
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("contact_time", "air_time", "stance_time", "cycle_time", "landing_x",
                     "loaded", "anchored", "took_off", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, command_name="twist", minimum_force=8., target_period=.70,
                 period_tolerance=.10, period_scale=.30, minimum_stance=.30,
                 stance_scale=.20, maximum_air=.55, air_scale=.20,
                 contact_confirmation=.06, release_confirmation=.04,
                 startup_grace=.80, event_seconds=.10, metric_prefix="slope"):
        positive = (minimum_force, target_period, period_tolerance, period_scale,
                    minimum_stance, stance_scale, maximum_air, air_scale,
                    contact_confirmation, release_confirmation, startup_grace, event_seconds)
        if not all(math.isfinite(value) and value > 0. for value in positive):
            raise ValueError("Slope step timing parameters must be finite and positive")
        if not contact_confirmation < minimum_stance < target_period - period_tolerance:
            raise ValueError("Slope stance and contact confirmation must fit within the target period")
        if not metric_prefix:
            raise ValueError("Step timing metric prefix must not be empty")
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        command = env.command_manager.get_command(command_name)
        contact = env.scene["elevation_contacts"].data
        found = contact.found.reshape(env.num_envs, 4, -1)
        force = contact.force.reshape(env.num_envs, 4, -1, 3)
        normal = contact.normal.reshape(env.num_envs, 4, -1, 3)
        feet_x = robot.site_pos_w[:, self.foot_ids, 0]
        finite = (torch.isfinite(command).all(dim=1) & torch.isfinite(feet_x).all(dim=1)
                  & torch.isfinite(found).all(dim=(1, 2))
                  & torch.isfinite(force).all(dim=(1, 2, 3))
                  & torch.isfinite(normal).all(dim=(1, 2, 3)))
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"]
                  & ~state["complete"] & finite & (command[:, 0] > .08))
        self.reset(torch.where(fresh & ~active)[0])
        mask = fresh & active
        raw_loaded = ((found > 0) & (normal[..., 2] < -.5)
                      & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        contact_time = torch.where(raw_loaded, self.contact_time + env.step_dt, 0.)
        air_time = torch.where(raw_loaded, 0., self.air_time + env.step_dt)
        self.contact_time.copy_(torch.where(mask[:, None], contact_time, self.contact_time))
        self.air_time.copy_(torch.where(mask[:, None], air_time, self.air_time))
        loaded = torch.where(raw_loaded, self.loaded | (self.contact_time >= contact_confirmation - 1.e-6),
                             self.loaded & (self.air_time < release_confirmation - 1.e-6))
        landing = mask[:, None] & loaded & ~self.loaded
        takeoff = mask[:, None] & ~loaded & self.loaded
        self.cycle_time += (mask[:, None] & self.anchored).float() * env.step_dt
        completed = landing & self.anchored & self.took_off
        short_cycle = ((target_period - period_tolerance - self.cycle_time) / period_scale).clamp(0., 1.).square()
        short_stance = ((minimum_stance - self.stance_time) / stance_scale).clamp(0., 1.).square()
        ready = mask & (env.episode_length_buf * env.step_dt >= startup_grace)
        event_cost = ((short_cycle * completed).mean(dim=1)
                      + (short_stance * takeoff).mean(dim=1)) * event_seconds / env.step_dt
        for index, leg in enumerate(LEG_NAMES):
            valid = completed[:, index] & ready
            if valid.any():
                env.extras["log"][f"Metrics/{metric_prefix}_cycle_{leg}_s"] = self.cycle_time[valid, index].detach()
                env.extras["log"][f"Metrics/{metric_prefix}_landing_stride_{leg}_x_m"] = (
                    feet_x[valid, index] - self.landing_x[valid, index]).detach()
            ended = takeoff[:, index] & ready
            if ended.any():
                env.extras["log"][f"Metrics/{metric_prefix}_stance_{leg}_s"] = self.stance_time[ended, index].detach()
        self.stance_time.copy_(torch.where(mask[:, None],
            torch.where(loaded, self.stance_time + env.step_dt, 0.), self.stance_time))
        self.cycle_time.masked_fill_(landing, 0.)
        self.landing_x.copy_(torch.where(landing, feet_x.nan_to_num(), self.landing_x))
        self.anchored |= landing
        self.took_off |= takeoff & self.anchored
        self.took_off &= ~landing
        self.loaded.copy_(torch.where(mask[:, None], loaded, self.loaded))
        overdue = ((self.cycle_time - target_period - period_tolerance) / period_scale).clamp(0., 1.).square()
        prolonged_air = ((self.air_time - maximum_air) / air_scale).clamp(0., 1.).square()
        continuous = .5 * overdue.mean(dim=1) + .5 * prolonged_air.max(dim=1).values
        cost = torch.where(ready, event_cost + continuous, 0.)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        return self.cached


SlopeStepTimingPenalty = StepTimingPenalty
