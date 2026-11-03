"""Independent forward tracking and load-bearing support for paired terrain tasks."""

import math

import torch

from .elevation import update_elevation_state
from .high_step_basic import basic_high_step_velocity


def elevation_basic_velocity(env, command_name="twist", std=.10):
    """Track forward speed without multiplying by step or posture quality."""
    state = update_elevation_state(env)
    allowed = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
    score = basic_high_step_velocity(env, command_name, std)
    return torch.where(allowed, score, 0.)


def elevation_basic_support_deficit(env, minimum_force=8.):
    """Penalize fewer than two feet carrying load on ground-facing contacts."""
    if not math.isfinite(minimum_force) or minimum_force <= 0.:
        raise ValueError("Support force must be finite and positive")
    sensor = env.scene["elevation_contacts"].data
    found = sensor.found.reshape(env.num_envs, 4, -1) > 0
    force = sensor.force.reshape(env.num_envs, 4, -1, 3)
    normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
    finite = torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
    loaded = (found & finite & (normal[..., 2] < -.5)
              & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
    count = loaded.sum(dim=1)
    env.extras["log"]["Metrics/elevation_loaded_feet"] = count.detach().float()
    return (2 - count).clamp_min(0).float()


class ElevationBasicStallPenalty:
    """Penalize a forward-commanded robot that remains nearly stationary."""

    def __init__(self, cfg, env):
        del cfg
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.elapsed[ids] = 0.
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, command_name="twist", minimum_forward_velocity=.05,
                 grace_period=.60, rise_time=.40):
        if (not math.isfinite(minimum_forward_velocity) or minimum_forward_velocity <= 0.
                or not math.isfinite(grace_period) or grace_period < 0.
                or not math.isfinite(rise_time) or rise_time <= 0.):
            raise ValueError("Stall penalty parameters must be finite and positive.")
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        command = env.command_manager.get_command(command_name)
        velocity = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
        finite = torch.isfinite(command).all(dim=1) & torch.isfinite(velocity)
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"]
                  & ~state["complete"] & finite & (command[:, 0] > .08))
        stationary = active & (velocity < minimum_forward_velocity)
        elapsed = torch.where(stationary, self.elapsed + env.step_dt, torch.zeros_like(self.elapsed))
        self.elapsed.copy_(torch.where(fresh, elapsed, self.elapsed))
        cost = ((self.elapsed - grace_period) / rise_time).clamp(0., 1.)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        env.extras["log"]["Metrics/elevation_stall_duration_s"] = self.elapsed.detach().mean()
        return self.cached


class ElevationBasicProlongedSwingPenalty:
    """Penalize one foot remaining airborne while the body is stalled."""

    def __init__(self, cfg, env):
        del cfg
        self.air_time = torch.zeros(env.num_envs, 4, device=env.device)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.air_time[ids] = 0.
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, command_name="twist", minimum_force=8.,
                 minimum_forward_velocity=.05, max_air_time=.35, rise_time=.20):
        parameters = (minimum_force, minimum_forward_velocity, max_air_time, rise_time)
        if not all(math.isfinite(value) and value > 0. for value in parameters):
            raise ValueError("Prolonged swing parameters must be finite and positive.")
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        command = env.command_manager.get_command(command_name)
        velocity = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        finite_contact = torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
        loaded = (found & finite_contact & (normal[..., 2] < -.5)
                  & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        finite = torch.isfinite(command).all(dim=1) & torch.isfinite(velocity)
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"]
                  & ~state["complete"] & finite & (command[:, 0] > .08)
                  & (velocity < minimum_forward_velocity))
        air_time = torch.where(active[:, None] & ~loaded, self.air_time + env.step_dt,
                               torch.zeros_like(self.air_time))
        self.air_time.copy_(torch.where(fresh[:, None], air_time, self.air_time))
        excess = ((self.air_time - max_air_time) / rise_time).clamp(0., 1.)
        cost = excess.max(dim=1).values
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        env.extras["log"]["Metrics/elevation_max_stalled_swing_s"] = self.air_time.max(dim=1).values.detach().mean()
        return self.cached
