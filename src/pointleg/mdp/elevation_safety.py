"""Geometry-based belly clearance and substep-latched landing costs."""

import math

import mujoco
import numpy as np
import torch

from mjlab.utils.lab_api.math import matrix_from_quat

from .elevation import elevation_state
from .elevation_clearance import elevation_surface_height
from .elevation_gait import elevation_surface_frame
from ..robot_cfg import BASE_BODY_NAME, LEG_NAMES
from ..terrains.elevation import CENTER_Y, LANE_WIDTH, TILE_SIZE


def walking_active(env):
    state = elevation_state(env)
    return (state["initialized"] & ~state["failed"] & ~state["invalid"]
            & ~state["complete"] & torch.isfinite(state["speed"]) & (state["speed"] > .1))


class ElevationBodyClearancePenalty:
    """Sample the actual base collision boxes instead of prescribing root height."""

    def __init__(self, cfg, env):
        if any(not math.isfinite(v) or v <= 0. for v in cfg.params.values()):
            raise ValueError("Body clearance parameters must be finite and positive")
        robot = env.scene["robot"]
        body_ids, _ = robot.find_bodies(BASE_BODY_NAME)
        if len(body_ids) != 1:
            raise ValueError("Body clearance requires one base body")
        host = env.sim.mj_model
        geoms = robot.indexing.geom_ids.cpu().numpy()
        body = int(robot.indexing.body_ids[body_ids[0]])
        enabled = (host.geom_contype[geoms] != 0) | (host.geom_conaffinity[geoms] != 0)
        boxes = enabled & (host.geom_bodyid[geoms] == body) & (host.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_BOX)
        indices = np.flatnonzero(boxes)
        if not len(indices):
            raise ValueError("Body clearance requires a base collision box")
        self.ids = torch.tensor(indices, device=env.device)
        sizes = torch.as_tensor(host.geom_size[geoms[boxes]].copy(), device=env.device, dtype=torch.float32)
        grid = sizes.new_tensor([(x, y, -1.) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
        self.points = sizes[:, None, :] * grid[None]
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.minimum = torch.zeros_like(self.cached)
        self.episode_minimum = torch.full_like(self.cached, torch.inf)
        self.samples = torch.zeros_like(self.cached)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        env._elevation_body_clearance_term = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.cached[ids] = self.minimum[ids] = self.samples[ids] = 0.
        self.episode_minimum[ids] = torch.inf
        self.last_step[ids] = -1

    def __call__(self, env, minimum_clearance, deviation_scale):
        active = walking_active(env)
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return torch.where(active, self.cached, 0.)
        state = elevation_state(env)
        pose = env.scene["robot"].data.geom_pose_w[:, self.ids]
        finite = torch.isfinite(pose).all(dim=(1, 2)) & torch.isfinite(state["corner"]).all(dim=1)
        safe = pose.nan_to_num(nan=0., posinf=0., neginf=0.)
        rotation = matrix_from_quat(safe[..., 3:7])
        points = safe[..., None, :3] + torch.einsum("bgij,gpj->bgpi", rotation, self.points)
        local = points.flatten(1, 2) - state["corner"][:, None, :]
        inside = ((local[..., 1] - CENTER_Y).abs() <= LANE_WIDTH / 2) & (local[..., 0] >= 0.) & (local[..., 0] <= TILE_SIZE[0])
        height = torch.where(inside, elevation_surface_height(env, local[..., 0]), 0.)
        _, normal = elevation_surface_frame(env, local[..., 0])
        normal_z = torch.where(inside, normal[..., 2], 1.)
        minimum = ((local[..., 2] - height) * normal_z).min(dim=1).values
        finite &= torch.isfinite(minimum)
        measured = fresh & finite & state["initialized"]
        value = ((minimum_clearance - minimum.nan_to_num()) / deviation_scale).clamp(0., 1.).square()
        self.cached.copy_(torch.where(fresh, torch.where(finite & active, value, 0.), self.cached))
        self.minimum.copy_(torch.where(measured, minimum, self.minimum))
        self.episode_minimum.copy_(torch.where(measured, torch.minimum(self.episode_minimum, minimum), self.episode_minimum))
        self.samples += measured
        self.last_step[fresh] = env.common_step_counter
        if measured.any():
            env.extras["log"]["Metrics/elevation_body_min_clearance_m"] = minimum[measured].detach().clone()
        return torch.where(active, self.cached, 0.)


class ElevationTouchdownSpeedPenalty:
    """Latch pre-contact sphere-center velocity once for each physical landing."""

    def __init__(self, cfg, env):
        if any(not math.isfinite(v) or v <= 0. for v in cfg.params.values()):
            raise ValueError("Touchdown parameters must be finite and positive")
        robot = env.scene["robot"]
        self.ids = []
        for leg in LEG_NAMES:
            ids, _ = robot.find_sites(f"{leg}_foot_contact_site")
            if len(ids) != 1:
                raise ValueError("Touchdown speed requires four sphere-center sites")
            self.ids.append(ids[0])
        shape = (env.num_envs, 4)
        self.previous_velocity = torch.zeros((*shape, 3), device=env.device)
        for name in ("previous_valid", "was_supported"):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=env.device))
        for name in ("pending_cost", "pending_speed", "pending_count", "speed_sum", "count", "maximum"):
            setattr(self, name, torch.zeros(shape, device=env.device))
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.last_physics_step = torch.full_like(self.last_step, -1)
        self.params = dict(cfg.params)
        env._elevation_touchdown_term = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("previous_velocity", "previous_valid", "was_supported", "pending_cost",
                     "pending_speed", "pending_count", "speed_sum", "count", "maximum", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = self.last_physics_step[ids] = -1

    def collect(self, env):
        counter = env._sim_step_counter
        fresh = self.last_physics_step != counter
        if not fresh.any():
            return
        robot = env.scene["robot"].data
        velocity = robot.site_lin_vel_w[:, self.ids]
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normals = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        finite_contact = torch.isfinite(force).all(dim=-1) & torch.isfinite(normals).all(dim=-1)
        norm = normals.nan_to_num().norm(dim=-1)
        support = (found & finite_contact & (norm > 1.e-6) & (normals[..., 2] < -.5)
                   & (force.nan_to_num().norm(dim=-1) > env._elevation_settings.contact_force_threshold))
        loaded = support.any(dim=2)
        normal = torch.where(support[..., None], normals / norm.clamp_min(1.e-6)[..., None], 0.).sum(dim=2)
        normal /= normal.norm(dim=-1, keepdim=True).clamp_min(1.e-6)
        active = walking_active(env)[:, None]
        event = loaded & ~self.was_supported & self.previous_valid & active & fresh[:, None]
        approach = (self.previous_velocity * normal).sum(dim=-1).clamp_min(0.)
        event &= torch.isfinite(approach)
        cost = ((approach - self.params["speed_tolerance"]).clamp_min(0.) / self.params["deviation_scale"]).square().clamp_max(1.)
        self.pending_cost += torch.where(event, cost, 0.)
        self.pending_speed += torch.where(event, approach, 0.)
        self.pending_count += event
        self.speed_sum += torch.where(event, approach, 0.)
        self.count += event
        self.maximum.copy_(torch.where(event, torch.maximum(self.maximum, approach), self.maximum))
        valid_velocity = torch.isfinite(velocity).all(dim=-1) & active
        self.previous_velocity.copy_(torch.where(fresh[:, None, None], velocity.nan_to_num(), self.previous_velocity))
        self.previous_valid.copy_(torch.where(fresh[:, None], valid_velocity, self.previous_valid))
        self.was_supported.copy_(torch.where(fresh[:, None], loaded & active, self.was_supported))
        self.last_physics_step[fresh] = counter

    def __call__(self, env, speed_tolerance, deviation_scale, event_seconds):
        active = walking_active(env)
        fresh = self.last_step != env.common_step_counter
        if fresh.any():
            value = self.pending_cost.sum(dim=1) * event_seconds / env.step_dt
            self.cached.copy_(torch.where(fresh, torch.where(active, value, 0.), self.cached))
            for i, leg in enumerate(LEG_NAMES):
                seen = fresh & (self.pending_count[:, i] > 0)
                if seen.any():
                    env.extras["log"][f"Metrics/elevation_touchdown_speed_{leg}_m_s"] = (
                        self.pending_speed[seen, i] / self.pending_count[seen, i]).detach().clone()
            self.pending_cost[fresh] = self.pending_speed[fresh] = self.pending_count[fresh] = 0.
            self.last_step[fresh] = env.common_step_counter
        return torch.where(active, self.cached, 0.)
