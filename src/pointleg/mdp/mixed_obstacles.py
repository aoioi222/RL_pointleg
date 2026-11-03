"""Per-episode obstacle direction, independent tire reset and crossing checks."""

from dataclasses import dataclass
import math

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from .commands import UniformThresholdVelocityCommand, UniformThresholdVelocityCommandCfg
from .terrain_utils import terrain_type_mask
from ..robot_cfg import FOOT_SITE_NAMES
from ..terrains.mixed_obstacles import (
    FENCE_APPROACH_DISTANCE, FENCE_NAMES, FENCE_X, RING_CENTER_Y, RING_COLUMNS,
    RING_FIRST_X, RING_HEIGHT, RING_LANE_HALF_WIDTH, RING_NAMES,
    RING_APPROACH_DISTANCE, RING_OUTER_RADIUS, RING_ROWS, RING_SPACING,
    RING_TERRAIN, START_X, OBSTACLE_SPEED_RANGE,
    parking_centers, ring_centers, ring_inertia,
)


def obstacle_state(env):
    if not hasattr(env, "_mixed_obstacles"):
        zeros = lambda: torch.zeros(env.num_envs, device=env.device)
        env._mixed_obstacles = {
            "direction": torch.ones(env.num_envs, device=env.device),
            "speed": torch.full((env.num_envs,), 0.25, device=env.device),
            "stable": zeros(), "entered": zeros().bool(), "complete": zeros().bool(),
            "last_step": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            "peak_push": zeros(), "push_cost": zeros(), "peak_tilt": zeros(),
            "ring_initial": torch.zeros(env.num_envs, len(RING_NAMES), 3, device=env.device),
            "ring_active": torch.zeros(env.num_envs, len(RING_NAMES), dtype=torch.bool, device=env.device),
            "end_x": zeros(), "entry_x": zeros(), "start_x": zeros(), "ring_height": zeros(),
            "best_travel": zeros(), "stalled_s": zeros(),
        }
    return env._mixed_obstacles


@requires_model_fields(
    "body_gravcomp", "geom_size", "geom_rbound", "geom_aabb",
    "body_inertia", "body_iquat", "body_ipos", recompute=RecomputeLevel.set_const_0,
)
def reset_mixed_obstacles(env, env_ids, forward_only=False, ring_start_x_range=(-0.3, 0.3)):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    state = obstacle_state(env)
    fence = terrain_type_mask(env, FENCE_NAMES)
    ring = terrain_type_mask(env, (RING_TERRAIN,))
    special = fence | ring
    ring_ids = ids[ring[ids]]
    state["direction"][ids] = 1.0
    fence_ids = ids[fence[ids]]
    if not forward_only:
        state["direction"][fence_ids] = torch.where(torch.rand(len(fence_ids), device=env.device) < 0.5, -1.0, 1.0)
    state["speed"][ids] = torch.empty(len(ids), device=env.device).uniform_(*OBSTACLE_SPEED_RANGE)
    for key in ("stable", "entered", "complete", "peak_push", "push_cost", "peak_tilt", "best_travel", "stalled_s"):
        state[key][ids] = 0
    state["last_step"][ids] = -1
    generator = env.scene.terrain.cfg.terrain_generator
    level = env.scene.terrain.terrain_levels.float()
    low, high = generator.difficulty_range
    difficulty = low + (high - low) * level / max(generator.num_rows - 1, 1)
    progress = (level / max(generator.num_rows - 1, 1)).clamp(0.0, 1.0)
    row_count = (1 + (progress * (RING_ROWS - 1) + 1.0e-6).floor()).long()
    state["ring_height"][ids] = torch.where(ring[ids], RING_HEIGHT * difficulty[ids], 0.0)
    robot = env.scene["robot"]
    origin = env.scene.env_origins
    selected = ids[special[ids]]
    pose = robot.data.default_root_state[selected].clone()
    pose[:, :3] += origin[selected]
    selected_fence = fence[selected]
    pose[selected_fence, 0] += (
        FENCE_X - START_X - state["direction"][selected[selected_fence]] * FENCE_APPROACH_DISTANCE
    )
    pose[ring[selected], 0] += (
        RING_FIRST_X - RING_OUTER_RADIUS - START_X - RING_APPROACH_DISTANCE
        + torch.empty(len(ring_ids), device=env.device).uniform_(*ring_start_x_range)
    )
    pose[:, 3:7] = pose.new_tensor([1.0, 0.0, 0.0, 0.0])
    pose[:, 7:] = 0.0
    robot.write_root_state_to_sim(pose, env_ids=selected)
    state["start_x"][selected] = pose[:, 0]
    state["end_x"][fence_ids] = origin[fence_ids, 0] + FENCE_X - START_X
    state["entry_x"][fence_ids] = state["end_x"][fence_ids]
    state["end_x"][ring_ids] = (
        origin[ring_ids, 0] + RING_FIRST_X - START_X
        + (row_count[ring_ids] - 1) * RING_SPACING + RING_OUTER_RADIUS
    )
    state["entry_x"][ring_ids] = (
        origin[ring_ids, 0] + RING_FIRST_X - START_X - RING_OUTER_RADIUS
    )
    centers = torch.as_tensor(ring_centers(), device=env.device, dtype=torch.float32)
    parking = torch.as_tensor(parking_centers(), device=env.device, dtype=torch.float32)
    for index, name in enumerate(RING_NAMES):
        active = ring[ids] & (index // RING_COLUMNS < row_count[ids])
        height = torch.where(active, RING_HEIGHT * difficulty[ids], RING_HEIGHT)
        positions = parking[index].repeat(len(ids), 1)
        positions[active] = (
            origin[ids[active]] + centers[index]
            - centers.new_tensor([START_X, RING_CENTER_Y, 0.0])
        )
        positions[active, 2] = origin[ids[active], 2] + height[active] / 2
        root = env.scene[name].data.default_root_state[ids].clone()
        root[:, :3] = positions
        root[:, 3:7] = root.new_tensor([1.0, 0.0, 0.0, 0.0])
        root[:, 7:] = 0.0
        env.scene[name].write_root_state_to_sim(root, env_ids=ids)
        env.sim.model.body_gravcomp[ids, env.scene[name].indexing.root_body_id] = (~active).float()
        entity = env.scene[name]
        geom_ids = entity.indexing.geom_ids.to(dtype=ids.dtype)
        world_grid, geom_grid = torch.meshgrid(ids, geom_ids, indexing="ij")
        size = env.sim.model.geom_size[world_grid, geom_grid].clone()
        size[:, :, 2] = height[:, None] / 2
        env.sim.model.geom_size[world_grid, geom_grid] = size
        env.sim.model.geom_rbound[world_grid, geom_grid] = torch.linalg.vector_norm(size, dim=-1)
        env.sim.model.geom_aabb[world_grid, geom_grid, 0] = 0.0
        env.sim.model.geom_aabb[world_grid, geom_grid, 1] = size
        xy, _, z = ring_inertia(height)
        body_id = entity.indexing.root_body_id
        env.sim.model.body_inertia[ids, body_id] = torch.stack((xy, xy, torch.full_like(height, z)), dim=1)
        env.sim.model.body_iquat[ids, body_id] = height.new_tensor([1.0, 0.0, 0.0, 0.0])
        env.sim.model.body_ipos[ids, body_id] = 0.0
        state["ring_initial"][ids, index] = positions
        state["ring_active"][ids, index] = active


class MixedObstacleVelocityCommand(UniformThresholdVelocityCommand):
    def _apply_obstacle_commands(self):
        state = obstacle_state(self._env)
        special = terrain_type_mask(self._env, (*FENCE_NAMES, RING_TERRAIN))
        self.is_standing_env[special] = False
        self.is_forward_env[special] = False
        self.is_backward_env[special] = False
        self.is_lateral_env[special] = False
        self.is_yaw_env[special] = False
        self.is_heading_env[special] = False
        self.vel_command_b[special, 0] = state["direction"][special] * state["speed"][special]
        self.vel_command_b[special, 1:] = 0.0
        self.vel_command_w[special] = self.vel_command_b[special]
        heading = self.robot.data.heading_w[special]
        self.progress_direction_w[special, 0] = heading.cos() * self.vel_command_b[special, 0]
        self.progress_direction_w[special, 1] = heading.sin() * self.vel_command_b[special, 0]

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._apply_obstacle_commands()

    def _update_command(self):
        super()._update_command()
        self._apply_obstacle_commands()


@dataclass(kw_only=True)
class MixedObstacleVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    def build(self, env):
        return MixedObstacleVelocityCommand(self, env)


def update_obstacle_state(env):
    state = obstacle_state(env)
    tick = env.common_step_counter
    fresh = state["last_step"] != tick
    if not fresh.any():
        return state
    fence = terrain_type_mask(env, FENCE_NAMES)
    ring = terrain_type_mask(env, (RING_TERRAIN,))
    special = fence | ring
    if "foot_ids" not in state:
        state["foot_ids"] = [env.scene["robot"].find_sites(name)[0][0] for name in FOOT_SITE_NAMES]
    robot = env.scene["robot"].data
    feet = robot.site_pos_w[:, state["foot_ids"]]
    direction = state["direction"]
    signed_travel = direction * (robot.root_link_pos_w[:, 0] - state["start_x"])
    advanced = signed_travel >= state["best_travel"] + .03
    state["best_travel"] = torch.where(fresh & advanced, signed_travel, state["best_travel"])
    state["stalled_s"] = torch.where(
        fresh, torch.where(special & ~advanced, state["stalled_s"] + env.step_dt, 0.), state["stalled_s"],
    )
    root_progress = direction * (robot.root_link_pos_w[:, 0] - state["end_x"])
    foot_progress = direction[:, None] * (feet[:, :, 0] - state["end_x"][:, None])
    push = torch.stack([
        torch.linalg.vector_norm(env.scene[name].data.root_link_pos_w[:, :2] - state["ring_initial"][:, i, :2], dim=1)
        for i, name in enumerate(RING_NAMES)
    ], dim=1)
    push = torch.where(state["ring_active"], push, 0.0)
    peak = torch.maximum(state["peak_push"], push.max(dim=1).values)
    state["peak_push"] = torch.where(fresh, peak, state["peak_push"])
    cost = ((push - 0.01).clamp_min(0.0) / 0.04).square().clamp_max(4.0).sum(dim=1)
    up = torch.stack([
        1.0 - 2.0 * env.scene[name].data.root_link_quat_w[:, 1:3].square().sum(dim=1)
        for name in RING_NAMES
    ], dim=1)
    tilt = torch.where(state["ring_active"], torch.acos(up.clamp(-1.0, 1.0)), 0.0)
    state["peak_tilt"] = torch.where(fresh, torch.maximum(state["peak_tilt"], torch.rad2deg(tilt).max(dim=1).values), state["peak_tilt"])
    cost += ((tilt - math.radians(5)).clamp_min(0) / math.radians(15)).square().clamp_max(4).sum(dim=1)
    state["push_cost"] = torch.where(fresh, cost, state["push_cost"])
    entry_progress = direction[:, None] * (feet[:, :, 0] - state["entry_x"][:, None])
    entered = (entry_progress.max(dim=1).values > 0.0) & special
    state["entered"] |= entered & fresh
    lateral = (robot.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]).abs()
    completion_half_width = torch.where(ring, RING_LANE_HALF_WIDTH - 0.10, 0.65)
    supported = (env.scene["feet_ground_contact"].data.found > 0).sum(dim=1) >= 2
    base_clear = ~(env.scene["base_ground_contact"].data.found > 0).any(dim=1)
    stable = (
        special & state["entered"] & (root_progress > 0.60)
        & (foot_progress.min(dim=1).values > 0.10)
        & (robot.projected_gravity_b[:, 2] < -math.cos(math.radians(15.0)))
        & (lateral < completion_half_width) & supported & base_clear
        & (~ring | ((state["peak_push"] <= 0.05) & (state["peak_tilt"] <= 15.0)))
    )
    state["stable"] = torch.where(fresh, torch.where(stable, state["stable"] + env.step_dt, 0.0), state["stable"])
    state["complete"] |= fresh & (state["stable"] >= 0.5)
    state["last_step"][:] = tick
    return state


def obstacle_completed(env):
    return update_obstacle_state(env)["complete"]


def obstacle_stalled(env, timeout_s=8.0):
    state = update_obstacle_state(env)
    special = terrain_type_mask(env, (*FENCE_NAMES, RING_TERRAIN))
    return special & ~state["complete"] & (state["stalled_s"] >= timeout_s)


def obstacle_failed(env):
    state = update_obstacle_state(env)
    special = terrain_type_mask(env, (*FENCE_NAMES, RING_TERRAIN))
    ring = terrain_type_mask(env, (RING_TERRAIN,))
    robot = env.scene["robot"].data
    lateral = (robot.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]).abs()
    failure_half_width = torch.where(ring, RING_LANE_HALF_WIDTH, 0.75)
    signed_travel = state["direction"] * (robot.root_link_pos_w[:, 0] - state["start_x"])
    ring_valid = torch.stack([
        torch.isfinite(env.scene[name].data.root_link_pos_w).all(dim=1)
        & torch.isfinite(env.scene[name].data.root_link_lin_vel_w).all(dim=1)
        for name in RING_NAMES
    ], dim=1).all(dim=1)
    return special & ((lateral > failure_half_width) | (signed_travel < -0.35) | ~ring_valid)


def obstacle_completion_reward(env):
    return obstacle_completed(env).float() / env.step_dt


def ring_push_penalty(env):
    return update_obstacle_state(env)["push_cost"]


def obstacle_metric(env, key):
    return update_obstacle_state(env)[key].float()


def protect_obstacles_from_push(env, env_ids, velocity_range, asset_cfg):
    from mjlab.envs.mdp import push_by_setting_velocity
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    special = terrain_type_mask(env, (*FENCE_NAMES, RING_TERRAIN))
    ids = env_ids[~special[env_ids]]
    if len(ids):
        push_by_setting_velocity(env, ids, velocity_range=velocity_range, asset_cfg=asset_cfg)
