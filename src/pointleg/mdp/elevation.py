"""Contact-grounded elevation tasks, one-shot rewards and per-type curricula."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat, matrix_from_quat

from ..robot_cfg import BASE_BODY_NAME, EFFORT_LEG, FOOT_SITE_NAMES
from ..terrains.elevation import (
    CENTER_Y, FLAT_NAME, LANE_WIDTH, STAIR_TREAD_COUNT, TERRAIN_NAMES, WALL_CENTERS, WALL_NAME,
    WALL_THICKNESS,
    walking_terrain_names,
)
from .commands import UniformThresholdVelocityCommand, UniformThresholdVelocityCommandCfg
from .terminations import joint_pos_out_of_limits


@dataclass(frozen=True)
class ElevationSettings:
    jump: bool = False
    speed_range: tuple[float, float] = (0.2, 0.5)
    trigger_distance_range: tuple[float, float] = (0.9, 1.1)
    initial_forward_speed_range: tuple[float, float] = (0.0, 0.15)
    approach_offset_range: tuple[float, float] = (-0.3, 0.3)
    walking_approach_distance: float = 1.0
    lateral_reset_range: float = 0.015
    yaw_reset_range: float = math.radians(2)
    contact_force_threshold: float = 1.0
    minimum_flight_duration: float = 0.01
    top_contact_tolerance: float = 0.008
    stable_duration: float = 0.5
    jump_handoff_duration: float = 0.12
    stall_grace_s: float = 1.0
    stall_window_s: float = 3.0
    stall_min_progress_ratio: float = 0.2
    stall_min_distance_m: float = 0.05
    gamma: float = 0.995
    terrain_kind: str | None = None

    def __post_init__(self):
        if self.terrain_kind is not None:
            if self.jump:
                raise ValueError("Jump cannot select a walking terrain scope")
            walking_terrain_names(self.terrain_kind)
        for name in ("speed_range", "trigger_distance_range", "initial_forward_speed_range"):
            low, high = getattr(self, name)
            if not math.isfinite(low + high) or low > high or low < 0:
                raise ValueError(f"Invalid {name}")
        if self.speed_range[0] <= 0 or not 0 < self.gamma < 1:
            raise ValueError("Elevation needs positive forward speed and a valid discount")
        low, high = self.approach_offset_range
        if not math.isfinite(low + high) or not -0.3 <= low <= high <= 0.3:
            raise ValueError("Approach offsets must lie within [-0.3, 0.3] meters")
        for name in ("contact_force_threshold", "minimum_flight_duration", "stable_duration", "jump_handoff_duration",
                     "stall_window_s", "stall_min_distance_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.stall_grace_s) or self.stall_grace_s < 0:
            raise ValueError("stall_grace_s must be finite and nonnegative")
        if not math.isfinite(self.walking_approach_distance) or self.walking_approach_distance < .9:
            raise ValueError("Walking approach distance must reserve at least 0.9 m before reset offsets")
        if not math.isfinite(self.stall_min_progress_ratio) or not 0 < self.stall_min_progress_ratio <= 1:
            raise ValueError("stall_min_progress_ratio must lie in (0, 1]")


def _ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
    return env_ids


def elevation_state(env):
    if hasattr(env, "_elevation"):
        return env._elevation
    count, device = env.num_envs, env.device
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros((count, *shape), device=device, dtype=dtype)
    state = {name: zeros() for name in (
        "start_x", "entry_x", "exit_x", "wall_x", "wall_height", "top_height",
        "angle", "speed", "stable", "air_time", "pending_new", "new_events", "stair_platform_length",
        "phi", "shaping", "slip", "max_slip", "edge", "contact_cost", "elapsed",
        "stall_anchor_x", "stall_elapsed", "stall_progress", "forward_frontier", "forward_progress",
    )}
    for name in ("initialized", "outside", "bad_ground", "bad_wall", "missed_wall", "had_support",
                 "airborne", "root_crossed", "crossed", "landed", "complete", "failed", "invalid", "stalled",
                 "stairs_clearance_failed"):
        state[name] = zeros(dtype=torch.bool)
    state.update(
        corner=zeros(3), step_surfaces=zeros(STAIR_TREAD_COUNT, 3), covered=zeros(STAIR_TREAD_COUNT, dtype=torch.bool),
        foot_passed=zeros(4, dtype=torch.bool), previous_feet=zeros(4, 3),
        foot_initialized=zeros(4, dtype=torch.bool), foot_loaded=zeros(4, dtype=torch.bool),
        swing_active=zeros(4, dtype=torch.bool),
        swing_origin_z=zeros(4), swing_peak_z=zeros(4),
        previous_root=zeros(3), support_count=zeros(), phase=zeros(dtype=torch.long),
        reason=zeros(dtype=torch.long), geometry_level=zeros(dtype=torch.long),
        last_step=torch.full((count,), -1, device=device, dtype=torch.long),
        collect_tick=-1, updated_tick=-1,
    )
    state["foot_ids"] = env.scene["robot"].find_sites(FOOT_SITE_NAMES, preserve_order=True)[0]
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    indices = getattr(terrain, "terrain_level_indices", None)
    profiles = [[generator.sub_terrains[kind].generated_profile(
        (level if indices is None else int(indices[level, column])) / 4)
        for column, kind in enumerate(generator.sub_terrains)] for level in range(5)]
    state["profiles"] = profiles
    state["spawn_table"] = torch.tensor(np.array([[p.spawn for p in row] for row in profiles]), device=device, dtype=torch.float32)
    env._elevation = state
    return state


def fixed_stair_heights(env):
    terrains = env.cfg.scene.terrain.terrain_generator.sub_terrains
    return {index: sub.stair_height_override for index, (name, sub) in enumerate(terrains.items())
            if name.startswith("stairs") and sub.stair_height_override is not None}


def select_elevation_terrain(env, env_ids, settings: ElevationSettings, randomize_levels=False):
    """Restrict semantic types before the generic scene reset writes robot poses."""
    env._elevation_settings = settings
    ids = _ids(env, env_ids)
    state = elevation_state(env)
    terrain = env.scene.terrain
    names = list(terrain.cfg.terrain_generator.sub_terrains)
    expected = list(TERRAIN_NAMES) if settings.jump else [*TERRAIN_NAMES, FLAT_NAME]
    if names != expected:
        raise ValueError("Elevation requires its task terrain types in canonical order")
    if not getattr(env, "_elevation_fixed_cases", False):
        if settings.jump:
            terrain.terrain_types[ids] = 4
        else:
            allowed_names = walking_terrain_names(settings.terrain_kind)
            walking_types = torch.tensor([names.index(name) for name in allowed_names], device=env.device)
            terrain.terrain_types[ids] = walking_types[ids.remainder(len(walking_types))]
        if randomize_levels:
            terrain.terrain_levels[ids] = torch.randint(0, 5, (len(ids),), device=env.device)
        elif hasattr(env, "_elevation_curriculum"):
            caps = env._elevation_curriculum.level_caps[terrain.terrain_types[ids]]
            terrain.terrain_levels[ids] = terrain.terrain_levels[ids].clamp_min(0).minimum(caps)
        else:
            terrain.terrain_levels[ids] = 0
        for kind in fixed_stair_heights(env):
            stair_ids = ids[terrain.terrain_types[ids] == kind]
            terrain.terrain_levels[stair_ids] = torch.randint(0, 5, (len(stair_ids),), device=env.device)
    kinds = terrain.terrain_types[ids]
    if (settings.jump and (kinds != 4).any()) or (not settings.jump and (kinds == 4).any()):
        raise ValueError("Fixed elevation cases include a terrain forbidden for this task")
    if settings.terrain_kind is not None:
        allowed = torch.tensor([names.index(name) for name in walking_terrain_names(settings.terrain_kind)], device=env.device)
        if not torch.isin(kinds, allowed).all():
            raise ValueError("Fixed elevation cases must match the specialized terrain scope")
    if ((terrain.terrain_levels[ids] < 0) | (terrain.terrain_levels[ids] > 4)).any():
        raise ValueError("Elevation levels must lie in [0, 4]")
    terrain.env_origins[ids] = terrain.get_terrain_origins(terrain.terrain_levels[ids], kinds)
    levels = terrain.terrain_levels[ids]
    resolver = getattr(terrain, "get_geometry_levels", None)
    if resolver is not None:
        levels = resolver(levels, kinds)
    state["corner"][ids] = terrain.env_origins[ids] - state["spawn_table"][levels, kinds]


def reset_elevation(env, env_ids, settings: ElevationSettings, randomize=True, randomize_approach=None):
    ids = _ids(env, env_ids)
    state = elevation_state(env)
    terrain = env.scene.terrain
    kinds, levels = terrain.terrain_types[ids], terrain.terrain_levels[ids]
    resolver = getattr(terrain, "get_geometry_levels", None)
    if resolver is not None:
        levels = resolver(levels, kinds)
    state["geometry_level"][ids] = levels
    selected = [state["profiles"][int(level)][int(kind)]
                for level, kind in zip(levels.cpu().tolist(), kinds.cpu().tolist())]
    for key in ("stable", "air_time", "pending_new", "new_events", "phi", "shaping", "elapsed", "slip", "max_slip", "edge", "contact_cost",
                "stall_elapsed", "stall_progress"):
        state[key][ids] = 0
    for key in ("outside", "bad_ground", "bad_wall", "missed_wall", "had_support", "airborne", "root_crossed", "crossed", "landed", "complete", "failed", "invalid", "covered", "foot_passed", "phase", "reason", "stalled", "stairs_clearance_failed"):
        state[key][ids] = 0
    for key in ("foot_initialized", "foot_loaded", "swing_active"):
        state[key][ids] = 0
    state["swing_origin_z"][ids] = 0
    state["swing_peak_z"][ids] = 0
    state["last_step"][ids] = -1
    state["updated_tick"] = -1
    state["initialized"][ids] = True
    for key, attr in (("entry_x", "entry_x"), ("exit_x", "exit_x"), ("wall_height", "wall_height"), ("top_height", "top_height"), ("angle", "slope_angle"), ("stair_platform_length", "stair_platform_length")):
        state[key][ids] = torch.tensor([getattr(p, attr) for p in selected], device=env.device, dtype=state[key].dtype)
    state["step_surfaces"][ids] = 0
    for index, profile in zip(ids.cpu().tolist(), selected):
        if profile.step_surfaces:
            state["step_surfaces"][index] = torch.tensor(profile.step_surfaces, device=env.device)
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[ids].clone()
    pose[:, :3] += terrain.env_origins[ids]
    pose[:, 3:7] = pose.new_tensor([1., 0., 0., 0.])
    pose[:, 7:] = 0
    if not settings.jump:
        obstacle = kinds < 4
        pose[obstacle, 0] = (state["corner"][ids[obstacle], 0] + state["entry_x"][ids[obstacle]]
                             - settings.walking_approach_distance)
    state["speed"][ids] = sum(settings.speed_range) / 2
    if randomize:
        state["speed"][ids] = torch.empty(len(ids), device=env.device).uniform_(*settings.speed_range)
        pose[:, 1] += torch.empty(len(ids), device=env.device).uniform_(-settings.lateral_reset_range, settings.lateral_reset_range)
        yaw = torch.empty(len(ids), device=env.device).uniform_(-settings.yaw_reset_range, settings.yaw_reset_range)
        pose[:, 3], pose[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        pose[:, 7] = torch.empty(len(ids), device=env.device).uniform_(*settings.initial_forward_speed_range)
    approach_enabled = randomize if randomize_approach is None else randomize_approach
    if approach_enabled and not settings.jump:
        pose[:, 0] += torch.empty(len(ids), device=env.device).uniform_(*settings.approach_offset_range)
    if settings.jump:
        wall_index = torch.randint(0, 3, (len(ids),), device=env.device) if randomize else torch.zeros(len(ids), device=env.device, dtype=torch.long)
        wall_index[:] = getattr(env, "_elevation_wall_index", wall_index)
        wall_x = torch.tensor(WALL_CENTERS, device=env.device)[wall_index]
        distance = torch.full((len(ids),), sum(settings.trigger_distance_range) / 2, device=env.device)
        if randomize:
            distance.uniform_(*settings.trigger_distance_range)
        pose[:, 0] = state["corner"][ids, 0] + wall_x - distance
        state["wall_x"][ids] = wall_x
        state["entry_x"][ids], state["exit_x"][ids] = wall_x - 0.04, wall_x + 0.04
        override = terrain.cfg.terrain_generator.sub_terrains[WALL_NAME].wall_height_override
        if override is not None:
            state["wall_height"][ids] = override
    state["start_x"][ids] = pose[:, 0] - state["corner"][ids, 0]
    state["forward_frontier"][ids] = state["start_x"][ids]
    state["forward_progress"][ids] = 0.
    state["previous_root"][ids] = pose[:, :3]
    state["stall_anchor_x"][ids] = pose[:, 0]
    state["previous_feet"][ids] = 0
    for name in ("_elevation_touchdown_term", "_elevation_body_clearance_term"):
        term = getattr(env, name, None)
        if term is not None:
            term.reset(ids)
    robot.write_root_state_to_sim(pose, env_ids=ids)


class ElevationVelocityCommand(UniformThresholdVelocityCommand):
    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
        """Show the reset-assigned speed without enabling a three-axis joystick."""
        lower, upper = self.cfg.ranges.lin_vel_x
        with server.gui.add_folder(name.capitalize()):
            server.gui.add_markdown(
                f"前向速度范围：{lower:g}～{upper:g} m/s。\n\n"
                "速度在环境重置时设定；横向速度与转向速度固定为 0，禁用手柄覆盖。"
            )

    def _apply_forward(self):
        state = elevation_state(self._env)
        for name in ("is_standing_env", "is_heading_env", "is_lateral_env", "is_yaw_env", "is_backward_env"):
            getattr(self, name)[:] = False
        self.is_forward_env[:] = True
        self.vel_command_b[:, 0] = state["speed"].clamp_min(0.01)
        self.vel_command_b[:, 1:] = 0
        self.vel_command_w[:] = self.vel_command_b
        self.progress_direction_w[:] = 0
        self.progress_direction_w[:, 0] = self.vel_command_b[:, 0]

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._apply_forward()

    def _update_command(self):
        super()._update_command()
        self._apply_forward()


@dataclass(kw_only=True)
class ElevationVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    def build(self, env):
        return ElevationVelocityCommand(self, env)


def geometry_bounds(env):
    """Conservative world bounds of all robot collision shapes, including feet."""
    state = elevation_state(env)
    robot = env.scene["robot"]
    if "collision_ids" not in state:
        host = env.sim.mj_model
        global_ids = robot.indexing.geom_ids.cpu().numpy()
        enabled = (host.geom_contype[global_ids] != 0) | (host.geom_conaffinity[global_ids] != 0)
        state["collision_ids"] = torch.tensor(np.flatnonzero(enabled), device=env.device)
        sizes = host.geom_size[global_ids[enabled]].copy()
        types = host.geom_type[global_ids[enabled]]
        import mujoco
        sizes[types == mujoco.mjtGeom.mjGEOM_SPHERE] = sizes[types == mujoco.mjtGeom.mjGEOM_SPHERE, :1]
        for i, kind in enumerate(types):
            if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                sizes[i] = (sizes[i, 0], sizes[i, 0], sizes[i, 1] + (sizes[i, 0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0))
        state["collision_half_size"] = torch.tensor(sizes, device=env.device, dtype=torch.float32)
    pose = robot.data.geom_pose_w[:, state["collision_ids"]]
    rotation = matrix_from_quat(pose[..., 3:7]).abs()
    extent = torch.matmul(rotation, state["collision_half_size"][None, :, :, None]).squeeze(-1)
    return pose[..., :3] - extent, pose[..., :3] + extent


def elevation_foot_bottom(env, foot_ids=None, foot_radius=0.025):
    """Recover the world-lowest point of the spherical foot from its sole site."""
    robot = env.scene["robot"].data
    if foot_ids is None:
        foot_ids = elevation_state(env)["foot_ids"]
    feet = robot.site_pos_w[:, foot_ids]
    rotation = matrix_from_quat(robot.site_quat_w[:, foot_ids])
    return feet + foot_radius * (rotation[..., :, 2] - feet.new_tensor([0., 0., 1.]))


def collect_elevation_contacts(env):
    """Latch short physical events before the control-rate reward is computed."""
    state = elevation_state(env)
    settings = env._elevation_settings
    if state["collect_tick"] != env.common_step_counter:
        state["pending_new"][:] = 0
        state["collect_tick"] = env.common_step_counter
    sensor = env.scene["elevation_contacts"].data
    slots = env.scene["elevation_contacts"].cfg.num_slots
    found = sensor.found.reshape(env.num_envs, 4, slots) > 0
    force_world = sensor.force.reshape(env.num_envs, 4, slots, 3)
    force = force_world.norm(dim=-1)
    pos = sensor.pos.reshape(env.num_envs, 4, slots, 3)
    normal = sensor.normal.reshape(env.num_envs, 4, slots, 3)
    effective = found & (force > settings.contact_force_threshold)
    support = effective & (normal[..., 2] < -0.5)
    local = pos - state["corner"][:, None, None, :]
    state["outside"] |= (support & ((local[..., 1] - CENTER_Y).abs() > LANE_WIDTH / 2 + 0.001)).any(dim=(1, 2))
    state["support_count"] = support.any(dim=2).sum(dim=1).float()
    previous_support = state["had_support"].clone()
    state["had_support"] |= state["support_count"] >= 2
    kinds = env.scene.terrain.terrain_types
    stair = kinds < 2
    tread = state["step_surfaces"]
    hit = (support[..., None]
           & (local[..., 0, None] >= tread[:, None, None, :, 0] - 0.001)
           & (local[..., 0, None] <= tread[:, None, None, :, 1] + 0.001)
           & ((local[..., 2, None] - tread[:, None, None, :, 2]).abs() <= settings.top_contact_tolerance))
    covered = hit.any(dim=(1, 2)) & stair[:, None]
    state["pending_new"] += (covered & ~state["covered"]).sum(dim=1)
    state["covered"] |= covered
    feet = elevation_foot_bottom(env, state["foot_ids"])
    foot_local = feet - state["corner"][:, None, :]
    loaded = support.any(dim=2)
    initialized = state["foot_initialized"]
    starting_support = loaded & ~initialized
    state["swing_origin_z"] = torch.where(starting_support, foot_local[..., 2], state["swing_origin_z"])
    state["swing_peak_z"] = torch.where(starting_support, foot_local[..., 2], state["swing_peak_z"])
    takeoff = initialized & state["foot_loaded"] & ~loaded
    state["swing_origin_z"] = torch.where(takeoff, foot_local[..., 2], state["swing_origin_z"])
    state["swing_peak_z"] = torch.where(takeoff, foot_local[..., 2], state["swing_peak_z"])
    state["swing_active"] |= takeoff
    airborne_foot = state["swing_active"] & ~loaded
    state["swing_peak_z"] = torch.where(
        airborne_foot, torch.maximum(state["swing_peak_z"], foot_local[..., 2]), state["swing_peak_z"])
    foot_hit = hit.any(dim=2)
    higher_tread = foot_hit & (tread[:, None, :, 2] > state["swing_origin_z"][:, :, None] + .001)
    insufficient_lift = state["swing_peak_z"] - state["swing_origin_z"] < .12 - 1.e-6
    state["stairs_clearance_failed"] |= (
        (kinds == 0) & (state["swing_active"] & higher_tread.any(dim=2) & insufficient_lift).any(dim=1))
    state["swing_active"] &= ~loaded
    state["foot_loaded"] = loaded
    state["foot_initialized"] |= loaded
    robot_data = env.scene["robot"].data
    offset = pos - robot_data.site_pos_w[:, state["foot_ids"], None, :]
    angular = robot_data.site_ang_vel_w[:, state["foot_ids"], None, :].expand_as(offset)
    tangent = robot_data.site_lin_vel_w[:, state["foot_ids"], None, :] + torch.linalg.cross(angular, offset)
    tangent = tangent - (tangent * normal).sum(dim=-1, keepdim=True) * normal
    slip = (tangent.square().sum(dim=-1) / 0.15 ** 2).clamp_max(1)
    state["slip"] = (slip * support).sum(dim=(1, 2)) / support.sum(dim=(1, 2)).clamp_min(1)
    state["max_slip"] = (slip * support).amax(dim=(1, 2))
    remaining = LANE_WIDTH / 2 - (local[..., 1] - CENTER_Y).abs() - 0.025
    edge = ((0.05 - remaining).clamp_min(0) / 0.05).square().clamp_max(1)
    state["edge"] = (edge * support).sum(dim=(1, 2)) / support.sum(dim=(1, 2)).clamp_min(1)
    body = env.scene["elevation_body_contacts"].data
    body_effective = (body.found > 0) & (body.force.norm(dim=-1) > settings.contact_force_threshold)
    body_local = body.pos - state["corner"][:, None, :]
    body_wall = ((body_local[..., 0] - state["wall_x"][:, None]).abs() <= WALL_THICKNESS / 2 + 0.008) & (body_local[..., 2] > 0.01)
    if "base_contact_slots" not in state:
        body_sensor = env.scene["elevation_body_contacts"]
        state["base_contact_slots"] = torch.tensor(
            [name.split("/")[-1] == BASE_BODY_NAME for name in body_sensor.primary_names],
            device=env.device).repeat_interleave(body_sensor.cfg.num_slots)
    ground = ~body_wall if settings.jump else torch.ones_like(body_wall)
    state["bad_ground"] |= (body_effective & state["base_contact_slots"][None] & ground).any(dim=1)
    foot_bad = (effective & ~support).float().flatten(1)
    body_bad = body_effective.float()
    if settings.jump:
        state["contact_cost"] = (body_bad.mean(dim=1) + foot_bad.mean(dim=1)).clamp_max(1)
    else:
        state["contact_cost"] = (.5 * (body_bad.mean(dim=1) + foot_bad.mean(dim=1))
                                 + .5 * torch.maximum(body_bad.max(dim=1).values, foot_bad.max(dim=1).values)).clamp_max(1)
        touchdown = getattr(env, "_elevation_touchdown_term", None)
        if touchdown is not None:
            touchdown.collect(env)
    if settings.jump:
        root = env.scene["robot"].data.root_link_pos_w
        feet = elevation_foot_bottom(env, state["foot_ids"])
        any_contact = effective.any(dim=(1, 2))
        state["air_time"] = torch.where(any_contact, 0.0, state["air_time"] + env.physics_dt)
        takeoff = previous_support & (state["air_time"] >= settings.minimum_flight_duration - 1e-6) & (env.scene["robot"].data.root_link_lin_vel_w[:, 0] > 0)
        state["pending_new"] += (takeoff & ~state["airborne"]).float()
        state["airborne"] |= takeoff
        plane = state["corner"][:, 0] + state["wall_x"]
        old = state["previous_feet"]
        moved = (old[:, :, 0] < plane[:, None]) & (feet[:, :, 0] >= plane[:, None]) & state["airborne"][:, None]
        alpha = ((plane[:, None] - old[:, :, 0]) / (feet[:, :, 0] - old[:, :, 0]).clamp_min(1e-8)).clamp(0, 1)
        crossing = old + alpha[..., None] * (feet - old)
        valid_cross = ((crossing[..., 2] - state["corner"][:, 2, None] >= state["wall_height"][:, None] - 0.002)
                       & ((crossing[..., 1] - state["corner"][:, 1, None] - CENTER_Y).abs() <= LANE_WIDTH / 2))
        state["missed_wall"] |= (moved & ~valid_cross).any(dim=1)
        state["foot_passed"] |= moved & valid_cross
        root_cross = (state["previous_root"][:, 0] < plane) & (root[:, 0] >= plane)
        state["root_crossed"] |= root_cross & state["airborne"] & ~any_contact
        on_wall = ((local[..., 0] - state["wall_x"][:, None, None]).abs() <= WALL_THICKNESS / 2 + 0.008) & (local[..., 2] > 0.01)
        foot_wall_load = support | (effective & (force_world[..., 2].abs() > settings.contact_force_threshold))
        body_wall_load = (body.normal[..., 2] < -0.5) | (body.force[..., 2].abs() > settings.contact_force_threshold)
        state["bad_wall"] |= (foot_wall_load & on_wall).any(dim=(1, 2)) | (body_effective & body_wall & body_wall_load).any(dim=1)
        land = state["airborne"] & support.any(dim=(1, 2)) & ((local[..., 0] > state["exit_x"][:, None, None]) & support).any(dim=(1, 2))
        state["pending_new"] += (land & ~state["landed"]).float()
        state["landed"] |= land
        state["previous_feet"][:] = feet
        state["previous_root"][:] = root
    return state["support_count"]


def slope_exit_supported_feet(env, minimum_force=8.):
    """Require actual load-bearing contacts on the destination platform of a ramp."""
    state = elevation_state(env)
    sensor = env.scene["elevation_contacts"].data
    shape = (env.num_envs, 4, -1)
    found = sensor.found.reshape(shape)
    pos = sensor.pos.reshape(*shape, 3)
    force = sensor.force.reshape(*shape, 3)
    normal = sensor.normal.reshape(*shape, 3)
    local = pos - state["corner"][:, None, None, :]
    kind = env.scene.terrain.terrain_types
    ramp = (kind == 2) | (kind == 3)
    height = torch.where(kind == 2, state["top_height"], 0.)
    finite = (torch.isfinite(found) & torch.isfinite(pos).all(dim=-1)
              & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1))
    normal_force = (force * normal).sum(dim=-1).abs()
    loaded = finite & (found > 0) & (normal[..., 2] < -.5) & (normal_force >= minimum_force)
    on_platform = ((local[..., 0] >= state["exit_x"][:, None, None])
                   & (local[..., 0] <= env.cfg.scene.terrain.terrain_generator.size[0])
                   & ((local[..., 1] - CENTER_Y).abs() <= LANE_WIDTH / 2)
                   & ((local[..., 2] - height[:, None, None]).abs()
                      <= env._elevation_settings.top_contact_tolerance))
    return (loaded & on_platform & ramp[:, None, None]).any(dim=2)


def stairs_exit_supported_feet(env, minimum_force=8.):
    """Require all feet to bear load on the destination platform of a stair task."""
    state = elevation_state(env)
    sensor = env.scene["elevation_contacts"].data
    shape = (env.num_envs, 4, -1)
    found = sensor.found.reshape(shape)
    pos = sensor.pos.reshape(*shape, 3)
    force = sensor.force.reshape(*shape, 3)
    normal = sensor.normal.reshape(*shape, 3)
    local = pos - state["corner"][:, None, None, :]
    kind = env.scene.terrain.terrain_types
    stairs = (kind == 0) | (kind == 1)
    height = torch.where(kind == 0, state["top_height"], 0.)
    finite = (torch.isfinite(found) & torch.isfinite(pos).all(dim=-1)
              & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1))
    normal_force = (force * normal).sum(dim=-1).abs()
    loaded = finite & (found > 0) & (normal[..., 2] < -.5) & (normal_force >= minimum_force)
    on_platform = ((local[..., 0] >= state["exit_x"][:, None, None])
                   & (local[..., 0] <= env.cfg.scene.terrain.terrain_generator.size[0])
                   & ((local[..., 1] - CENTER_Y).abs() <= LANE_WIDTH / 2)
                   & ((local[..., 2] - height[:, None, None]).abs()
                      <= env._elevation_settings.top_contact_tolerance))
    return (loaded & on_platform & stairs[:, None, None]).any(dim=2)


def update_elevation_state(env):
    state = elevation_state(env)
    if state["updated_tick"] == env.common_step_counter:
        return state
    settings = env._elevation_settings
    robot = env.scene["robot"].data
    local = robot.root_link_pos_w - state["corner"]
    low, high = geometry_bounds(env)
    rear = low[..., 0].min(dim=1).values - state["corner"][:, 0]
    roll, pitch, _ = euler_xyz_from_quat(robot.root_link_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    valid = (torch.isfinite(robot.root_link_pos_w).all(dim=1)
             & torch.isfinite(robot.root_link_quat_w).all(dim=1)
             & torch.isfinite(robot.root_link_lin_vel_b).all(dim=1)
             & torch.isfinite(robot.root_link_ang_vel_b).all(dim=1)
             & torch.isfinite(robot.joint_pos).all(dim=1))
    state["invalid"] |= ~valid
    base_contact = state["bad_ground"].clone()
    if not settings.jump:
        base_contact |= env.scene["base_ground_contact"].data.found.gt(0).any(dim=1)
    joint_failure = joint_pos_out_of_limits(env)
    tilt = (roll.abs() > math.radians(70)) | (pitch.abs() > math.radians(70))
    elapsed = env.episode_length_buf.float() * env.step_dt
    if not settings.jump:
        warming_up = elapsed <= settings.stall_grace_s
        state["stall_anchor_x"] = torch.where(warming_up, robot.root_link_pos_w[:, 0], state["stall_anchor_x"])
        state["stall_elapsed"] = torch.where(warming_up, 0.0, state["stall_elapsed"] + env.step_dt)
        state["stall_progress"] = robot.root_link_pos_w[:, 0] - state["stall_anchor_x"]
        ready = state["stall_elapsed"] >= settings.stall_window_s - 1e-5
        required = (state["speed"] * settings.stall_window_s * settings.stall_min_progress_ratio).clamp_min(settings.stall_min_distance_m)
        state["stalled"] |= ready & (state["stall_progress"] < required)
        advance = ready & ~state["stalled"]
        state["stall_anchor_x"] = torch.where(advance, robot.root_link_pos_w[:, 0], state["stall_anchor_x"])
        state["stall_elapsed"] = torch.where(advance, 0.0, state["stall_elapsed"])
    deadline = elapsed >= env.cfg.episode_length_s - 1e-6
    wrong_route = ((local[:, 1] - CENTER_Y).abs() > 0.8) | (local[:, 0] < state["start_x"] - 0.5)
    missed_step = (env.scene.terrain.terrain_types < 2) & (rear > state["exit_x"] + 0.01) & ~state["covered"].all(dim=1)
    clearance_failure = state["stairs_clearance_failed"] & (settings.terrain_kind != "stairs")
    failure = state["outside"] | state["bad_wall"] | state["missed_wall"] | base_contact | joint_failure | tilt | wrong_route | missed_step | state["invalid"] | state["stalled"] | clearance_failure
    state["reason"][:] = 0
    for code, condition in ((1, state["outside"]), (2, missed_step), (3, state["bad_wall"] | state["missed_wall"]), (4, base_contact | tilt), (5, joint_failure), (6, wrong_route), (9, state["stalled"]), (7, deadline), (8, state["invalid"]), (10, clearance_failure)):
        state["reason"] = torch.where(condition, code, state["reason"])
    if settings.jump:
        crossed = state["root_crossed"] & state["foot_passed"].all(dim=1) & (rear > state["exit_x"] + 0.01)
        state["pending_new"] += (crossed & ~state["crossed"]).float()
        state["crossed"] |= crossed
        eligible = state["crossed"] & state["landed"]
        state["phase"] = torch.where(state["landed"], 2, torch.where(state["airborne"], 1, 0))
        hold = settings.jump_handoff_duration
    else:
        eligible = (rear > state["exit_x"] + 0.10) & ((env.scene.terrain.terrain_types >= 2) | state["covered"].all(dim=1))
        state["phase"] = eligible.long() * 2
        hold = settings.stable_duration
    stable = (eligible & ~failure & (state["support_count"] >= 2)
              & (roll.abs() <= math.radians(12)) & (pitch.abs() <= math.radians(18))
              & (robot.root_link_ang_vel_b.norm(dim=1) <= 1.0)
              & (robot.root_link_lin_vel_b[:, 1].abs() <= 0.15)
              & ((robot.root_link_lin_vel_b[:, 0] - state["speed"]).abs() <= 0.25))
    if settings.terrain_kind == "slope":
        stable = slope_exit_supported_feet(env).all(dim=1) & state["initialized"] & ~failure
        state["phase"] = stable.long() * 2
        hold = env.step_dt
    elif settings.terrain_kind == "stairs":
        stable = (stairs_exit_supported_feet(env).all(dim=1) & state["covered"].all(dim=1)
                  & state["initialized"] & ~failure)
        state["phase"] = stable.long() * 2
        hold = env.step_dt
    state["stable"] = torch.where(stable, state["stable"] + env.step_dt, 0)
    state["complete"] = (state["stable"] >= hold - 1e-6) & ~failure
    failure |= deadline & ~state["complete"]
    state["failed"] = failure
    state["complete"] &= ~failure
    state["new_events"] = torch.where(failure, 0, state["pending_new"])
    if not settings.jump:
        advancing = state["initialized"] & valid & ~failure & ~state["complete"]
        next_frontier = torch.maximum(state["forward_frontier"], local[:, 0].nan_to_num())
        state["forward_progress"] = torch.where(advancing, next_frontier - state["forward_frontier"], 0.)
        state["forward_frontier"] = torch.where(advancing, next_frontier, state["forward_frontier"])
    next_phi = ((local[:, 0] - state["start_x"]) / (state["exit_x"] + 0.7 - state["start_x"]).clamp_min(0.1)).clamp(-1, 1)
    next_phi = torch.where(failure | state["complete"], 0, next_phi)
    state["shaping"] = settings.gamma ** (env.step_dt / 0.02) * next_phi - state["phi"]
    state["phi"] = next_phi
    state["elapsed"] = elapsed
    state["last_step"][:] = env.common_step_counter
    state["updated_tick"] = env.common_step_counter
    return state


def elevation_failed(env):
    return update_elevation_state(env)["failed"]


def elevation_completed(env):
    return update_elevation_state(env)["complete"]


def elevation_event(env, key):
    value = update_elevation_state(env)[key].float()
    if key == "new_events" and not env._elevation_settings.jump:
        value = value * (3.0 / STAIR_TREAD_COUNT)
    return value / env.step_dt


def deadband_cost(value, tolerance, scale):
    return ((value.abs() - tolerance).clamp_min(0) / scale).square().clamp_max(1)


def elevation_forward_velocity_reward(env, std=.1):
    """Reward command tracking only when the body reaches new forward ground."""
    if not math.isfinite(std) or std <= 0.:
        raise ValueError("Velocity tracking width must be finite and positive")
    if env._elevation_settings.jump:
        raise ValueError("Walking velocity rewards must not be enabled for Jump")
    state = update_elevation_state(env)
    desired = state["speed"]
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    progress = state["forward_progress"]
    finite = torch.isfinite(desired) & torch.isfinite(velocity) & torch.isfinite(progress)
    active = (state["initialized"] & ~state["failed"] & ~state["invalid"]
              & ~state["complete"] & finite & (desired > 0.) & (state["support_count"] >= 2))
    target = torch.nan_to_num(desired, nan=0., posinf=0., neginf=0.).clamp_min(1.e-6)
    speed = torch.nan_to_num(velocity, nan=0., posinf=0., neginf=0.)
    credit = (torch.nan_to_num(progress, nan=0., posinf=0., neginf=0.)
              / (target * env.step_dt)).clamp(0., 1.)
    tracking = torch.exp(-((speed - target) / std).square())
    reward = credit * (speed / target).clamp(0., 1.) * tracking
    gait = getattr(env, "_elevation_gait_term", None)
    if gait is not None:
        gait(env, **gait.params)
        reward = reward * gait.gait["renewal"]
    return torch.where(active, reward, 0.)


def elevation_support_pitch(env):
    """Fit the loaded support contacts, falling back to the known ramp profile."""
    state = elevation_state(env)
    robot = env.scene["robot"].data
    x = robot.root_link_pos_w[:, 0] - state["corner"][:, 0]
    kind = env.scene.terrain.terrain_types
    fraction = ((x - 2.7) / .3).clamp(0., 1.) * ((5.3 - x) / .3).clamp(0., 1.)
    fallback = torch.where(kind == 2, -state["angle"], torch.where(kind == 3, state["angle"], 0.)) * fraction
    sensor = env.scene["elevation_contacts"]
    data = sensor.data
    finite = torch.isfinite(data.pos).all(dim=-1) & torch.isfinite(data.force).all(dim=-1)
    supported = (data.found > 0) & (data.normal[..., 2] < -.5) & (data.force.norm(dim=-1) >= 8.) & finite
    weight = supported.float() / supported.sum(dim=1, keepdim=True).clamp_min(1)
    pos = torch.nan_to_num(data.pos, nan=0., posinf=0., neginf=0.)
    mean = (pos * weight[..., None]).sum(dim=1, keepdim=True)
    delta = pos - mean
    variance = (delta[..., 0].square() * weight).sum(dim=1)
    covariance = (delta[..., 0] * delta[..., 2] * weight).sum(dim=1)
    fitted = -torch.atan2(covariance, variance.clamp_min(1.e-6))
    loaded_feet = supported.reshape(env.num_envs, 4, sensor.cfg.num_slots).any(dim=2).sum(dim=1)
    reliable = (loaded_feet >= 2) & (variance >= .005) & (kind < 4)
    return torch.where(reliable, fitted.clamp(-math.radians(35), math.radians(35)), fallback)


def elevation_cost(env, key, tolerance=None, scale=None):
    state = update_elevation_state(env)
    robot = env.scene["robot"].data
    if key == "slip" and not env._elevation_settings.jump:
        return .5 * state["slip"] + .5 * state["max_slip"]
    if key in ("slip", "edge", "contact_cost"):
        return state[key]
    if key == "time":
        return torch.ones(env.num_envs, device=env.device)
    if key == "velocity":
        desired = state["speed"]
        value = 1 - torch.exp(-((robot.root_link_lin_vel_b[:, 0] - desired) / 0.25).square())
        return value * (state["landed"].float() * 0.25 if env._elevation_settings.jump else 1)
    if key == "center":
        return deadband_cost(
            robot.root_link_pos_w[:, 1] - state["corner"][:, 1] - CENTER_Y,
            0.02 if tolerance is None else tolerance, 0.18 if scale is None else scale,
        )
    if key == "heading":
        error = torch.atan2(torch.sin(robot.heading_w), torch.cos(robot.heading_w))
        return deadband_cost(
            error, math.radians(3) if tolerance is None else tolerance,
            math.radians(12) if scale is None else scale,
        )
    if key == "lateral":
        return deadband_cost(
            robot.root_link_lin_vel_b[:, 1], 0.03 if tolerance is None else tolerance,
            0.15 if scale is None else scale,
        )
    if key == "yaw_rate":
        return deadband_cost(robot.root_link_ang_vel_b[:, 2], tolerance, scale)
    if key == "attitude":
        roll, pitch, _ = euler_xyz_from_quat(robot.root_link_quat_w)
        roll, pitch = torch.atan2(torch.sin(roll), torch.cos(roll)), torch.atan2(torch.sin(pitch), torch.cos(pitch))
        x = robot.root_link_pos_w[:, 0] - state["corner"][:, 0]
        kind = env.scene.terrain.terrain_types
        reference = elevation_support_pitch(env)
        near_stairs = (kind < 2) & (x >= state["entry_x"] - 0.5) & (x <= state["exit_x"] + 0.5)
        pitch_tolerance = torch.where(near_stairs, math.radians(15), math.radians(5))
        on_ramp = ((kind == 2) | (kind == 3)) & (x >= 2.7) & (x <= 5.3)
        pitch_tolerance = torch.where(on_ramp, math.radians(10), pitch_tolerance)
        if env._elevation_settings.jump:
            pitch_tolerance = math.radians(35)
        return (deadband_cost(roll, math.radians(5), math.radians(15))
                + deadband_cost(pitch - reference, pitch_tolerance, math.radians(20))) / 2
    if key == "angular":
        omega = robot.root_link_ang_vel_b
        yaw = omega[:, 2].square() if env._elevation_settings.jump else 0.
        return ((omega[:, 0].square() + yaw + 0.25 * omega[:, 1].square()) / 4).clamp_max(1)
    if key == "torque":
        return (robot.qfrc_actuator / EFFORT_LEG).square().mean(dim=1).clamp_max(1)
    if key == "joint":
        limits = robot.soft_joint_pos_limits
        span = (limits[..., 1] - limits[..., 0]).clamp_min(0.01)
        excess = ((limits[..., 0] - robot.joint_pos).clamp_min(0) + (robot.joint_pos - limits[..., 1]).clamp_min(0))
        return (excess / (0.05 * span)).square().mean(dim=1).clamp_max(1)
    if key == "action":
        delta = env.action_manager.action - env.action_manager.prev_action
        value = (delta / (env.step_dt * 10)).square().mean(dim=1)
        return value.clamp_max(1) if env._elevation_settings.jump else torch.log1p(value)
    raise KeyError(key)


def elevation_alignment_metric(env, key):
    robot = env.scene["robot"].data
    if key == "lateral_velocity":
        return robot.root_link_lin_vel_w[:, 1]
    if key == "lateral_speed_abs":
        return robot.root_link_lin_vel_w[:, 1].abs()
    if key == "center_error_abs":
        state = update_elevation_state(env)
        return (robot.root_link_pos_w[:, 1] - state["corner"][:, 1] - CENTER_Y).abs()
    if key == "heading_error_deg":
        return torch.rad2deg(torch.atan2(torch.sin(robot.heading_w), torch.cos(robot.heading_w)).abs())
    if key == "support_pitch_deg":
        return torch.rad2deg(elevation_support_pitch(env))
    raise KeyError(key)


def elevation_metric(env, key):
    state = update_elevation_state(env)
    if key == "covered":
        return state["covered"].sum(dim=1).float()
    return state[key].float()


class ElevationTerrainCurriculum:
    def __init__(self, cfg, env):
        self.window_size = int(cfg.params.get("window_size", 100))
        self.threshold = float(cfg.params.get("threshold", 0.8))
        if self.window_size < 1 or not 0 <= self.threshold <= 1:
            raise ValueError("Invalid elevation curriculum window or threshold")
        self.terrain_names = tuple(env.cfg.scene.terrain.terrain_generator.sub_terrains)
        self.fixed_stair_heights = fixed_stair_heights(env)
        self.level_caps = torch.zeros(len(self.terrain_names), device=env.device, dtype=torch.long)
        self.windows = [deque(maxlen=self.window_size) for _ in self.terrain_names]
        env._elevation_curriculum = self

    def __call__(self, env, env_ids, **kwargs):
        del kwargs
        ids = _ids(env, env_ids)
        state = elevation_state(env)
        terrain = env.scene.terrain
        valid = state["initialized"][ids] & ~state["invalid"][ids] & (env.episode_length_buf[ids] > 0)
        finished = ids[valid]
        for index in finished.cpu().tolist():
            kind, level = int(terrain.terrain_types[index]), int(terrain.terrain_levels[index])
            if kind in self.fixed_stair_heights:
                continue
            if self.terrain_names[kind] == FLAT_NAME:
                terrain.terrain_levels[index] = 0
                continue
            success = bool(state["complete"][index]) and not bool(state["failed"][index])
            if level == int(self.level_caps[kind]):
                self.windows[kind].append(success)
                if len(self.windows[kind]) == self.window_size and sum(self.windows[kind]) / self.window_size >= self.threshold and self.level_caps[kind] < 4:
                    self.level_caps[kind] += 1
                    self.windows[kind].clear()
            terrain.terrain_levels[index] = max(0, min(level + (1 if success else -1), int(self.level_caps[kind])))
        settings = getattr(env, "_elevation_settings", None)
        if settings is None:
            settings = env.cfg.events["reset_elevation"].params["settings"]
        active_kind = settings.terrain_kind
        active_names = self.terrain_names if active_kind is None else walking_terrain_names(active_kind)
        active = {i for i, name in enumerate(self.terrain_names) if name in active_names}
        values = {name + "_cap": self.level_caps[i].float()
                  for i, name in enumerate(self.terrain_names) if i in active and i not in self.fixed_stair_heights}
        values.update({self.terrain_names[i] + "_fixed_height_m": self.level_caps.new_tensor(height, dtype=torch.float32)
                       for i, height in self.fixed_stair_heights.items() if i in active})
        return values

    def state_dict(self):
        return {"level_caps": self.level_caps.cpu().tolist(), "windows": [list(w) for w in self.windows], "window_size": self.window_size}

    def load_state_dict(self, value):
        caps = torch.as_tensor(value["level_caps"], device=self.level_caps.device, dtype=torch.long)
        if caps.shape != self.level_caps.shape or ((caps < 0) | (caps > 4)).any() or len(value["windows"]) != len(self.terrain_names):
            raise ValueError("Invalid saved elevation curriculum")
        if value.get("window_size", self.window_size) != self.window_size:
            raise ValueError("Curriculum window changed; use weights-only transfer instead of resume")
        self.level_caps[:] = caps
        self.windows = [deque(map(bool, w), maxlen=self.window_size) for w in value["windows"]]
        for kind in self.fixed_stair_heights:
            self.level_caps[kind] = 0
            self.windows[kind].clear()
