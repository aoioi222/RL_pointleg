"""State, commands, observations and rewards for the stepping-stone task."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from .commands import UniformThresholdVelocityCommand
from .elevation import deadband_cost
from .fixed_stone_gait import HOLD, FixedStoneGait
from ..robot_cfg import FOOT_BODY_NAMES, FOOT_SITE_NAMES, LEG_NAMES
from ..terrains.stepping_stones import (
    MAX_STONES, PLATFORM_HEIGHT, PLATFORM_SIZE, right_platform_start, stone_centers,
)


GROUND_CONTACT_Z_TOL = 0.025
TOP_CONTACT_Z_TOL = 0.030
TOP_EDGE_TOL = 0.003
LANDING_EDGE_MARGIN = 0.03
EXIT_FIRST_LANDING_MIN_M = 0.05
EXIT_FIRST_LANDING_MAX_M = 0.20
MIN_TOP_NORMAL_Z = 0.7
STONE_CONTACT_TIME_S = 0.020
EXIT_CONTACT_TIME_S = 0.020
EXIT_SETTLE_TIME_S = 0.70
EXIT_SETTLE_TIMEOUT_S = 5.0
EXIT_HEIGHT_TOL_M = 0.04
EXIT_TILT_TOL_RAD = math.radians(10.0)
EXIT_LINEAR_SPEED_MPS = 0.05
EXIT_ANGULAR_SPEED_RADPS = 0.15
EXIT_JOINT_ERROR_RAD = 0.45
SPAWN_X = 0.33
FOOT_RADIUS = 0.025
STANDSTILL_PROGRESS = 0.10
STEPPING_FORWARD_SPEED_RANGE = (0.3, 0.4)
FORWARD_STEP_CONFIRM_S = 0.08
EXIT_SURFACE = MAX_STONES + 1
SUPPORT_HANDOFF_CONFIRM_S = 0.02
STARTUP_RISE_WINDOW_S = 1.0
# Midpoint of the front hip abduction pivots in the base frame.
STARTUP_FRONT_X = 0.32826
STARTUP_FRONT_Z = 0.054
KNEE_REVERSE_FULL_RAD = math.radians(5.0)
NO_FORWARD_STEP_GRACE_S = 0.60
RIGHT_REAR_FOLLOW_GRACE_S = 1.0
MIN_FOOT_SUPPORT_N = 8.0
SHIN_SUPPORT_N = 8.0
SHIN_IMPACT_N = 25.0
SHIN_CONFIRM_S = 0.02
SUPPORT_PROGRESS_INFO_KEY = "pointleg_stepping_support_progress"


class LeftFrontHandoff:
    """Track confirmed support surfaces, forward steps and left-front eligibility."""

    def __init__(self, num_envs, device):
        self.first_right_rear_step_complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.left_front_blocked = torch.zeros_like(self.first_right_rear_step_complete)
        self.right_rear_follow_pending = torch.zeros_like(self.first_right_rear_step_complete)
        self.right_rear_step_credit = torch.zeros_like(self.first_right_rear_step_complete)
        self.left_front_follow_blocked = torch.zeros_like(self.first_right_rear_step_complete)
        self.loaded_substeps = torch.zeros(num_envs, len(LEG_NAMES), dtype=torch.long, device=device)
        self.loading_surface = torch.full_like(self.loaded_substeps, -1)
        self.last_surface = torch.full_like(self.loaded_substeps, -1)
        self.last_rewarded_surface = torch.full_like(self.loaded_substeps, -1)
        self.left_front_route_surface = torch.full_like(self.loaded_substeps[:, 0], -1)

    def reset(self, env_ids):
        for buffer in (self.first_right_rear_step_complete, self.left_front_blocked, self.loaded_substeps,
                       self.right_rear_follow_pending, self.right_rear_step_credit, self.left_front_follow_blocked):
            buffer[env_ids] = 0
        for buffer in (self.loading_surface, self.last_surface, self.last_rewarded_surface, self.left_front_route_surface):
            buffer[env_ids] = -1

    def left_front_target_surface(self, left_stones):
        origin = self.left_front_route_surface
        surfaces = torch.arange(1, left_stones.shape[1] + 1, device=origin.device)
        ahead = left_stones & (surfaces > origin[:, None])
        next_left_stone = torch.where(ahead, surfaces, EXIT_SURFACE).amin(dim=1)
        return torch.where(origin >= 0, next_left_stone, EXIT_SURFACE)

    def update(self, surface, effective_support, active, physics_dt, *, traversing, left_front_target=EXIT_SURFACE):
        stable = math.ceil(FORWARD_STEP_CONFIRM_S / physics_dt)
        self.loaded_substeps.masked_fill_(self.loading_surface != surface, 0)
        self.loading_surface.copy_(surface)
        self.loaded_substeps.add_(1).clamp_max_(stable).masked_fill_(~effective_support, 0)
        stable_support = self.loaded_substeps >= stable
        has_origin = self.last_surface >= 0
        forward_step = active[:, None] & stable_support & has_origin & (surface > self.last_surface)
        initial_support = ~has_origin & (self.loaded_substeps >= math.ceil(STONE_CONTACT_TIME_S / physics_dt))
        confirmed_support = active[:, None] & (initial_support | stable_support)
        self.last_surface.copy_(torch.where(confirmed_support, surface, self.last_surface))
        left_front_on_route = confirmed_support[:, 0] & (surface[:, 0] <= left_front_target)
        self.left_front_route_surface.copy_(torch.where(left_front_on_route, surface[:, 0], self.left_front_route_surface))
        self.first_right_rear_step_complete |= forward_step[:, 3]

        holding = active & self.first_right_rear_step_complete & ~stable_support[:, 3]
        violation = holding & (surface[:, 0] < 0) & ~self.left_front_blocked
        blocked_landing = self.left_front_blocked | violation
        self.left_front_blocked.copy_(blocked_landing & ~stable_support[:, 0])
        blocked_landing |= surface[:, 0] > left_front_target
        rewarded_step = forward_step & (surface > self.last_rewarded_surface)

        right_rear_step = rewarded_step[:, 3]
        waiting_for_rear = traversing & self.right_rear_follow_pending & ~right_rear_step
        too_early = waiting_for_rear & ((surface[:, 0] < 0) | (surface[:, 0] > self.last_rewarded_surface[:, 0]))
        follow_blocked = traversing & (self.left_front_follow_blocked | too_early)
        self.left_front_follow_blocked.copy_(follow_blocked & ~stable_support[:, 0])
        blocked_landing |= follow_blocked

        rewarded_step[:, 0] &= ~blocked_landing
        left_front_step = rewarded_step[:, 0]
        rear_credit = self.right_rear_step_credit | (right_rear_step & ~self.right_rear_follow_pending)
        needs_follow = waiting_for_rear | (left_front_step & ~rear_credit)
        self.right_rear_follow_pending.copy_(traversing & needs_follow & (self.last_surface[:, 3] != EXIT_SURFACE))
        self.right_rear_step_credit.copy_(traversing & rear_credit & ~left_front_step)

        self.last_rewarded_surface.copy_(torch.where(rewarded_step, surface, self.last_rewarded_surface))
        return holding, violation, blocked_landing, rewarded_step

    def support_progress(self, physics_dt):
        return self.loaded_substeps / math.ceil(FORWARD_STEP_CONFIRM_S / physics_dt)


def _ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _geometry_tables(env):
    generator = env.scene.terrain.cfg.terrain_generator
    difficulties = np.linspace(*generator.difficulty_range, generator.num_rows)
    centers, exit_starts, half_sizes, stone_counts = [], [], [], []
    for cfg in generator.sub_terrains.values():
        counts, gaps = zip(*(cfg.count_and_gap(difficulty) for difficulty in difficulties))
        # Fill every observation slot and mask inactive stones separately.
        centers.append([stone_centers(MAX_STONES, gap, size=cfg.stone_size) for gap in gaps])
        exit_starts.append([right_platform_start(count, gap, size=cfg.stone_size) for count, gap in zip(counts, gaps)])
        half_sizes.append(cfg.stone_size / 2)
        stone_counts.append(counts)
    return (
        torch.as_tensor(np.asarray(centers), dtype=torch.float32, device=env.device),
        torch.tensor(exit_starts, dtype=torch.float32, device=env.device),
        torch.tensor(half_sizes, dtype=torch.float32, device=env.device),
        torch.tensor(stone_counts, device=env.device),
    )


def _state(env):
    if hasattr(env, "_stepping_stones"):
        return env._stepping_stones
    device = env.device
    zeros = lambda: torch.zeros(env.num_envs, device=device)
    robot = env.scene["robot"]
    foot_site_ids = []
    for site_name in FOOT_SITE_NAMES:
        ids, _ = robot.find_sites(site_name)
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one foot site named '{site_name}'.")
        foot_site_ids.append(ids[0])
    contact_sensor = env.scene["stepping_foot_contact"]
    hip_abduction_ids = []
    knee_ids = []
    knee_body_ids = []
    for leg in LEG_NAMES:
        ids, _ = robot.find_joints(f"{leg}_hip_abduction_joint")
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one hip abduction joint for '{leg}'.")
        hip_abduction_ids.append(ids[0])
        knee, _ = robot.find_joints(f"{leg}_knee_joint")
        if len(knee) != 1:
            raise ValueError(f"Expected exactly one knee joint for '{leg}'.")
        knee_ids.append(knee[0])
        knee_body, _ = robot.find_bodies(f"{leg}_knee_Link")
        if len(knee_body) != 1:
            raise ValueError(f"Expected exactly one knee body for '{leg}'.")
        knee_body_ids.append(knee_body[0])
    contact_ids = []
    for body_name in FOOT_BODY_NAMES:
        ids = [index for index, name in enumerate(contact_sensor.primary_names) if name.endswith(body_name)]
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one contact channel for '{body_name}'.")
        contact_ids.append(ids[0])
    shin_names = env.scene["stepping_shin_contact"].primary_names
    shin_ids = []
    for leg in LEG_NAMES:
        ids = [index for index, name in enumerate(shin_names) if name.endswith(f"{leg}_shin_collision")]
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one shin contact channel for '{leg}'.")
        shin_ids.append(ids[0])
    env._stepping_stones = {
        "geometry_tables": _geometry_tables(env),
        "active_count": torch.full((env.num_envs,), 2, dtype=torch.long, device=device),
        "visited": torch.zeros(env.num_envs, MAX_STONES, dtype=torch.bool, device=device),
        "failure": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "success": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "stage": torch.full((env.num_envs,), 2, dtype=torch.long, device=device),
        "episode_speed": torch.full((env.num_envs,), sum(STEPPING_FORWARD_SPEED_RANGE) / 2.0, device=device),
        "stone_contact_substeps": torch.zeros(
            env.num_envs, len(FOOT_BODY_NAMES), MAX_STONES, dtype=torch.long, device=device
        ),
        "stone_loaded_substeps": torch.zeros(
            env.num_envs, len(FOOT_BODY_NAMES), MAX_STONES, dtype=torch.long, device=device
        ),
        "stone_center_awarded": torch.zeros(env.num_envs, 4, MAX_STONES, dtype=torch.bool, device=device),
        "stone_center_pulse": zeros(),
        "stone_edge_pulse": zeros(),
        "stone_centers": torch.zeros(env.num_envs, MAX_STONES, 3, device=device),
        "stone_half_size": zeros(),
        "right_platform_start": zeros(),
        "stall_anchor_x": zeros(),
        "stall_age": zeros(),
        "stall_tick": torch.full((env.num_envs,), -1, dtype=torch.long, device=device),
        "flight_step_time": zeros(),
        "left_front_unsupported_time": zeros(),
        "left_front_liftoff_pulse": zeros(),
        "left_front_overlap_count": zeros(),
        "left_front_overreach_cost_time": zeros(),
        "left_front_previous_height": torch.full((env.num_envs,), torch.nan, device=device),
        "left_front_handoff": LeftFrontHandoff(env.num_envs, device),
        "premature_support_handoff_substeps": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "premature_support_handoff_pulse": zeros(),
        "premature_support_handoffs": zeros(),
        "low_support_time": zeros(),
        "shin_stone_cost_time": zeros(),
        "foot_side_cost_time": zeros(),
        "knee_reverse_cost_time": zeros(),
        "shin_loaded_substeps": torch.zeros(env.num_envs, 4, dtype=torch.long, device=device),
        "shin_released_substeps": torch.zeros(env.num_envs, 4, dtype=torch.long, device=device),
        "shin_support_latched": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device),
        "shin_support_pulse": zeros(),
        "shin_support_this_step": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "platform_transition": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "exit_contact_substeps": torch.zeros(env.num_envs, 4, dtype=torch.long, device=device),
        "exit_visited": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device),
        "exit_settle_substeps": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "exit_stable_substeps": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "exit_best_stable_substeps": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "exit_best_quality": zeros(),
        "exit_stability_pulse": zeros(),
        "legal_support": torch.zeros(env.num_envs, len(FOOT_BODY_NAMES), dtype=torch.bool, device=device),
        "cover_pulse": zeros(),
        "success_pulse": zeros(),
        "foot_step_pulse": zeros(),
        "right_rear_step_pulse": zeros(),
        "right_rear_forward_steps": zeros(),
        "right_rear_follow_substeps": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "right_rear_follow_wait_s": zeros(),
        "right_rear_overdue_time": zeros(),
        "no_forward_step_substeps": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "no_forward_step_cost_time": zeros(),
        "no_forward_step_time_s": zeros(),
        "step_seen": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device),
        "step_air": torch.zeros(env.num_envs, 4, dtype=torch.long, device=device),
        "startup_armed": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "startup_progress": zeros(),
        "startup_progress_pulse": zeros(),
        "startup_early_pulse": zeros(),
        "startup_closed": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "startup_spawn_z": zeros(),
        "startup_rise_substeps": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "startup_rise_cost_time": zeros(),
        "startup_peak_body_rise_m": zeros(),
        "startup_peak_front_rise_m": zeros(),
        "traverse_peak_body_sink_m": zeros(),
        "peak_foot_clearance_m": zeros(),
        "last_control_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=device),
        "foot_site_ids": foot_site_ids,
        "contact_ids": contact_ids,
        "shin_ids": shin_ids,
        "hip_abduction_ids": hip_abduction_ids,
        "knee_ids": knee_ids,
        "knee_body_ids": knee_body_ids,
    }
    if "fixed_gait" in env.cfg.observations["actor"].terms:
        env._stepping_stones["fixed_gait"] = FixedStoneGait(env.num_envs, device)
    return env._stepping_stones


def _active_mask(state: dict[str, torch.Tensor]) -> torch.Tensor:
    index = torch.arange(MAX_STONES, device=state["active_count"].device)
    return index[None, :] < state["active_count"][:, None]


def _relative_xy(env, points_w: torch.Tensor) -> torch.Tensor:
    origin = env.scene.env_origins
    return points_w - origin[:, None, :]


def _foot_bottom(env, state) -> torch.Tensor:
    """Convert the rotating sole sites to the spherical feet's world-lowest points."""
    robot = env.scene["robot"].data
    sites = robot.site_pos_w[:, state["foot_site_ids"]]
    rotation = matrix_from_quat(robot.site_quat_w[:, state["foot_site_ids"]])
    return sites + FOOT_RADIUS * (rotation[..., :, 2] - sites.new_tensor((0.0, 0.0, 1.0)))


def _front_hip_offset_w(robot) -> torch.Tensor:
    rotation = matrix_from_quat(robot.root_link_quat_w)
    return STARTUP_FRONT_X * rotation[:, :, 0] + STARTUP_FRONT_Z * rotation[:, :, 2]


def _top_contact_masks(env, state, found: torch.Tensor, position_w: torch.Tensor, normal_z: torch.Tensor):
    local = _relative_xy(env, position_w)
    # The sensor normal points from the foot to the terrain.
    top = found & (normal_z <= -MIN_TOP_NORMAL_Z) & ((local[..., 2] - PLATFORM_HEIGHT).abs() <= TOP_CONTACT_Z_TOL)
    stone_delta = local[:, :, None, :] - state["stone_centers"][:, None, :, :]
    stone_limit = state["stone_half_size"][:, None, None] + TOP_EDGE_TOL
    stone_top = (
        top[:, :, None]
        & (stone_delta[..., 0].abs() <= stone_limit)
        & (stone_delta[..., 1].abs() <= stone_limit)
        & _active_mask(state)[:, None, :]
    )
    left_top = (
        top
        & (local[:, :, 0] >= -TOP_EDGE_TOL)
        & (local[:, :, 0] <= PLATFORM_SIZE + TOP_EDGE_TOL)
        & (local[:, :, 1].abs() <= PLATFORM_SIZE / 2 + TOP_EDGE_TOL)
    )
    right_start = state["right_platform_start"]
    right_top = (
        top
        & (local[:, :, 0] >= right_start[:, None] - TOP_EDGE_TOL)
        & (local[:, :, 0] <= right_start[:, None] + PLATFORM_SIZE + TOP_EDGE_TOL)
        & (local[:, :, 1].abs() <= PLATFORM_SIZE / 2 + TOP_EDGE_TOL)
    )
    ground = found & (local[:, :, 2] <= GROUND_CONTACT_Z_TOL)
    return stone_top, left_top, right_top, ground


def reset_stepping_stones(env, env_ids, terrain_level: int | None = None, randomize_levels: bool = False) -> None:
    """Place each robot in its nominal standing pose on the left platform."""
    ids = _ids(env, env_ids)
    state = _state(env)
    terrain = env.scene.terrain
    if terrain is None:
        raise RuntimeError("Stepping stones requires generated terrain.")
    if randomize_levels and terrain_level is not None:
        raise ValueError("Choose either a fixed terrain level or random levels.")
    centers, exit_starts, half_sizes, counts = state["geometry_tables"]
    num_levels = counts.shape[1]
    if randomize_levels:
        terrain.terrain_levels[ids] = torch.randint(0, num_levels, (len(ids),), device=env.device)
        terrain.terrain_types[ids] = torch.randint(0, half_sizes.numel(), (len(ids),), device=env.device)
        terrain.env_origins[ids] = terrain.get_terrain_origins(terrain.terrain_levels[ids], terrain.terrain_types[ids])
    elif terrain_level is not None:
        if not 0 <= terrain_level < num_levels:
            raise ValueError(f"Stepping-stone terrain level must be between 0 and {num_levels - 1}.")
        terrain.terrain_levels[ids] = terrain_level
        terrain.env_origins[ids] = terrain.get_terrain_origins(terrain.terrain_levels[ids], terrain.terrain_types[ids])
    types = terrain.terrain_types[ids]
    physical_levels = terrain.get_geometry_levels(terrain.terrain_levels[ids], types)
    state["active_count"][ids] = counts[types, physical_levels]
    state["stone_centers"][ids] = centers[types, physical_levels]
    state["stone_half_size"][ids] = half_sizes[types]
    state["right_platform_start"][ids] = exit_starts[types, physical_levels]
    state["visited"][ids] = False
    state["failure"][ids] = False
    state["success"][ids] = False
    state["stage"][ids] = 2
    state["stone_contact_substeps"][ids] = 0
    state["stone_loaded_substeps"][ids] = 0
    state["stone_center_awarded"][ids] = False
    state["stone_center_pulse"][ids] = 0.0
    state["stone_edge_pulse"][ids] = 0.0
    state["stall_anchor_x"][ids] = SPAWN_X
    state["stall_age"][ids] = 0.0
    state["stall_tick"][ids] = -1
    state["flight_step_time"][ids] = 0.0
    state["left_front_unsupported_time"][ids] = 0.0
    state["left_front_liftoff_pulse"][ids] = 0.0
    state["left_front_overlap_count"][ids] = 0.0
    state["left_front_overreach_cost_time"][ids] = 0.0
    state["left_front_previous_height"][ids] = torch.nan
    state["left_front_handoff"].reset(ids)
    if "fixed_gait" in state:
        state["fixed_gait"].reset(ids)
    state["premature_support_handoff_substeps"][ids] = 0
    state["premature_support_handoff_pulse"][ids] = 0.0
    state["premature_support_handoffs"][ids] = 0.0
    state["low_support_time"][ids] = 0.0
    state["shin_stone_cost_time"][ids] = 0.0
    state["foot_side_cost_time"][ids] = 0.0
    state["knee_reverse_cost_time"][ids] = 0.0
    state["shin_loaded_substeps"][ids] = 0
    state["shin_released_substeps"][ids] = 0
    state["shin_support_latched"][ids] = False
    state["shin_support_pulse"][ids] = 0.0
    state["shin_support_this_step"][ids] = False
    state["platform_transition"][ids] = False
    state["exit_contact_substeps"][ids] = 0
    state["exit_visited"][ids] = False
    state["exit_settle_substeps"][ids] = 0
    state["exit_stable_substeps"][ids] = 0
    state["exit_best_stable_substeps"][ids] = 0
    state["exit_best_quality"][ids] = 0.0
    state["exit_stability_pulse"][ids] = 0.0
    state["legal_support"][ids] = False
    state["cover_pulse"][ids] = 0.0
    state["success_pulse"][ids] = 0.0
    state["foot_step_pulse"][ids] = 0.0
    state["right_rear_step_pulse"][ids] = 0.0
    state["right_rear_forward_steps"][ids] = 0.0
    state["right_rear_follow_substeps"][ids] = 0
    state["right_rear_follow_wait_s"][ids] = 0.0
    state["right_rear_overdue_time"][ids] = 0.0
    state["no_forward_step_substeps"][ids] = 0
    state["no_forward_step_cost_time"][ids] = 0.0
    state["no_forward_step_time_s"][ids] = 0.0
    state["step_seen"][ids] = False
    state["step_air"][ids] = 0
    state["startup_armed"][ids] = False
    state["startup_progress"][ids] = 0
    state["startup_progress_pulse"][ids] = 0
    state["startup_early_pulse"][ids] = 0
    state["startup_closed"][ids] = False
    state["startup_rise_substeps"][ids] = math.ceil(STARTUP_RISE_WINDOW_S / env.physics_dt)
    state["startup_rise_cost_time"][ids] = 0.0
    state["startup_peak_body_rise_m"][ids] = 0.0
    state["startup_peak_front_rise_m"][ids] = 0.0
    state["traverse_peak_body_sink_m"][ids] = 0.0
    state["peak_foot_clearance_m"][ids] = 0.0
    state["last_control_step"][ids] = -1
    state["episode_speed"][ids] = torch.empty(len(ids), device=env.device).uniform_(*STEPPING_FORWARD_SPEED_RANGE)
    robot = env.scene["robot"]
    root_state = robot.data.default_root_state[ids].clone()
    root_state[:, 0] = env.scene.env_origins[ids, 0] + SPAWN_X
    root_state[:, 1] = env.scene.env_origins[ids, 1]
    root_state[:, 2] += env.scene.env_origins[ids, 2] + PLATFORM_HEIGHT
    state["startup_spawn_z"][ids] = root_state[:, 2]
    root_state[:, 3:7] = root_state.new_tensor((1.0, 0.0, 0.0, 0.0))
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=ids)


def _collect_exit_settle(env, state, robot, all_covered, right_top, effective_support):
    exit_steps = math.ceil(EXIT_CONTACT_TIME_S / env.physics_dt)
    exit_contact = state["exit_contact_substeps"]
    exit_contact.add_(1).clamp_max_(exit_steps).masked_fill_(~right_top, 0)
    state["exit_visited"] |= (exit_contact >= exit_steps) & ~state["failure"][:, None]
    reached = all_covered & state["exit_visited"].all(dim=1) & ~state["failure"]
    if "fixed_gait" in state:
        reached &= state["fixed_gait"].done
    state["stage"].masked_fill_(reached, 3)
    settling = (state["stage"] == 3) & ~state["failure"] & ~state["success"]
    state["platform_transition"] &= state["stage"] == 2
    state["exit_settle_substeps"] += settling.long()

    exit_support = right_top & effective_support
    gravity = robot.projected_gravity_b
    tilt = torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=1), -gravity[:, 2])
    errors = torch.stack((
        (robot.root_link_pos_w[:, 2] - state["startup_spawn_z"]).abs() / EXIT_HEIGHT_TOL_M,
        tilt / EXIT_TILT_TOL_RAD,
        torch.linalg.vector_norm(robot.root_link_lin_vel_w, dim=1) / EXIT_LINEAR_SPEED_MPS,
        torch.linalg.vector_norm(robot.root_link_ang_vel_w, dim=1) / EXIT_ANGULAR_SPEED_RADPS,
        (robot.joint_pos - robot.default_joint_pos).abs().amax(dim=1) / EXIT_JOINT_ERROR_RAD,
    ), dim=1)
    valid_pose = torch.isfinite(errors).all(dim=1)
    quiet_pose = torch.exp(-errors.square().mean(dim=1))
    quality = exit_support.float().mean(dim=1) * (0.5 + 0.5 * quiet_pose)
    quality = torch.where(settling & valid_pose, quality, 0.0)
    best_quality = state["exit_best_quality"]
    state["exit_stability_pulse"] += (quality - best_quality).clamp_min(0.0)
    best_quality.copy_(torch.maximum(best_quality, quality))

    required = math.ceil(EXIT_SETTLE_TIME_S / env.physics_dt)
    stable = settling & exit_support.all(dim=1) & valid_pose & (errors <= 1.0).all(dim=1)
    stable_steps = state["exit_stable_substeps"]
    stable_steps.add_(stable.long()).clamp_max_(required).masked_fill_(~stable & ~state["success"], 0)
    best_stable = state["exit_best_stable_substeps"]
    best_stable.copy_(torch.maximum(best_stable, stable_steps))
    success_event = settling & (stable_steps >= required)
    state["success"] |= success_event
    state["success_pulse"] += success_event.float()


def _collect_no_forward_step_cost(state, robot, surface, completed_step, physics_dt):
    startup_done = state["startup_closed"] | (state["startup_rise_substeps"] == 0)
    traversing = (state["stage"] == 2) & ~state["failure"] & startup_done
    grace = math.ceil(NO_FORWARD_STEP_GRACE_S / physics_dt)
    elapsed = state["no_forward_step_substeps"]
    elapsed.add_(1).clamp_max_(grace + 1).masked_fill_(~traversing | completed_step.any(dim=1), 0)
    overdue = elapsed > grace

    last_surface = state["left_front_handoff"].last_surface
    feet_on_same_surfaces = ((surface >= 0) & (surface == last_surface)).all(dim=1)
    linear_speed = torch.linalg.vector_norm(robot.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(robot.root_link_ang_vel_w, dim=1)
    linear_cost = ((linear_speed - 0.03) / 0.12).clamp(0.0, 1.0)
    angular_cost = ((angular_speed - 0.15) / 0.60).clamp(0.0, 1.0)
    rocking = torch.maximum(linear_cost, angular_cost).nan_to_num(nan=0.0)
    motion_cost = torch.where(feet_on_same_surfaces, rocking, 0.0)
    cost = torch.where(overdue, 0.2 + 0.8 * motion_cost, 0.0)
    state["no_forward_step_cost_time"] += cost * physics_dt
    state["no_forward_step_time_s"] += overdue.float() * physics_dt


def _collect_left_front_foothold_cost(state, foot_local, knee_angle, surface, target, traversing, physics_dt):
    x, y, z = foot_local.unbind(dim=1)
    height = z - PLATFORM_HEIGHT
    previous_height = state["left_front_previous_height"]
    descending = (height < previous_height - 0.03 * physics_dt) & torch.isfinite(previous_height)
    previous_height.copy_(height)

    target_is_stone = target != EXIT_SURFACE
    slot = (target - 1).clamp_max(MAX_STONES - 1)
    stone_xy = state["stone_centers"][:, :, :2].gather(1, slot[:, None, None].expand(-1, 1, 2)).squeeze(1)
    target_y = torch.where(target_is_stone, stone_xy[:, 1], 0.0)
    half_size = torch.where(target_is_stone, state["stone_half_size"], PLATFORM_SIZE / 2)
    exit_start = state["right_platform_start"]
    front_edge = torch.where(target_is_stone, stone_xy[:, 0] + half_size, exit_start + EXIT_FIRST_LANDING_MAX_M)
    safe_x_min = torch.where(target_is_stone, stone_xy[:, 0] - half_size + LANDING_EDGE_MARGIN,
                             exit_start + EXIT_FIRST_LANDING_MIN_M)
    safe_x_max = torch.where(target_is_stone, front_edge - LANDING_EDGE_MARGIN, front_edge)
    handoff = state["left_front_handoff"]
    air = surface < 0

    past_front = ((x - front_edge - 0.02).clamp_min(0.0) / 0.07).square().clamp_max(1.0)
    past_front = torch.where(surface != handoff.last_surface[:, 0], past_front, 0.0)

    outside_x = torch.maximum(safe_x_min - x, x - safe_x_max)
    outside_y = (y - target_y).abs() - half_size + LANDING_EDGE_MARGIN
    outside_safe_top = torch.maximum(outside_x, outside_y).clamp_min(0.0)
    near_top = (1.0 - height / 0.07).clamp(0.0, 1.0)
    missed_landing = (outside_safe_top / 0.05).square().clamp_max(1.0) * near_top
    missed_landing = torch.where(descending & air, missed_landing, 0.0)

    flexion_deficit = (math.radians(20.0) + knee_angle).clamp_min(0.0)
    straight_knee = (flexion_deficit / math.radians(15.0)).square().clamp_max(1.0)
    straight_knee = torch.where(air & torch.isfinite(knee_angle), straight_knee, 0.0)
    route_origin = handoff.left_front_route_surface
    active = traversing & (route_origin >= 0) & (route_origin != EXIT_SURFACE) & torch.isfinite(foot_local).all(dim=1)
    cost = torch.maximum(torch.maximum(past_front, missed_landing), straight_knee)
    state["left_front_overreach_cost_time"] += torch.where(active, cost, 0.0) * physics_dt


def collect_stepping_stones_substep(env) -> torch.Tensor:
    """Track consecutive top contacts and latch ground failures at physics frequency."""
    state = _state(env)
    new_control_step = state["last_control_step"] != env.common_step_counter
    if "fixed_gait" in state:
        state["fixed_gait"].clear_control(new_control_step)
    for key in ("cover_pulse", "success_pulse", "exit_stability_pulse", "foot_step_pulse", "startup_progress_pulse", "startup_early_pulse",
                "startup_rise_cost_time", "no_forward_step_cost_time", "right_rear_step_pulse", "right_rear_overdue_time",
                "stone_center_pulse", "stone_edge_pulse", "flight_step_time", "left_front_unsupported_time",
                "left_front_liftoff_pulse", "left_front_overreach_cost_time", "premature_support_handoff_pulse", "low_support_time",
                "shin_stone_cost_time", "foot_side_cost_time", "knee_reverse_cost_time",
                "shin_support_pulse", "shin_support_this_step"):
        state[key].masked_fill_(new_control_step, 0.0)
    state["last_control_step"].fill_(env.common_step_counter)

    sensor = env.scene["stepping_foot_contact"].data
    if sensor.found is None or sensor.pos is None or sensor.normal is None or sensor.force is None:
        raise RuntimeError("Stepping-stone foot contacts require found, pos, normal and force.")
    contacts = state["contact_ids"]
    found = sensor.found[:, contacts] > 0
    contact_pos = sensor.pos[:, contacts]
    normal_z = sensor.normal[:, contacts, 2]
    stone_top, left_top, right_top, ground = _top_contact_masks(env, state, found, contact_pos, normal_z)
    state["legal_support"] = stone_top.any(dim=2) | left_top | right_top
    physics_dt = env.physics_dt
    shin = env.scene["stepping_shin_contact"].data
    if shin.found is None or shin.force is None or shin.pos is None:
        raise RuntimeError("Shin contact sensor requires found, force and pos fields.")
    local_shin = shin.pos - env.scene.env_origins[:, None, :]
    delta_shin = local_shin[:, :, None, :] - state["stone_centers"][:, None, :, :]
    shin_stone_limit = state["stone_half_size"][:, None, None, None] + 0.01
    on_stone = ((delta_shin[..., :2].abs() <= shin_stone_limit).all(dim=-1)
                & (local_shin[:, :, None, 2] >= -0.01)
                & (local_shin[:, :, None, 2] <= PLATFORM_HEIGHT + 0.01)
                & _active_mask(state)[:, None, :]).any(dim=-1)
    exit_start = state["right_platform_start"][:, None]
    sx, sy, sz = local_shin.unbind(dim=-1)
    left_platform = (sx >= -0.01) & (sx <= PLATFORM_SIZE + 0.01)
    right_platform = (sx >= exit_start - 0.01) & (sx <= exit_start + PLATFORM_SIZE + 0.01)
    platform_contact = ((left_platform | right_platform) & (sy.abs() <= PLATFORM_SIZE / 2 + 0.01)
                        & (sz > 0.01) & (sz <= PLATFORM_HEIGHT + 0.01))
    force = torch.linalg.vector_norm(shin.force, dim=-1)
    cost = ((force - 2.0).clamp_min(0.0) / 18.0).clamp_max(1.0)
    raised_contact = (shin.found > 0) & (on_stone | platform_contact) & torch.isfinite(force)
    cost = torch.where(raised_contact, cost, 0.0)
    state["shin_stone_cost_time"] += cost.amax(dim=1) * physics_dt
    # Sensor forces point from the primary (shin) toward the terrain.
    vertical = torch.where(raised_contact, (-shin.force[..., 2]).clamp_min(0.0), 0.0)
    shin_load = vertical.reshape(env.num_envs, len(LEG_NAMES), -1).sum(dim=-1)[:, state["shin_ids"]]
    loaded = shin_load >= SHIN_SUPPORT_N
    confirm_steps = math.ceil(SHIN_CONFIRM_S / physics_dt)
    loaded_steps = state["shin_loaded_substeps"]
    released_steps = state["shin_released_substeps"]
    latched = state["shin_support_latched"]
    loaded_steps.add_(1).clamp_max_(confirm_steps).masked_fill_(~loaded, 0)
    released_steps.add_(1).clamp_max_(confirm_steps).masked_fill_(loaded, 0)
    event = ~latched & ((loaded_steps >= confirm_steps) | (shin_load >= SHIN_IMPACT_N))
    latched.masked_fill_(released_steps >= confirm_steps, False)
    latched.logical_or_(event)
    state["shin_support_pulse"] += event.sum(dim=1)
    state["shin_support_this_step"] |= (latched & loaded).any(dim=1)

    # The foot sensor force is in contact coordinates, with normal force in slot zero.
    foot_normal_force = sensor.force[:, contacts, 0]
    contact_height = contact_pos[..., 2] - env.scene.env_origins[:, 2:3]
    side_contact = (found & ~ground & (normal_z.abs() < MIN_TOP_NORMAL_Z)
                    & (contact_height <= PLATFORM_HEIGHT + TOP_CONTACT_Z_TOL))
    side_cost = ((foot_normal_force - 2.0) / 18.0).clamp(0.0, 1.0)
    side_cost = torch.where(side_contact & torch.isfinite(foot_normal_force), side_cost, 0.0)
    state["foot_side_cost_time"] += side_cost.amax(dim=1) * physics_dt
    loaded_foot = torch.isfinite(foot_normal_force) & (foot_normal_force >= MIN_FOOT_SUPPORT_N) & ~loaded
    loaded_stone_top = stone_top & loaded_foot[:, :, None]
    support_count = found.sum(dim=1)
    state["flight_step_time"] += (support_count == 0).float() * physics_dt
    state["failure"] |= ground.any(dim=1)
    state["success"] &= ~state["failure"]

    foot_bottom = _foot_bottom(env, state)
    foot_z = foot_bottom[..., 2] - env.scene.env_origins[:, 2:3]
    clearance = torch.where(torch.isfinite(foot_z), foot_z - PLATFORM_HEIGHT, 0.0)
    state["peak_foot_clearance_m"] = torch.maximum(state["peak_foot_clearance_m"], clearance.amax(dim=1))

    robot = env.scene["robot"].data
    knee_angles = robot.joint_pos[:, state["knee_ids"]]
    knee_reverse = (knee_angles / KNEE_REVERSE_FULL_RAD).clamp(0.0, 1.0).square()
    knee_reverse = torch.where(torch.isfinite(knee_angles), knee_reverse, 0.0)
    state["knee_reverse_cost_time"] += knee_reverse.amax(dim=1) * physics_dt
    startup_substeps = state["startup_rise_substeps"]
    rise = robot.root_link_pos_w[:, 2] - state["startup_spawn_z"]
    upward_speed = robot.root_link_lin_vel_w[:, 2]
    monitor_rise = (startup_substeps > 0) & torch.isfinite(rise) & torch.isfinite(upward_speed)
    rise_cost = ((rise - 0.02).clamp_min(0.0) / 0.04).square()
    speed_cost = ((upward_speed - 0.10).clamp_min(0.0) / 0.30).square()
    front_offset = _front_hip_offset_w(robot)
    front_rise = rise + front_offset[:, 2] - STARTUP_FRONT_Z
    angular_velocity = robot.root_link_ang_vel_w
    front_speed = upward_speed + (angular_velocity[:, 0] * front_offset[:, 1]
                                  - angular_velocity[:, 1] * front_offset[:, 0])
    monitor_front = (startup_substeps > 0) & torch.isfinite(front_rise) & torch.isfinite(front_speed)
    front_cost = torch.maximum(((front_rise - 0.02).clamp_min(0.0) / 0.04).square(),
                               ((front_speed - 0.10).clamp_min(0.0) / 0.30).square())
    startup_cost = torch.maximum(torch.where(monitor_rise, torch.maximum(rise_cost, speed_cost), 0.0),
                                 torch.where(monitor_front, front_cost, 0.0))
    state["startup_rise_cost_time"] += startup_cost.clamp_max(1.0) * physics_dt
    peak = state["startup_peak_body_rise_m"]
    peak.copy_(torch.maximum(peak, torch.where(monitor_rise, rise, 0.0)))
    front_peak = state["startup_peak_front_rise_m"]
    front_peak.copy_(torch.maximum(front_peak, torch.where(monitor_front, front_rise, 0.0)))
    startup_substeps.sub_(1).clamp_min_(0)

    support = state["legal_support"]
    step_seen = state["step_seen"]
    step_seen |= support
    confirm = math.ceil(FORWARD_STEP_CONFIRM_S / physics_dt)
    effective_support = support & loaded_foot
    active = ~state["failure"] & ~state["success"]
    stone_surface = stone_top.long().argmax(dim=2) + 1
    surface = torch.where(stone_top.any(dim=2), stone_surface, -1)
    surface.masked_fill_(left_top, 0).masked_fill_(right_top, EXIT_SURFACE)
    handoff = state["left_front_handoff"]
    traversing = active & (state["stage"] == 2)
    sink = -torch.minimum(rise, front_rise)
    sink_peak = state["traverse_peak_body_sink_m"]
    sink_peak.copy_(torch.maximum(sink_peak, torch.where(traversing & torch.isfinite(sink), sink, 0.0)))
    left_stones = (state["stone_centers"][:, :, 1] > 0) & _active_mask(state)
    left_front_target = handoff.left_front_target_surface(left_stones).masked_fill_(~traversing, EXIT_SURFACE)
    left_front_local = foot_bottom[:, 0] - env.scene.env_origins
    left_front_knee = robot.joint_pos[:, state["knee_ids"][0]]
    _collect_left_front_foothold_cost(
        state, left_front_local, left_front_knee, surface[:, 0], left_front_target, traversing, physics_dt,
    )
    holding_left_front, violation, blocked_landing, rewarded_step = handoff.update(
        surface, effective_support, active, physics_dt, traversing=traversing, left_front_target=left_front_target,
    )
    if "fixed_gait" in state:
        gait = state["fixed_gait"]
        local_feet = foot_bottom - env.scene.env_origins[:, None, :]
        gait.update(local_feet, surface, effective_support, state["active_count"], state["stone_centers"],
                    state["right_platform_start"], traversing, physics_dt,
                    stone_half_width=state["stone_half_size"][:, None])
        blocked_landing = torch.zeros_like(blocked_landing)
        rewarded_step = gait.correct_landing & gait.pulses["complete"].bool()[:, None]
    state["left_front_unsupported_time"] += (holding_left_front & ~effective_support[:, 0]).float() * physics_dt
    state["left_front_liftoff_pulse"] += violation.float()
    state["left_front_overlap_count"] += violation.float()
    state["foot_step_pulse"] += rewarded_step.sum(dim=1)
    state["right_rear_step_pulse"] += rewarded_step[:, 3].float()
    state["right_rear_forward_steps"] += rewarded_step[:, 3].float()

    handoff_steps = math.ceil(SUPPORT_HANDOFF_CONFIRM_S / physics_dt)
    overlap = state["premature_support_handoff_substeps"]
    right_front_and_left_rear_unloaded = (step_seen[:, 1] & step_seen[:, 2]) & ~effective_support[:, 1] & ~effective_support[:, 2]
    left_front_stable = handoff.loaded_substeps[:, 0] >= confirm
    premature_handoff = active & right_front_and_left_rear_unloaded & (~left_front_stable | (overlap > 0))
    overlap.add_(1).clamp_max_(handoff_steps + 1).masked_fill_(~premature_handoff, 0)
    handoff_event = (overlap == handoff_steps).float()
    state["premature_support_handoff_pulse"] += handoff_event
    state["premature_support_handoffs"] += handoff_event

    too_few_supports = (effective_support.sum(dim=1) < 2) & (step_seen.sum(dim=1) >= 2)
    state["low_support_time"] += (active & too_few_supports).float() * physics_dt
    step_air = state["step_air"]
    step_air.add_(1).clamp_max_(confirm).masked_fill_(support | ~step_seen, 0)
    state["startup_armed"] |= left_top.all(dim=1) & ~state["failure"]
    startup_active = ~state["startup_closed"] & state["startup_armed"] & ~state["failure"]
    root_x = robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress = ((root_x - SPAWN_X) / 0.05).clamp(0.0, 1.0)
    progress = torch.where(torch.isfinite(progress) & support.all(dim=1) & startup_active, progress, 0.0)
    improvement = (progress - state["startup_progress"]).clamp_min(0.0)
    state["startup_progress_pulse"] += improvement
    state["startup_progress"] = torch.maximum(state["startup_progress"], progress)
    early = startup_active & (state["startup_progress"] < 1.0 - 1.e-5) & (step_air >= confirm).any(dim=1)
    state["startup_early_pulse"] += early.float()
    state["startup_closed"] |= early | (state["startup_progress"] >= 1.0 - 1.e-5)
    route_x = robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    active = _active_mask(state)
    all_covered_before = (~active | state["visited"]).all(dim=1)
    all_on_right = right_top.all(dim=1)
    right_start = state["right_platform_start"]
    state["platform_transition"] = all_covered_before & (route_x >= right_start - 0.5) & ~all_on_right

    required_contact_steps = math.ceil(STONE_CONTACT_TIME_S / physics_dt)
    contact_substeps = state["stone_contact_substeps"]
    loaded_substeps = state["stone_loaded_substeps"]
    live = ~state["failure"][:, None, None]
    touchdown = stone_top & (contact_substeps == required_contact_steps - 1) & live
    loaded_touchdown = loaded_stone_top & (loaded_substeps == required_contact_steps - 1) & live
    if "fixed_gait" in state:
        loaded_touchdown = loaded_stone_top & state["fixed_gait"].correct_landing[:, :, None] & live
    loaded_touchdown[:, 0] &= ~blocked_landing[:, None]
    local_contact = sensor.pos[:, contacts, :2] - env.scene.env_origins[:, None, :2]
    offset = local_contact[:, :, None, :] - state["stone_centers"][:, None, :, :2]
    edge = offset.abs().amax(dim=-1) >= state["stone_half_size"][:, None, None] - LANDING_EDGE_MARGIN
    quality = torch.exp(-offset.square().sum(dim=-1) / (0.04 ** 2))
    fresh = loaded_touchdown & ~state["stone_center_awarded"]
    state["stone_center_pulse"] += torch.where(fresh & ~edge, quality, 0.0).sum(dim=(1, 2))
    state["stone_center_awarded"] |= loaded_touchdown
    state["stone_edge_pulse"] += (touchdown & edge).float().sum(dim=(1, 2))
    contact_substeps.add_(1).clamp_max_(required_contact_steps).masked_fill_(~stone_top, 0)
    loaded_substeps.add_(1).clamp_max_(required_contact_steps).masked_fill_(~loaded_stone_top, 0)
    new_stones = loaded_touchdown.any(dim=1) & ~state["visited"]
    state["visited"] |= new_stones
    state["cover_pulse"] += new_stones.sum(dim=1) / state["active_count"]

    all_covered = (~active | state["visited"]).all(dim=1)
    _collect_exit_settle(env, state, robot, all_covered, right_top, effective_support)
    _collect_no_forward_step_cost(state, robot, surface, rewarded_step, physics_dt)
    waiting_for_rear = handoff.right_rear_follow_pending & (state["stage"] == 2)
    grace = math.ceil(RIGHT_REAR_FOLLOW_GRACE_S / physics_dt)
    waiting_steps = state["right_rear_follow_substeps"]
    waiting_steps.add_(1).clamp_max_(grace + 1).masked_fill_(~waiting_for_rear, 0)
    state["right_rear_follow_wait_s"] += waiting_for_rear.float() * physics_dt
    state["right_rear_overdue_time"] += (waiting_steps > grace).float() * physics_dt

    for key in ("cover_pulse", "success_pulse", "exit_stability_pulse", "foot_step_pulse", "startup_progress_pulse", "stone_center_pulse"):
        state[key].masked_fill_(state["failure"], 0.0)
    return support_count.float() / len(FOOT_BODY_NAMES)


class SteppingStonesVelocityCommand(UniformThresholdVelocityCommand):
    """Apply the sampled forward command until the finish hold."""

    def _apply_task_command(self) -> None:
        state = _state(self._env)
        holding = state["stage"] != 2
        self.is_standing_env[:] = holding
        self.is_forward_env[:] = ~holding
        self.is_heading_env[:] = False
        self.is_world_env[:] = False
        self.is_lateral_env[:] = False
        self.is_yaw_env[:] = False
        self.is_backward_env[:] = False
        self.vel_command_b[:, 0] = torch.where(holding, 0.0, state["episode_speed"])
        self.vel_command_b[:, 1:] = 0.0
        self.vel_command_w[:] = self.vel_command_b
        heading = self.robot.data.heading_w
        self.progress_direction_w[:, 0] = torch.cos(heading) * self.vel_command_b[:, 0]
        self.progress_direction_w[:, 1] = torch.sin(heading) * self.vel_command_b[:, 0]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        del env_ids
        self._apply_task_command()

    def _update_command(self) -> None:
        self._apply_task_command()


@dataclass(kw_only=True)
class SteppingStonesVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the fixed-direction stepping-stone velocity command."""

    def build(self, env):
        return SteppingStonesVelocityCommand(self, env)


def stepping_failed(env) -> torch.Tensor:
    return _state(env)["failure"]


def stepping_bad_orientation(env, limit_angle: float = 1.0) -> torch.Tensor:
    """Apply the orientation termination outside platform transitions."""
    return envs_mdp.bad_orientation(env, limit_angle=limit_angle) & ~_state(env)["platform_transition"]


def stepping_joint_pos_out_of_limits(env, limit_margin: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate on joint limits except the reverse-bending side of the knees."""
    robot = env.scene[asset_cfg.name].data
    limits = robot.joint_pos_limits
    assert limits is not None
    below_lower = robot.joint_pos <= limits[..., 0] + limit_margin
    above_upper = robot.joint_pos >= limits[..., 1] - limit_margin
    above_upper[:, _state(env)["knee_ids"]] = False
    return (below_lower | above_upper)[:, asset_cfg.joint_ids].any(dim=1)


def _standstill_time(env) -> torch.Tensor:
    state = _state(env)
    fresh = state["stall_tick"] != env.common_step_counter
    x = env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    command = env.command_manager.get_command("twist")[:, 0]
    active = (
        (command > 0.0) & torch.isfinite(x) & torch.isfinite(command)
        & ~state["failure"] & ~state["success"] & (state["stage"] == 2)
    )
    advancing = active & ((x - state["stall_anchor_x"]) >= STANDSTILL_PROGRESS)
    state["stall_anchor_x"] = torch.where(fresh & advancing, x, state["stall_anchor_x"])
    age = torch.where(active & ~advancing, state["stall_age"] + env.step_dt, 0.0)
    state["stall_age"] = torch.where(fresh, age, state["stall_age"])
    state["stall_tick"].fill_(env.common_step_counter)
    return state["stall_age"]


def stepping_stalled(env, timeout_s: float) -> torch.Tensor:
    """End a traversal that has made no new forward progress for too long."""
    return _standstill_time(env) >= timeout_s - 1.0e-5


def stepping_succeeded(env) -> torch.Tensor:
    state = _state(env)
    physical_failure = state["failure"].clone()
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        params = env.cfg.terminations["bad_orientation"].params
        physical_failure |= stepping_bad_orientation(env, **params)
    else:
        # A valid finish at the deadline keeps its success result.
        for name in manager.active_terms:
            if name not in ("stepping_succeeded", "stepping_timeout") and not manager.get_term_cfg(name).time_out:
                physical_failure |= manager.get_term(name)
    state["failure"] |= state["success"] & physical_failure
    state["success"] &= ~physical_failure
    return state["success"]


def stepping_settle_timeout(env) -> torch.Tensor:
    state = _state(env)
    return (state["exit_settle_substeps"] >= math.ceil(EXIT_SETTLE_TIMEOUT_S / env.physics_dt)) & ~state["success"]


def stepping_timeout(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def stepping_failure_reward(env) -> torch.Tensor:
    state = _state(env)
    terminated = getattr(env, "reset_terminated", state["failure"])
    return (terminated & ~state["success"]).float() / env.step_dt


def stepping_early_failure_penalty(env, grace_time_s: float) -> torch.Tensor:
    """Charge the unused startup time when a traversal fails early."""
    state = _state(env)
    failed = (state["failure"] | getattr(env, "reset_terminated", state["failure"])) & ~state["success"]
    horizon = min(grace_time_s, env.max_episode_length * env.step_dt)
    remaining = (horizon - env.episode_length_buf * env.step_dt).clamp_min(0.0)
    return torch.where(failed, remaining, 0.0) / env.step_dt


def stepping_backward_velocity_penalty(env, tolerance: float, scale: float) -> torch.Tensor:
    """Penalize body motion opposite the course's positive X direction."""
    state = _state(env)
    velocity = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
    cost = ((-velocity - tolerance).clamp_min(0.0) / scale).square().clamp_max(1.0)
    active = (state["stage"] == 2) & ~state["failure"] & ~state["success"] & torch.isfinite(velocity)
    return torch.where(active, cost, 0.0)


def _positive_event_reward(env, key: str) -> torch.Tensor:
    state = _state(env)
    failed = state["failure"] | (getattr(env, "reset_terminated", state["failure"]) & ~state["success"])
    return torch.where(failed, 0.0, state[key]) / env.step_dt


def cover_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "cover_pulse")


def stone_center_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "stone_center_pulse")


def stone_edge_penalty(env) -> torch.Tensor:
    return _state(env)["stone_edge_pulse"] / env.step_dt


def foot_forward_step_reward(env) -> torch.Tensor:
    """Score each foot independently when its own forward landing is confirmed."""
    return _positive_event_reward(env, "foot_step_pulse")


def right_rear_forward_step_reward(env) -> torch.Tensor:
    """Reward the right rear reaching a new forward support surface."""
    return _positive_event_reward(env, "right_rear_step_pulse")


def right_rear_overdue_penalty(env) -> torch.Tensor:
    """Charge time spent waiting beyond the right-rear follow grace period."""
    return (_state(env)["right_rear_overdue_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_no_forward_step_penalty(env) -> torch.Tensor:
    """Charge prolonged lack of foot progress and extra body motion on planted feet."""
    return (_state(env)["no_forward_step_cost_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_left_front_support_penalty(env) -> torch.Tensor:
    """Charge time without left-front support until the right rear lands securely."""
    return (_state(env)["left_front_unsupported_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_left_front_liftoff_penalty(env) -> torch.Tensor:
    """Charge each confirmed left-front release during the right-rear step."""
    return _state(env)["left_front_liftoff_pulse"] / env.step_dt


def stepping_left_front_overreach_penalty(env) -> torch.Tensor:
    """Charge left-front overreach, misaligned descent and near-straight swings."""
    return (_state(env)["left_front_overreach_cost_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_premature_support_handoff_penalty(env) -> torch.Tensor:
    """Charge a simultaneous release before the left front has stable support."""
    return _state(env)["premature_support_handoff_pulse"] / env.step_dt


def stepping_low_support_penalty(env) -> torch.Tensor:
    """Charge time with fewer than two feet bearing load."""
    return (_state(env)["low_support_time"] / env.step_dt).clamp(0.0, 1.0)


def startup_progress_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "startup_progress_pulse")


def startup_early_step_penalty(env) -> torch.Tensor:
    return _state(env)["startup_early_pulse"] / env.step_dt


def startup_body_rise_penalty(env) -> torch.Tensor:
    """Charge excessive rise or upward velocity during the startup window."""
    return (_state(env)["startup_rise_cost_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_foot_side_penalty(env) -> torch.Tensor:
    """Penalize loaded foot contacts on raised terrain faces."""
    return (_state(env)["foot_side_cost_time"] / env.step_dt).clamp(0.0, 1.0)


def success_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "success_pulse")


def exit_stability_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "exit_stability_pulse")


def _forward_speed_ratio(env, command_name: str) -> torch.Tensor:
    state = _state(env)
    command = env.command_manager.get_command(command_name)[:, 0]
    speed = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
    terminated = getattr(env, "reset_terminated", state["failure"] | state["success"])
    active = (
        (state["stage"] == 2) & (command > 0.0)
        & torch.isfinite(speed) & torch.isfinite(command)
        & ~state["failure"] & ~state["success"] & ~terminated
    )
    return torch.where(active, speed / command.clamp_min(1.0e-6), 0.0)


def forward_velocity_reward(env, command_name: str = "twist") -> torch.Tensor:
    """Score commanded forward speed only after the current gait phase begins a legal swing."""
    speed_ratio = _forward_speed_ratio(env, command_name).clamp(-1.0, 2.0)
    reward = 2.0 * speed_ratio - speed_ratio.square()
    state = _state(env)
    blocked = state["shin_support_this_step"]
    gait = state.get("fixed_gait")
    if gait is not None:
        blocked = blocked | gait.left_front_rear_transfer_lost_support | gait.left_front_landing_wait_this_step
        legal_swing_started = gait.lifted.any(dim=1)
        reward = reward.masked_fill(~legal_swing_started & (reward > 0), 0.0)
    return reward.masked_fill(blocked & (reward > 0), 0.0)


def stepping_body_stall_penalty(env, grace_s: float, ramp_s: float) -> torch.Tensor:
    """Charge lack of new forward body progress while traversal is commanded."""
    state = _state(env)
    age = _standstill_time(env)
    command = env.command_manager.get_command("twist")[:, 0]
    terminated = getattr(env, "reset_terminated", state["failure"] | state["success"])
    active = ((state["stage"] == 2) & (command > 0.0) & torch.isfinite(command)
              & ~state["failure"] & ~state["success"] & ~terminated)
    cost = ((age - grace_s) / ramp_s).clamp(0.0, 1.0)
    return torch.where(active, cost, 0.0)


def stepping_body_sink_penalty(env, tolerance: float, scale: float) -> torch.Tensor:
    """Penalize the greater drop of the base or front hips during traversal."""
    state = _state(env)
    robot = env.scene["robot"].data
    root_drop = state["startup_spawn_z"] - robot.root_link_pos_w[:, 2]
    front_drop = root_drop + STARTUP_FRONT_Z - _front_hip_offset_w(robot)[:, 2]
    drop = torch.maximum(root_drop, front_drop)
    terminated = getattr(env, "reset_terminated", state["failure"] | state["success"])
    active = (state["stage"] == 2) & ~state["failure"] & ~state["success"] & ~terminated
    cost = ((drop - tolerance) / scale).clamp(0.0, 1.0).square()
    return torch.where(active & torch.isfinite(drop), cost, 0.0)


def stepping_lateral_velocity_penalty(env, tolerance: float, scale: float) -> torch.Tensor:
    """Penalize body-frame lateral speed outside the allowed deadband."""
    lateral_speed = env.scene["robot"].data.root_link_lin_vel_b[:, 1]
    return deadband_cost(lateral_speed, tolerance, scale)


def stepping_foot_row_alignment_penalty(env, tolerance: float, scale: float) -> torch.Tensor:
    """Penalize the largest cross-track foot deviation from its own stone row."""
    state = _state(env)
    foot_y = _foot_bottom(env, state)[..., 1] - env.scene.env_origins[:, 1:2]
    row_y = state["stone_centers"][:, :2, 1].repeat(1, 2)
    error = foot_y - row_y
    cost = torch.where(torch.isfinite(error), deadband_cost(error, tolerance, scale), 0.0).amax(dim=1)
    terminated = getattr(env, "reset_terminated", state["failure"] | state["success"])
    active = (state["stage"] == 2) & ~state["failure"] & ~state["success"] & ~terminated
    return torch.where(active, cost, 0.0)


def stepping_hip_abduction_symmetry_reward(
    env, angle_scale: float, command_name: str = "twist"
) -> torch.Tensor:
    """Reward mirrored hip abduction while the body advances."""
    state = _state(env)
    angles = env.scene["robot"].data.joint_pos[:, state["hip_abduction_ids"]]
    errors = torch.stack((angles[:, 0] + angles[:, 1], angles[:, 2] + angles[:, 3]), dim=1)
    symmetry = torch.exp(-(errors / angle_scale).square().mean(dim=1))
    forward_fraction = _forward_speed_ratio(env, command_name).clamp(0.0, 1.0)
    return torch.where(torch.isfinite(symmetry), symmetry * forward_fraction, 0.0)


def stepping_leg_clearance_penalty(env, clearance: float) -> torch.Tensor:
    """Penalize each foot approaching another foot or another leg's shin."""
    state = _state(env)
    robot = env.scene["robot"].data
    feet = _foot_bottom(env, state) + robot.site_pos_w.new_tensor((0.0, 0.0, FOOT_RADIUS))
    knees = robot.body_link_pos_w[:, state["knee_body_ids"]]

    def point_segment(point, start, end):
        segment = end - start
        fraction = ((point - start) * segment).sum(dim=-1) / segment.square().sum(dim=-1).clamp_min(1.e-8)
        closest = start + fraction.clamp(0.0, 1.0)[..., None] * segment
        return torch.linalg.vector_norm(point - closest, dim=-1)

    foot_distance = torch.linalg.vector_norm(feet[:, :, None, :] - feet[:, None, :, :], dim=-1)
    shin_distance = point_segment(feet[:, :, None, :], knees[:, None, :, :], feet[:, None, :, :])
    distance = torch.minimum(foot_distance, shin_distance)
    distance = distance.masked_fill(torch.eye(4, dtype=torch.bool, device=env.device)[None], torch.inf)
    cost = ((clearance - distance).clamp_min(0.0) / clearance).square().amax(dim=(1, 2))
    return torch.where(torch.isfinite(cost), cost, 0.0)


def stepping_knee_min_angle_penalty(env, minimum_degrees: float, scale_degrees: float) -> torch.Tensor:
    """Penalize insufficient knee flexion in the model's normal direction."""
    angles = env.scene["robot"].data.joint_pos[:, _state(env)["knee_ids"]]
    flexion = -angles
    scale = math.radians(scale_degrees)
    deficit = ((math.radians(minimum_degrees) - flexion) / scale).clamp(0.0, 1.0)
    cost = deficit.square().mean(dim=1)
    return torch.where(torch.isfinite(cost), cost, 0.0)


def stepping_knee_reverse_penalty(env) -> torch.Tensor:
    """Charge the worst reverse-bending knee per physics substep."""
    return (_state(env)["knee_reverse_cost_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_hip_abduction_target_penalty(env, target: float, tolerance: float, scale: float) -> torch.Tensor:
    """Apply a soft mirrored hip-abduction target with a deadband."""
    angles = env.scene["robot"].data.joint_pos[:, _state(env)["hip_abduction_ids"]]
    desired = angles.new_tensor((target, -target, target, -target))
    cost = deadband_cost(angles - desired, tolerance, scale).mean(dim=1)
    return torch.where(torch.isfinite(cost), cost, 0.0)


def route_alignment_reward(
    env, heading_scale_degrees: float, lateral_speed_scale: float, command_name: str = "twist"
) -> torch.Tensor:
    """Reward facing and moving along the positive world X axis."""
    robot = env.scene["robot"].data
    lateral_speed = robot.root_link_lin_vel_w[:, 1]
    heading = robot.heading_w
    heading_error = torch.atan2(torch.sin(heading), torch.cos(heading))
    forward_fraction = _forward_speed_ratio(env, command_name).clamp(0.0, 1.0)
    heading_cost = (heading_error / math.radians(heading_scale_degrees)).square()
    lateral_cost = (lateral_speed / lateral_speed_scale).square()
    aligned = forward_fraction * torch.exp(-heading_cost - lateral_cost)
    return torch.where(torch.isfinite(heading_error) & torch.isfinite(lateral_speed), aligned, 0.0)


def stepping_body_level_reward(env, std_degrees: float, command_name: str = "twist") -> torch.Tensor:
    """Reward a level body in proportion to forward speed."""
    gravity = env.scene["robot"].data.projected_gravity_b
    horizontal = torch.linalg.vector_norm(gravity[:, :2], dim=1)
    tilt = torch.atan2(horizontal, -gravity[:, 2])
    forward_fraction = _forward_speed_ratio(env, command_name).clamp(0.0, 1.0)
    level = torch.exp(-(tilt / math.radians(std_degrees)).square())
    return torch.where(torch.isfinite(tilt), forward_fraction * level, 0.0)


def stepping_body_tilt_penalty(env, tolerance_degrees: float, scale_degrees: float) -> torch.Tensor:
    """Penalize body tilt beyond the allowed tolerance."""
    gravity = env.scene["robot"].data.projected_gravity_b
    tilt = torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=1), -gravity[:, 2])
    cost = ((tilt - math.radians(tolerance_degrees)).clamp_min(0.0) / math.radians(scale_degrees)).square().clamp_max(1.0)
    return torch.where(torch.isfinite(tilt), cost, 0.0)


def stepping_body_height_reward(
    env, target: float, tolerance: float, scale: float, command_name: str = "twist"
) -> torch.Tensor:
    """Reward maintaining body height relative to level support tops while advancing."""
    height = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - PLATFORM_HEIGHT
    error = (height - target).abs().sub(tolerance).clamp_min(0.0)
    score = torch.exp(-(error / scale).square())
    forward_fraction = _forward_speed_ratio(env, command_name).clamp(0.0, 1.0)
    return torch.where(torch.isfinite(height), score * forward_fraction, 0.0)


def stepping_swing_height_penalty(env, target_height: float, scale: float) -> torch.Tensor:
    """Penalize swing feet rising above the target relative to the support top."""
    state = _state(env)
    height = _foot_bottom(env, state)[..., 2] - env.scene.env_origins[:, 2:3] - PLATFORM_HEIGHT
    sensor = env.scene["stepping_foot_contact"].data
    airborne = ~(sensor.found[:, state["contact_ids"]] > 0)
    active = (state["stage"] == 2) & ~state["failure"] & ~state["success"]
    excess = ((height - target_height).clamp_min(0.0) / scale).square().clamp_max(1.0)
    return torch.where(active[:, None] & airborne & torch.isfinite(height), excess, 0.0).sum(dim=1)


def stepping_shin_stone_penalty(env) -> torch.Tensor:
    """Return the time-averaged force cost for shin contacts with raised surfaces."""
    return (_state(env)["shin_stone_cost_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_shin_support_penalty(env) -> torch.Tensor:
    return _state(env)["shin_support_pulse"] / env.step_dt


def stepping_flight_penalty(env) -> torch.Tensor:
    """Return the fraction of physical substeps without foot contact."""
    return (_state(env)["flight_step_time"] / env.step_dt).clamp(0.0, 1.0)


def stepping_foot_slip_penalty(env, speed_tolerance: float, speed_scale: float) -> torch.Tensor:
    """Penalize horizontal movement of feet supporting on a legal top surface."""
    state = _state(env)
    robot = env.scene["robot"].data
    speed = torch.linalg.vector_norm(robot.site_lin_vel_w[:, state["foot_site_ids"], :2], dim=2)
    active = (state["stage"] == 2) & ~state["failure"] & ~state["success"]
    supported = active[:, None] & state["legal_support"] & torch.isfinite(speed)
    cost = ((speed - speed_tolerance).clamp_min(0.0) / speed_scale).square().clamp_max(1.0)
    return torch.where(supported, cost, 0.0).sum(dim=1)


def stone_relative_positions(env) -> torch.Tensor:
    state = _state(env)
    centers = state["stone_centers"]
    points = centers + env.scene.env_origins[:, None, :]
    root = env.scene["robot"].data.root_link_pos_w
    delta = points - root[:, None, :]
    heading = env.scene["robot"].data.heading_w[:, None]
    local_x = torch.cos(heading) * delta[:, :, 0] + torch.sin(heading) * delta[:, :, 1]
    local_y = -torch.sin(heading) * delta[:, :, 0] + torch.cos(heading) * delta[:, :, 1]
    return torch.stack((local_x, local_y, delta[:, :, 2]), dim=-1).reshape(env.num_envs, -1)


def active_stone_mask(env) -> torch.Tensor:
    return _active_mask(_state(env)).float()


def visited_stone_mask(env) -> torch.Tensor:
    return _state(env)["visited"].float()


def platform_relative_positions(env) -> torch.Tensor:
    state = _state(env)
    count = state["active_count"].float()
    left = torch.tensor((PLATFORM_SIZE / 2, 0.0, PLATFORM_HEIGHT), device=env.device).repeat(env.num_envs, 1)
    right = torch.stack((state["right_platform_start"] + PLATFORM_SIZE / 2, torch.zeros_like(count), torch.full_like(count, PLATFORM_HEIGHT)), dim=1)
    points = torch.stack((left, right), dim=1) + env.scene.env_origins[:, None, :]
    root = env.scene["robot"].data.root_link_pos_w
    delta = points - root[:, None, :]
    heading = env.scene["robot"].data.heading_w[:, None]
    local_x = torch.cos(heading) * delta[:, :, 0] + torch.sin(heading) * delta[:, :, 1]
    local_y = -torch.sin(heading) * delta[:, :, 0] + torch.cos(heading) * delta[:, :, 1]
    return torch.stack((local_x, local_y, delta[:, :, 2]), dim=-1).reshape(env.num_envs, -1)


def stage_observation(env) -> torch.Tensor:
    return torch.nn.functional.one_hot(_state(env)["stage"], num_classes=4).float()


def fixed_gait_observation(env) -> torch.Tensor:
    state = _state(env)
    robot = env.scene["robot"].data
    root = robot.root_link_pos_w - env.scene.env_origins
    gait = state["fixed_gait"]
    active = (state["stage"] == 2) & ~state["failure"] & ~state["success"]
    gait.initialize(_foot_bottom(env, state) - env.scene.env_origins[:, None, :], state["active_count"],
                    state["stone_centers"], state["right_platform_start"], active)
    return gait.observation(state["active_count"], root, robot.heading_w)


def fixed_gait_reward(env, key: str, event: bool = False) -> torch.Tensor:
    state = _state(env)
    gait = state["fixed_gait"]
    value = gait.pulses[key] if event else gait.costs[key]
    failed = state["failure"] | getattr(env, "reset_terminated", state["failure"])
    if key in ("progress", "complete"):
        value = torch.where(failed, 0.0, value)
    return value / env.step_dt


def strict_ground_observation(env) -> torch.Tensor:
    return torch.ones((env.num_envs, 1), device=env.device)


def foot_contact_observation(env, support_progress: bool = False) -> torch.Tensor:
    state = _state(env)
    if support_progress:
        return state["left_front_handoff"].support_progress(env.physics_dt)
    sensor = env.scene["stepping_foot_contact"].data
    return (sensor.found[:, state["contact_ids"]] > 0).float()


def settle_progress_observation(env) -> torch.Tensor:
    state = _state(env)
    arrival = state["exit_visited"].float().mean(dim=1)
    hold = state["exit_stable_substeps"].float() / math.ceil(EXIT_SETTLE_TIME_S / env.physics_dt)
    return torch.where(state["stage"] == 3, hold, arrival).unsqueeze(1)


def route_direction_observation(env) -> torch.Tensor:
    heading = env.scene["robot"].data.heading_w
    return torch.stack((torch.cos(heading), -torch.sin(heading)), dim=1)


def navigation_valid_observation(env) -> torch.Tensor:
    return torch.ones(env.num_envs, 1, device=env.device)


def remaining_time_observation(env) -> torch.Tensor:
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(0.0, 1.0).unsqueeze(1)


def stepping_metric(env, key: str) -> torch.Tensor:
    state = _state(env)
    if key.startswith("gait_"):
        gait = state["fixed_gait"]
        moving = gait.destinations(state["active_count"]) != HOLD
        return {"gait_phase": gait.phase.float(), "gait_completed_steps": gait.completed,
                "gait_violations": gait.violations, "gait_phase_age_s": gait.age,
                "gait_left_front_rear_transfer_violations": gait.left_front_rear_transfer_violations,
                "gait_ready": gait.ready.float(),
                "gait_lifted_fraction": (gait.lifted & moving).sum(dim=1) / moving.sum(dim=1).clamp_min(1),
                "gait_pair_ready": (gait.pair_lifted | (moving.sum(dim=1) == 1)).float()}[key]
    if key == "success":
        return state["success"].float()
    if key == "failure":
        terminated = getattr(env, "reset_terminated", state["failure"])
        return ((state["failure"] | terminated) & ~state["success"]).float()
    if key == "covered_fraction":
        return (state["visited"] & _active_mask(state)).sum(dim=1).float() / state["active_count"].float()
    if key == "no_forward_step_time_s":
        return state["no_forward_step_time_s"]
    if key == "right_rear_forward_steps":
        return state["right_rear_forward_steps"]
    if key == "right_rear_follow_wait_s":
        return state["right_rear_follow_wait_s"]
    if key == "stone_count":
        return state["active_count"].float()
    if key == "stone_gap_m":
        centers = state["stone_centers"]
        return centers[:, 1, 0] - centers[:, 0, 0] - 2.0 * state["stone_half_size"]
    if key == "exit_reached":
        return (state["stage"] == 3).float()
    if key == "exit_best_stable_time_s":
        return state["exit_best_stable_substeps"].float() * env.physics_dt
    if key == "forward_speed_mps":
        speed = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
        return torch.where(torch.isfinite(speed), speed, 0.0)
    if key == "peak_foot_clearance_m":
        return state["peak_foot_clearance_m"]
    if key == "startup_peak_body_rise_m":
        return state["startup_peak_body_rise_m"]
    if key == "startup_peak_front_rise_m":
        return state["startup_peak_front_rise_m"]
    if key == "traverse_peak_body_sink_m":
        return state["traverse_peak_body_sink_m"]
    if key == "premature_support_handoffs":
        return state["premature_support_handoffs"]
    if key == "left_front_overlap_count":
        return state["left_front_overlap_count"]
    raise ValueError(f"Unknown stepping-stone metric '{key}'.")


class SteppingStoneCurriculum:
    """Promote terrain levels based on complete passes."""

    def __init__(self, cfg: CurriculumTermCfg, env):
        params = cfg.params
        self.window_size = int(params.get("window_size", 100))
        self.success_threshold = float(params.get("success_threshold", 0.80))
        self.demote_threshold = float(params.get("demote_threshold", 0.40))
        self.promote_at_current_cap = bool(params.get("promote_at_current_cap", False))
        if self.window_size <= 0 or not 0.0 <= self.demote_threshold < self.success_threshold <= 1.0:
            raise ValueError("Invalid stepping-stone curriculum thresholds.")
        self.level_cap = int(params.get("initial_level_cap", 0))
        self.window: deque[bool] = deque(maxlen=self.window_size)
        self.maximum = env.cfg.scene.terrain.terrain_generator.num_rows - 1
        env._stepping_stone_curriculum = self

    def state_dict(self):
        return {"level_cap": self.level_cap, "window": list(self.window), "window_size": self.window_size}

    def load_state_dict(self, value):
        cap, window = value["level_cap"], value["window"]
        if (not isinstance(cap, int) or not 0 <= cap <= self.maximum or not isinstance(window, list)
                or not all(isinstance(item, bool) for item in window)):
            raise ValueError("Invalid saved stepping-stone curriculum")
        self.level_cap = cap
        self.window.clear()
        self.window.extend(window)

    def __call__(self, env, env_ids, **kwargs):
        del kwargs
        ids = _ids(env, env_ids)
        state = _state(env)
        terrain = env.scene.terrain
        if terrain is None:
            raise RuntimeError("Stepping-stone curriculum requires generated terrain.")
        completed = env.episode_length_buf[ids] > 0
        success = state["success"][ids] & completed
        current = terrain.terrain_levels[ids]
        eligible = completed & (current == self.level_cap) if self.promote_at_current_cap else completed
        for value in success[eligible].detach().cpu().tolist():
            self.window.append(bool(value))
        rate = sum(self.window) / len(self.window) if self.window else 0.0
        maximum = terrain.cfg.terrain_generator.num_rows - 1
        if len(self.window) == self.window_size and rate >= self.success_threshold and self.level_cap < maximum:
            self.level_cap += 1
            self.window.clear()
        move_up = success
        failed = completed & ~success
        move_down = failed & (rate < self.demote_threshold)
        next_level = (current + move_up.long() - move_down.long()).clamp(min=0, max=self.level_cap)
        terrain.terrain_levels[ids] = next_level
        terrain.env_origins[ids] = terrain.get_terrain_origins(next_level, terrain.terrain_types[ids])
        return {
            "level_mean": terrain.terrain_levels.float().mean(),
            "level_cap": torch.tensor(float(self.level_cap), device=env.device),
            "success_rate": torch.tensor(rate, device=env.device),
            "window_episodes": torch.tensor(float(len(self.window)), device=env.device),
        }
