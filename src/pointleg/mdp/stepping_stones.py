"""State, commands, observations and rewards for the stepping-stone task."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from .commands import UniformThresholdVelocityCommand
from ..robot_cfg import FOOT_BODY_NAMES, FOOT_SITE_NAMES
from ..terrains.stepping_stones import (
    MAX_STONES,
    PLATFORM_HEIGHT,
    PLATFORM_SIZE,
    STONE_SIZE,
    right_platform_start,
    stone_centers,
)


GROUND_CONTACT_Z_TOL = 0.025
TOP_CONTACT_Z_TOL = 0.030
TOP_EDGE_TOL = 0.003
REQUIRED_CLEARANCE = PLATFORM_HEIGHT + 0.04
BRAKE_TIME_S = 0.50
STABLE_TIME_S = 0.50
MAX_AIRBORNE_TIME_S = 0.020
FOOT_RADIUS = 0.025
MIN_SWING_TIME_S = 0.020
LANDING_MARGIN = 0.020
GOAL_REACHED_DISTANCE = 0.050


def _ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


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
    contact_ids = []
    for body_name in FOOT_BODY_NAMES:
        ids = [index for index, name in enumerate(contact_sensor.primary_names) if name.endswith(body_name)]
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one contact channel for '{body_name}'.")
        contact_ids.append(ids[0])

    relative = robot.data.site_pos_w[:, foot_site_ids] - robot.data.root_link_pos_w[:, None, :]
    heading = robot.data.heading_w[:, None]
    forward = torch.cos(heading) * relative[:, :, 0] + torch.sin(heading) * relative[:, :, 1]
    front_foot_offset = forward.max(dim=1).values.median().clamp(0.05, 0.50)
    if not torch.isfinite(front_foot_offset):
        front_foot_offset = torch.tensor(0.20, device=device)

    env._stepping_stones = {
        "active_count": torch.full((env.num_envs,), 2, dtype=torch.long, device=device),
        "visited": torch.zeros(env.num_envs, MAX_STONES, dtype=torch.bool, device=device),
        "strict_ground_check": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "failure": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "success": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "stage": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "episode_speed": torch.full((env.num_envs,), 0.25, device=device),
        "spawn_x": zeros(),
        "front_foot_offset": front_foot_offset,
        "airborne_time": zeros(),
        "brake_time": zeros(),
        "stable_time": zeros(),
        "previous_progress": zeros(),
        "progress_ready": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "was_airborne": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device),
        "swing_peak_z": torch.zeros(env.num_envs, 4, device=device),
        "previous_support": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device),
        "swing_valid": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device),
        "swing_time": torch.zeros(env.num_envs, 4, device=device),
        "swing_landing_count": zeros(),
        "low_clearance_count": zeros(),
        "goal_id": torch.full((env.num_envs,), -2, dtype=torch.long, device=device),
        "goal_best_distance": torch.full((env.num_envs,), torch.inf, device=device),
        "goal_reward_tick": torch.full((env.num_envs,), -1, dtype=torch.long, device=device),
        "goal_reward_value": zeros(),
        "mount_lift_awarded": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=device),
        "mount_pulse": zeros(),
        "cover_pulse": zeros(),
        "landing_pulse": zeros(),
        "lift_pulse": zeros(),
        "success_pulse": zeros(),
        "last_control_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=device),
        "foot_site_ids": foot_site_ids,
        "contact_ids": contact_ids,
        "last_support_count": zeros(),
    }
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


def _top_contact_masks(env, state, found: torch.Tensor, position_w: torch.Tensor):
    local = _relative_xy(env, position_w)
    count = state["active_count"]
    stone_template = torch.as_tensor(stone_centers(MAX_STONES), dtype=local.dtype, device=local.device)
    stone_delta = local[:, :, None, :] - stone_template[None, None, :, :]
    stone_top = (
        found[:, :, None]
        & (stone_delta[..., 0].abs() <= STONE_SIZE / 2 + TOP_EDGE_TOL)
        & (stone_delta[..., 1].abs() <= STONE_SIZE / 2 + TOP_EDGE_TOL)
        & ((local[:, :, None, 2] - PLATFORM_HEIGHT).abs() <= TOP_CONTACT_Z_TOL)
        & _active_mask(state)[:, None, :]
    )
    left_top = (
        found
        & (local[:, :, 0] >= -TOP_EDGE_TOL)
        & (local[:, :, 0] <= PLATFORM_SIZE + TOP_EDGE_TOL)
        & (local[:, :, 1].abs() <= PLATFORM_SIZE / 2 + TOP_EDGE_TOL)
        & ((local[:, :, 2] - PLATFORM_HEIGHT).abs() <= TOP_CONTACT_Z_TOL)
    )
    right_start = 0.90 + 0.30 * count.float()
    right_top = (
        found
        & (local[:, :, 0] >= right_start[:, None] - TOP_EDGE_TOL)
        & (local[:, :, 0] <= right_start[:, None] + PLATFORM_SIZE + TOP_EDGE_TOL)
        & (local[:, :, 1].abs() <= PLATFORM_SIZE / 2 + TOP_EDGE_TOL)
        & ((local[:, :, 2] - PLATFORM_HEIGHT).abs() <= TOP_CONTACT_Z_TOL)
    )
    ground = found & (local[:, :, 2] <= GROUND_CONTACT_Z_TOL)
    return local, stone_top, left_top, right_top, ground


def reset_stepping_stones(env, env_ids) -> None:
    """Reset the task state and place each robot before its left platform."""
    ids = _ids(env, env_ids)
    state = _state(env)
    terrain = env.scene.terrain
    if terrain is None:
        raise RuntimeError("Stepping stones requires generated terrain.")
    count = terrain.terrain_levels[ids].to(torch.long) + 2
    count = count.clamp(min=2, max=MAX_STONES)
    state["active_count"][ids] = count
    state["visited"][ids] = False
    state["strict_ground_check"][ids] = False
    state["failure"][ids] = False
    state["success"][ids] = False
    state["stage"][ids] = 0
    state["airborne_time"][ids] = 0.0
    state["brake_time"][ids] = 0.0
    state["stable_time"][ids] = 0.0
    state["previous_progress"][ids] = 0.0
    state["progress_ready"][ids] = False
    state["was_airborne"][ids] = False
    state["swing_peak_z"][ids] = 0.0
    state["previous_support"][ids] = False
    state["swing_valid"][ids] = False
    state["swing_time"][ids] = 0.0
    state["swing_landing_count"][ids] = 0.0
    state["low_clearance_count"][ids] = 0.0
    state["goal_id"][ids] = -2
    state["goal_best_distance"][ids] = torch.inf
    state["goal_reward_tick"][ids] = -1
    state["goal_reward_value"][ids] = 0.0
    state["mount_lift_awarded"][ids] = False
    state["last_control_step"][ids] = -1
    for key in ("mount_pulse", "cover_pulse", "landing_pulse", "lift_pulse", "success_pulse", "last_support_count"):
        state[key][ids] = 0.0

    distance = torch.empty(len(ids), device=env.device).uniform_(0.15, 0.35)
    state["episode_speed"][ids] = torch.empty(len(ids), device=env.device).uniform_(0.15, 0.35)
    local_x = -distance - state["front_foot_offset"]
    state["spawn_x"][ids] = local_x
    robot = env.scene["robot"]
    root_state = robot.data.default_root_state[ids].clone()
    root_state[:, 0] = env.scene.env_origins[ids, 0] + local_x
    root_state[:, 1] = env.scene.env_origins[ids, 1]
    root_state[:, 2] = env.scene.env_origins[ids, 2] + root_state[:, 2]
    root_state[:, 3:7] = root_state.new_tensor((1.0, 0.0, 0.0, 0.0))
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=ids)


def collect_stepping_stones_substep(env) -> torch.Tensor:
    """Latch physical-step contacts so brief ground contacts cannot be missed."""
    state = _state(env)
    token = torch.full((env.num_envs,), env.common_step_counter, dtype=torch.long, device=env.device)
    new_control_step = state["last_control_step"] != token
    if new_control_step.any():
        for key in ("mount_pulse", "cover_pulse", "landing_pulse", "lift_pulse", "success_pulse"):
            state[key][new_control_step] = 0.0
        state["last_control_step"][new_control_step] = token[new_control_step]

    sensor = env.scene["stepping_foot_contact"].data
    if sensor.found is None or sensor.pos is None:
        raise RuntimeError("Stepping-stone contact sensor requires found and pos fields.")
    found = sensor.found[:, state["contact_ids"]] > 0
    position_w = sensor.pos[:, state["contact_ids"]]
    local_contact, stone_top, left_top, right_top, ground = _top_contact_masks(env, state, found, position_w)
    robot = env.scene["robot"].data
    foot_w = _foot_bottom(env, state)
    foot_local = _relative_xy(env, foot_w)
    mounted_now = left_top.all(dim=1)
    strict_now = state["strict_ground_check"] | mounted_now
    mount_event = mounted_now & ~state["strict_ground_check"]
    state["strict_ground_check"] |= mounted_now
    state["mount_pulse"] += mount_event.float()

    support_count = found.sum(dim=1)
    state["last_support_count"] = support_count.float()
    physics_dt = getattr(env, "physics_dt", env.step_dt)
    state["airborne_time"] = torch.where(
        support_count == 0,
        state["airborne_time"] + physics_dt,
        torch.zeros_like(state["airborne_time"]),
    )
    route_x = robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    route_invalid = ~strict_now & (route_x > 0.90)
    failed_now = (strict_now & ground.any(dim=1)) | (state["airborne_time"] > MAX_AIRBORNE_TIME_S) | route_invalid
    state["failure"] |= failed_now

    was_airborne = state["was_airborne"]
    peak_before_landing = state["swing_peak_z"]
    airborne = ~found
    legal_support = stone_top.any(dim=2) | left_top | right_top | (ground & ~strict_now[:, None])
    supported = legal_support.sum(dim=1) >= 2
    valid_landing = found & was_airborne
    valid_swing_landing = (
        valid_landing & state["swing_valid"] & (state["swing_time"] >= MIN_SWING_TIME_S)
        & legal_support & supported[:, None] & ~state["failure"][:, None]
    )
    lift_quality = ((peak_before_landing - PLATFORM_HEIGHT) / (REQUIRED_CLEARANCE - PLATFORM_HEIGHT)).clamp(0.0, 1.0)
    first_mount_contact = left_top & ~state["mount_lift_awarded"]
    first_mount_lift = first_mount_contact & valid_swing_landing
    state["mount_lift_awarded"] |= first_mount_contact
    state["lift_pulse"] += (0.5 * lift_quality * first_mount_lift.float()).sum(dim=1)

    new_stones = stone_top.any(dim=1) & ~state["visited"] & strict_now[:, None] & ~state["failure"][:, None]
    margin = torch.minimum(
        STONE_SIZE / 2 - (local_contact[:, :, None, 0] - torch.as_tensor(stone_centers(MAX_STONES), device=env.device, dtype=local_contact.dtype)[None, None, :, 0]).abs(),
        STONE_SIZE / 2 - (local_contact[:, :, None, 1] - torch.as_tensor(stone_centers(MAX_STONES), device=env.device, dtype=local_contact.dtype)[None, None, :, 1]).abs(),
    )
    landing_quality = (margin / LANDING_MARGIN).clamp(0.0, 1.0)
    landing_candidates = stone_top & valid_landing[:, :, None]
    best_landing = torch.where(landing_candidates, landing_quality, 0.0).max(dim=1).values
    lift_candidates = stone_top & valid_swing_landing[:, :, None]
    best_lift = torch.where(lift_candidates, lift_quality[:, :, None], 0.0).max(dim=1).values
    state["visited"] |= new_stones
    count = state["active_count"].float()
    state["cover_pulse"] += new_stones.sum(dim=1).float() / count
    state["landing_pulse"] += (best_landing * new_stones.float()).sum(dim=1) / count
    state["lift_pulse"] += (best_lift * new_stones.float()).sum(dim=1) * (2.0 / count)

    measured_landing = valid_swing_landing & (left_top | stone_top.any(dim=2) | right_top)
    state["swing_landing_count"] += measured_landing.sum(dim=1)
    state["low_clearance_count"] += (measured_landing & (peak_before_landing < REQUIRED_CLEARANCE - 1.0e-6)).sum(dim=1)
    starting_swing = airborne & state["previous_support"]
    state["swing_valid"] = (
        airborne & (starting_swing | state["swing_valid"])
        & supported[:, None] & ~state["failure"][:, None]
    )
    state["swing_time"] = torch.where(
        state["swing_valid"], state["swing_time"] + physics_dt, 0.0
    )
    state["swing_peak_z"] = torch.where(
        state["swing_valid"], torch.maximum(state["swing_peak_z"], foot_local[:, :, 2]), 0.0
    )
    state["previous_support"] = legal_support

    state["was_airborne"] = airborne
    active = _active_mask(state)
    all_covered = torch.logical_or(~active, state["visited"]).all(dim=1)
    all_right = right_top.all(dim=1)
    settle_requested = all_covered & all_right & ~state["failure"]
    state["brake_time"] = torch.where(
        settle_requested,
        state["brake_time"] + physics_dt,
        torch.zeros_like(state["brake_time"]),
    )
    speed = torch.linalg.vector_norm(robot.root_link_lin_vel_w[:, :2], dim=1)
    angular_speed = torch.linalg.vector_norm(robot.root_link_ang_vel_w, dim=1)
    upright = -robot.projected_gravity_b[:, 2]
    stable = settle_requested & (state["brake_time"] >= BRAKE_TIME_S) & (speed < 0.05) & (angular_speed < 0.15) & (upright > math.cos(math.radians(10.0)))
    state["stable_time"] = torch.where(stable, state["stable_time"] + physics_dt, torch.zeros_like(state["stable_time"]))
    success_event = (state["stable_time"] >= STABLE_TIME_S) & ~state["success"] & ~state["failure"]
    state["success"] |= success_event
    state["success_pulse"] += success_event.float()
    state["stage"] = torch.where(
        state["strict_ground_check"],
        torch.where(settle_requested, torch.full_like(state["stage"], 3), torch.full_like(state["stage"], 2)),
        torch.where(left_top.any(dim=1), torch.ones_like(state["stage"]), torch.zeros_like(state["stage"])),
    )
    for key in ("mount_pulse", "cover_pulse", "landing_pulse", "lift_pulse", "success_pulse"):
        state[key].masked_fill_(state["failure"], 0.0)
    return support_count.float() / 4.0


class SteppingStonesVelocityCommand(UniformThresholdVelocityCommand):
    """Keep a sampled forward command until the exit platform requires braking."""

    def _apply_task_command(self) -> None:
        state = _state(self._env)
        brake = (1.0 - state["brake_time"] / BRAKE_TIME_S).clamp(0.0, 1.0)
        self.is_standing_env[:] = False
        self.is_forward_env[:] = True
        self.is_heading_env[:] = False
        self.is_world_env[:] = False
        self.is_lateral_env[:] = False
        self.is_yaw_env[:] = False
        self.is_backward_env[:] = False
        self.vel_command_b[:, 0] = state["episode_speed"] * brake
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


def stepping_succeeded(env) -> torch.Tensor:
    return _state(env)["success"]


def stepping_timeout(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def stepping_failure_reward(env) -> torch.Tensor:
    state = _state(env)
    terminated = getattr(env, "reset_terminated", state["failure"])
    return (terminated & ~state["success"]).float() / env.step_dt


def _positive_event_reward(env, key: str) -> torch.Tensor:
    state = _state(env)
    failed = state["failure"] | (getattr(env, "reset_terminated", state["failure"]) & ~state["success"])
    return torch.where(failed, 0.0, state[key]) / env.step_dt


def mount_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "mount_pulse")


def cover_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "cover_pulse")


def landing_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "landing_pulse")


def lift_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "lift_pulse")


def success_reward(env) -> torch.Tensor:
    return _positive_event_reward(env, "success_pulse")


def _body_goal(env):
    """Guide the body between paired stones without assigning a foot or contact order."""
    state = _state(env)
    active = _active_mask(state)
    remaining = active & ~state["visited"]
    first = remaining.long().argmax(dim=1)
    indices = torch.arange(MAX_STONES, device=env.device)
    group = active & (indices[None, :] // 2 == first[:, None] // 2)
    centers = torch.as_tensor(stone_centers(MAX_STONES), dtype=torch.float32, device=env.device)
    target_x = (group * centers[None, :, 0]).sum(dim=1) / group.sum(dim=1).clamp_min(1)
    all_covered = ~remaining.any(dim=1)
    exit_x = 0.90 + 0.30 * state["active_count"].float() + PLATFORM_SIZE / 2
    target_x = torch.where(all_covered, exit_x, target_x)
    target_x = torch.where(state["strict_ground_check"], target_x, PLATFORM_SIZE / 2)
    goal_id = torch.where(all_covered, MAX_STONES, first // 2)
    goal_id = torch.where(state["strict_ground_check"], goal_id, -1)
    target = torch.stack((target_x, torch.zeros_like(target_x)), dim=1) + env.scene.env_origins[:, :2]
    return target, goal_id


def goal_velocity_reward(env, command_name: str = "twist") -> torch.Tensor:
    """Reward bounded goal-directed speed only while improving the closest distance."""
    state = _state(env)
    target, goal_id = _body_goal(env)
    robot = env.scene["robot"].data
    delta = target - robot.root_link_pos_w[:, :2]
    distance = torch.linalg.vector_norm(delta, dim=1)
    direction = delta / distance.clamp_min(1.0e-6)[:, None]
    command = env.command_manager.get_command(command_name)[:, 0]
    projection = (robot.root_link_lin_vel_w[:, :2] * direction).sum(dim=1)
    value = (projection / command.clamp_min(0.05)).clamp(max=1.0)
    changed = state["goal_id"] != goal_id
    improving = distance < state["goal_best_distance"] - 1.0e-6
    value = torch.where((value <= 0.0) | improving, value, 0.0)
    active = (command > 0.05) & (state["stage"] != 3) & (distance > GOAL_REACHED_DISTANCE) & ~changed
    stage_scale = torch.where(state["stage"] == 1, 0.30, 1.0)
    value = torch.where(active, value * stage_scale, 0.0)
    new_step = state["goal_reward_tick"] != env.common_step_counter
    state["goal_reward_value"] = torch.where(new_step, value, state["goal_reward_value"])
    state["goal_best_distance"] = torch.where(
        new_step, torch.where(changed, distance, torch.minimum(distance, state["goal_best_distance"])),
        state["goal_best_distance"],
    )
    state["goal_id"] = torch.where(new_step, goal_id, state["goal_id"])
    state["goal_reward_tick"].fill_(env.common_step_counter)
    terminated = getattr(env, "reset_terminated", state["failure"] | state["success"])
    return torch.where(state["failure"] | state["success"] | terminated, 0.0, state["goal_reward_value"])


def forward_velocity_reward(env, command_name: str = "twist") -> torch.Tensor:
    state = _state(env)
    command = env.command_manager.get_command(command_name)[:, 0]
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    active = (command > 0.05) & ~state["failure"] & ~state["success"]
    speed_gate = (actual / command.clamp_min(0.05)).clamp(0.0, 1.0)
    tracking = torch.exp(-((actual - command) / 0.15).square())
    climbing = state["stage"] == 1
    stage_scale = torch.where(climbing, 0.30, 1.0)
    return active.float() * speed_gate * tracking * stage_scale


def route_progress_reward(env, gamma: float = 0.9995) -> torch.Tensor:
    state = _state(env)
    count = state["active_count"]
    stop_x = 0.90 + 0.30 * count.float() + PLATFORM_SIZE / 2
    root_x = env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    current = ((root_x - state["spawn_x"]) / (stop_x - state["spawn_x"]).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    previous = state["previous_progress"]
    valid = state["progress_ready"].clone()
    terminated = getattr(env, "reset_terminated", state["failure"] | state["success"])
    next_value = torch.where(terminated & ~state["success"], torch.zeros_like(current), current)
    delta = gamma * next_value - previous
    state["previous_progress"] = current
    state["progress_ready"].fill_(True)
    return torch.where(valid, delta / env.step_dt, torch.zeros_like(delta))


def time_cost(env) -> torch.Tensor:
    state = _state(env)
    return (~state["success"] & ~state["failure"]).float()


def support_deficit(env) -> torch.Tensor:
    count = _state(env)["last_support_count"]
    return (2.0 - count).clamp_min(0.0).square()


def clearance_penalty(env) -> torch.Tensor:
    state = _state(env)
    robot = env.scene["robot"].data
    feet = _relative_xy(env, _foot_bottom(env, state))
    sensor = env.scene["stepping_foot_contact"].data
    found = sensor.found[:, state["contact_ids"]] > 0
    count = state["active_count"]
    fixed_edges = torch.cat(
        (
            torch.zeros(1, device=env.device, dtype=feet.dtype),
            torch.as_tensor(
                stone_centers(MAX_STONES)[:, 0] - STONE_SIZE / 2,
                device=env.device,
                dtype=feet.dtype,
            ),
        )
    )
    right_edge = (0.90 + 0.30 * count.float())[:, None]
    edge_distance = torch.minimum(
        (feet[:, :, None, 0] - fixed_edges[None, None, :]).abs().amin(dim=2),
        (feet[:, :, 0] - right_edge).abs(),
    )
    crossing = ~found & (edge_distance < 0.10)
    deficit = ((REQUIRED_CLEARANCE - feet[:, :, 2]).clamp_min(0.0) / 0.04).square()
    return (deficit * crossing).sum(dim=1) / crossing.sum(dim=1).clamp_min(1)


def over_height_penalty(env) -> torch.Tensor:
    state = _state(env)
    robot = env.scene["robot"].data
    feet = _relative_xy(env, _foot_bottom(env, state))
    sensor = env.scene["stepping_foot_contact"].data
    airborne = sensor.found[:, state["contact_ids"]] == 0
    excess = ((feet[:, :, 2] - (REQUIRED_CLEARANCE + 0.06)).clamp_min(0.0) / 0.06).square()
    return (excess * airborne).sum(dim=1) / airborne.sum(dim=1).clamp_min(1)


def edge_penalty(env) -> torch.Tensor:
    state = _state(env)
    sensor = env.scene["stepping_foot_contact"].data
    found = sensor.found[:, state["contact_ids"]] > 0
    _, stone_top, _, _, _ = _top_contact_masks(env, state, found, sensor.pos[:, state["contact_ids"]])
    local = _relative_xy(env, sensor.pos[:, state["contact_ids"]])
    centers = torch.as_tensor(stone_centers(MAX_STONES), device=env.device, dtype=local.dtype)
    dx = STONE_SIZE / 2 - (local[:, :, None, 0] - centers[None, None, :, 0]).abs()
    dy = STONE_SIZE / 2 - (local[:, :, None, 1] - centers[None, None, :, 1]).abs()
    margin = torch.minimum(dx, dy)
    risk = ((LANDING_MARGIN - margin).clamp_min(0.0) / LANDING_MARGIN).square()
    mask = stone_top
    return (risk * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)


def side_collision_penalty(env) -> torch.Tensor:
    state = _state(env)
    sensor = env.scene["stepping_foot_contact"].data
    found = sensor.found[:, state["contact_ids"]] > 0
    _, stone_top, left_top, right_top, ground = _top_contact_masks(env, state, found, sensor.pos[:, state["contact_ids"]])
    top = stone_top.any(dim=2) | left_top | right_top
    return (found & ~top & ~ground).float().mean(dim=1)


def settle_speed_penalty(env) -> torch.Tensor:
    state = _state(env)
    settle = state["brake_time"] > 0.0
    speed = torch.linalg.vector_norm(env.scene["robot"].data.root_link_lin_vel_w[:, :2], dim=1)
    return settle.float() * (speed / 0.10).square()


def settle_angular_penalty(env) -> torch.Tensor:
    state = _state(env)
    settle = state["brake_time"] > 0.0
    speed = torch.linalg.vector_norm(env.scene["robot"].data.root_link_ang_vel_w, dim=1)
    return settle.float() * (speed / 0.20).square()


def stone_relative_positions(env) -> torch.Tensor:
    state = _state(env)
    centers = torch.as_tensor(stone_centers(MAX_STONES), device=env.device, dtype=torch.float32)
    points = centers[None, :, :] + env.scene.env_origins[:, None, :]
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
    right = torch.stack((0.90 + 0.30 * count + PLATFORM_SIZE / 2, torch.zeros_like(count), torch.full_like(count, PLATFORM_HEIGHT)), dim=1)
    points = torch.stack((left, right), dim=1) + env.scene.env_origins[:, None, :]
    root = env.scene["robot"].data.root_link_pos_w
    delta = points - root[:, None, :]
    heading = env.scene["robot"].data.heading_w[:, None]
    local_x = torch.cos(heading) * delta[:, :, 0] + torch.sin(heading) * delta[:, :, 1]
    local_y = -torch.sin(heading) * delta[:, :, 0] + torch.cos(heading) * delta[:, :, 1]
    return torch.stack((local_x, local_y, delta[:, :, 2]), dim=-1).reshape(env.num_envs, -1)


def stage_observation(env) -> torch.Tensor:
    state = _state(env)
    return torch.nn.functional.one_hot(state["stage"], num_classes=4).float()


def strict_ground_observation(env) -> torch.Tensor:
    return _state(env)["strict_ground_check"].float().unsqueeze(1)


def foot_contact_observation(env) -> torch.Tensor:
    state = _state(env)
    sensor = env.scene["stepping_foot_contact"].data
    return (sensor.found[:, state["contact_ids"]] > 0).float()


def settle_progress_observation(env) -> torch.Tensor:
    return (_state(env)["stable_time"] / STABLE_TIME_S).clamp(0.0, 1.0).unsqueeze(1)


def route_direction_observation(env) -> torch.Tensor:
    heading = env.scene["robot"].data.heading_w
    return torch.stack((torch.cos(heading), -torch.sin(heading)), dim=1)


def navigation_valid_observation(env) -> torch.Tensor:
    return torch.ones(env.num_envs, 1, device=env.device)


def remaining_time_observation(env) -> torch.Tensor:
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(0.0, 1.0).unsqueeze(1)


def stepping_metric(env, key: str) -> torch.Tensor:
    state = _state(env)
    if key == "success":
        return state["success"].float()
    if key == "failure":
        return state["failure"].float()
    if key == "covered_fraction":
        return (state["visited"] & _active_mask(state)).sum(dim=1).float() / state["active_count"].float()
    if key == "stone_count":
        return state["active_count"].float()
    if key == "goal_distance":
        target, _ = _body_goal(env)
        return torch.linalg.vector_norm(target - env.scene["robot"].data.root_link_pos_w[:, :2], dim=1)
    if key == "low_clearance_fraction":
        return state["low_clearance_count"] / state["swing_landing_count"].clamp_min(1)
    if key == "swing_landing_count":
        return state["swing_landing_count"]
    if key == "edge_contact_fraction":
        sensor = env.scene["stepping_foot_contact"].data
        found = sensor.found[:, state["contact_ids"]] > 0
        local, top, _, _, _ = _top_contact_masks(env, state, found, sensor.pos[:, state["contact_ids"]])
        centers = torch.as_tensor(stone_centers(MAX_STONES), device=env.device, dtype=local.dtype)
        margin = (STONE_SIZE / 2 - (local[:, :, None, :2] - centers[None, None, :, :2]).abs()).amin(dim=-1)
        return ((margin < LANDING_MARGIN) & top).sum(dim=(1, 2)).float() / top.sum(dim=(1, 2)).clamp_min(1)
    raise ValueError(f"Unknown stepping-stone metric '{key}'.")


class SteppingStoneCurriculum:
    """Promote only complete passes from two to eight stones."""

    def __init__(self, cfg: CurriculumTermCfg, env):
        params = cfg.params
        self.window_size = int(params.get("window_size", 100))
        self.success_threshold = float(params.get("success_threshold", 0.80))
        self.demote_threshold = float(params.get("demote_threshold", 0.40))
        if self.window_size <= 0 or not 0.0 <= self.demote_threshold < self.success_threshold <= 1.0:
            raise ValueError("Invalid stepping-stone curriculum thresholds.")
        self.level_cap = int(params.get("initial_level_cap", 0))
        self.window: deque[bool] = deque(maxlen=self.window_size)

    def __call__(self, env, env_ids, **kwargs):
        del kwargs
        ids = _ids(env, env_ids)
        state = _state(env)
        terrain = env.scene.terrain
        if terrain is None:
            raise RuntimeError("Stepping-stone curriculum requires generated terrain.")
        completed = env.episode_length_buf[ids] > 0
        success = state["success"][ids] & completed
        for value in success[completed].detach().cpu().tolist():
            self.window.append(bool(value))
        rate = sum(self.window) / len(self.window) if self.window else 0.0
        maximum = terrain.cfg.terrain_generator.num_rows - 1
        if len(self.window) == self.window_size and rate >= self.success_threshold and self.level_cap < maximum:
            self.level_cap += 1
            self.window.clear()
        current = terrain.terrain_levels[ids]
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
