"""One triggered jump with substep contact failures and four-foot settling."""

from collections import deque
from dataclasses import dataclass
import math

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

from ..robot_cfg import EFFORT_LEG, FOOT_SITE_NAMES
from ..terrains.jump import (
    CENTER_Y, LAUNCH_EDGE, LANE_WIDTH, LEVELS_PER_AXIS, LEVEL_COUNT,
    PLATFORM_START, PLATFORM_END, SPAWN_X, TERRAIN_NAMES, TILE_SIZE, jump_profile,
)
from .terminations import joint_pos_out_of_limits
from .elevation import elevation_foot_bottom


@dataclass(frozen=True)
class JumpSettings:
    distance_min: float = .400
    distance_max: float = .500
    stable_duration: float = .5
    launch_deadline: float = .6
    minimum_flight_duration: float = .02
    episode_duration: float = 3.
    max_launch_foot_drift: float = .04
    max_launch_root_drift: float = .08
    top_tolerance: float = .008
    support_force: float = 1.
    stable_linear_speed: float = .10
    stable_angular_speed: float = .50
    stable_tilt: float = math.radians(12)
    gamma: float = .995


def _ids(env, env_ids):
    return torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]


def jump_state(env):
    if hasattr(env, "_jump"):
        return env._jump
    def zeros(*shape, dtype=torch.float):
        return torch.zeros((env.num_envs, *shape), device=env.device, dtype=dtype)
    state = {key: zeros() for key in (
        "height", "gap", "wall_rise", "landing_edge", "elapsed", "flight_time", "stable_time",
        "takeoff_latency", "distance_min", "distance_max", "distance_mean", "peak_clearance",
        "clearance_increment", "phi", "shaping", "new_events", "launch_front_min", "launch_front_max",
        "slip", "launch_root_x", "landing_speed", "settle_fraction", "landing_events",
    )}
    state.update({key: zeros(dtype=torch.bool) for key in (
        "initialized", "supported", "airborne", "landed", "complete", "failed", "invalid",
        "bad_contact", "body_contact", "wall_contact", "late_launch", "distance_failed",
        "runup", "multiple_jump", "timeout", "have_launch_reference",
    )})
    state.update(corner=zeros(3), initial_front=zeros(2), last_front=zeros(2), rear_touch_x=zeros(2),
                 front_referenced=zeros(2, dtype=torch.bool),
                 rear_touched=zeros(2, dtype=torch.bool), source_support=zeros(4, dtype=torch.bool),
                 target_support=zeros(4, dtype=torch.bool),
                 height_level=zeros(dtype=torch.long), gap_level=zeros(dtype=torch.long),
                 phase=zeros(dtype=torch.long), reason=zeros(dtype=torch.long),
                 updated_tick=-1, event_tick=-1)
    state["foot_ids"] = env.scene["robot"].find_sites(FOOT_SITE_NAMES, preserve_order=True)[0]
    env._jump = state
    return state


def select_jump_terrain(env, env_ids, settings, randomize_levels=False):
    env._jump_settings = settings
    ids = _ids(env, env_ids)
    state, terrain = jump_state(env), env.scene.terrain
    if tuple(terrain.cfg.terrain_generator.sub_terrains) != TERRAIN_NAMES:
        raise ValueError("Jump requires platform_gap and platform_wall in canonical order")
    if not getattr(env, "_jump_fixed_cases", False):
        kinds = ids.remainder(2)
        terrain.terrain_types[ids] = kinds
        caps = torch.full((len(ids), 2), LEVELS_PER_AXIS - 1 if randomize_levels else 0,
                          device=env.device, dtype=torch.long)
        if not randomize_levels and hasattr(env, "_jump_curriculum"):
            caps = env._jump_curriculum.level_caps[kinds]
        levels = torch.floor(torch.rand(len(ids), 2, device=env.device) * (caps + 1)).long()
        levels = torch.where(torch.rand_like(levels.float()) < .5, caps, levels)
        terrain.terrain_levels[ids] = levels[:, 0] * LEVELS_PER_AXIS + levels[:, 1]
    kinds, levels = terrain.terrain_types[ids], terrain.terrain_levels[ids]
    if ((kinds < 0) | (kinds > 1) | (levels < 0) | (levels >= LEVEL_COUNT)).any():
        raise ValueError("Jump fixed case is out of range")
    resolved = terrain.get_geometry_levels(levels, kinds)
    if not torch.equal(resolved, levels):
        raise ValueError("Jump needs all 50 physical height/gap cases; compact level aliasing is forbidden")
    state["height_level"][ids] = levels // LEVELS_PER_AXIS
    state["gap_level"][ids] = levels % LEVELS_PER_AXIS
    profiles = [jump_profile(TERRAIN_NAMES[int(k)], int(h), int(g)) for k, h, g in
                zip(kinds.cpu(), state["height_level"][ids].cpu(), state["gap_level"][ids].cpu())]
    terrain.env_origins[ids] = terrain.get_terrain_origins(levels, kinds)
    for key, attr in (("height", "platform_height"), ("gap", "gap"),
                      ("wall_rise", "wall_rise"), ("landing_edge", "landing_edge")):
        state[key][ids] = torch.tensor([getattr(p, attr) for p in profiles], device=env.device)
    state["corner"][ids] = terrain.env_origins[ids] - torch.tensor(
        [p.spawn.tolist() for p in profiles], device=env.device)


def reset_jump(env, env_ids, settings, sole_offset=0.):
    """Start the skill at a fixed pose; episode time zero is the trigger."""
    ids = _ids(env, env_ids)
    state = jump_state(env)
    preserve = {"height", "gap", "wall_rise", "landing_edge", "height_level", "gap_level", "corner"}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and key not in preserve:
            value[ids] = 0
    state["initialized"][ids] = True
    state["updated_tick"] = -1
    state["event_tick"] = -1
    state["takeoff_latency"][ids] = settings.launch_deadline
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[ids].clone()
    pose[:, :3] += env.scene.terrain.env_origins[ids]
    pose[:, 2] += sole_offset
    pose[:, 3:7] = pose.new_tensor([1., 0., 0., 0.])
    pose[:, 7:] = 0
    robot.write_root_state_to_sim(pose, env_ids=ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids], robot.data.default_joint_vel[ids], env_ids=ids)
    state["launch_root_x"][ids] = pose[:, 0] - state["corner"][ids, 0]


def _top_masks(local, normal, state, tolerance):
    width_ok = (local[..., 1] - CENTER_Y).abs() <= LANE_WIDTH / 2 - .001
    top = ((local[..., 2] - state["height"][:, None, None]).abs() <= tolerance) & (normal[..., 2] < -.9) & width_ok
    x = local[..., 0]
    source = top & (x >= PLATFORM_START + .001) & (x <= LAUNCH_EDGE - .001)
    target = top & (x >= state["landing_edge"][:, None, None] + .001) & (x <= PLATFORM_END - .001)
    return source, target


def collect_jump_contacts(env):
    """Latch every physical substep so a brief collision cannot disappear."""
    state, settings = jump_state(env), env._jump_settings
    if state["event_tick"] != env.common_step_counter:
        state["new_events"][:] = 0
        state["landing_events"][:] = 0
        state["clearance_increment"][:] = 0
        state["event_tick"] = env.common_step_counter
    robot = env.scene["robot"].data
    sensor = env.scene["jump_feet"]
    slots = sensor.cfg.num_slots
    data = sensor.data
    found = data.found.reshape(env.num_envs, 4, slots) > 0
    force = data.force.reshape(env.num_envs, 4, slots, 3).norm(dim=-1)
    pos = data.pos.reshape(env.num_envs, 4, slots, 3) - state["corner"][:, None, None]
    normal = data.normal.reshape(env.num_envs, 4, slots, 3)
    source, target = _top_masks(pos, normal, state, settings.top_tolerance)
    effective = found & (force > settings.support_force)
    state["source_support"] = (effective & source).any(dim=2)
    state["target_support"] = (effective & target).any(dim=2)
    forbidden = found & ~(source | target)
    state["bad_contact"] |= forbidden.any(dim=(1, 2))
    wall_x = LAUNCH_EDGE + state["gap"] / 2
    wall_hit = (pos[..., 0] - wall_x[:, None, None]).abs() <= .040 + settings.top_tolerance
    state["wall_contact"] |= (forbidden & wall_hit & (state["wall_rise"] > 0)[:, None, None]).any(dim=(1, 2))
    body = env.scene["jump_body"].data
    state["body_contact"] |= (body.found > 0).any(dim=1)
    body_x = body.pos[..., 0] - state["corner"][:, 0, None]
    state["wall_contact"] |= ((body.found > 0) & ((body_x - wall_x[:, None]).abs() <= .048)
                              & (state["wall_rise"] > 0)[:, None]).any(dim=1)
    state["elapsed"] += env.physics_dt
    feet_x = robot.site_pos_w[:, state["foot_ids"], 0] - state["corner"][:, 0, None]
    ready = ~state["have_launch_reference"]
    state["initial_front"][ready] = feet_x[ready, :2]
    state["last_front"][ready] = feet_x[ready, :2]
    state["have_launch_reference"][:] = True
    front_source = state["source_support"][:, :2] & ~state["airborne"][:, None]
    contact_x = torch.where(effective & source, pos[..., 0], 0.).sum(dim=2) / (effective & source).sum(dim=2).clamp_min(1)
    first_front = front_source & ~state["front_referenced"]
    state["initial_front"] = torch.where(first_front, contact_x[:, :2], state["initial_front"])
    state["front_referenced"] |= front_source
    state["last_front"] = torch.where(front_source, contact_x[:, :2], state["last_front"])
    state["supported"] |= state["source_support"].all(dim=1)
    any_contact = found.any(dim=(1, 2)) | (body.found > 0).any(dim=1)
    state["flight_time"] = torch.where(any_contact, 0., state["flight_time"] + env.physics_dt)
    takeoff = (state["supported"] & ~state["airborne"]
               & (state["flight_time"] >= settings.minimum_flight_duration - 1e-6)
               & (robot.root_link_lin_vel_w[:, 0] > .05) & (robot.root_link_lin_vel_w[:, 2] > .05))
    state["takeoff_latency"] = torch.where(takeoff, state["elapsed"], state["takeoff_latency"])
    state["launch_front_min"] = torch.where(takeoff, state["last_front"].min(dim=1).values, state["launch_front_min"])
    state["launch_front_max"] = torch.where(takeoff, state["last_front"].max(dim=1).values, state["launch_front_max"])
    state["new_events"] += takeoff.float()
    state["airborne"] |= takeoff
    state["late_launch"] |= ~state["airborne"] & (state["elapsed"] >= settings.launch_deadline - 1e-6)
    drift = torch.where(front_source, (contact_x[:, :2] - state["initial_front"]).abs(), 0.).max(dim=1).values
    root_drift = (robot.root_link_pos_w[:, 0] - state["corner"][:, 0] - state["launch_root_x"]).abs()
    state["runup"] |= ~state["airborne"] & ((drift > settings.max_launch_foot_drift) | (root_drift > settings.max_launch_root_drift))
    premature = state["target_support"].any(dim=1) & ~state["airborne"]
    return_to_source = (found & source).any(dim=(1, 2)) & state["airborne"]
    state["bad_contact"] |= premature | return_to_source
    target_rear = state["target_support"][:, 2:] & state["airborne"][:, None]
    new_rear = target_rear & ~state["rear_touched"]
    target_x = torch.where(effective & target, pos[..., 0], 0.).sum(dim=2) / (effective & target).sum(dim=2).clamp_min(1)
    state["rear_touch_x"] = torch.where(new_rear, target_x[:, 2:], state["rear_touch_x"])
    state["rear_touched"] |= target_rear
    measured = state["rear_touched"].all(dim=1)
    distance_min = state["rear_touch_x"].min(dim=1).values - state["launch_front_max"]
    distance_max = state["rear_touch_x"].max(dim=1).values - state["launch_front_min"]
    state["distance_min"] = torch.where(measured, distance_min, 0.)
    state["distance_max"] = torch.where(measured, distance_max, 0.)
    state["distance_mean"] = torch.where(measured, state["rear_touch_x"].mean(dim=1) - state["last_front"].mean(dim=1), 0.)
    range_ok = (distance_min > settings.distance_min + 1e-6) & (distance_max < settings.distance_max - 1e-6)
    state["distance_failed"] |= measured & ~range_ok
    current_min = target_x[:, 2:].min(dim=1).values - state["launch_front_max"]
    current_max = target_x[:, 2:].max(dim=1).values - state["launch_front_min"]
    current_range_ok = (current_min > settings.distance_min + 1e-6) & (current_max < settings.distance_max - 1e-6)
    state["distance_failed"] |= target_rear.all(dim=1) & ~current_range_ok
    target_contact = state["target_support"].any(dim=1) & state["airborne"]
    first_land = target_contact & ~state["landed"]
    state["landing_events"] += first_land.float()
    state["landing_speed"] = torch.where(first_land, robot.root_link_lin_vel_w.norm(dim=1), state["landing_speed"])
    state["landed"] |= target_contact
    state["multiple_jump"] |= state["landed"] & (state["flight_time"] >= settings.minimum_flight_duration - 1e-6)
    roll, pitch, _ = euler_xyz_from_quat(robot.root_link_quat_w)
    roll, pitch = torch.atan2(torch.sin(roll), torch.cos(roll)), torch.atan2(torch.sin(pitch), torch.cos(pitch))
    stable = (state["airborne"] & measured & range_ok & current_range_ok & state["target_support"].all(dim=1)
              & (roll.abs() < settings.stable_tilt) & (pitch.abs() < settings.stable_tilt)
              & (robot.root_link_lin_vel_w.norm(dim=1) < settings.stable_linear_speed)
              & (robot.root_link_ang_vel_b.norm(dim=1) < settings.stable_angular_speed))
    state["stable_time"] = torch.where(stable, state["stable_time"] + env.physics_dt, 0.)
    state["settle_fraction"] = (state["stable_time"] / settings.stable_duration).clamp_max(1)
    heights = elevation_foot_bottom(env, state["foot_ids"])[..., 2] - state["corner"][:, 2, None] - state["height"][:, None]
    clearance = heights.min(dim=1).values.clamp_min(0)
    peak = torch.where(state["airborne"], torch.maximum(state["peak_clearance"], clearance), state["peak_clearance"])
    target_clearance = (state["wall_rise"] + .03).clamp_min(.08)
    state["clearance_increment"] += (peak.clamp_max(target_clearance) - state["peak_clearance"].clamp_max(target_clearance)) / target_clearance
    state["peak_clearance"] = peak
    velocity = robot.site_lin_vel_w[:, state["foot_ids"], :2].square().sum(dim=-1)
    support = state["source_support"] | state["target_support"]
    state["slip"] = ((velocity / .15 ** 2).clamp_max(1) * support).sum(dim=1) / support.sum(dim=1).clamp_min(1)
    return support.sum(dim=1).float()


def update_jump_state(env):
    state, settings = jump_state(env), env._jump_settings
    if state["updated_tick"] == env.common_step_counter:
        return state
    robot = env.scene["robot"].data
    valid = torch.isfinite(torch.cat((robot.root_link_pos_w, robot.root_link_quat_w,
                                     robot.root_link_lin_vel_w, robot.root_link_ang_vel_b, robot.joint_pos), dim=1)).all(dim=1)
    state["invalid"] |= ~valid
    local = robot.root_link_pos_w - state["corner"]
    outside = ((local[:, 1] - CENTER_Y).abs() > LANE_WIDTH / 2) | (local[:, 0] < PLATFORM_START) | (local[:, 0] > TILE_SIZE[0])
    roll, pitch, _ = euler_xyz_from_quat(robot.root_link_quat_w)
    tilt = (torch.atan2(torch.sin(roll), torch.cos(roll)).abs() > math.radians(70)) | (torch.atan2(torch.sin(pitch), torch.cos(pitch)).abs() > math.radians(70))
    joint = joint_pos_out_of_limits(env)
    failure = (state["bad_contact"] | state["body_contact"] | state["wall_contact"] | state["late_launch"]
               | state["runup"] | state["distance_failed"] | state["multiple_jump"] | outside | tilt | joint | state["invalid"])
    state["complete"] = (state["stable_time"] >= settings.stable_duration - 1e-6) & ~failure
    state["timeout"] = (state["elapsed"] >= env.cfg.episode_length_s - 1e-6) & ~state["complete"]
    failure |= state["timeout"]
    state["failed"] |= failure
    state["complete"] &= ~state["failed"]
    state["reason"][:] = 0
    for code, condition in ((1, state["bad_contact"]), (2, state["body_contact"]), (3, state["wall_contact"]),
                            (4, state["late_launch"]), (5, state["runup"]), (6, state["distance_failed"]),
                            (7, state["multiple_jump"]), (8, outside | tilt | joint), (9, state["timeout"]), (10, state["invalid"])):
        state["reason"] = torch.where(condition, code, state["reason"])
    state["phase"] = torch.where(state["landed"], 2, state["airborne"].long())
    rear_x = robot.site_pos_w[:, state["foot_ids"][2:], 0].min(dim=1).values - state["corner"][:, 0]
    target = state["initial_front"].mean(dim=1) + (settings.distance_min + settings.distance_max) / 2
    initial_rear = SPAWN_X - .055
    next_phi = ((rear_x - initial_rear) / (target - initial_rear).clamp_min(.1)).clamp(-1, 1)
    next_phi = torch.where(state["failed"] | state["complete"], 0., next_phi)
    state["shaping"] = settings.gamma ** (env.step_dt / .02) * next_phi - state["phi"]
    state["phi"] = next_phi
    state["updated_tick"] = env.common_step_counter
    return state


def jump_failed(env):
    return update_jump_state(env)["failed"]


def jump_completed(env):
    return update_jump_state(env)["complete"]


def jump_event(env, key):
    state = update_jump_state(env)
    value = state[key].float()
    if key not in ("shaping", "failed"):
        value = torch.where(state["failed"], 0., value)
    return value / env.step_dt


def jump_cost(env, key):
    state = update_jump_state(env)
    robot = env.scene["robot"].data
    if key == "time":
        return torch.ones(env.num_envs, device=env.device)
    if key == "lateral":
        return (robot.root_link_lin_vel_w[:, 1] / .2).square().clamp_max(1)
    if key == "center":
        return ((robot.root_link_pos_w[:, 1] - state["corner"][:, 1] - CENTER_Y) / .15).square().clamp_max(1)
    if key == "heading":
        return (torch.atan2(torch.sin(robot.heading_w), torch.cos(robot.heading_w)) / math.radians(15)).square().clamp_max(1)
    if key == "attitude":
        roll, pitch, _ = euler_xyz_from_quat(robot.root_link_quat_w)
        pitch_tol = torch.where(state["landed"], math.radians(12), math.radians(40))
        return ((roll / math.radians(15)).square() + ((pitch.abs() - pitch_tol).clamp_min(0) / math.radians(20)).square()).clamp_max(1)
    if key == "torque":
        return (robot.qfrc_actuator / EFFORT_LEG).square().mean(dim=1).clamp_max(1)
    if key == "action":
        return ((env.action_manager.action - env.action_manager.prev_action) / (env.step_dt * 10)).square().mean(dim=1).clamp_max(1)
    if key == "settle_speed":
        return (robot.root_link_lin_vel_w.norm(dim=1) / .3).square().clamp_max(1) * state["landed"]
    if key == "settle_angular":
        return (robot.root_link_ang_vel_b.norm(dim=1) / 1.).square().clamp_max(1) * state["landed"]
    if key == "slip":
        return state["slip"]
    raise KeyError(key)


def jump_metric(env, key):
    state = update_jump_state(env)
    if key == "four_foot_support":
        return state["target_support"].all(dim=1).float()
    if key == "distance_valid":
        return (state["rear_touched"].all(dim=1) & ~state["distance_failed"]).float()
    return state[key].float()


class JumpTerrainCurriculum:
    """Unlock height and gap independently using actual terminal successes."""

    def __init__(self, cfg, env):
        self.window_size = int(cfg.params.get("window_size", 100))
        self.threshold = float(cfg.params.get("threshold", .8))
        if self.window_size < 1 or not 0 < self.threshold <= 1:
            raise ValueError("Invalid jump curriculum")
        self.level_caps = torch.zeros(2, 2, device=env.device, dtype=torch.long)
        self.windows = [[deque(maxlen=self.window_size) for _ in range(2)] for _ in range(2)]
        self.outcomes = [deque(maxlen=self.window_size) for _ in range(2)]
        env._jump_curriculum = self

    def __call__(self, env, env_ids, **kwargs):
        del kwargs
        state = jump_state(env)
        ids = _ids(env, env_ids)
        finished = ids[state["initialized"][ids] & ~state["invalid"][ids] & (env.episode_length_buf[ids] > 0)]
        for index in finished.cpu().tolist():
            kind = int(env.scene.terrain.terrain_types[index])
            success = bool(state["complete"][index]) and not bool(state["failed"][index])
            self.outcomes[kind].append(success)
            levels = (int(state["height_level"][index]), int(state["gap_level"][index]))
            for axis, level in enumerate(levels):
                window = self.windows[kind][axis]
                if level == int(self.level_caps[kind, axis]):
                    window.append(success)
                    if len(window) == self.window_size and sum(window) / len(window) >= self.threshold and level < LEVELS_PER_AXIS - 1:
                        self.level_caps[kind, axis] += 1
                        window.clear()
        metrics = {}
        for kind, name in enumerate(TERRAIN_NAMES):
            for axis, axis_name in enumerate(("height", "gap")):
                metrics[f"{name}_{axis_name}_cap"] = self.level_caps[kind, axis].float()
            if self.outcomes[kind]:
                metrics[f"{name}_success_rate"] = sum(self.outcomes[kind]) / len(self.outcomes[kind])
        return metrics

    def state_dict(self):
        return {"level_caps": self.level_caps.cpu().tolist(), "window_size": self.window_size,
                "threshold": self.threshold, "windows": [[list(w) for w in row] for row in self.windows],
                "outcomes": [list(w) for w in self.outcomes]}

    def load_state_dict(self, value):
        caps = torch.as_tensor(value["level_caps"], device=self.level_caps.device, dtype=torch.long)
        if (caps.shape != (2, 2) or ((caps < 0) | (caps >= LEVELS_PER_AXIS)).any()
                or value["window_size"] != self.window_size or value["threshold"] != self.threshold
                or len(value["windows"]) != 2 or any(len(row) != 2 for row in value["windows"])):
            raise ValueError("Jump curriculum changed; use weights-only transfer")
        self.level_caps[:] = caps
        self.windows = [[deque(map(bool, w), maxlen=self.window_size) for w in row] for row in value["windows"]]
        self.outcomes = [deque(map(bool, w), maxlen=self.window_size) for w in value["outcomes"]]


class JumpCommand(CommandTerm):
    """Expose trigger and elapsed time in the existing three command slots."""

    @property
    def command(self):
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt
        return torch.stack((torch.ones_like(elapsed), (elapsed / self.cfg.duration).clamp(0, 1), torch.zeros_like(elapsed)), dim=1)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
        with server.gui.add_folder(name):
            server.gui.add_markdown("重置即触发一次固定跳跃；命令输入为触发标志、归一化已用时间、零。")


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
    duration: float = 3.

    def build(self, env):
        return JumpCommand(self, env)
