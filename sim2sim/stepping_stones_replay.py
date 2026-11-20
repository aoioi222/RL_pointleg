"""Build the trained stepping-stone observations from native MuJoCo data."""

import math

import mujoco
import numpy as np
import torch

from pointleg.mdp.stepping_stones import (
    EXIT_ANGULAR_SPEED_RADPS,
    EXIT_CONTACT_TIME_S,
    EXIT_HEIGHT_TOL_M,
    EXIT_JOINT_ERROR_RAD,
    EXIT_LINEAR_SPEED_MPS,
    EXIT_SURFACE,
    EXIT_SETTLE_TIME_S,
    EXIT_SETTLE_TIMEOUT_S,
    EXIT_TILT_TOL_RAD,
    GROUND_CONTACT_Z_TOL,
    LeftFrontHandoff,
    MIN_FOOT_SUPPORT_N,
    MIN_TOP_NORMAL_Z,
    SHIN_SUPPORT_N,
    SPAWN_X,
    STEPPING_FORWARD_SPEED_RANGE,
    STONE_CONTACT_TIME_S,
    TOP_CONTACT_Z_TOL,
    TOP_EDGE_TOL,
)
from pointleg.robot_cfg import BASE_BODY_NAME, FOOT_BODY_NAMES, LEG_JOINT_NAMES, LEG_NAMES
from pointleg.terrains.stepping_stones import MAX_STONES
from pointleg.mdp.fixed_stone_gait import FixedStoneGait


STEPPING_TASK = "PointLeg-SteppingStones-v0"
LEFT_FRONT = LEG_NAMES.index("fl")
RIGHT_REAR = LEG_NAMES.index("rr")
RIGHT_FRONT = LEG_NAMES.index("fr")
STEPPING_TRACE_FIELDS = (
    "time_s", "episode_time_s", "rr_foot_x_m", "rr_foot_z_m", "rr_target", "rr_target_x_m",
    "rr_foot_minus_target_x_m", "rr_foot_normal_n", "rr_foot_surface", "rr_foot_supported",
    "rr_foot_contact_x_m", "rr_foot_contact_y_m", "rr_foot_contact_z_m",
    "rr_shin_upward_n", "rr_shin_surface", "rr_shin_contact_x_m", "rr_shin_contact_y_m",
    "rr_shin_contact_z_m", "ground_feet", "failed", "finished",
    "fr_foot_x_m", "fr_foot_z_m", "fr_target", "fr_target_x_m", "fr_foot_minus_target_x_m",
    "fr_foot_normal_n", "fr_foot_surface", "fr_foot_supported",
    "fr_foot_contact_x_m", "fr_foot_contact_y_m", "fr_foot_contact_z_m",
    "fr_shin_upward_n", "fr_stone_landing", "fr_landing_offset_x_m",
    "settling", "exit_stable_time_s", "exit_settle_timeout",
    "lf_foot_normal_n", "lf_shin_upward_n", "lf_foot_supported",
    "rr_transfer_active", "lf_overlap_event", "lf_landing_blocked",
    "lf_stone_landing", "lf_stone_landing_eligible", "lf_new_stone_covered",
    "rr_first_step_complete", "rr_follow_pending",
)


def stepping_observation(env, name):
    return torch.tensor(env.stepping.values[name], dtype=torch.float32, device=env.device)[None]


class SteppingStonesReplay:
    def __init__(self, model, data, trace=None, *, standing_height, standing_joint_pos):
        self.model = model
        self.data = data
        self.trace = trace
        course_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stepping_stones")
        if course_id < 0:
            raise ValueError("Stepping stones requires a --scene containing the stepping_stones body")
        entry_id = model.geom("stones_entry_platform").id
        exit_id = model.geom("stones_exit_platform").id
        self.body_id = model.body(BASE_BODY_NAME).id
        self.standing_height = standing_height
        self.standing_joint_pos = standing_joint_pos
        self.joint_qpos_ids = np.array([model.joint(name).qposadr[0] for name in LEG_JOINT_NAMES])
        self.body_velocity = np.zeros(6)
        self.feet = {model.geom(f"{name}_collision").id: i for i, name in enumerate(FOOT_BODY_NAMES)}
        self.shins = {model.geom(f"{leg}_shin_collision").id: i for i, leg in enumerate(LEG_NAMES)}
        self.foot_geoms = tuple(self.feet)
        self.contact_force = np.zeros(6)
        robot_root = model.body_rootid[self.body_id]
        self.terrain_geoms = model.body_rootid[model.geom_bodyid] != robot_root
        stone_ids = [
            int(i) for i in np.flatnonzero(model.geom_bodyid == course_id)
            if model.geom(int(i)).name.startswith("stone_")
        ]
        self.count = len(stone_ids)
        if not 2 <= self.count <= MAX_STONES:
            raise ValueError(f"Stepping-stone scene must have 2 to {MAX_STONES} named stone geoms")
        self.platforms = self._top_centers([entry_id, exit_id])
        axis = self.platforms[1, :2] - self.platforms[0, :2]
        length = np.linalg.norm(axis)
        if length <= 0:
            raise ValueError("Stepping-stone entry and exit must have distinct positions")
        self.direction = axis / length
        self.spawn_offset = SPAWN_X - model.geom_size[entry_id, 0]
        centers = self._top_centers(stone_ids)
        order = np.argsort(centers[:, :2] @ self.direction)
        self.stones = np.empty((MAX_STONES, 3))
        self.stones[:self.count] = centers[order]
        # Training keeps projected slots beyond the last stone and masks them as inactive.
        pitch = np.mean(np.diff(self.stones[:self.count, :2] @ self.direction))
        if pitch <= 0:
            raise ValueError("Stepping stones must advance from entry toward exit")
        for i in range(self.count, MAX_STONES):
            self.stones[i] = self.stones[i - 2]
            self.stones[i, :2] += 2 * pitch * self.direction
        self.stone_indices = {stone_ids[j]: i for i, j in enumerate(order)}
        self.stone_half_width = float(model.geom_size[stone_ids[0], 0])
        self.stone_names = [model.geom(stone_ids[j]).name for j in order]
        lateral = np.array([-self.direction[1], self.direction[0]])
        left_stones = (self.stones[:self.count, :2] - self.platforms[0, :2]) @ lateral > 0
        self.left_stones = torch.from_numpy(left_stones)[None]
        self.right_stones = np.flatnonzero(~left_stones)
        self.stone_x = self.stones[:self.count, :2] @ self.direction
        self.raised_surfaces = set(stone_ids) | {entry_id, exit_id}
        self.entry_id = entry_id
        self.exit_id = exit_id
        self.floor_height = self.platforms[0, 2] - 2 * model.geom_size[entry_id, 2]
        self.left_front_handoff = LeftFrontHandoff(1, "cpu")
        self.use_support_progress = False
        self.fixed_enabled = False
        self.fixed_gait = FixedStoneGait(1, "cpu")
        self.handoff_active = torch.ones(1, dtype=torch.bool)
        self.reset()

    def _top_centers(self, ids):
        if np.any(self.model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("Stepping-stone platforms and stones must be box geoms")
        positions = self.data.geom_xpos[ids].copy()
        local_z = self.data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
        if not np.allclose(local_z, (0, 0, 1), atol=1e-5):
            raise ValueError("Stepping-stone tops must be horizontal")
        positions[:, 2] += self.model.geom_size[ids, 2]
        return positions

    def reset(self):
        self.start_time = self.data.time
        self.visited = np.zeros(MAX_STONES, dtype=bool)
        self.exit_visited = np.zeros(4, dtype=bool)
        self.stone_steps = np.zeros((4, self.count), dtype=int)
        self.exit_steps = np.zeros(4, dtype=int)
        self.failed = False
        self.finished = False
        self.settling = False
        self.settle_timed_out = False
        self.settle_substeps = 0
        self.stable_substeps = 0
        self.left_front_handoff.reset(slice(None))
        self.fixed_gait.reset(slice(None))
        self.last_surface = [self.entry_id] * len(self.feet)
        self.was_supported = [False] * len(self.feet)
        self.swing_target = [None] * len(self.feet)

    def _foot_contacts(self):
        nearest = [None] * len(self.feet)
        for index, contact in enumerate(self.data.contact):
            first, second = contact.geom
            if first in self.feet and self.terrain_geoms[second]:
                foot, surface, normal = self.feet[first], second, contact.frame[2]
            elif second in self.feet and self.terrain_geoms[first]:
                foot, surface, normal = self.feet[second], first, -contact.frame[2]
            else:
                continue
            if contact.efc_address < 0 or (nearest[foot] is not None and nearest[foot][0] <= contact.dist):
                continue
            nearest[foot] = (contact.dist, surface, normal, contact.pos, index)
        return nearest

    def _shin_load(self):
        load = np.zeros(len(LEG_NAMES))
        strongest_rr = -1.0
        rr_contact = None
        for index, contact in enumerate(self.data.contact):
            first, second = contact.geom
            if first in self.shins and second in self.raised_surfaces:
                leg, surface, sign = self.shins[first], second, -1
            elif second in self.shins and first in self.raised_surfaces:
                leg, surface, sign = self.shins[second], first, 1
            else:
                continue
            if contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, self.contact_force)
            vertical = max(0., sign * (self.contact_force[:3] @ contact.frame.reshape(3, 3)[:, 2]))
            load[leg] += vertical
            if leg == RIGHT_REAR and vertical > strongest_rr:
                strongest_rr = vertical
                rr_contact = (self.model.geom(surface).name, *contact.pos)
        return load, rr_contact

    def _right_foot_trace(self, foot, contact, normal_force, supported):
        geom = self.foot_geoms[foot]
        position = self.data.geom_xpos[geom]
        x = position[:2] @ self.direction
        previous = self.stone_indices.get(self.last_surface[foot], -1)
        if supported and self.was_supported[foot] and previous >= 0:
            target = self.stone_names[previous]
            target_x = self.stone_x[previous]
        else:
            if self.swing_target[foot] is None:
                self.swing_target[foot] = self._next_right_foot_target(self.last_surface[foot], previous)
            target, target_x = self.swing_target[foot]
        foot_contact = (self.model.geom(contact[1]).name, *contact[3]) if contact is not None else ("", "", "", "")
        if supported:
            self.last_surface[foot] = contact[1]
            self.swing_target[foot] = None
        self.was_supported[foot] = supported
        return (
            x, position[2] - self.model.geom_size[geom, 0], target, target_x, x - target_x, normal_force,
            foot_contact[0], int(supported), *foot_contact[1:],
        )

    def _next_right_foot_target(self, last_surface, previous):
        candidates = self.right_stones[self.right_stones > previous]
        if last_surface != self.exit_id and candidates.size:
            stone = candidates[0]
            return self.stone_names[stone], self.stone_x[stone]
        return self.model.geom(self.exit_id).name, self.platforms[1, :2] @ self.direction

    def _trace_feet(self, contacts, normal_force, supported, shin_load, rr_shin_contact, ground_feet, left_front_trace):
        rr = self._right_foot_trace(RIGHT_REAR, contacts[RIGHT_REAR], normal_force[RIGHT_REAR], supported[RIGHT_REAR])
        fr = self._right_foot_trace(RIGHT_FRONT, contacts[RIGHT_FRONT], normal_force[RIGHT_FRONT], supported[RIGHT_FRONT])
        fr_stone, fr_offset = "", ""
        contact = contacts[RIGHT_FRONT]
        if not self.failed and supported[RIGHT_FRONT]:
            stone = self.stone_indices.get(contact[1])
            confirmed = math.ceil(STONE_CONTACT_TIME_S / self.model.opt.timestep)
            if stone is not None and self.stone_steps[RIGHT_FRONT, stone] == confirmed:
                fr_stone = self.stone_names[stone]
                fr_offset = contact[3][:2] @ self.direction - self.stone_x[stone]
        self.trace((
            self.data.time, self.data.time - self.start_time, *rr,
            shin_load[RIGHT_REAR], *(rr_shin_contact or ("", "", "", "")),
            " ".join(ground_feet), int(self.failed), int(self.finished),
            *fr, shin_load[RIGHT_FRONT], fr_stone, fr_offset,
            int(self.settling), self.stable_substeps * self.model.opt.timestep, int(self.settle_timed_out),
            normal_force[LEFT_FRONT], shin_load[LEFT_FRONT], int(supported[LEFT_FRONT]), *left_front_trace,
            int(self.left_front_handoff.first_right_rear_step_complete.item()),
            int(self.left_front_handoff.right_rear_follow_pending.item()),
        ))

    def _record_exit_settle(self, on_exit, clean_support):
        ordered = not self.fixed_enabled or self.fixed_gait.done.item()
        self.settling |= self.visited[:self.count].all() and self.exit_visited.all() and ordered
        if not self.settling:
            return
        dt = self.model.opt.timestep
        self.settle_substeps += 1
        stable = (on_exit & clean_support).all() and self._body_is_stable()
        self.stable_substeps = self.stable_substeps + 1 if stable else 0
        self.finished = self.stable_substeps >= math.ceil(EXIT_SETTLE_TIME_S / dt)
        self.settle_timed_out = not self.finished and self.settle_substeps >= math.ceil(EXIT_SETTLE_TIMEOUT_S / dt)
        self.failed |= self.settle_timed_out

    def _body_is_stable(self):
        height = self.data.xpos[self.body_id, 2] - self.platforms[1, 2]
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        tilt = math.atan2(np.linalg.norm(rotation[2, :2]), rotation[2, 2])
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.body_id, self.body_velocity, 0)
        joint_error = np.max(np.abs(self.data.qpos[self.joint_qpos_ids] - self.standing_joint_pos))
        return (
            abs(height - self.standing_height) <= EXIT_HEIGHT_TOL_M and tilt <= EXIT_TILT_TOL_RAD
            and np.linalg.norm(self.body_velocity[3:]) <= EXIT_LINEAR_SPEED_MPS
            and np.linalg.norm(self.body_velocity[:3]) <= EXIT_ANGULAR_SPEED_RADPS
            and joint_error <= EXIT_JOINT_ERROR_RAD
        )

    def record_substep(self):
        if self.failed or self.finished:
            return
        on_stone = np.zeros_like(self.stone_steps, dtype=bool)
        on_exit = np.zeros_like(self.exit_steps, dtype=bool)
        support_surface = np.full(len(self.feet), -1, dtype=np.int64)
        contacts = self._foot_contacts()
        shin_load, rr_shin_contact = self._shin_load()
        normal_force = np.zeros(len(self.feet))
        clean_support = np.zeros(len(self.feet), dtype=bool)
        ground_feet = []
        for foot, contact in enumerate(contacts):
            if contact is None:
                continue
            _, surface, normal_z, position, index = contact
            mujoco.mj_contactForce(self.model, self.data, index, self.contact_force)
            normal_force[foot] = self.contact_force[0]
            if position[2] <= self.floor_height + GROUND_CONTACT_Z_TOL:
                self.failed = True
                ground_feet.append(LEG_NAMES[foot])
                continue
            stone = self.stone_indices.get(surface)
            top = self.stones[stone] if stone is not None else self.platforms[int(surface == self.exit_id)]
            if surface not in self.raised_surfaces or normal_z > -MIN_TOP_NORMAL_Z:
                continue
            if abs(position[2] - top[2]) > TOP_CONTACT_Z_TOL:
                continue
            delta = position - self.data.geom_xpos[surface]
            local_xy = delta @ self.data.geom_xmat[surface].reshape(3, 3)[:, :2]
            if np.any(np.abs(local_xy) > self.model.geom_size[surface, :2] + TOP_EDGE_TOL):
                continue
            if stone is not None:
                support_surface[foot] = stone + 1
            else:
                support_surface[foot] = EXIT_SURFACE if surface == self.exit_id else 0
            clean_support[foot] = normal_force[foot] >= MIN_FOOT_SUPPORT_N and shin_load[foot] < SHIN_SUPPORT_N
            if surface == self.exit_id:
                on_exit[foot] = True
            elif stone is not None and clean_support[foot]:
                on_stone[foot, stone] = True
        left_front_trace = ("", "", "", "", "", "")
        if not self.failed:
            dt = self.model.opt.timestep
            target = EXIT_SURFACE if self.settling else self.left_front_handoff.left_front_target_surface(self.left_stones)
            holding, violation, blocked_landing, _ = self.left_front_handoff.update(
                torch.from_numpy(support_surface)[None], torch.from_numpy(clean_support)[None], self.handoff_active, dt,
                traversing=self.handoff_active & (not self.settling), left_front_target=target,
            )
            blocked = blocked_landing.item()
            self.stone_steps = np.where(on_stone, self.stone_steps + 1, 0)
            touchdowns = self.stone_steps == math.ceil(STONE_CONTACT_TIME_S / dt)
            if self.fixed_enabled:
                self._update_fixed_gait(support_surface, clean_support, dt)
                touchdowns = on_stone & self.fixed_gait.correct_landing[0].numpy()[:, None]
                blocked = False
            if self.trace is not None:
                lf_stones = np.flatnonzero(touchdowns[LEFT_FRONT])
                lf_stone, lf_eligible, lf_covered = "", "", ""
                if lf_stones.size:
                    stone = lf_stones[0]
                    lf_stone = self.stone_names[stone]
                    lf_eligible = int(not blocked)
                    lf_covered = int(not blocked and not self.visited[stone])
                left_front_trace = (int(holding.item()), int(violation.item()), int(blocked), lf_stone, lf_eligible, lf_covered)
            touchdowns[LEFT_FRONT] &= not blocked
            self.exit_steps = np.where(on_exit, self.exit_steps + 1, 0)
            self.visited[:self.count] |= touchdowns.any(axis=0)
            self.exit_visited |= self.exit_steps >= math.ceil(EXIT_CONTACT_TIME_S / dt)
            self._record_exit_settle(on_exit, clean_support)
        if self.trace is not None:
            self._trace_feet(contacts, normal_force, clean_support, shin_load, rr_shin_contact, ground_feet, left_front_trace)

    def _course_points(self, points):
        relative = points - self.platforms[0]
        lateral = np.array((-self.direction[1], self.direction[0]))
        result = relative.copy()
        result[..., 0] = relative[..., :2] @ self.direction + self.model.geom_size[self.entry_id, 0]
        result[..., 1] = relative[..., :2] @ lateral
        result[..., 2] = relative[..., 2] + .20
        return result

    def _fixed_inputs(self):
        sites = np.array([self.data.site_xpos[self.model.site(f"{leg}_foot_site").id] for leg in LEG_NAMES])
        rotations = np.array([self.data.site_xmat[self.model.site(f"{leg}_foot_site").id] for leg in LEG_NAMES]).reshape(4, 3, 3)
        sites += .025 * (rotations[:, :, 2] - np.array((0., 0., 1.)))
        positions = torch.tensor(self._course_points(sites), dtype=torch.float32)[None]
        stones = torch.tensor(self._course_points(self.stones), dtype=torch.float32)[None]
        exit_start = self._course_points(self.platforms[1])[0] - self.model.geom_size[self.exit_id, 0]
        return positions, stones, torch.tensor([exit_start], dtype=torch.float32)

    def _update_fixed_gait(self, surface, support, dt):
        positions, stones, exit_start = self._fixed_inputs()
        self.fixed_gait.update(positions, torch.tensor(surface)[None], torch.tensor(support)[None],
                               torch.tensor([self.count]), stones, exit_start, self.handoff_active & (not self.settling), dt,
                               stone_half_width=self.stone_half_width)

    def command(self, requested_speed):
        if self.failed or self.settling or requested_speed <= 0:
            return 0.0
        return float(np.clip(requested_speed, *STEPPING_FORWARD_SPEED_RANGE))

    def refresh_observations(self, episode_length_s):
        root = self.data.xpos[self.body_id]
        heading = self.data.xmat[self.body_id].reshape(3, 3)[:, 0]
        yaw = np.arctan2(heading[1], heading[0])
        if self.fixed_enabled:
            positions, stones, exit_start = self._fixed_inputs()
            active = self.handoff_active & (not self.failed and not self.settling and not self.finished)
            self.fixed_gait.initialize(positions, torch.tensor([self.count]), stones, exit_start, active)
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_heading = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        stage = (0, 0, 0, 1) if self.settling else (0, 0, 1, 0)
        settle_progress = (self.stable_substeps / math.ceil(EXIT_SETTLE_TIME_S / self.model.opt.timestep)
                           if self.settling else self.exit_visited.mean())
        remaining = np.clip(1 - (self.data.time - self.start_time) / episode_length_s, 0, 1)
        if self.use_support_progress:
            foot_contact = self.left_front_handoff.support_progress(self.model.opt.timestep)[0]
        else:
            foot_contact = [contact is not None for contact in self._foot_contacts()]
        self.values = {
            "stone_relative_positions": ((self.stones - root) @ world_to_heading).reshape(-1),
            "active_stone_mask": np.arange(MAX_STONES) < self.count,
            "visited_stone_mask": self.visited,
            "platform_relative_positions": ((self.platforms - root) @ world_to_heading).reshape(-1),
            "stage": stage,
            "strict_ground_check": (1.0,),
            "foot_contact": foot_contact,
            "settle_progress": (settle_progress,),
            "route_direction": self.direction @ world_to_heading[:2, :2],
            "navigation_valid": (1.0,),
            "remaining_time": (remaining,),
        }
        course_yaw = np.arctan2(self.direction[1], self.direction[0])
        self.values["fixed_gait"] = self.fixed_gait.observation(
            torch.tensor([self.count]), torch.tensor(self._course_points(root), dtype=torch.float32)[None],
            torch.tensor([yaw - course_yaw], dtype=torch.float32),
        )[0].numpy()
