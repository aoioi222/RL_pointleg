"""Stair-specific metrics and hip alignment constraints."""

import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from ..robot_cfg import LEG_NAMES
from ..terrains.elevation import STAIR_TREAD_COUNT, STEPS_PER_FLIGHT, STEP_DEPTH
from .elevation import elevation_foot_bottom, elevation_state, stairs_exit_supported_feet, update_elevation_state
from .rewards import hip_abduction_excess


def stairs_task_metric(env, key):
    """Expose stair coverage, hard-clearance failures and loaded destination feet."""
    state = elevation_state(env)
    if key in ("exit_loaded_feet", "middle_loaded_feet"):
        return stairs_exit_supported_feet(env).sum(dim=1).float()
    if key == "covered_treads":
        return state["covered"][:, :STEPS_PER_FLIGHT].sum(dim=1).float()
    if key == "clearance_failed":
        return state["stairs_clearance_failed"].float()
    if key in ("up_fraction", "down_fraction"):
        kind = 0 if key == "up_fraction" else 1
        return (env.scene.terrain.terrain_types == kind).float()
    raise KeyError(key)


def stairs_hip_abduction_zero(env, asset_cfg, tolerance=0., deviation_scale=.25):
    """Keep all abduction joints at their zero-degree default during stair traversal."""
    cost = hip_abduction_excess(env, asset_cfg, tolerance, deviation_scale)
    robot = env.scene[asset_cfg.name]
    for leg in LEG_NAMES:
        index = robot.joint_names.index(f"{leg}_hip_abduction_joint")
        angle = torch.rad2deg(robot.data.joint_pos[:, index])
        env.extras["log"][f"Metrics/stairs_hip_abduction_{leg}_deg"] = angle.detach()
    return cost


class StairsRightFrontHipLimitPenalty:
    """Penalize right-front thigh swing ahead of the body-down reference."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.hip, _ = robot.find_bodies("fr_hip_pitch_Link")
        self.knee, _ = robot.find_bodies("fr_knee_Link")
        if len(self.hip) != 1 or len(self.knee) != 1:
            raise ValueError("Expected one right-front hip and knee body")

    def __call__(self, env, maximum_forward_degrees=20., scale_degrees=10.):
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        vector = robot.body_link_pos_w[:, self.knee[0]] - robot.body_link_pos_w[:, self.hip[0]]
        quat = robot.root_link_quat_w
        finite = torch.isfinite(vector).all(dim=1) & torch.isfinite(quat).all(dim=1)
        local = quat_apply_inverse(quat.nan_to_num(), vector.nan_to_num())
        angle = torch.atan2(local[:, 0], -local[:, 2])
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"] & finite
        excess = ((angle - math.radians(maximum_forward_degrees)).clamp_min(0.) / math.radians(scale_degrees))
        if active.any():
            env.extras["log"]["Metrics/stairs_fr_hip_forward_deg"] = torch.rad2deg(angle[active]).detach().clone()
        return torch.where(active, excess.square().clamp_max(4.), 0.)


class StairsRightFrontKneeFlexLimitPenalty:
    """Penalize right-front extension past the minimum normal knee flexion."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.hip, _ = robot.find_bodies("fr_hip_pitch_Link")
        self.knee, _ = robot.find_bodies("fr_knee_Link")
        self.foot, _ = robot.find_sites(("fr_foot_site",), preserve_order=True)
        if len(self.hip) != 1 or len(self.knee) != 1 or len(self.foot) != 1:
            raise ValueError("Expected one right-front hip, knee and foot")

    def __call__(self, env, minimum_flexion_degrees=30., scale_degrees=10.):
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        thigh = robot.body_link_pos_w[:, self.knee[0]] - robot.body_link_pos_w[:, self.hip[0]]
        shin = robot.site_pos_w[:, self.foot[0]] - robot.body_link_pos_w[:, self.knee[0]]
        quat = robot.root_link_quat_w
        finite = (torch.isfinite(thigh).all(dim=1) & torch.isfinite(shin).all(dim=1)
                  & torch.isfinite(quat).all(dim=1))
        thigh = quat_apply_inverse(quat.nan_to_num(), thigh.nan_to_num())
        shin = quat_apply_inverse(quat.nan_to_num(), shin.nan_to_num())
        cross = thigh[:, 0] * shin[:, 2] - thigh[:, 2] * shin[:, 0]
        dot = thigh[:, 0] * shin[:, 0] + thigh[:, 2] * shin[:, 2]
        flexion = torch.atan2(cross, dot)
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"] & finite
        deficit = ((math.radians(minimum_flexion_degrees) - flexion).clamp_min(0.) / math.radians(scale_degrees))
        if active.any():
            env.extras["log"]["Metrics/stairs_fr_knee_flexion_deg"] = torch.rad2deg(flexion[active]).detach().clone()
        return torch.where(active, deficit.square().clamp_max(4.), 0.)


class StairsRearThighAboveBodyPenalty:
    """Discourage rear knees rising above their hip in the body frame."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.hips, self.knees = [], []
        for leg in ("rl", "rr"):
            for suffix, indices in (("hip_pitch_Link", self.hips), ("knee_Link", self.knees)):
                found, _ = robot.find_bodies(f"{leg}_{suffix}")
                if len(found) != 1:
                    raise ValueError(f"Expected one {leg}_{suffix} body")
                indices.append(found[0])

    def __call__(self, env, scale_degrees=20.):
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        vector = robot.body_link_pos_w[:, self.knees] - robot.body_link_pos_w[:, self.hips]
        quat = robot.root_link_quat_w
        finite = torch.isfinite(vector).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=1)
        local = quat_apply_inverse(quat.nan_to_num()[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
                                   vector.nan_to_num().reshape(-1, 3)).reshape(env.num_envs, 2, 3)
        angle = torch.atan2(local[..., 2], local[..., 0].abs().clamp_min(1.e-6))
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"] & finite
        for index, leg in enumerate(("rl", "rr")):
            if active.any():
                env.extras["log"][f"Metrics/stairs_rear_thigh_above_body_{leg}_deg"] = torch.rad2deg(angle[active, index]).detach().clone()
        cost = (angle.clamp_min(0.) / math.radians(scale_degrees)).square().clamp_max(1.).mean(dim=1)
        return torch.where(active, cost, 0.)


class StairsFrontSupportPenalty:
    """Discourage prolonged simultaneous loss of front-foot support."""

    def __init__(self, cfg, env):
        self.age = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.cached = torch.zeros_like(self.age)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.age[ids] = self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., grace=0., rise_time=.12):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        data = env.scene["elevation_contacts"].data
        found = data.found.reshape(env.num_envs, 4, -1) > 0
        force = data.force.reshape(env.num_envs, 4, -1, 3)
        normal = data.normal.reshape(env.num_envs, 4, -1, 3)
        loaded = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                  & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
        both = active & ~loaded[:, 0] & ~loaded[:, 1]
        self.age.copy_(torch.where(fresh, torch.where(both, self.age + env.step_dt, 0.), self.age))
        penalty = torch.where(both, .5 + .5 * ((self.age - grace) / rise_time).clamp(0., 1.), 0.)
        self.cached.copy_(torch.where(fresh, penalty, self.cached))
        self.last_step[fresh] = env.common_step_counter
        env.extras["log"]["Metrics/stairs_front_unloaded_s"] = self.age.detach().clone()
        return self.cached


class StairsFrontStepSequencePenalty:
    """Penalize front-foot takeoff before the other front foot is stably loaded."""

    def __init__(self, cfg, env):
        self.age = torch.zeros(env.num_envs, 2, device=env.device)
        self.loaded = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.anchored = torch.zeros_like(self.loaded)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("age", "loaded", "anchored", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., stable_time=.10, event_seconds=.10):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        data = env.scene["elevation_contacts"].data
        found = data.found.reshape(env.num_envs, 4, -1)[:, :2] > 0
        force = data.force.reshape(env.num_envs, 4, -1, 3)[:, :2]
        normal = data.normal.reshape(env.num_envs, 4, -1, 3)[:, :2]
        loaded = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                  & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
        mask = fresh & active
        takeoff = mask[:, None] & self.anchored & self.loaded & ~loaded
        other_age = self.age.flip(dims=[1])
        other_loaded = loaded.flip(dims=[1])
        premature = takeoff & (~other_loaded | (other_age < stable_time - 1.e-6))
        self.age.copy_(torch.where(fresh[:, None], torch.where(active[:, None] & loaded, self.age + env.step_dt, 0.), self.age))
        self.loaded.copy_(torch.where(fresh[:, None], loaded & active[:, None], self.loaded))
        self.anchored.copy_(torch.where(fresh[:, None], (self.anchored | loaded) & active[:, None], self.anchored))
        cost = premature.float().sum(dim=1) * event_seconds / env.step_dt
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        env.extras["log"]["Metrics/stairs_front_premature_takeoff"] = premature.any(dim=1).float().detach()
        return self.cached


class StairsFrontReachPenalty:
    """Penalize excessive forward reach of each front foot relative to its hip."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.feet = robot.find_sites(tuple(f"{leg}_foot_site" for leg in ("fl", "fr")), preserve_order=True)[0]
        self.hips = []
        for leg in ("fl", "fr"):
            ids, _ = robot.find_bodies(f"{leg}_hip_pitch_Link")
            if len(ids) != 1:
                raise ValueError(f"Expected one {leg} hip body")
            self.hips.append(ids[0])

    def __call__(self, env, tolerance=.22, scale=.12):
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        offset = robot.site_pos_w[:, self.feet] - robot.body_link_pos_w[:, self.hips]
        quat = robot.root_link_quat_w
        finite = torch.isfinite(offset).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=1)
        local = quat_apply_inverse(quat.nan_to_num()[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
                                   offset.nan_to_num().reshape(-1, 3)).reshape(env.num_envs, 2, 3)
        reach = local[..., 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"] & finite
                  & (env.scene.terrain.terrain_types == 0))
        for index, leg in enumerate(("fl", "fr")):
            if active.any():
                env.extras["log"][f"Metrics/stairs_front_reach_{leg}_m"] = reach[active, index].detach().clone()
        cost = ((reach - tolerance).clamp_min(0.) / scale).square().clamp_max(1.).mean(dim=1)
        return torch.where(active, cost, 0.)


class StairsShinContactPenalty:
    """Discourage loading the shin against terrain during stair descent."""

    def __init__(self, cfg, env):
        sensor = env.scene["elevation_body_contacts"]
        names = [name.split("/")[-1] for name in sensor.primary_names]
        self.indices = []
        for leg in LEG_NAMES:
            name = f"{leg}_knee_Link"
            if names.count(name) != 1:
                raise ValueError(f"Expected one stair body contact sensor for {name}")
            self.indices.append(names.index(name))
        self.slots = sensor.cfg.num_slots

    def __call__(self, env, threshold=2., force_scale=18.):
        state = update_elevation_state(env)
        data = env.scene["elevation_body_contacts"].data
        found = data.found.reshape(env.num_envs, -1, self.slots)[:, self.indices] > 0
        force = data.force.reshape(env.num_envs, -1, self.slots, 3)[:, self.indices]
        valid = found & torch.isfinite(force).all(dim=-1)
        magnitude = torch.where(valid, force.nan_to_num().norm(dim=-1), 0.).amax(dim=2)
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & (env.scene.terrain.terrain_types == 1))
        cost = ((magnitude - threshold).clamp_min(0.) / force_scale).clamp_max(1.)
        for index, leg in enumerate(LEG_NAMES):
            env.extras["log"][f"Metrics/stairs_down_shin_force_{leg}_n"] = torch.where(active, magnitude[:, index], 0.).detach()
        env.extras["log"]["Metrics/stairs_down_shin_contact_feet"] = torch.where(active, (magnitude > threshold).sum(dim=1), 0.).float().detach()
        return torch.where(active, cost.amax(dim=1), 0.)


class StairsSwingSpeedPenalty:
    """Penalize excessive sagittal joint speeds only while the foot is unloaded."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.joints = [robot.joint_names.index(f"{leg}_{joint}_joint")
                       for leg in LEG_NAMES for joint in ("hip_pitch", "knee")]

    def __call__(self, env, minimum_force=8., hip_limit=4., knee_limit=6., speed_scale=4.):
        state = update_elevation_state(env)
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        valid = torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
        loaded = (found & valid & (normal[..., 2] < -.5)
                  & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        vel = env.scene["robot"].data.joint_vel[:, self.joints].reshape(env.num_envs, 4, 2)
        finite = torch.isfinite(vel).all(dim=(1, 2))
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"] & finite
        swing = active[:, None] & ~loaded
        speed = vel.nan_to_num().abs()
        limits = speed.new_tensor([hip_limit, knee_limit])
        cost = (((speed - limits).clamp_min(0.) / speed_scale).square().clamp_max(1.).mean(dim=2)
                * swing).mean(dim=1)
        for index, leg in enumerate(LEG_NAMES):
            if swing[:, index].any():
                for joint, name in enumerate(("hip", "knee")):
                    env.extras["log"][f"Metrics/stairs_swing_{name}_speed_{leg}_rad_s"] = speed[swing[:, index], index, joint].detach().clone()
        return cost


class StairsLeftFrontTrackingPenalty:
    """Penalize left-front and right-rear pose differences in comparable phases."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.joints = [robot.joint_names.index(f"{leg}_{joint}_joint")
                       for leg in ("fl", "rr") for joint in ("hip_pitch", "knee")]
        self.feet = robot.find_sites(tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True)[0]

    def __call__(self, env, minimum_force=8., tolerance_degrees=25., scale_degrees=30., unequal_height=.03, terrain_weight=.25):
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        q = robot.joint_pos[:, self.joints].reshape(env.num_envs, 2, 2)
        feet = elevation_foot_bottom(env, self.feet)
        command = env.command_manager.get_command("twist")[:, 0]
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        support = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                   & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        finite = torch.isfinite(q).all(dim=(1, 2)) & torch.isfinite(feet).all(dim=(1, 2)) & torch.isfinite(command)
        comparable = (support[:, 0] == support[:, 3]) & (support.sum(dim=1) >= 2)
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & finite & (command > .08) & comparable)
        difference = torch.atan2(torch.sin(q[:, 0] - q[:, 1]), torch.cos(q[:, 0] - q[:, 1])).nan_to_num()
        cost = ((difference.abs() - math.radians(tolerance_degrees)).clamp_min(0.)
                / math.radians(scale_degrees)).square().clamp_max(1.).mean(dim=1)
        mixed = (feet[:, 0, 2] - feet[:, 3, 2]).abs() > unequal_height
        cost *= torch.where(mixed, terrain_weight, 1.)
        if active.any():
            for index, joint in enumerate(("hip", "knee")):
                env.extras["log"][f"Metrics/stairs_lf_rr_{joint}_error_deg"] = torch.rad2deg(difference[active, index].abs()).detach().clone()
        return torch.where(active, cost, 0.)


class StairsThighAlignmentPenalty:
    """Keep thigh inclination within a soft band above the stair gradient."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.hips, self.knees = [], []
        for leg in LEG_NAMES:
            for suffix, ids in (("hip_pitch_Link", self.hips), ("knee_Link", self.knees)):
                found, _ = robot.find_bodies(f"{leg}_{suffix}")
                if len(found) != 1:
                    raise ValueError(f"Expected one {leg}_{suffix} body")
                ids.append(found[0])

    def __call__(self, env, offset_degrees=15., tolerance_degrees=10., scale_degrees=30., approach_margin=.4):
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        vector = robot.body_link_pos_w[:, self.knees] - robot.body_link_pos_w[:, self.hips]
        quat = robot.root_link_quat_w
        finite = (torch.isfinite(vector).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=1)
                  & torch.isfinite(state["top_height"]) & (state["top_height"] > 0.))
        local = quat_apply_inverse(quat.nan_to_num()[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
                                   vector.nan_to_num().reshape(-1, 3)).reshape(env.num_envs, 4, 3)
        body_angle = torch.atan2(local[..., 2].abs(), local[..., 0].abs().clamp_min(1.e-6))
        angle = torch.atan2(vector[..., 2].abs(), vector[..., 0].abs().clamp_min(1.e-6))
        slope = torch.atan2(state["top_height"].nan_to_num() / STAIR_TREAD_COUNT,
                            torch.full_like(state["top_height"], STEP_DEPTH))
        target = slope[:, None] + math.radians(offset_degrees)
        local_x = robot.root_link_pos_w[:, 0] - state["corner"][:, 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"] & finite
                  & (env.scene.terrain.terrain_types == 0)
                  & (local_x >= state["entry_x"] - approach_margin)
                  & (local_x <= state["exit_x"] + approach_margin))
        excess = ((angle.sub(target).abs() - math.radians(tolerance_degrees)).clamp_min(0.)
                  / math.radians(scale_degrees)).square().clamp_max(1.)
        for index, leg in enumerate(LEG_NAMES):
            if active.any():
                env.extras["log"][f"Metrics/stairs_thigh_body_angle_{leg}_deg"] = torch.rad2deg(body_angle[active, index]).detach().clone()
                env.extras["log"][f"Metrics/stairs_thigh_world_angle_{leg}_deg"] = torch.rad2deg(angle[active, index]).detach().clone()
        if active.any():
            env.extras["log"]["Metrics/stairs_thigh_target_angle_deg"] = torch.rad2deg(target[active, 0]).detach().clone()
        return torch.where(active, excess.mean(dim=1), 0.)


class StairsSwingKneeFlexReward:
    """Reward flexion during genuine, supported forward swing."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.joints = [robot.joint_names.index(f"{leg}_knee_joint") for leg in LEG_NAMES]
        self.feet = robot.find_sites(tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True)[0]
        self.support_z = torch.zeros(env.num_envs, 4, device=env.device)
        self.air_time = torch.zeros_like(self.support_z)
        self.anchored = torch.zeros_like(self.support_z, dtype=torch.bool)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("support_z", "air_time", "anchored", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., start_flexion=1.125, target_flexion=1.6,
                 minimum_lift=.015, maximum_air=.45, minimum_speed=.05, target_speed=.15):
        if not (0. < start_flexion < target_flexion and 0. < minimum_speed < target_speed
                and minimum_force > 0. and minimum_lift > 0. and maximum_air > 0.):
            raise ValueError("Invalid stair swing knee reward parameters")
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        support = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                   & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        z = elevation_foot_bottom(env, self.feet)[..., 2]
        q = robot.joint_pos[:, self.joints]
        velocity = robot.root_link_lin_vel_b[:, 0]
        command = env.command_manager.get_command("twist")[:, 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & torch.isfinite(z).all(dim=1) & torch.isfinite(q).all(dim=1)
                  & torch.isfinite(velocity) & torch.isfinite(command) & (command > .08))
        mask = fresh & active
        self.support_z.copy_(torch.where(mask[:, None] & support, z, self.support_z))
        self.anchored |= mask[:, None] & support
        airborne = ~found.any(dim=2) & self.anchored
        self.air_time.copy_(torch.where(mask[:, None], torch.where(airborne, self.air_time + env.step_dt, 0.), self.air_time))
        eligible = (mask[:, None] & airborne & (self.air_time <= maximum_air)
                    & ((z - self.support_z) >= minimum_lift) & (support.sum(dim=1) >= 2)[:, None])
        flex = ((-q - start_flexion) / (target_flexion - start_flexion)).clamp(0., 1.)
        speed_factor = ((velocity - minimum_speed) / (target_speed - minimum_speed)).clamp(0., 1.)
        reward = (flex * eligible).mean(dim=1) * speed_factor
        self.cached.copy_(torch.where(fresh, torch.where(active, reward, 0.), self.cached))
        self.last_step[fresh] = env.common_step_counter
        for index, leg in enumerate(LEG_NAMES):
            valid = eligible[:, index] & (speed_factor > 0.)
            if valid.any():
                env.extras["log"][f"Metrics/stairs_swing_knee_{leg}_deg"] = torch.rad2deg(q[valid, index]).detach().clone()
        return self.cached


class StairsSwingLiftPenalty:
    """Penalize insufficient foot lift from the last loaded position."""

    def __init__(self, cfg, env):
        self.foot_ids = env.scene["robot"].find_sites(tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True)[0]
        shape = (env.num_envs, 4)
        self.support_z = torch.zeros(shape, device=env.device)
        self.peak = torch.zeros_like(self.support_z)
        self.air_time = torch.zeros_like(self.support_z)
        self.loaded = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.anchored = torch.zeros_like(self.loaded)
        self.swinging = torch.zeros_like(self.loaded)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("support_z", "peak", "air_time", "loaded", "anchored", "swinging", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., minimum_air_time=.04,
                 grace_time=.08, event_seconds=.10):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        sensor = env.scene["elevation_contacts"].data
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        loaded = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                  & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        z = elevation_foot_bottom(env, self.foot_ids)[..., 2]
        command = env.command_manager.get_command("twist")[:, 0]
        target_lift = state["top_height"] / STAIR_TREAD_COUNT
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & torch.isfinite(z).all(dim=1) & torch.isfinite(command) & (command > .08)
                  & torch.isfinite(target_lift) & (target_lift > 0.))
        mask = fresh & active
        takeoff = mask[:, None] & self.loaded & ~loaded & self.anchored
        landing = mask[:, None] & loaded & self.swinging
        peak = torch.where(takeoff, (z - self.support_z).clamp_min(0.), self.peak)
        observing = mask[:, None] & (self.swinging | takeoff)
        peak = torch.where(observing, torch.maximum(peak, (z - self.support_z).clamp_min(0.)), peak)
        air_time = torch.where(takeoff, env.step_dt, torch.where(observing & ~loaded, self.air_time + env.step_dt, self.air_time))
        target = target_lift.nan_to_num(nan=0., posinf=0., neginf=0.).clamp_min(1.e-6)[:, None]
        deficit = ((target - peak) / target).clamp(0., 1.).square()
        event = landing & (air_time >= minimum_air_time - 1.e-6)
        ongoing = observing & ~loaded & (air_time >= grace_time)
        cost = (deficit * ongoing).mean(dim=1) + (deficit * event).mean(dim=1) * event_seconds / env.step_dt
        for index, leg in enumerate(LEG_NAMES):
            valid = event[:, index]
            if valid.any():
                env.extras["log"][f"Metrics/stairs_swing_lift_{leg}_m"] = peak[valid, index].detach().clone()
        self.support_z.copy_(torch.where(mask[:, None] & loaded, z, self.support_z))
        self.peak.copy_(torch.where(mask[:, None], torch.where(landing, 0., peak), self.peak))
        self.air_time.copy_(torch.where(mask[:, None], torch.where(landing, 0., air_time), self.air_time))
        self.swinging.copy_(torch.where(mask[:, None], (self.swinging | takeoff) & ~landing, self.swinging))
        self.anchored |= mask[:, None] & loaded
        self.loaded.copy_(torch.where(mask[:, None], loaded, self.loaded))
        self.cached.copy_(torch.where(fresh, torch.where(active, cost, 0.), self.cached))
        self.last_step[fresh] = env.common_step_counter
        if active.any():
            env.extras["log"]["Metrics/stairs_target_lift_m"] = target_lift[active].detach().clone()
        return self.cached


class StairsStuckRecoveryReward:
    """Pay bounded progress toward clearing a riser after a foot stalls nearby."""

    def __init__(self, cfg, env):
        self.feet = env.scene["robot"].find_sites(tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True)[0]
        shape = (env.num_envs, 4)
        self.anchor = torch.zeros(*shape, 3, device=env.device)
        self.root_anchor = torch.zeros(shape, device=env.device)
        self.age = torch.zeros(shape, device=env.device)
        self.duration = torch.zeros(shape, device=env.device)
        self.edge = torch.zeros(shape, device=env.device)
        self.top = torch.zeros(shape, device=env.device)
        self.right = torch.zeros(shape, device=env.device)
        self.best = torch.zeros(*shape, 3, device=env.device)
        self.watching = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.recovering = torch.zeros_like(self.watching)
        self.used = torch.zeros(*shape, STEPS_PER_FLIGHT, dtype=torch.bool, device=env.device)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("anchor", "root_anchor", "age", "duration", "edge", "top", "right", "best",
                     "watching", "recovering", "used", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., stall_time=.35, foot_motion=.012,
                 root_motion=.025, max_speed=.05, approach=.10, clearance=.02,
                 backward_distance=.03, landing_margin=.04, timeout=1.5):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        pos = sensor.pos.reshape(env.num_envs, 4, -1, 3) - state["corner"][:, None, None, :]
        valid = torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1) & torch.isfinite(pos).all(dim=-1)
        support = found & valid & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)
        loaded = support.any(dim=2)
        feet = elevation_foot_bottom(env, self.feet) - state["corner"][:, None, :]
        root = robot.root_link_pos_w[:, 0]
        speed = robot.root_link_lin_vel_b[:, 0]
        command = env.command_manager.get_command("twist")[:, 0]
        tread = state["step_surfaces"][:, :STEPS_PER_FLIGHT]
        finite = (torch.isfinite(feet).all(dim=(1, 2)) & torch.isfinite(root) & torch.isfinite(speed)
                  & torch.isfinite(command) & torch.isfinite(tread).all(dim=(1, 2)))
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & finite & (command > .08) & (env.scene.terrain.terrain_types == 0))
        near = ((tread[:, None, :, 0] - feet[..., 0, None] >= -.025)
                & (tread[:, None, :, 0] - feet[..., 0, None] <= approach)
                & (feet[..., 2, None] < tread[:, None, :, 2] + clearance)
                & (feet[..., 2, None] > tread[:, None, :, 2] - state["top_height"][:, None, None] / STAIR_TREAD_COUNT - .05)
                & ~self.used)
        index = near.float().argmax(dim=2)
        chosen = torch.gather(tread[:, None, :, :].expand(-1, 4, -1, -1), 2,
                              index[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        candidate = (active[:, None] & fresh[:, None] & ~loaded & (loaded.sum(dim=1) >= 2)[:, None]
                     & (speed.abs() < max_speed)[:, None] & near.any(dim=2) & ~self.recovering)
        moved = ((feet - self.anchor).norm(dim=2) > foot_motion) | ((root[:, None] - self.root_anchor).abs() > root_motion)
        start_watch = candidate & (~self.watching | moved)
        self.anchor.copy_(torch.where(start_watch[..., None], feet.nan_to_num(), self.anchor))
        self.root_anchor.copy_(torch.where(start_watch, root[:, None].nan_to_num(), self.root_anchor))
        self.age.copy_(torch.where(candidate, torch.where(start_watch, 0., self.age + env.step_dt), 0.))
        self.watching.copy_(candidate)
        trigger = candidate & (self.age >= stall_time - 1.e-6)
        self.edge.copy_(torch.where(trigger, chosen[..., 0], self.edge))
        self.right.copy_(torch.where(trigger, chosen[..., 1], self.right))
        self.top.copy_(torch.where(trigger, chosen[..., 2], self.top))
        self.duration.copy_(torch.where(trigger, 0., self.duration))
        self.best.copy_(torch.where(trigger[..., None], 0., self.best))
        self.used.scatter_(2, index[..., None], self.used.gather(2, index[..., None]) | trigger[..., None])
        self.recovering |= trigger

        running = self.recovering & active[:, None] & fresh[:, None]
        self.duration.copy_(torch.where(running, self.duration + env.step_dt, self.duration))
        lift_target = (self.top + clearance - self.anchor[..., 2]).clamp_min(.02)
        lift = ((feet[..., 2] - self.anchor[..., 2]) / lift_target).clamp(0., 1.)
        back = ((self.anchor[..., 0] - feet[..., 0]) / backward_distance).clamp(0., 1.)
        forward = ((feet[..., 0] - self.edge) / landing_margin).clamp(0., 1.)
        high = feet[..., 2] >= self.top + clearance
        scores = torch.stack((back, lift, forward * high), dim=2).nan_to_num()
        scores *= running[..., None] & ~loaded[..., None]
        best = torch.maximum(self.best, scores)
        increments = (best - self.best) * feet.new_tensor([.03, .12, .15])
        self.best.copy_(best)
        landed = (support & (pos[..., 0] >= self.edge[..., None] + landing_margin)
                  & (pos[..., 0] <= self.right[..., None])
                  & ((pos[..., 2] - self.top[..., None]).abs() <= env._elevation_settings.top_contact_tolerance)).any(dim=2)
        landed &= running & (self.best[..., 2] > 0.)
        reward = (increments.sum(dim=(1, 2)) + .15 * landed.sum(dim=1)) / env.step_dt
        self.recovering &= active[:, None] & ~loaded & (self.duration < timeout)
        self.cached.copy_(torch.where(fresh, reward, self.cached))
        self.last_step[fresh] = env.common_step_counter
        env.extras["log"]["Metrics/stairs_stuck_recovery_feet"] = self.recovering.float().sum(dim=1).detach()
        return self.cached


class StairsFootPlacement:
    """Score each foot for confirmed support inside a tread."""

    def __init__(self, cfg, env):
        self.age = torch.zeros(env.num_envs, 4, STAIR_TREAD_COUNT, device=env.device)
        self.paid = torch.zeros_like(self.age, dtype=torch.bool)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.age[ids] = 0.
        self.paid[ids] = False
        self.cached[ids] = 0.
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., confirmation=.10, target_margin=.08):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        pos = sensor.pos.reshape(env.num_envs, 4, -1, 3) - state["corner"][:, None, None, :]
        valid = torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1) & torch.isfinite(pos).all(dim=-1)
        support = found & valid & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)
        tread = state["step_surfaces"]
        x = pos[..., 0, None]
        left, right, height = (tread[:, None, None, :, i] for i in range(3))
        hit = (support[..., None] & (x >= left) & (x <= right)
               & ((pos[..., 2, None] - height).abs() <= env._elevation_settings.top_contact_tolerance))
        margin = torch.minimum(x - left, right - x).clamp_min(0.)
        quality = torch.where(hit, (margin / target_margin).clamp(0., 1.), 0.).amax(dim=2)
        touched = hit.any(dim=2)
        active = state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
        mask = fresh & active
        age = torch.where(touched, self.age + env.step_dt, 0.)
        self.age.copy_(torch.where(mask[:, None, None], age, self.age))
        ready = (self.age >= confirmation - 1.e-6) & ~self.paid & mask[:, None, None]
        reward = (quality.square() * ready).sum(dim=(1, 2)) / env.step_dt
        self.paid |= ready
        self.cached.copy_(torch.where(fresh, reward, self.cached))
        self.last_step[fresh] = env.common_step_counter
        return self.cached


class StairsFourFootStandstill:
    """Penalize sustained four-foot support without meaningful forward progress."""

    def __init__(self, cfg, env):
        self.anchor = torch.zeros(env.num_envs, device=env.device)
        self.age = torch.zeros_like(self.anchor)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.cached = torch.zeros_like(self.anchor)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.anchor[ids] = self.age[ids] = self.cached[ids] = 0.
        self.initialized[ids] = False
        self.last_step[ids] = -1

    def __call__(self, env, minimum_force=8., grace_period=1.0, rise_time=.5, minimum_progress=.02):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        sensor = env.scene["elevation_contacts"].data
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        loaded = (found & torch.isfinite(force).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
                  & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)).any(dim=2)
        x = env.scene["robot"].data.root_link_pos_w[:, 0]
        command = env.command_manager.get_command("twist")[:, 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & torch.isfinite(x) & torch.isfinite(command) & (command > .08))
        standing = active & loaded.all(dim=1)
        start = standing & ~self.initialized
        self.anchor.copy_(torch.where(fresh & start, x, self.anchor))
        advancing = standing & ((x - self.anchor) >= minimum_progress)
        self.anchor.copy_(torch.where(fresh & advancing, x, self.anchor))
        age = torch.where(standing & ~advancing, self.age + env.step_dt, 0.)
        self.age.copy_(torch.where(fresh, age, self.age))
        self.initialized.copy_(torch.where(fresh, standing, self.initialized))
        cost = ((self.age - grace_period) / rise_time).clamp(0., 1.)
        self.cached.copy_(torch.where(fresh, cost, self.cached))
        self.last_step[fresh] = env.common_step_counter
        env.extras["log"]["Metrics/stairs_four_foot_standstill_s"] = self.age.detach().mean()
        return self.cached


class StairsProgressReward:
    """Reward new body progress, loaded propulsion and stable tread contacts."""

    def __init__(self, cfg, env):
        self.foot_ids = env.scene["robot"].find_sites(tuple(f"{leg}_foot_site" for leg in LEG_NAMES), preserve_order=True)[0]
        self.loaded = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self.foot_start = torch.zeros(env.num_envs, 4, device=env.device)
        self.root_start = torch.zeros_like(self.foot_start)
        self.distance = torch.zeros_like(self.foot_start)
        self.stance = torch.zeros_like(self.foot_start)
        self.slip = torch.zeros_like(self.foot_start)
        self.tread_time = torch.zeros(env.num_envs, 6, device=env.device)
        self.tread_paid = torch.zeros(env.num_envs, 6, dtype=torch.bool, device=env.device)
        self.cached = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("loaded", "foot_start", "root_start", "distance", "stance", "slip", "tread_time", "tread_paid", "cached"):
            getattr(self, name)[ids] = 0
        self.last_step[ids] = -1

    def __call__(self, env, target_distance=.07, minimum_stance=.16, target_stance=.30,
                 max_slip=.025, minimum_force=8., tread_hold=.10, edge_margin=.03):
        fresh = self.last_step != env.common_step_counter
        if not fresh.any():
            return self.cached
        state = update_elevation_state(env)
        robot = env.scene["robot"].data
        sensor = env.scene["elevation_contacts"].data
        found = sensor.found.reshape(env.num_envs, 4, -1) > 0
        force = sensor.force.reshape(env.num_envs, 4, -1, 3)
        normal = sensor.normal.reshape(env.num_envs, 4, -1, 3)
        pos = sensor.pos.reshape(env.num_envs, 4, -1, 3)
        valid = torch.isfinite(force).all(dim=-1) & torch.isfinite(pos).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
        support = found & valid & (normal[..., 2] < -.5) & (force.nan_to_num().norm(dim=-1) >= minimum_force)
        loaded = support.any(dim=2)
        root = robot.root_link_pos_w[:, 0, None]
        foot = robot.site_pos_w[:, self.foot_ids, 0]
        active = (state["initialized"] & ~state["failed"] & ~state["invalid"] & ~state["complete"]
                  & torch.isfinite(root).all(dim=1) & torch.isfinite(foot).all(dim=1))
        mask = fresh & active
        landing = loaded & ~self.loaded & mask[:, None]
        takeoff = ~loaded & self.loaded & mask[:, None]
        self.root_start = torch.where(landing, root, self.root_start)
        self.foot_start = torch.where(landing, foot, self.foot_start)
        travel = (root - self.root_start).clamp_min(0.)
        sweep = (root - self.root_start - foot + self.foot_start).clamp_min(0.)
        distance = torch.minimum(travel, sweep)
        self.distance = torch.where(loaded & mask[:, None], distance, self.distance)
        slip = (foot - self.foot_start).abs()
        self.slip = torch.where(landing, 0., torch.where(loaded & mask[:, None], torch.maximum(self.slip, slip), self.slip))
        self.stance = torch.where(mask[:, None], torch.where(loaded, torch.where(landing, env.step_dt, self.stance + env.step_dt), self.stance), self.stance)
        quality = (self.distance / target_distance).clamp(0., 1.).square() * (self.stance / target_stance).clamp(0., 1.) * (1. - self.slip / max_slip).clamp(0., 1.)
        propulsion = (takeoff * (self.stance >= minimum_stance) * quality).sum(dim=1)
        local = pos - state["corner"][:, None, None, :]
        tread = state["step_surfaces"]
        hit = (support[..., None] & (local[..., 0, None] >= tread[:, None, None, :, 0] + edge_margin)
               & (local[..., 0, None] <= tread[:, None, None, :, 1] - edge_margin)
               & ((local[..., 2, None] - tread[:, None, None, :, 2]).abs() <= env._elevation_settings.top_contact_tolerance)).any(dim=(1, 2))
        self.tread_time = torch.where(mask[:, None], torch.where(hit, self.tread_time + env.step_dt, 0.), self.tread_time)
        new = (self.tread_time >= tread_hold - 1.e-6) & ~self.tread_paid & mask[:, None]
        self.tread_paid |= new
        command = env.command_manager.get_command("twist")[:, 0].clamp_min(1.e-6)
        progress = (state["forward_progress"] / (command * env.step_dt)).clamp(0., 1.)
        reward = 2. * progress + (.3 * propulsion + new.sum(dim=1).float()) / env.step_dt
        self.cached = torch.where(fresh, torch.where(active, reward, 0.), self.cached)
        self.loaded = torch.where(mask[:, None], loaded, self.loaded)
        self.last_step[fresh] = env.common_step_counter
        return self.cached
