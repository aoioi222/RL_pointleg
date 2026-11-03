"""Track the rear-right swing pose to correct a persistently straight front-left leg."""

import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse
from .walking_gait import loaded_feet, ordered_contact_ids


class HighStepLeftFrontPoseReward:
    """Match actual hip/knee pose and knee height while the rear-right leg swings."""

    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.contacts = ordered_contact_ids(env.scene[cfg.params["sensor_name"]])
        self.joints = [robot.joint_names.index(f"{leg}_{joint}_joint")
                       for leg in ("fl", "rr") for joint in ("hip_pitch", "knee")]
        self.hips, self.knees = [], []
        for leg in ("fl", "rr"):
            for name, output in ((f"{leg}_hip_pitch_Link", self.hips),
                                 (f"{leg}_knee_Link", self.knees)):
                ids, _ = robot.find_bodies(name)
                if len(ids) != 1:
                    raise ValueError(f"Expected one pose reference for {name}")
                output.append(ids[0])

    def __call__(self, env, sensor_name, command_name, minimum_support_force=8.,
                 minimum_forward_velocity=.05, minimum_rear_lift=.03,
                 hip_scale_degrees=20., knee_scale_degrees=25., knee_height_scale=.04):
        if not all(math.isfinite(v) and v > 0. for v in
                   (minimum_support_force, minimum_forward_velocity, minimum_rear_lift,
                    hip_scale_degrees, knee_scale_degrees, knee_height_scale)):
            raise ValueError("Left-front pose scales must be finite and positive")
        basic = env._high_step_basic_term
        basic(env, **basic.params)
        robot = env.scene["robot"].data
        q = robot.joint_pos[:, self.joints].reshape(env.num_envs, 2, 2)
        hip = robot.body_link_pos_w[:, self.hips]
        knee = robot.body_link_pos_w[:, self.knees]
        quat = robot.root_link_quat_w
        command = env.command_manager.get_command(command_name)
        velocity = robot.root_link_lin_vel_b[:, 0]
        finite = (torch.isfinite(q).all(dim=(1, 2)) & torch.isfinite(hip).all(dim=(1, 2))
                  & torch.isfinite(knee).all(dim=(1, 2)) & torch.isfinite(quat).all(dim=1)
                  & torch.isfinite(command).all(dim=1) & torch.isfinite(velocity))
        support = loaded_feet(env.scene[sensor_name], self.contacts, minimum_support_force)
        allowed = (finite & (command[:, 0] > .08) & (velocity > minimum_forward_velocity)
                   & basic.swinging[:, 3] & basic.eligible_swing[:, 3]
                   & (basic.air[:, 3] <= basic.params["max_air_time"] + 1.e-6)
                   & (basic.lift_peak[:, 3] >= minimum_rear_lift)
                   & (support.sum(dim=1) >= 2))
        terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
        if terminated is not None:
            allowed &= ~terminated
        local_knee = quat_apply_inverse(
            quat.nan_to_num()[:, None].expand(-1, 2, -1).reshape(-1, 4),
            (knee - hip).nan_to_num().reshape(-1, 3),
        ).reshape(env.num_envs, 2, 3)
        joint_difference = q[:, 0].nan_to_num() - q[:, 1].nan_to_num()
        scale = q.new_tensor([math.radians(hip_scale_degrees), math.radians(knee_scale_degrees)])
        joint_error = (joint_difference / scale).square().mean(dim=1)
        height_difference = local_knee[:, 0, 2] - local_knee[:, 1, 2]
        error = .5 * joint_error + .5 * (height_difference / knee_height_scale).square()
        score = torch.where(allowed, torch.exp(-error), 0.)
        if allowed.any():
            for index, joint in enumerate(("hip", "knee")):
                env.extras["log"][f"Metrics/high_step_lf_rr_{joint}_pose_error_deg"] = (
                    torch.rad2deg(joint_difference[allowed, index].abs()).detach().clone())
            env.extras["log"]["Metrics/high_step_lf_rr_knee_height_error_m"] = (
                height_difference[allowed].abs().detach().clone())
        return score
