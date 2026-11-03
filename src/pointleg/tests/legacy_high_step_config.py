"""Frozen pre-simplification configuration for historical reward unit tests only."""

import math

from mjlab.managers.reward_manager import RewardTermCfg

from pointleg.mdp.high_step import (
    HighStepContactForcePenalty, HighStepCycleReward,
    HighStepDriftBiasPenalty, HighStepMeanLegSymmetryPenalty,
    high_step_feedback, high_step_phase_penalty, high_step_pitch_penalty,
    high_step_yaw_rate_penalty,
)
from pointleg.mdp.gait_bias import HipAbductionBiasPenalty
from pointleg.mdp.high_step_coordination import (
    HighStepSwingRetractionPenalty, high_step_static_joint_posture,
)
from pointleg.mdp.walking_gait import DiagonalGaitPenalty
from pointleg.mdp.high_step_support import (
    HighStepSupportCorrection, high_step_contact_slip_penalty,
    high_step_short_stance_penalty, high_step_velocity_reward,
)
from pointleg.mdp.rewards import TranslationFootWorkspacePenalty, lin_vel_z_l2
from pointleg.config.env_cfgs import flat_env_cfg
from pointleg.config.evaluation import configure_evaluation


def high_step_env_cfg(play: bool = False):
    """Preserve the walking interface while replacing long-swing incentives."""
    cfg = flat_env_cfg(False)
    cfg.scene.num_envs = 512
    contact = next(sensor for sensor in cfg.scene.sensors if sensor.name == "feet_ground_contact")
    contact.fields = ("found", "force", "pos", "normal", "tangent")
    contact.global_frame = True
    contact.history_length = cfg.decimation
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.15, 0.25)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.10
    # 正向范围直接采样，避免基础前进分组把低速指令抬到每秒0.3米。
    command.rel_forward_envs = 0.0
    command.rel_backward_envs = 0.0
    command.rel_lateral_envs = command.rel_yaw_envs = 0.0
    command.flat_lateral_probability = command.flat_yaw_probability = 0.0
    command.linear_deadband = 0.08
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = 0.0
    command.resampling_time_range = (6.0, 8.0)
    command.__post_init__()
    cfg.rewards.pop("foot_swing_height")
    cfg.rewards.pop("feet_air_time")
    cfg.rewards["track_lin_vel"].params["std"] = 0.20
    cfg.rewards["track_lin_vel"].func = high_step_velocity_reward
    cfg.rewards.pop("long_foot_air_time")
    cfg.rewards["short_stance"].func = high_step_short_stance_penalty
    cfg.rewards["short_stance"].params["min_stance_time"] = 0.24
    cfg.rewards["short_stance"].weight = -0.10
    cfg.rewards["feet_slip"] = RewardTermCfg(func=high_step_contact_slip_penalty, weight=-.2)
    cfg.rewards.pop("leg_motion_penalty")
    cfg.rewards["action_rate"].weight = -0.005
    cfg.rewards["static_joint_posture"].func = high_step_static_joint_posture
    cfg.rewards["command_orthogonal_drift"].weight = -3.0
    cfg.rewards["command_orthogonal_drift"].params.update(
        speed_tolerance=0.015, deviation_scale=0.10,
    )
    cfg.rewards["high_step_drift_bias"] = RewardTermCfg(
        func=HighStepDriftBiasPenalty, weight=-1.5,
        params={
            "command_name": "twist", "smoothing_time": 0.6,
            "warmup_time": 1.0, "speed_tolerance": 0.005,
            "speed_scale": 0.04, "yaw_tolerance_degrees": 1.0,
            "yaw_scale_degrees": 9.0,
        },
    )
    cfg.rewards["hip_abduction_excess"].weight = -1.5
    cfg.rewards["hip_abduction_excess"].params["tolerance"] = math.radians(4.0)
    cfg.rewards["hip_abduction_bias"] = RewardTermCfg(
        func=HipAbductionBiasPenalty, weight=-1.0,
        params={
            "smoothing_time": 0.5, "warmup_time": 1.0,
            "tolerance_degrees": 3.0, "deviation_scale_degrees": 8.0,
        },
    )
    cfg.rewards["foot_lateral_placement"].weight = -1.5
    cfg.rewards["foot_lateral_placement"].params["center_tolerance"] = 0.015
    cfg.rewards["high_step_yaw_rate"] = RewardTermCfg(
        func=high_step_yaw_rate_penalty, weight=-1.0,
        params={
            "command_name": "twist", "tolerance_degrees": 2.0,
            "deviation_degrees": 20.0,
        },
    )
    cfg.rewards["high_step_contact_force"] = RewardTermCfg(
        func=HighStepContactForcePenalty, weight=-0.5,
        params={
            "sensor_name": "feet_ground_contact", "command_name": "twist",
            "force_threshold": 150.0, "force_scale": 100.0,
            "startup_grace": 0.6,
        },
    )
    cfg.rewards["high_step_cycle"] = RewardTermCfg(
        func=HighStepCycleReward, weight=0.38,
        params={
            "sensor_name": "feet_ground_contact", "command_name": "twist",
            "target_height": 0.15, "height_std": 0.03,
            "target_cycle": 0.60, "cycle_std": 0.12,
            "timing_weight": 0.50, "minimum_support_force": 8.,
            "min_air_time": 0.10, "max_air_time": 0.36,
            "min_stance_time": 0.16, "max_step_length": 0.18,
            "max_target_step_length": 0.15,
        },
    )
    cfg.rewards["high_step_phase"] = RewardTermCfg(
        func=high_step_phase_penalty, weight=-0.4,
        params={
            "sensor_name": "feet_ground_contact", "command_name": "twist",
            "max_stance_time": 0.85, "max_air_time": 0.36,
            "deviation_scale": 0.18, "stance_weight": 0.,
        },
    )
    cfg.rewards["high_step_lift_progress"] = RewardTermCfg(
        func=high_step_feedback, weight=1., params={"key": "lift"},
    )
    cfg.rewards["high_step_cycle_symmetry"] = RewardTermCfg(
        func=high_step_feedback, weight=-.4, params={"key": "symmetry"},
    )
    cfg.rewards["diagonal_gait"] = RewardTermCfg(
        func=DiagonalGaitPenalty, weight=-.6,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["high_step_workspace"] = RewardTermCfg(
        func=TranslationFootWorkspacePenalty, weight=-0.5,
        params={
            "sensor_name": "feet_ground_contact", "command_name": "twist",
            "include_forward": True, "nominal_forward": -.02, "forward_margin": 0.14,
            "lateral_margin": 0.04, "minimum_command_speed": 0.08,
        },
    )
    cfg.rewards["high_step_swing_retraction"] = RewardTermCfg(
        func=HighStepSwingRetractionPenalty, weight=-0.2,
        params={
            "sensor_name": "feet_ground_contact", "command_name": "twist",
            "activation_height": 0.06, "nominal_forward": -.02,
            "fore_aft_margin": 0.14, "deviation_scale": 0.08,
            "minimum_support_force": 8.,
        },
    )
    cfg.rewards["high_step_body_bounce"] = RewardTermCfg(func=lin_vel_z_l2, weight=-0.5)
    cfg.rewards["high_step_pitch"] = RewardTermCfg(
        func=high_step_pitch_penalty, weight=-1.5,
        params={"tolerance_degrees": 3.0, "deviation_degrees": 8.0},
    )
    cfg.rewards["high_step_mean_leg_symmetry"] = RewardTermCfg(
        func=HighStepMeanLegSymmetryPenalty, weight=-0.2,
        params={
            "command_name": "twist", "smoothing_time": 1.0, "warmup_time": 1.2,
            "tolerance_degrees": 4.0, "deviation_degrees": 8.0,
            "front_rear_weight": .75, "front_rear_hip_tolerance": 8.,
            "front_rear_knee_tolerance": 12., "front_rear_hip_scale": 15.,
            "front_rear_knee_scale": 20.,
        },
    )
    cfg.rewards["high_step_support"] = RewardTermCfg(
        func=HighStepSupportCorrection, weight=-.3,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist",
                "minimum_force": 8., "minimum_stance": .16, "target_stance": .35,
                "target_backstroke": .07, "maximum_drift": .025, "maximum_stance": .85,
                "overstay_scale": .25, "renewal_grace": .6, "renewal_timeout": 1.2,
                "slip_grace": .04, "slip_tolerance": .02, "slip_scale": .10},
    )
    if play:
        configure_evaluation(cfg)
    return cfg



def load_env_cfg(task_id, play=False):
    """Use the archived HighStep objective; other tasks use the live registry."""
    from mjlab.tasks.registry import load_env_cfg as live_env_cfg
    if task_id == "PointLeg-HighStep-v0":
        return high_step_env_cfg(play)
    return live_env_cfg(task_id, play=play)

