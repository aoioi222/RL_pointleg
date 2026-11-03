"""Flat-ground posture and walking foundation with independent objectives."""

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from pointleg.mdp.high_step import (
    HighStepContactForcePenalty, HighStepMeanLegSymmetryPenalty,
    high_step_pitch_penalty, high_step_yaw_rate_penalty,
)
from pointleg.mdp.high_step_hip_swing import HighStepHipSwingReward
from pointleg.mdp.high_step_front_workspace import HighStepFrontWorkspacePenalty
from pointleg.mdp.high_step_diagonal import HighStepDiagonalTrackingReward
from pointleg.mdp.high_step_lf_pose import HighStepLeftFrontPoseReward
from pointleg.mdp.high_step_basic import (
    BasicHighStepSupportPenalty, basic_backstroke_reward, basic_height_event, basic_height_progress,
    basic_height_overshoot,
    basic_high_step_slip, basic_high_step_velocity, basic_support_deficit,
)
from .env_cfgs import flat_env_cfg
from .evaluation import configure_evaluation


def high_step_env_cfg(play: bool = False):
    """Train a forward stepping foundation with posture and active-lift terms."""
    cfg = flat_env_cfg(False)
    cfg.scene.num_envs = 512
    contact = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    contact.fields = ("found", "force", "pos", "normal", "tangent")
    contact.global_frame = True
    contact.history_length = cfg.decimation
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.15, 0.25)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.10
    command.rel_forward_envs = command.rel_backward_envs = 0.0
    command.rel_lateral_envs = command.rel_yaw_envs = 0.0
    command.flat_lateral_probability = command.flat_yaw_probability = 0.0
    command.linear_deadband = 0.08
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = 0.0
    command.resampling_time_range = (6.0, 8.0)
    command.__post_init__()

    # 显式白名单避免平地配置中的其他奖励随继承重新进入训练。
    retained = (
        "upright", "body_ang_vel", "stand_still", "joint_torques",
        "action_rate", "joint_pos_limits", "body_collision", "is_terminated",
        "command_orthogonal_drift",
    )
    rewards = {name: cfg.rewards[name] for name in retained}
    rewards["action_rate"].weight = -0.005
    rewards["command_orthogonal_drift"].weight = -3.0
    rewards["command_orthogonal_drift"].params.update(
        speed_tolerance=0.010, deviation_scale=0.10,
    )
    rewards["hip_abduction_excess"] = cfg.rewards["hip_abduction_excess"]
    rewards["hip_abduction_excess"].weight = -3.6
    rewards["hip_abduction_excess"].params.update(tolerance=0., deviation_scale=.25)
    rewards["high_step_pitch"] = RewardTermCfg(
        func=high_step_pitch_penalty, weight=-1.0,
        params={"tolerance_degrees": 2., "deviation_degrees": 8.},
    )
    rewards["front_foot_workspace"] = RewardTermCfg(
        func=HighStepFrontWorkspacePenalty, weight=-.7,
        params={"command_name": "twist", "smoothing_time": .6, "warmup_time": 1.,
                "forward_tolerance": .04, "deviation_scale": .08},
    )
    rewards["track_lin_vel"] = RewardTermCfg(
        func=basic_high_step_velocity, weight=2.0,
        params={"command_name": "twist", "std": 0.10},
    )
    rewards["high_step_yaw_rate"] = RewardTermCfg(
        func=high_step_yaw_rate_penalty, weight=-0.5,
        params={"command_name": "twist", "tolerance_degrees": 2.0,
                "deviation_degrees": 20.0},
    )
    rewards["high_step_mean_leg_symmetry"] = RewardTermCfg(
        func=HighStepMeanLegSymmetryPenalty, weight=-0.2,
        params={
            "command_name": "twist", "smoothing_time": 1.0, "warmup_time": 1.2,
            "tolerance_degrees": 4.0, "deviation_degrees": 8.0,
            "front_rear_weight": .75, "front_rear_hip_tolerance": 8.,
            "front_rear_knee_tolerance": 12., "front_rear_hip_scale": 15.,
            "front_rear_knee_scale": 20.,
        },
    )
    rewards["support_deficit"] = RewardTermCfg(
        func=basic_support_deficit, weight=-2.0,
        params={"sensor_name": "feet_ground_contact", "minimum_force": 8.0},
    )
    rewards["support_timing"] = RewardTermCfg(
        func=BasicHighStepSupportPenalty, weight=-0.1,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist",
                "minimum_force": 8.0, "min_stance_time": .16,
                "max_stance_time": .85, "max_air_time": .36,
                "stance_scale": .25, "air_scale": .18,
                "short_event_scale": .10, "startup_grace": .60,
                "slip_grace": .04, "slip_tolerance": .02, "slip_scale": .10,
                "target_height": .15, "min_air_time": .10,
                "min_forward_progress": .025, "lift_progress_distance": .07,
                "short_stance_target": .24, "overshoot_scale": .05},
    )
    # 抬升按起脚后的主动上升计分，使用真实承重新增前进距离分配奖金。
    rewards["active_lift_event"] = RewardTermCfg(func=basic_height_event, weight=.35)
    rewards["active_lift_progress"] = RewardTermCfg(func=basic_height_progress, weight=.08)
    rewards["active_lift_overshoot"] = RewardTermCfg(func=basic_height_overshoot, weight=-.5)
    rewards["stance_backstroke"] = RewardTermCfg(
        func=basic_backstroke_reward, weight=.10,
        params={"target_distance": .07, "minimum_stance": .16,
                "target_stance": .30, "max_slip": .025},
    )
    rewards["hip_swing_coordination"] = RewardTermCfg(
        func=HighStepHipSwingReward, weight=.06,
        params={"sensor_name": "feet_ground_contact", "minimum_support_force": 8.,
                "target_hip_range_degrees": 20., "target_knee_range": .03,
                "min_body_progress": .01},
    )
    rewards["diagonal_rear_tracking"] = RewardTermCfg(
        func=HighStepDiagonalTrackingReward, weight=.2,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist",
                "minimum_support_force": 8., "minimum_forward_velocity": .05,
                "minimum_rear_motion_degrees": 5., "full_rear_motion_degrees": 20.,
                "joint_error_scale_degrees": 15.},
    )
    rewards["lf_rr_pose_tracking"] = RewardTermCfg(
        func=HighStepLeftFrontPoseReward, weight=.25,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist",
                "minimum_support_force": 8., "minimum_forward_velocity": .05,
                "minimum_rear_lift": .03, "hip_scale_degrees": 20.,
                "knee_scale_degrees": 25., "knee_height_scale": .04},
    )
    rewards["feet_slip"] = RewardTermCfg(func=basic_high_step_slip, weight=-0.2)
    cfg.rewards = rewards
    # 接触力只记录指标，不再参与奖励总和。
    cfg.metrics["high_step_contact_force"] = MetricsTermCfg(
        func=HighStepContactForcePenalty,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist",
                "force_threshold": 150., "force_scale": 100., "startup_grace": .6},
    )
    if play:
        configure_evaluation(cfg)
    return cfg
