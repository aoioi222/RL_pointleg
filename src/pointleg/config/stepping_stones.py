"""Configuration for the forward-only stepping-stone precision task."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..mdp.stepping_stones import (
    STEPPING_FORWARD_SPEED_RANGE,
    SteppingStoneCurriculum,
    SteppingStonesVelocityCommandCfg,
    active_stone_mask,
    collect_stepping_stones_substep,
    cover_reward,
    exit_stability_reward,
    stone_center_reward,
    stone_edge_penalty,
    stepping_low_support_penalty,
    startup_body_rise_penalty,
    stepping_foot_side_penalty,
    foot_contact_observation,
    fixed_gait_observation,
    fixed_gait_reward,
    forward_velocity_reward,
    stepping_body_sink_penalty,
    stepping_body_stall_penalty,
    navigation_valid_observation,
    platform_relative_positions,
    remaining_time_observation,
    reset_stepping_stones,
    route_direction_observation,
    settle_progress_observation,
    stage_observation,
    stepping_bad_orientation,
    stepping_joint_pos_out_of_limits,
    stepping_body_tilt_penalty,
    stepping_failed,
    stepping_failure_reward,
    stepping_early_failure_penalty,
    stepping_backward_velocity_penalty,
    stepping_flight_penalty,
    stepping_shin_stone_penalty,
    stepping_shin_support_penalty,
    stepping_swing_height_penalty,
    stepping_foot_slip_penalty,
    stepping_hip_abduction_symmetry_reward,
    stepping_hip_abduction_target_penalty,
    stepping_knee_min_angle_penalty,
    stepping_knee_reverse_penalty,
    stepping_leg_clearance_penalty,
    stepping_lateral_velocity_penalty,
    stepping_foot_row_alignment_penalty,
    stepping_metric,
    stepping_stalled,
    stepping_succeeded,
    stepping_settle_timeout,
    stepping_timeout,
    stone_relative_positions,
    strict_ground_observation,
    success_reward,
    visited_stone_mask,
)
from ..robot_cfg import FOOT_BODY_NAMES
from ..terrains.stepping_stones import COMPETITION_STONE_COUNTS, SteppingStonesTerrainCfg
from .env_cfgs import flat_env_cfg


def configure_competition_six(cfg, *, gap_curriculum: bool = False) -> None:
    """Select the final six-stone scene or the two-, four-, and six-stone curriculum."""
    difficulty = 1.0
    terrain = cfg.scene.terrain
    generator = terrain.terrain_generator
    competition = replace(
        generator.sub_terrains["competition_stones"], proportion=1.0, six_stone_gap_curriculum=gap_curriculum,
    )
    terrain.terrain_generator = replace(
        generator,
        num_rows=len(COMPETITION_STONE_COUNTS) if gap_curriculum else 1,
        num_cols=1,
        layout_shape=None,
        difficulty_range=(0.0, 1.0) if gap_curriculum else (difficulty, difficulty),
        sub_terrains={"competition_stones": competition},
    )
    terrain.max_init_terrain_level = 0
    if gap_curriculum:
        cfg.curriculum["stepping_stone_levels"].params.update(
            window_size=256, promote_at_current_cap=True,
        )
    else:
        cfg.curriculum = {}


def make_stepping_stones_env_cfg(play: bool = False):
    """Build the task with fixed geometry and physical foot-contact adjudication."""
    cfg = flat_env_cfg(play=False)
    robot = cfg.scene.entities["robot"]
    # Align the front feet under the hip pitch axes in X and match touchdown heights.
    joint_pos = {
        "(fl|rl)_hip_abduction_joint": 0.15,
        "(fr|rr)_hip_abduction_joint": -0.15,
        "fl_hip_pitch_joint": 0.751,
        "fr_hip_pitch_joint": 0.741,
        "(rl|rr)_hip_pitch_joint": 0.80,
        "fl_knee_joint": -1.404,
        "fr_knee_joint": -1.388,
        "(rl|rr)_knee_joint": -1.40,
    }
    robot.init_state = replace(robot.init_state, pos=(0.0, 0.0, 0.362), joint_pos=joint_pos)
    cfg.episode_length_s = 60.0
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=2.0,
            num_rows=len(COMPETITION_STONE_COUNTS),
            num_cols=1,
            layout_shape=None,
            curriculum=True,
            sub_terrains={
                "competition_stones": SteppingStonesTerrainCfg(proportion=1.0, competition=True),
            },
            difficulty_range=(0.0, 1.0),
            color_scheme="none",
        ),
        max_init_terrain_level=0,
    )

    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(
            name="stepping_shin_contact",
            primary=ContactMatch(mode="geom", pattern=".*_shin_collision", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal", "tangent"),
            reduce="maxforce", num_slots=4, global_frame=True,
        ),
        ContactSensorCfg(
            name="stepping_foot_contact",
            primary=ContactMatch(mode="body", pattern=FOOT_BODY_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "pos", "normal", "force"),
            reduce="mindist",
            num_slots=1,
            history_length=0,
        ),
    )

    cfg.commands["twist"] = SteppingStonesVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(60.0, 60.0),
        heading_command=False,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=1.0,
        init_velocity_prob=0.0,
        ranges=SteppingStonesVelocityCommandCfg.Ranges(
            lin_vel_x=STEPPING_FORWARD_SPEED_RANGE, lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None
        ),
    )

    cfg.events.pop("reset_base", None)
    cfg.events.pop("push_robot", None)
    cfg.events["reset_stepping_stones"] = EventTermCfg(func=reset_stepping_stones, mode="reset")

    task_terms = {
        "stone_relative_positions": ObservationTermCfg(
            func=stone_relative_positions, scale=0.25, noise=Unoise(n_min=-0.002, n_max=0.002)
        ),
        "active_stone_mask": ObservationTermCfg(func=active_stone_mask),
        "visited_stone_mask": ObservationTermCfg(func=visited_stone_mask),
        "platform_relative_positions": ObservationTermCfg(
            func=platform_relative_positions, scale=0.25, noise=Unoise(n_min=-0.002, n_max=0.002)
        ),
        "stage": ObservationTermCfg(func=stage_observation),
        "strict_ground_check": ObservationTermCfg(func=strict_ground_observation),
        "foot_contact": ObservationTermCfg(func=foot_contact_observation, params={"support_progress": not play}),
        "settle_progress": ObservationTermCfg(func=settle_progress_observation),
        "route_direction": ObservationTermCfg(func=route_direction_observation),
        "navigation_valid": ObservationTermCfg(func=navigation_valid_observation),
        "remaining_time": ObservationTermCfg(func=remaining_time_observation),
        "fixed_gait": ObservationTermCfg(func=fixed_gait_observation),
    }
    cfg.observations["actor"].terms.update(task_terms)
    cfg.observations["critic"].terms.update(deepcopy(task_terms))

    action_rate = cfg.rewards["action_rate"]
    joint_pos_limits = replace(cfg.rewards["joint_pos_limits"], weight=-5.0)
    cfg.rewards = {
        "stepping_cover": RewardTermCfg(func=cover_reward, weight=12.0),
        "stepping_stone_center": RewardTermCfg(func=stone_center_reward, weight=0.5),
        "stepping_stone_edge": RewardTermCfg(func=stone_edge_penalty, weight=-3.5),
        "stepping_low_support": RewardTermCfg(func=stepping_low_support_penalty, weight=-12.0),
        "stepping_startup_body_rise": RewardTermCfg(func=startup_body_rise_penalty, weight=-6.0),
        "stepping_foot_side": RewardTermCfg(func=stepping_foot_side_penalty, weight=-8.0),
        "stepping_success": RewardTermCfg(func=success_reward, weight=40.0),
        "stepping_exit_stability": RewardTermCfg(func=exit_stability_reward, weight=4.0),
        "stepping_failure": RewardTermCfg(
            func=stepping_failure_reward, weight=-25.0, params={"ground_contact_multiplier": 2.0},
        ),
        "stepping_early_failure": RewardTermCfg(
            func=stepping_early_failure_penalty, weight=0.0, params={"grace_time_s": 60.0},
        ),
        "stepping_forward_velocity": RewardTermCfg(func=forward_velocity_reward, weight=5.0),
        "stepping_body_sink": RewardTermCfg(
            func=stepping_body_sink_penalty, weight=-8.0,
            params={"tolerance": 0.02, "scale": 0.06},
        ),
        "stepping_body_stall": RewardTermCfg(
            func=stepping_body_stall_penalty, weight=-50.0,
            params={"grace_s": 0.30, "ramp_s": 0.30},
        ),
        "stepping_backward_velocity": RewardTermCfg(
            func=stepping_backward_velocity_penalty, weight=-3.0,
            params={"tolerance": 0.02, "scale": 0.10},
        ),
        "stepping_lateral_velocity": RewardTermCfg(
            func=stepping_lateral_velocity_penalty, weight=-3.0,
            params={"tolerance": 0.01, "scale": 0.10},
        ),
        "stepping_foot_row_alignment": RewardTermCfg(
            func=stepping_foot_row_alignment_penalty, weight=-4.0,
            params={"tolerance": 0.04, "scale": 0.06},
        ),
        "stepping_hip_abduction_symmetry": RewardTermCfg(
            func=stepping_hip_abduction_symmetry_reward, weight=0.5,
            params={"angle_scale": 0.35},
        ),
        "stepping_hip_abduction_target": RewardTermCfg(
            func=stepping_hip_abduction_target_penalty, weight=-1.0,
            params={"target": 0.15, "tolerance": 0.05, "scale": 0.15},
        ),
        "stepping_knee_min_angle": RewardTermCfg(
            func=stepping_knee_min_angle_penalty, weight=-2.0,
            params={"minimum_degrees": 5.0, "scale_degrees": 5.0},
        ),
        "stepping_knee_reverse": RewardTermCfg(func=stepping_knee_reverse_penalty, weight=-50.0),
        "stepping_leg_clearance": RewardTermCfg(
            func=stepping_leg_clearance_penalty, weight=-4.0, params={"clearance": 0.05},
        ),
        "stepping_body_tilt": RewardTermCfg(
            func=stepping_body_tilt_penalty, weight=-2.0,
            params={"tolerance_degrees": 8.0, "scale_degrees": 15.0},
        ),
        "stepping_flight": RewardTermCfg(func=stepping_flight_penalty, weight=-2.0),
        "stepping_shin_stone": RewardTermCfg(func=stepping_shin_stone_penalty, weight=-8.0),
        "stepping_shin_support": RewardTermCfg(func=stepping_shin_support_penalty, weight=-1.0),
        "stepping_swing_height": RewardTermCfg(
            func=stepping_swing_height_penalty, weight=-2.0, params={"target_height": 0.07, "scale": 0.07},
        ),
        "stepping_foot_slip": RewardTermCfg(
            func=stepping_foot_slip_penalty, weight=-0.5,
            params={"speed_tolerance": 0.02, "speed_scale": 0.12},
        ),
        "joint_pos_limits": joint_pos_limits,
        "action_rate": action_rate,
    }

    for name, key, weight, event in (
        ("progress", "progress", 12.0, True),
        ("complete", "complete", 12.0, True),
        ("wrong_lift", "wrong_lift", -2.0, True),
        ("support", "support", -8.0, False),
        ("target", "target", -3.0, False),
        ("platform_edge", "edge", -5.0, False),
        ("clearance", "clearance", -1.5, False),
        ("left_rear_gap", "left_rear_gap", -8.0, False),
        ("delay", "delay", -8.0, False),
        ("backward", "backward", -4.0, False),
    ):
        cfg.rewards[f"stepping_gait_{name}"] = RewardTermCfg(
            func=fixed_gait_reward, weight=weight, params={"key": key, "event": event},
        )

    cfg.terminations.pop("time_out", None)
    cfg.terminations.pop("base_height_out_of_range", None)
    cfg.terminations["bad_orientation"] = TerminationTermCfg(func=stepping_bad_orientation, params={"limit_angle": 1.0})
    cfg.terminations["joint_pos_out_of_limits"] = replace(
        cfg.terminations["joint_pos_out_of_limits"], func=stepping_joint_pos_out_of_limits,
    )
    cfg.terminations["stepping_failed"] = TerminationTermCfg(func=stepping_failed)
    cfg.terminations["stepping_settle_timeout"] = TerminationTermCfg(func=stepping_settle_timeout)
    cfg.terminations["stepping_stalled"] = TerminationTermCfg(func=stepping_stalled, params={"timeout_s": 8.0})
    cfg.terminations["stepping_timeout"] = TerminationTermCfg(func=stepping_timeout)
    cfg.terminations["stepping_succeeded"] = TerminationTermCfg(func=stepping_succeeded)

    cfg.metrics["stepping_contact_latch"] = MetricsTermCfg(
        func=collect_stepping_stones_substep, per_substep=True
    )
    for key, reduce in (
        ("success", "last"),
        ("failure", "last"),
        ("covered_fraction", "last"),
        ("no_forward_step_time_s", "last"),
        ("right_rear_forward_steps", "last"),
        ("right_rear_follow_wait_s", "last"),
        ("stone_count", "last"),
        ("stone_gap_m", "last"),
        ("exit_reached", "last"),
        ("exit_best_stable_time_s", "last"),
        ("forward_speed_mps", "mean"),
        ("peak_foot_clearance_m", "last"),
        ("startup_peak_body_rise_m", "last"),
        ("startup_peak_front_rise_m", "last"),
        ("traverse_peak_body_sink_m", "last"),
        ("premature_support_handoffs", "last"),
        ("left_front_overlap_count", "last"),
        ("gait_phase", "last"),
        ("gait_completed_steps", "last"),
        ("gait_violations", "last"),
        ("gait_left_front_rear_transfer_violations", "last"),
        ("gait_phase_age_s", "last"),
        ("gait_ready", "last"),
        ("gait_lifted_fraction", "last"),
        ("gait_pair_ready", "last"),
    ):
        cfg.metrics[f"stepping_{key}"] = MetricsTermCfg(
            func=stepping_metric, params={"key": key}, reduce=reduce
        )
    cfg.curriculum = {
        "stepping_stone_levels": CurriculumTermCfg(
            func=SteppingStoneCurriculum,
            params={
                "window_size": 256,
                "success_threshold": 0.80,
                "demote_threshold": 0.40,
                "initial_level_cap": 0,
                "promote_at_current_cap": True,
            },
        )
    }

    if play:
        cfg.seed = 42
        cfg.events["reset_stepping_stones"].params["randomize_levels"] = True
        cfg.episode_length_s = int(1e9)
        cfg.curriculum = {}
        cfg.events = {
            name: term
            for name, term in cfg.events.items()
            if name in ("reset_scene", "reset_joints", "reset_stepping_stones")
        }
        for group in cfg.observations.values():
            group.enable_corruption = False
        for action in cfg.actions.values():
            if hasattr(action, "min_delay"):
                action.min_delay = action.max_delay = 0
    return cfg
