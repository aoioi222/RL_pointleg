"""Configuration for the forward-only stepping-stone precision task."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..mdp.stepping_stones import (
    SteppingStoneCurriculum,
    SteppingStonesVelocityCommandCfg,
    active_stone_mask,
    clearance_penalty,
    collect_stepping_stones_substep,
    cover_reward,
    edge_penalty,
    foot_contact_observation,
    goal_velocity_reward,
    landing_reward,
    lift_reward,
    mount_reward,
    navigation_valid_observation,
    over_height_penalty,
    platform_relative_positions,
    remaining_time_observation,
    reset_stepping_stones,
    route_direction_observation,
    route_progress_reward,
    settle_angular_penalty,
    settle_progress_observation,
    settle_speed_penalty,
    side_collision_penalty,
    stage_observation,
    stepping_failed,
    stepping_failure_reward,
    stepping_metric,
    stepping_succeeded,
    stepping_timeout,
    strict_ground_observation,
    success_reward,
    support_deficit,
    time_cost,
    visited_stone_mask,
    stone_relative_positions,
)
from ..robot_cfg import FOOT_BODY_NAMES
from ..terrains.stepping_stones import SteppingStonesTerrainCfg
from .env_cfgs import flat_env_cfg


def make_stepping_stones_env_cfg(play: bool = False):
    """Build the task with fixed geometry and physical foot-contact adjudication."""
    cfg = flat_env_cfg(play=False)
    cfg.episode_length_s = 60.0
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=2.0,
            num_rows=7,
            num_cols=1,
            layout_shape=(4, 4),
            curriculum=True,
            sub_terrains={"stepping_stones": SteppingStonesTerrainCfg(proportion=1.0)},
            difficulty_range=(0.0, 1.0),
            color_scheme="none",
        ),
        max_init_terrain_level=0,
    )

    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(
            name="stepping_foot_contact",
            primary=ContactMatch(mode="body", pattern=FOOT_BODY_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "pos"),
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
            lin_vel_x=(0.15, 0.35), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None
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
        "foot_contact": ObservationTermCfg(func=foot_contact_observation),
        "settle_progress": ObservationTermCfg(func=settle_progress_observation),
        "route_direction": ObservationTermCfg(func=route_direction_observation),
        "navigation_valid": ObservationTermCfg(func=navigation_valid_observation),
        "remaining_time": ObservationTermCfg(func=remaining_time_observation),
    }
    cfg.observations["actor"].terms.update(task_terms)
    cfg.observations["critic"].terms.update(deepcopy(task_terms))

    for key in (
        "track_lin_vel",
        "track_ang_vel",
        "is_terminated",
        "base_height_l2",
        "feet_air_time",
        "foot_swing_height",
        "short_stance",
        "stance_balance",
        "foot_lateral_placement",
        "support_deficit",
        "stand_still",
        "feet_contact_without_cmd",
        "static_joint_posture",
        "static_leg_symmetry",
        "command_orthogonal_drift",
    ):
        cfg.rewards.pop(key, None)
    cfg.rewards.update(
        {
            "stepping_mount": RewardTermCfg(func=mount_reward, weight=10.0),
            "stepping_cover": RewardTermCfg(func=cover_reward, weight=30.0),
            "stepping_landing": RewardTermCfg(func=landing_reward, weight=10.0),
            "stepping_lift": RewardTermCfg(func=lift_reward, weight=1.0),
            "stepping_success": RewardTermCfg(func=success_reward, weight=100.0),
            "stepping_failure": RewardTermCfg(func=stepping_failure_reward, weight=-150.0),
            "stepping_goal_velocity": RewardTermCfg(func=goal_velocity_reward, weight=1.0),
            "stepping_route_progress": RewardTermCfg(func=route_progress_reward, weight=10.0),
            "stepping_time_cost": RewardTermCfg(func=time_cost, weight=-0.2),
            "stepping_support": RewardTermCfg(func=support_deficit, weight=-0.5),
            "stepping_clearance": RewardTermCfg(func=clearance_penalty, weight=-1.0),
            "stepping_edge": RewardTermCfg(func=edge_penalty, weight=-0.3),
            "stepping_side_collision": RewardTermCfg(func=side_collision_penalty, weight=-0.5),
            "stepping_over_height": RewardTermCfg(func=over_height_penalty, weight=-0.1),
            "stepping_settle_speed": RewardTermCfg(func=settle_speed_penalty, weight=-1.0),
            "stepping_settle_angular": RewardTermCfg(func=settle_angular_penalty, weight=-0.3),
        }
    )

    cfg.terminations.pop("time_out", None)
    cfg.terminations.pop("base_height_out_of_range", None)
    cfg.terminations["stepping_failed"] = TerminationTermCfg(func=stepping_failed)
    cfg.terminations["stepping_succeeded"] = TerminationTermCfg(func=stepping_succeeded)
    cfg.terminations["stepping_timeout"] = TerminationTermCfg(func=stepping_timeout)

    cfg.metrics["stepping_contact_latch"] = MetricsTermCfg(
        func=collect_stepping_stones_substep, per_substep=True
    )
    cfg.metrics["stepping_success"] = MetricsTermCfg(
        func=stepping_metric, params={"key": "success"}, reduce="last"
    )
    cfg.metrics["stepping_failure"] = MetricsTermCfg(
        func=stepping_metric, params={"key": "failure"}, reduce="last"
    )
    cfg.metrics["stepping_covered_fraction"] = MetricsTermCfg(
        func=stepping_metric, params={"key": "covered_fraction"}, reduce="last"
    )
    cfg.metrics["stepping_stone_count"] = MetricsTermCfg(
        func=stepping_metric, params={"key": "stone_count"}, reduce="last"
    )
    for key, reduce in (
        ("goal_distance", "mean"),
        ("low_clearance_fraction", "last"),
        ("swing_landing_count", "last"),
        ("edge_contact_fraction", "mean"),
    ):
        cfg.metrics[f"stepping_{key}"] = MetricsTermCfg(
            func=stepping_metric, params={"key": key}, reduce=reduce
        )
    cfg.curriculum = {
        "stepping_stone_levels": CurriculumTermCfg(
            func=SteppingStoneCurriculum,
            params={
                "window_size": 100,
                "success_threshold": 0.80,
                "demote_threshold": 0.40,
                "initial_level_cap": 0,
            },
        )
    }

    if play:
        cfg.seed = 42
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
