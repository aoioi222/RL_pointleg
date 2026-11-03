"""Forward-only stair/ramp walking and separately triggered wall jumping."""

from copy import deepcopy
from dataclasses import dataclass
import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg

from ..mdp.elevation import (
    ElevationSettings, ElevationTerrainCurriculum, ElevationVelocityCommandCfg,
    collect_elevation_contacts, elevation_completed, elevation_cost, elevation_event,
    elevation_failed, elevation_metric, reset_elevation, select_elevation_terrain,
    elevation_alignment_metric,
    elevation_forward_velocity_reward,
)
from ..mdp.gait_bias import HipAbductionBiasPenalty
from ..mdp.elevation_clearance import ElevationSwingClearancePenalty, elevation_clearance_metric
from ..mdp.elevation_gait import ElevationFootWorkspacePenalty, elevation_gait_term
from ..mdp.elevation_support import ElevationMeanPosturePenalty
from ..mdp.elevation_safety import ElevationBodyClearancePenalty, ElevationTouchdownSpeedPenalty
from ..mdp.walking_gait import DiagonalGaitPenalty
from ..robot_cfg import BASE_BODY_NAME, FOOT_BODY_NAMES, get_robot_cfg
from ..terrains.elevation import TILE_SIZE, make_elevation_terrains


@dataclass
class ElevatedGridPatternCfg(GridPatternCfg):
    def generate_rays(self, mj_model, device):
        origins, directions = super().generate_rays(mj_model, device)
        origins[:, 2] += 1.0
        return origins, directions


def configure_walking_rewards(cfg):
    """Reuse Rough walking constraints with room for stair and ramp transitions."""
    from .env_cfgs import rough_env_cfg

    rough = rough_env_cfg()
    names = (
        "command_aligned_progress", "command_progress_deficit",
        "translation_stance_overstay", "translation_foot_workspace",
        "hip_abduction_excess", "knee_extension", "feet_air_time", "support_deficit",
        "stance_balance", "short_stance", "long_foot_air_time",
        "leg_joint_acc_l2", "joint_power", "contact_forces",
    )
    cfg.rewards.update({name: deepcopy(rough.rewards[name]) for name in names})
    cfg.rewards["command_aligned_progress"].weight = 0.0
    cfg.rewards["command_progress_deficit"].weight = 0.0
    cfg.rewards["velocity"] = RewardTermCfg(
        func=elevation_forward_velocity_reward, weight=1., params={"std": .1},
    )
    cfg.rewards["action"].weight = -.02
    cfg.rewards["leg_joint_acc_l2"].weight = -1.e-7
    cfg.rewards["translation_stance_overstay"].weight = 0.0
    cfg.rewards["translation_stance_overstay"].params.update(
        include_forward=True, stance_distance=0.20,
        minimum_stance_limit=0.50, maximum_stance_limit=1.60,
    )
    cfg.rewards["translation_foot_workspace"] = RewardTermCfg(
        func=ElevationFootWorkspacePenalty, weight=-0.8,
        params={"nominal_forward": -.02, "nominal_outward": .105,
                "forward_margin": .14, "lateral_margin": .12, "deviation_scale": .08,
                "terrain_extra_margin": .04},
    )
    cfg.rewards["feet_air_time"].weight = 0.0
    cfg.rewards["slip"].weight = -1.0
    cfg.rewards["first_contact_events"].weight = 0.2
    cfg.rewards["knee_extension"].weight = -0.25
    cfg.rewards["support_deficit"].weight = -0.25
    cfg.rewards["stance_balance"].weight = -0.15
    cfg.rewards["stance_balance"].params.update(smoothing_time=1.5, warmup_time=1.5, duty_tolerance=0.20)
    cfg.rewards["short_stance"].weight = -0.05
    cfg.rewards["short_stance"].params["min_stance_time"] = 0.16
    cfg.rewards["long_foot_air_time"].weight = -0.5
    cfg.rewards["long_foot_air_time"].params["max_air_time"] = 1.0
    cfg.rewards["lateral"].weight = -3.0
    cfg.rewards["lateral"].params.update(tolerance=0.015, scale=0.10)
    cfg.rewards["center"].weight = 0.0
    cfg.rewards["center"].params.update(tolerance=0.010, scale=0.10)
    cfg.rewards["heading"].weight = 0.0
    cfg.rewards["heading"].params.update(tolerance=math.radians(2.0), scale=math.radians(10.0))
    cfg.rewards["hip_abduction_bias"] = RewardTermCfg(
        func=HipAbductionBiasPenalty, weight=-0.5,
        params={
            "smoothing_time": 0.5, "warmup_time": 0.5,
            "tolerance_degrees": 4.0, "deviation_scale_degrees": 8.0,
        },
    )
    cfg.rewards["swing_clearance"] = RewardTermCfg(
        func=ElevationSwingClearancePenalty, weight=-0.4,
        params={
            "target_clearance": 0.15, "riser_clearance": 0.03, "lookahead": 0.16,
            "foot_radius": 0.025, "minimum_support_force": 8.0,
            "minimum_air_time": 0.02, "minimum_foot_progress": 0.03,
            "landing_cost_seconds": 0.10,
            "minimum_quality_air_time": .08, "minimum_stance_time": .10, "minimum_cycle_time": .40,
            "preferred_stride": .16, "maximum_stride": .26, "minimum_body_progress": .02,
            "reward_progress_scale": .16, "minimum_relative_lift": .025, "step_bonus": .16,
            "participation_warmup": .60, "participation_window": .90,
            "minimum_push_stance": .16, "target_push_stance": .35, "target_backstroke": .07,
            "maximum_stance_slip": .025, "maximum_stance_time": .85, "stance_time_scale": .25,
            "renewal_grace": .60, "renewal_timeout": 1.20,
            "landing_edge_margin": .05,
        },
    )
    cfg.rewards["step_quality"] = RewardTermCfg(func=elevation_gait_term, weight=1., params={"key": "quality"})
    cfg.rewards["foot_participation"] = RewardTermCfg(func=elevation_gait_term, weight=-1., params={"key": "participation"})
    cfg.rewards["stride_length"] = RewardTermCfg(func=elevation_gait_term, weight=-.4, params={"key": "stride"})
    cfg.rewards["stance_overstay"] = RewardTermCfg(func=elevation_gait_term, weight=-.3, params={"key": "stance"})
    cfg.rewards["body_clearance"] = RewardTermCfg(
        func=ElevationBodyClearancePenalty, weight=-.3,
        params={"minimum_clearance": .12, "deviation_scale": .06},
    )
    cfg.rewards["touchdown_speed"] = RewardTermCfg(
        func=ElevationTouchdownSpeedPenalty, weight=-.2,
        params={"speed_tolerance": .30, "deviation_scale": .50, "event_seconds": .10},
    )
    cfg.rewards["mean_leg_posture"] = RewardTermCfg(
        func=ElevationMeanPosturePenalty, weight=-.15,
        params={"smoothing_time": 1., "warmup_time": 1.2,
                "hip_tolerance_degrees": 8., "knee_tolerance_degrees": 12.,
                "hip_scale_degrees": 15., "knee_scale_degrees": 20.,
                "unequal_height": .03, "terrain_relaxation": 2.5,
                "terrain_weight": .25, "minimum_support_force": 8.},
    )
    cfg.rewards["time"].weight = 0.0
    cfg.rewards["yaw_rate"] = RewardTermCfg(
        func=elevation_cost, weight=-.75,
        params={"key": "yaw_rate", "tolerance": math.radians(2.), "scale": math.radians(20.)},
    )
    cfg.rewards["diagonal_gait"] = RewardTermCfg(
        func=DiagonalGaitPenalty, weight=-.2,
        params={"sensor_name": "elevation_contacts", "flat_only": True},
    )


def make_elevation_env_cfg(jump=False, play=False, terrain_kind=None):
    from .env_cfgs import _make_base_env_cfg
    from .evaluation import configure_evaluation

    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}
    settings = ElevationSettings(
        jump=jump, speed_range=(.2, .5) if jump else (.15, .25), terrain_kind=terrain_kind,
    )
    cfg.scene.num_envs = 256
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator", max_init_terrain_level=0,
        terrain_generator=TerrainGeneratorCfg(
            seed=42, size=TILE_SIZE, border_width=2.0, num_rows=5, num_cols=5 if jump else 6,
            layout_shape=(4, 4),
            curriculum=True, shuffle_curriculum_columns=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains=make_elevation_terrains(
                include_flat=not jump, stair_height_override=.12 if not jump else None,
            ),
        ),
    )
    cfg.commands["twist"] = ElevationVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(1000.0, 1000.0),
        rel_standing_envs=0, rel_forward_envs=1.0, heading_command=False,
        rel_heading_envs=0, linear_deadband=0.02, angular_deadband=0.01,
        ranges=ElevationVelocityCommandCfg.Ranges(
            lin_vel_x=settings.speed_range, lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0), heading=None,
        ),
    )
    cfg.events.pop("reset_base", None)
    cfg.events.pop("push_robot", None)
    cfg.events = {
        "select_elevation_terrain": EventTermCfg(
            func=select_elevation_terrain, mode="reset", params={"settings": settings}),
        **cfg.events,
        "reset_elevation": EventTermCfg(
            func=reset_elevation, mode="reset", params={"settings": settings, "randomize": True}),
    }
    sensors = list(cfg.scene.sensors)
    for sensor in sensors:
        if sensor.name == "height_scanner":
            sensor.pattern = ElevatedGridPatternCfg(resolution=0.08, size=(1.6, 1.0))
    sensors.extend((
        ContactSensorCfg(
            name="elevation_contacts",
            primary=ContactMatch(mode="body", pattern=FOOT_BODY_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal", "tangent"),
            reduce="maxforce", num_slots=4, history_length=4, global_frame=True,
        ),
        ContactSensorCfg(
            name="elevation_body_contacts",
            primary=ContactMatch(mode="body", entity="robot", pattern=(
                BASE_BODY_NAME, ".*_hip_abduction_Link", ".*_hip_pitch_Link", ".*_knee_Link")),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal", "tangent"), reduce="maxforce", num_slots=2, global_frame=True,
        ),
    ))
    cfg.scene.sensors = tuple(sensors)
    coefficients = {
        "velocity": 1.0, "center": 1.0, "heading": 0.5, "lateral": 0.5,
        "edge": 1.0, "slip": 0.5, "attitude": 0.25 if jump else 0.5,
        "angular": 0.1, "joint": 0.2, "torque": 0.02,
        "action": 0.02 if jump else 0.05, "contact_cost": 0.2, "time": 0.05,
    }
    cfg.rewards = {
        name: RewardTermCfg(func=elevation_cost, weight=-weight, params={"key": name})
        for name, weight in coefficients.items()
    }
    cfg.rewards.update({
        "legal_progress": RewardTermCfg(func=elevation_event, weight=5.0, params={"key": "shaping"}),
        "first_contact_events": RewardTermCfg(func=elevation_event, weight=0.5, params={"key": "new_events"}),
        "task_completed": RewardTermCfg(func=elevation_event, weight=20.0, params={"key": "complete"}),
        "task_failed": RewardTermCfg(func=elevation_event, weight=-50.0, params={"key": "failed"}),
    })
    if not jump:
        configure_walking_rewards(cfg)
    cfg.terminations = {
        "elevation_failed": TerminationTermCfg(func=elevation_failed),
        "elevation_completed": TerminationTermCfg(func=elevation_completed),
    }
    cfg.curriculum = {"elevation_levels": CurriculumTermCfg(
        func=ElevationTerrainCurriculum, params={"window_size": 100, "threshold": 0.8})}
    cfg.metrics = {"physics_support_count": MetricsTermCfg(func=collect_elevation_contacts, per_substep=True)}
    for name in ("complete", "failed", "invalid", "outside", "covered", "airborne", "crossed", "landed", "reason", "phase", "elapsed", "geometry_level", "stalled", "stall_elapsed", "stall_progress"):
        cfg.metrics["elevation_" + name] = MetricsTermCfg(
            func=elevation_metric, params={"key": name}, reduce="last")
    if not jump:
        for name in ("lateral_velocity", "lateral_speed_abs", "center_error_abs", "heading_error_deg", "support_pitch_deg"):
            cfg.metrics["elevation_" + name] = MetricsTermCfg(
                func=elevation_alignment_metric, params={"key": name},
            )
        for name in ("swing_peak_height", "near_riser_feet", "riser_deficit",
                     "low_swing_events_per_s", "completed_swings_per_s", "quality_steps_per_s",
                     "worst_leg_cycle_age_s", "stride_length", "target_peak_height"):
            cfg.metrics["elevation_" + name] = MetricsTermCfg(
                func=elevation_clearance_metric, params={"key": name},
                reduce="last" if name == "swing_peak_height" else "mean",
            )
        for name in ("true_lift_height", "landing_edge_margin"):
            cfg.metrics["elevation_" + name] = MetricsTermCfg(
                func=elevation_clearance_metric, params={"key": name}, reduce="last",
            )
    cfg.episode_length_s = 6.0 if jump else 60.0
    cfg.scale_rewards_by_dt = True
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 2048
    cfg.seed = 42
    if play:
        configure_evaluation(cfg)
        cfg.episode_length_s = 6.0 if jump else 60.0
        cfg.events["select_elevation_terrain"].params["randomize_levels"] = True
        cfg.events["reset_elevation"].params["randomize_approach"] = not jump
    return cfg
