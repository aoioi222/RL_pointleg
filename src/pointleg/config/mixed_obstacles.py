"""Mixed rough, fenced rough and movable tire-array training profile."""

from copy import deepcopy
from dataclasses import dataclass, fields

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg

from .env_cfgs import rough_env_cfg
from ..mdp.mixed_obstacles import (
    MixedObstacleVelocityCommandCfg, obstacle_completed, obstacle_failed,
    obstacle_completion_reward, obstacle_metric, protect_obstacles_from_push,
    reset_mixed_obstacles, ring_push_penalty, obstacle_stalled,
)
from ..mdp.obstacle_recovery import (
    FOOT_CENTER_SITES, ObstacleFootRecoveryReward, obstacle_recovery_metric,
)
from ..mdp.swing_metrics import FootSwingPeakHeightMetric
from ..robot_cfg import FOOT_BODY_NAMES
from ..terrains.mixed_obstacles import (
    FENCE_NAMES, RING_NAMES, RING_TERRAIN, FencedRoughTerrainCfg,
    RingArrayFloorCfg, ring_entity_cfg,
)


MIXED_EPISODE_LENGTH_S = 30.0


@dataclass
class RaisedFootGridPatternCfg(GridPatternCfg):
    height_offset: float = .35

    def generate_rays(self, mj_model, device):
        origins, directions = super().generate_rays(mj_model, device)
        origins[:, 2] += self.height_offset
        return origins, directions


def rough_mixed_env_cfg(play=False):
    cfg = rough_env_cfg(play=play)
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 2048)
    cfg.sim.njmax = max(cfg.sim.njmax or 0, 6144)
    cfg.episode_length_s = MIXED_EPISODE_LENGTH_S if not play else cfg.episode_length_s
    generator = cfg.scene.terrain.terrain_generator
    generator.difficulty_range = (1.0 / 3.0, 1.0)
    cfg.scene.terrain.max_init_terrain_level = 0
    if "terrain_levels" in cfg.curriculum:
        cfg.curriculum["terrain_levels"].params["initial_level_cap"] = 0
    rough = generator.sub_terrains["random_rough"]
    bumps = generator.sub_terrains["low_random_bumps"]
    rough.proportion = 0.026
    bumps.proportion = 0.024
    generator.sub_terrains.update({
        FENCE_NAMES[0]: FencedRoughTerrainCfg(background=deepcopy(rough), proportion=0.182, size=(8.0, 8.0)),
        FENCE_NAMES[1]: FencedRoughTerrainCfg(background=deepcopy(bumps), proportion=0.168, size=(8.0, 8.0)),
        RING_TERRAIN: RingArrayFloorCfg(proportion=0.60, size=(8.0, 8.0)),
    })
    generator.num_cols = len(generator.sub_terrains)
    for index, name in enumerate(RING_NAMES):
        cfg.scene.entities[name] = ring_entity_cfg(index)
    for sensor in cfg.scene.sensors:
        if sensor.name in ("feet_ground_contact", "base_ground_contact", "body_collision"):
            sensor.secondary = None
    cfg.scene.sensors += (
        ContactSensorCfg(
            name="obstacle_foot_contact",
            primary=ContactMatch(mode="body", pattern=FOOT_BODY_NAMES, entity="robot"),
            secondary=None, fields=("found", "force"), reduce="netforce",
        ),
        RayCastSensorCfg(
            name="obstacle_foot_scan",
            frame=tuple(ObjRef(type="site", name=name, entity="robot") for name in FOOT_CENTER_SITES),
            pattern=RaisedFootGridPatternCfg(size=(.16, .08), resolution=.02),
            ray_alignment="world", max_distance=1.,
            exclude_parent_body=True, include_geom_groups=(0,),
        ),
    )
    command = cfg.commands["twist"]
    cfg.commands["twist"] = MixedObstacleVelocityCommandCfg(**{
        field.name: deepcopy(getattr(command, field.name)) for field in fields(command)
    })
    cfg.events["reset_mixed_obstacles"] = EventTermCfg(
        func=reset_mixed_obstacles, mode="reset",
        params={"ring_start_x_range": (-0.3, 0.3)},
    )
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].func = protect_obstacles_from_push
    if "terrain_levels" in cfg.curriculum:
        cfg.curriculum["terrain_levels"].params["require_obstacle_completion"] = True
    cfg.terminations["obstacle_completed"] = TerminationTermCfg(func=obstacle_completed, time_out=True)
    cfg.terminations["obstacle_failed"] = TerminationTermCfg(func=obstacle_failed)
    cfg.terminations["obstacle_stalled"] = TerminationTermCfg(func=obstacle_stalled, params={"timeout_s": 8.0})
    cfg.rewards["obstacle_completed"] = RewardTermCfg(func=obstacle_completion_reward, weight=5.0)
    cfg.rewards["ring_push"] = RewardTermCfg(func=ring_push_penalty, weight=-0.5)
    cfg.rewards["obstacle_foot_recovery"] = RewardTermCfg(
        func=ObstacleFootRecoveryReward, weight=1.0,
        params={
            "contact_sensor_name": "obstacle_foot_contact",
            "height_sensor_name": "obstacle_foot_scan", "command_name": "twist",
            "clearance_m": .020, "foot_radius": .025,
            "block_duration": .04, "recovery_window": 2.0, "replant_duration": .10,
            "minimum_foot_progress": .12, "minimum_body_progress": .03,
            "minimum_horizontal_force": 6., "minimum_support_force": 8.,
            "horizontal_force_ratio": .05,
            "maximum_foot_speed": .08, "minimum_barrier_height": .015,
            "minimum_relative_lift": .020, "cooldown_s": .4,
            "reward_spacing": .20, "height_cost": 4.,
            "approach_height_cost": 2., "lift_progress_bonus": .15,
            "collision_cost": .30, "recovery_bonus": .10,
        },
    )
    for key in (
        "blocked_events_per_s", "recovered_events_per_s", "active_feet",
        "clearance_gap_m", "near_obstacle_feet", "side_contact_feet", "stalled_feet",
        "approach_feet", "lift_progress", "cleared_events_per_s",
    ):
        cfg.metrics[f"obstacle_foot_{key}"] = MetricsTermCfg(
            func=obstacle_recovery_metric, params={"key": key},
        )
    for name, key in (("obstacle_success", "complete"), ("obstacle_entered", "entered"), ("ring_peak_displacement_m", "peak_push"), ("ring_peak_tilt_deg", "peak_tilt")):
        cfg.metrics[name] = MetricsTermCfg(func=obstacle_metric, params={"key": key}, reduce="last")
    cfg.metrics["obstacle_stalled_s"] = MetricsTermCfg(func=obstacle_metric, params={"key": "stalled_s"}, reduce="last")
    cfg.metrics["swing_peak_height"] = MetricsTermCfg(
        func=FootSwingPeakHeightMetric, reduce="last",
        params={
            "sensor_name": "obstacle_foot_contact", "foot_radius": .025,
            "minimum_air_time": .08, "minimum_support_force": 8.,
        },
    )
    return cfg
