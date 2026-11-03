"""A single fixed jump policy for platform gaps and platform gaps with a wall."""

import mujoco
import numpy as np

from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg

from ..mdp.jump import (
    JumpCommandCfg, JumpSettings, JumpTerrainCurriculum, collect_jump_contacts,
    jump_completed, jump_cost, jump_event, jump_failed, jump_metric, reset_jump, select_jump_terrain,
)
from ..robot_cfg import FOOT_BODY_NAMES, get_robot_cfg
from ..terrains.jump import JumpTerrainCfg, LEVEL_COUNT, TERRAIN_NAMES, TILE_SIZE


def default_sole_offset(robot_cfg):
    """Place the lowest default ball sole on the top without a reset drop."""
    entity = Entity(robot_cfg)
    model = entity.spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    sole_z = np.array([data.geom_xpos[model.geom(name + "_collision").id, 2]
                       - model.geom_size[model.geom(name + "_collision").id, 0] for name in FOOT_BODY_NAMES])
    return -float(sole_z.min()) + .0005


def make_jump_env_cfg(play=False):
    from .env_cfgs import _make_base_env_cfg
    from .elevation import ElevatedGridPatternCfg
    from .evaluation import configure_evaluation

    cfg = _make_base_env_cfg()
    settings = JumpSettings()
    cfg.scene.entities = {"robot": get_robot_cfg()}
    cfg.scene.num_envs = 256
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator", max_init_terrain_level=0,
        terrain_generator=TerrainGeneratorCfg(
            seed=42, size=TILE_SIZE, border_width=1., num_rows=LEVEL_COUNT, num_cols=2,
            layout_shape=(10, 5), curriculum=True, shuffle_curriculum_columns=True,
            difficulty_range=(0., 1.),
            sub_terrains={name: JumpTerrainCfg(kind=name, size=TILE_SIZE, proportion=1.) for name in TERRAIN_NAMES},
        ),
    )
    cfg.commands = {"twist": JumpCommandCfg(resampling_time_range=(1000., 1000.), duration=settings.episode_duration)}
    cfg.events.pop("reset_base", None)
    cfg.events.pop("push_robot", None)
    cfg.events = {
        "select_jump_terrain": EventTermCfg(func=select_jump_terrain, mode="reset", params={"settings": settings}),
        **cfg.events,
        "reset_jump": EventTermCfg(func=reset_jump, mode="reset", params={
            "settings": settings, "sole_offset": default_sole_offset(cfg.scene.entities["robot"]),
        }),
    }
    sensors = list(cfg.scene.sensors)
    for sensor in sensors:
        if sensor.name == "height_scanner":
            sensor.pattern = ElevatedGridPatternCfg(resolution=.08, size=(1.6, 1.0))
    sensors.extend((
        ContactSensorCfg(
            name="jump_feet", primary=ContactMatch(mode="body", pattern=FOOT_BODY_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal", "tangent"), reduce="none",
            num_slots=4, global_frame=True,
        ),
        ContactSensorCfg(
            name="jump_body", primary=ContactMatch(mode="geom", pattern=".*_collision", entity="robot",
                                                   exclude=(".*_foot_Link_collision",)),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "pos"), reduce="none", num_slots=4,
        ),
    ))
    cfg.scene.sensors = tuple(sensors)
    cfg.rewards = {
        name: RewardTermCfg(func=jump_event, weight=weight, params={"key": key})
        for name, key, weight in (
            ("legal_progress", "shaping", 5.), ("takeoff", "new_events", 1.),
            ("clearance_progress", "clearance_increment", 2.), ("first_landing", "landing_events", 3.),
            ("task_completed", "complete", 30.), ("task_failed", "failed", -50.),
        )
    }
    cfg.rewards.update({key: RewardTermCfg(func=jump_cost, weight=-weight, params={"key": key})
                        for key, weight in (("time", .2), ("center", .5), ("lateral", .5), ("heading", .3),
                                            ("attitude", .2), ("torque", .02), ("action", .01),
                                            ("settle_speed", 1.), ("settle_angular", .2), ("slip", .2))})
    cfg.terminations = {"jump_failed": TerminationTermCfg(func=jump_failed),
                        "jump_completed": TerminationTermCfg(func=jump_completed)}
    cfg.curriculum = {"jump_levels": CurriculumTermCfg(func=JumpTerrainCurriculum,
                                                      params={"window_size": 100, "threshold": .8})}
    cfg.metrics = {"jump_support_count": MetricsTermCfg(func=collect_jump_contacts, per_substep=True)}
    for key in ("complete", "failed", "reason", "phase", "airborne", "landed", "four_foot_support",
                "distance_min", "distance_max", "distance_mean", "distance_valid", "peak_clearance",
                "takeoff_latency", "stable_time", "settle_fraction", "landing_speed", "bad_contact",
                "body_contact", "wall_contact", "late_launch", "runup", "distance_failed", "multiple_jump",
                "timeout", "invalid", "height", "gap", "wall_rise", "height_level", "gap_level"):
        cfg.metrics["jump_" + key] = MetricsTermCfg(func=jump_metric, params={"key": key}, reduce="last")
    cfg.episode_length_s = settings.episode_duration
    cfg.scale_rewards_by_dt = True
    cfg.seed = 42
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 2048
    if play:
        configure_evaluation(cfg)
        cfg.episode_length_s = settings.episode_duration
        cfg.events["select_jump_terrain"].params["randomize_levels"] = True
    return cfg
