"""Train one policy on both ascending and descending ramp curricula."""

from dataclasses import replace

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .elevation_basic import make_paired_elevation_env_cfg
from ..mdp.elevation_basic import ElevationBasicProlongedSwingPenalty, ElevationBasicStallPenalty
from ..mdp.elevation_support import ElevationMeanPosturePenalty
from ..mdp.slope import SlopeStepTimingPenalty, slope_hip_abduction_zero, slope_task_metric
from .rl_cfg import elevation_ppo_runner_cfg


def slope_env_cfg(play=False):
    cfg = make_paired_elevation_env_cfg("slope", play=play)
    original = cfg.events["reset_elevation"].params["settings"]
    settings = replace(original, speed_range=(.15, .30))
    for event in cfg.events.values():
        if event.params.get("settings") is original:
            event.params["settings"] = settings
    cfg.commands["twist"].ranges.lin_vel_x = settings.speed_range
    cfg.episode_length_s = 120.
    for key in ("up_fraction", "down_fraction", "exit_loaded_feet"):
        cfg.metrics["slope_" + key] = MetricsTermCfg(func=slope_task_metric, params={"key": key})
    cfg.rewards.update({
        "forward_stall": RewardTermCfg(
            func=ElevationBasicStallPenalty, weight=-1.0,
            params={"command_name": "twist", "minimum_forward_velocity": .05,
                    "grace_period": .60, "rise_time": .40},
        ),
        "prolonged_stalled_swing": RewardTermCfg(
            func=ElevationBasicProlongedSwingPenalty, weight=-1.0,
            params={"command_name": "twist", "minimum_force": 8.,
                    "minimum_forward_velocity": .05, "max_air_time": .35, "rise_time": .20},
        ),
        "hip_abduction_zero": RewardTermCfg(
            func=slope_hip_abduction_zero, weight=-7.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_abduction_joint",)),
                    "tolerance": 0., "deviation_scale": .25},
        ),
        "step_timing": RewardTermCfg(
            func=SlopeStepTimingPenalty, weight=-.6,
            params={"command_name": "twist", "minimum_force": 8., "target_period": .70,
                    "period_tolerance": .10, "period_scale": .30, "minimum_stance": .30,
                    "stance_scale": .20, "maximum_air": .55, "air_scale": .20,
                    "contact_confirmation": .06, "release_confirmation": .04,
                    "startup_grace": .80, "event_seconds": .10},
        ),
        "mean_leg_posture": RewardTermCfg(
            func=ElevationMeanPosturePenalty, weight=-.6,
            params={"smoothing_time": .8, "warmup_time": 1.2,
                    "hip_tolerance_degrees": 6., "knee_tolerance_degrees": 10.,
                    "hip_scale_degrees": 12., "knee_scale_degrees": 18.,
                    "unequal_height": .03, "terrain_relaxation": 2.5,
                    "terrain_weight": .25, "minimum_support_force": 8.},
        ),
    })
    return cfg


def slope_ppo_runner_cfg():
    cfg = elevation_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_slope"
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg
