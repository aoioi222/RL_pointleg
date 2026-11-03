"""Explicit nominal and randomized evaluation configuration profiles."""

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg


EvaluationMode = Literal["nominal", "robust"]


def configure_evaluation(
    cfg: ManagerBasedRlEnvCfg, mode: EvaluationMode = "nominal"
) -> ManagerBasedRlEnvCfg:
    """Freeze curricula; nominal mode removes hardware/reset/noise randomness."""
    if mode not in ("nominal", "robust"):
        raise ValueError(f"Unknown evaluation mode: {mode}")
    cfg.seed = 42 if cfg.seed is None else cfg.seed
    cfg.episode_length_s = int(1e9)
    cfg.curriculum = {}
    if mode == "nominal":
        cfg.events = {
            name: term for name, term in cfg.events.items()
            if name in ("reset_scene", "positive_rewards", "reset_mixed_obstacles",
                        "select_elevation_terrain", "reset_elevation", "select_jump_terrain", "reset_jump")
        }
        if "reset_elevation" in cfg.events:
            cfg.events["reset_elevation"].params["randomize"] = False
            cfg.events["reset_elevation"].params["randomize_approach"] = False
        for group in cfg.observations.values():
            group.enable_corruption = False
        for action in cfg.actions.values():
            if hasattr(action, "min_delay"):
                action.min_delay = action.max_delay = 0
    terrain = cfg.scene.terrain
    if terrain is not None and terrain.terrain_generator is not None:
        generator = terrain.terrain_generator
        generator.curriculum = True
        generator.num_cols = len(generator.sub_terrains)
        generator.num_rows = 25 if "reset_jump" in cfg.events else 5
    return cfg


def evaluation_env_cfg(
    task_id: str, mode: EvaluationMode = "nominal"
) -> ManagerBasedRlEnvCfg:
    """Create either evaluation mode from a fresh training configuration."""
    from mjlab.tasks.registry import load_env_cfg

    return configure_evaluation(load_env_cfg(task_id, play=False), mode)
