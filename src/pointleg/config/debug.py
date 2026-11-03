"""Small training profile that preserves the task's policy architecture."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg


def configure_debug_training(
    env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg
) -> None:
    """Reduce rollout and terrain size; keep randomization and task semantics."""
    env.scene.num_envs = 16
    terrain = env.scene.terrain
    if terrain is not None and terrain.terrain_generator is not None:
        generator = terrain.terrain_generator
        generator.num_rows = 25 if "reset_jump" in env.events else (5 if "reset_elevation" in env.events else 2)
        generator.num_cols = len(generator.sub_terrains)
        generator.border_width = 2.0
        terrain.max_init_terrain_level = 0
    agent.num_steps_per_env = 8
    agent.algorithm.num_mini_batches = 1
    agent.algorithm.num_learning_epochs = 1
    agent.max_iterations = 2
    agent.save_interval = 1
    agent.logger = "tensorboard"
    agent.upload_model = False
    agent.run_name = "debug"
