"""Verify every enabled task uses the compact atlas and consistent profiles."""

from copy import deepcopy
import unittest

import mujoco
import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator
from pointleg import TASK_SPECS
from pointleg.config.debug import configure_debug_training
from mjlab.tasks.registry import load_rl_cfg
from pointleg.mdp.elevation import reset_elevation, select_elevation_terrain
from pointleg.tests.test_elevation import fake_env


class TerrainLayoutTests(unittest.TestCase):
    def test_training_play_and_debug_configs_keep_task_geometry_layout(self):
        for task, specification in TASK_SPECS.items():
            if not specification.enabled:
                continue
            for play in (False, True):
                with self.subTest(task=task, play=play):
                    cfg = load_env_cfg(task, play=play)
                    expected = (10, 5) if task == "PointLeg-Jump-v0" else (4, 4)
                    self.assertEqual(cfg.scene.terrain.terrain_generator.layout_shape, expected)
                    if not play:
                        configure_debug_training(cfg, load_rl_cfg(task))
                        self.assertEqual(cfg.scene.terrain.terrain_generator.layout_shape, expected)

    def test_every_task_generates_its_requested_physical_cells(self):
        for task, specification in TASK_SPECS.items():
            if not specification.enabled:
                continue
            with self.subTest(task=task):
                cfg = load_env_cfg(task).scene.terrain.terrain_generator
                generator = TerrainGenerator(deepcopy(cfg))
                positions = []
                create = generator._create_terrain_geom
                def record(spec, position, *args):
                    positions.append(position.copy())
                    return create(spec, position, *args)
                generator._create_terrain_geom = record
                spec = mujoco.MjSpec()
                generator.compile(spec)
                spec.compile()
                expected = (10, 5) if task == "PointLeg-Jump-v0" else (4, 4)
                self.assertEqual(generator.grid_shape, expected)
                self.assertEqual(np.unique(positions, axis=0).shape, (expected[0] * expected[1], 3))
                self.assertEqual(np.unique(np.asarray(positions)[:, 0]).size, expected[0])
                self.assertEqual(np.unique(np.asarray(positions)[:, 1]).size, expected[1])

    def test_elevation_profiles_use_actual_mapped_geometry_level(self):
        env = fake_env()
        terrain = env.scene.terrain
        terrain.terrain_levels[:] = 1
        terrain.get_geometry_levels = lambda levels, kinds: torch.zeros_like(levels)
        terrain.terrain_level_indices = torch.tensor([[0], [0], [2], [2], [4]]).repeat(1, len(terrain.cfg.terrain_generator.sub_terrains))
        for sub in terrain.cfg.terrain_generator.sub_terrains.values():
            sub._generated_profiles = {level: profile for level, profile in sub._generated_profiles.items() if level in (0., .5, 1.)}
        del env._elevation
        env._elevation_fixed_cases = True
        settings = load_env_cfg("PointLeg-Elevation-v0").events["reset_elevation"].params["settings"]
        select_elevation_terrain(env, torch.arange(env.num_envs), settings)
        reset_elevation(env, torch.arange(env.num_envs), settings, randomize=False)
        for index, kind in enumerate(terrain.terrain_types.tolist()):
            profile = env._elevation["profiles"][0][kind]
            self.assertAlmostEqual(env._elevation["top_height"][index].item(), profile.top_height, places=5)


if __name__ == "__main__":
    unittest.main()
