"""Check fixed competition-height stairs and reward rollback without physics."""

from copy import deepcopy
import math
import unittest

import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.evaluate_elevation import evaluation_cases
from pointleg.mdp.elevation import ElevationTerrainCurriculum, reset_elevation, select_elevation_terrain
from pointleg.terrains.elevation import STEP_DEPTH, geometry_profile
from test_elevation import fake_env


class FixedStairTests(unittest.TestCase):
    def test_training_and_play_use_only_competition_height_stairs(self):
        for play in (False, True):
            cfg = load_env_cfg("PointLeg-Elevation-v0", play=play)
            terrains = cfg.scene.terrain.terrain_generator.sub_terrains
            for name, sub in terrains.items():
                self.assertEqual(sub.stair_height_override, .12 if name.startswith("stairs") else None)
            self.assertEqual(STEP_DEPTH, .32)
        env = fake_env(count=10, stair_height_override=.12)
        lengths = []
        for row in env._elevation["profiles"]:
            for kind in (0, 1):
                profile = row[kind]
                expected = np.arange(1, 7) * .12 if kind == 0 else np.arange(5, -1, -1) * .12
                np.testing.assert_allclose([t[2] for t in profile.step_surfaces], expected)
                self.assertAlmostEqual(profile.top_height, .72)
                self.assertAlmostEqual(profile.spawn[2], .72 if kind == 1 else 0.)
                lengths.append(profile.stair_platform_length)
        self.assertGreater(len(set(lengths)), 1)

    def test_stair_resets_sample_variants_without_a_height_curriculum(self):
        env = fake_env(count=100, stair_height_override=.12)
        course = ElevationTerrainCurriculum(env.cfg.curriculum["elevation_levels"], env)
        settings = env._elevation_settings
        with torch.random.fork_rng():
            torch.manual_seed(23)
            select_elevation_terrain(env, None, settings)
        ids = torch.where(env.scene.terrain.terrain_types < 2)[0]
        self.assertGreater(env.scene.terrain.terrain_levels[ids].unique().numel(), 1)
        reset_elevation(env, ids, settings, randomize=False)
        torch.testing.assert_close(env._elevation["top_height"][ids], torch.full((len(ids),), .72))
        env.episode_length_buf[:] = 10
        before = env.scene.terrain.terrain_levels[ids].clone()
        for success in (True, False):
            env._elevation["complete"][ids] = success
            env._elevation["failed"][ids] = not success
            for _ in range(110):
                values = course(env, ids)
            torch.testing.assert_close(env.scene.terrain.terrain_levels[ids], before)
        self.assertEqual([len(course.windows[i]) for i in (0, 1)], [0, 0])
        for name in ("stairs_up", "stairs_down"):
            self.assertNotIn(name + "_cap", values)
            self.assertAlmostEqual(values[name + "_fixed_height_m"].item(), .12)

    def test_slope_curriculum_and_saved_state_remain_independent(self):
        env = fake_env(count=5, stair_height_override=.12)
        cfg = deepcopy(env.cfg.curriculum["elevation_levels"])
        cfg.params.update(window_size=2, threshold=.8)
        course = ElevationTerrainCurriculum(cfg, env)
        env.episode_length_buf[:] = 10
        env._elevation["complete"][:] = True
        for _ in range(2):
            course(env, torch.tensor([2, 3]))
        self.assertEqual(course.level_caps.tolist(), [0, 0, 1, 1, 0, 0])
        saved = course.state_dict()
        saved["level_caps"][:2] = [4, 3]
        saved["windows"][:2] = [[True], [False]]
        course.load_state_dict(saved)
        self.assertEqual(course.level_caps.tolist(), [0, 0, 1, 1, 0, 0])
        self.assertEqual([list(course.windows[i]) for i in (0, 1)], [[], []])
        self.assertAlmostEqual(geometry_profile("slope_up", 0.).slope_angle, math.radians(20 / 3))
        self.assertAlmostEqual(geometry_profile("slope_up", 1.).slope_angle, math.radians(20))

    def test_reward_rollback_and_height_contract_are_explicit(self):
        cfg = load_env_cfg("PointLeg-Elevation-v0")
        for name in ("hip_swing", "leg_clearance", "flat_hip_fold"):
            self.assertNotIn(name, cfg.rewards)
        self.assertNotIn("hip_step_bonus", cfg.rewards["swing_clearance"].params)
        self.assertEqual(cfg.rewards["swing_clearance"].params["target_clearance"], .15)
        stairs = cfg.scene.terrain.terrain_generator.sub_terrains["stairs_up"]
        contract = stairs.layout_contract()
        self.assertEqual(contract["fixed_step_height_m"], .12)
        old = deepcopy(stairs)
        old.stair_height_override = None
        self.assertNotEqual(contract, old.layout_contract())
        self.assertEqual(old.layout_contract()["revision"], 2)
        rough = load_env_cfg("PointLeg-Rough-v0")
        self.assertEqual(list(cfg.observations["actor"].terms), list(rough.observations["actor"].terms))
        for case in evaluation_cases("PointLeg-Elevation-v0"):
            self.assertEqual(case["fixed_step_height_m"], .12 if case["terrain_type"].startswith("stairs") else None)

    def test_fixed_height_rejects_invalid_geometry(self):
        for height in (0., -.01, .13, math.nan, math.inf):
            with self.subTest(height=height), self.assertRaises(ValueError):
                geometry_profile("stairs_up", .5, stair_height_override=height)
        with self.assertRaises(ValueError):
            geometry_profile("slope_up", .5, stair_height_override=.12)


if __name__ == "__main__":
    unittest.main()
