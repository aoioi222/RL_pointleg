"""Check bounded swing costs against actual stair and ramp geometry."""

import unittest

import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.elevation import reset_elevation, select_elevation_terrain
from pointleg.mdp.elevation_clearance import elevation_surface_height
from pointleg.terrains.elevation import CENTER_Y
from pointleg.tests.test_elevation import fake_env


def clearance_env(count=1, kind=5, level=0):
    env = fake_env(count=count)
    env._elevation_fixed_cases = True
    env.scene.terrain.terrain_types[:] = kind
    env.scene.terrain.terrain_levels[:] = level
    select_elevation_terrain(env, None, env._elevation_settings)
    reset_elevation(env, None, env._elevation_settings, randomize=False)
    env.extras = {"log": {}}
    cfg = env.cfg.rewards["swing_clearance"]
    term = cfg.func(cfg, env)
    for foot in range(4):
        set_foot(env, foot, 1.4, 0., support=True)
    return env, term, cfg


def set_foot(env, foot, x, z, support=False, side=False):
    corner = env._elevation["corner"]
    env.scene["robot"].data.site_pos_w[:, foot, 0] = corner[:, 0] + x
    env.scene["robot"].data.site_pos_w[:, foot, 1] = corner[:, 1] + CENTER_Y
    env.scene["robot"].data.site_pos_w[:, foot, 2] = corner[:, 2] + z
    sensor = env.scene["elevation_contacts"].data
    slot = foot * 4
    sensor.found[:, slot:slot + 4] = 0
    sensor.force[:, slot:slot + 4] = 0
    sensor.normal[:, slot:slot + 4] = 0
    sensor.found[:, slot] = float(support or side)
    if support:
        sensor.force[:, slot, 2] = 40.
        sensor.normal[:, slot, 2] = -1.
    if side:
        sensor.force[:, slot, 0] = 20.
        sensor.normal[:, slot, 0] = 1.


def tick(env, term, cfg):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    env.extras["log"] = {}
    return term(env, **cfg.params).clone()


def completed_swing(height, step_dt=.02):
    env, term, cfg = clearance_env()
    env.step_dt = step_dt
    tick(env, term, cfg)
    set_foot(env, 0, 1.6, height)
    for _ in range(round(.08 / step_dt)):
        tick(env, term, cfg)
    set_foot(env, 0, 1.7, 0., support=True)
    cost = tick(env, term, cfg)
    return env, term, cfg, cost


class ElevationClearanceTests(unittest.TestCase):
    def test_only_elevation_walking_has_the_new_cost_and_no_new_sensor_or_observation(self):
        walking = load_env_cfg("PointLeg-Elevation-v0")
        self.assertEqual(walking.rewards["swing_clearance"].weight, -.4)
        self.assertIn("elevation_swing_peak_height", walking.metrics)
        self.assertEqual(walking.rewards["feet_air_time"].weight, 0.)
        for task in ("PointLeg-Jump-v0", "PointLeg-Flat-v0", "PointLeg-Rough-v0", "PointLeg-RoughMixed-v0"):
            self.assertNotIn("swing_clearance", load_env_cfg(task).rewards)
        self.assertEqual(list(walking.observations["actor"].terms), list(load_env_cfg("PointLeg-Rough-v0").observations["actor"].terms))
        self.assertNotIn("swing_clearance", {sensor.name for sensor in walking.scene.sensors})

    def test_surface_reference_matches_all_five_walking_profiles_at_two_levels(self):
        env = fake_env(count=5)
        env._elevation_fixed_cases = True
        names = list(env.scene.terrain.cfg.terrain_generator.sub_terrains)
        probes = torch.tensor([1.2, 2.999, 3.001, 3.16, 3.48, 3.8, 4.2, 5.1, 6.4, 7.9, 9.5]).repeat(5, 1)
        for level in (0, 4):
            env.scene.terrain.terrain_levels[:] = level
            select_elevation_terrain(env, None, env._elevation_settings)
            reset_elevation(env, None, env._elevation_settings, randomize=False)
            actual = elevation_surface_height(env, probes)
            for index, kind in enumerate(env.scene.terrain.terrain_types.tolist()):
                profile = env.scene.terrain.cfg.terrain_generator.sub_terrains[names[kind]].generated_profile(level / 4)
                np.testing.assert_allclose(actual[index].numpy(), profile.surface_height(probes[index].numpy()), atol=1.e-6)

    def test_low_flat_swing_is_penalized_once_after_forward_landing(self):
        env, term, cfg, cost = completed_swing(.01)
        actual = float(cost[0]) * cfg.weight * env.step_dt
        self.assertAlmostEqual(actual, -.04 * (1 - .01 / cfg.params["target_clearance"]) ** 2, places=6)
        torch.testing.assert_close(env.extras["log"]["Metrics/swing_peak_height"], torch.tensor([.01]))
        self.assertGreater(float(term.metrics["low_swing_events_per_s"][0]), 0.)
        self.assertEqual(float(tick(env, term, cfg)[0]), 0.)
        self.assertNotIn("Metrics/swing_peak_height", env.extras["log"])

    def test_height_above_target_never_creates_positive_reward(self):
        for height in (.15, .20, .25):
            _, _, cfg, cost = completed_swing(height)
            self.assertAlmostEqual(float(cost[0]) * cfg.weight, 0., places=6)

    def test_flat_liftoff_and_descent_are_not_continuously_penalized(self):
        env, term, cfg = clearance_env()
        tick(env, term, cfg)
        for x, z in ((1.45, .005), (1.50, .16), (1.55, .005)):
            set_foot(env, 0, x, z)
            self.assertEqual(float(tick(env, term, cfg)[0]), 0.)
        set_foot(env, 0, 1.6, 0., support=True)
        self.assertEqual(float(tick(env, term, cfg)[0]), 0.)

    def test_upward_riser_uses_local_step_top_plus_three_centimeters(self):
        env, term, cfg = clearance_env(kind=0, level=4)
        for foot in range(4):
            set_foot(env, foot, 2.8, 0., support=True)
        tick(env, term, cfg)
        set_foot(env, 0, 2.98, .05)
        low = tick(env, term, cfg)
        self.assertGreater(float(low[0]), 0.)
        self.assertEqual(float(term.metrics["near_riser_feet"][0]), 1.)
        set_foot(env, 0, 2.98, .15)
        high = tick(env, term, cfg)
        self.assertAlmostEqual(float(high[0]), 0., places=6)
        self.assertAlmostEqual(float(env._elevation["top_height"][0]), .72, places=6)

    def test_support_feet_and_all_airborne_phases_have_no_riser_cost(self):
        env, term, cfg = clearance_env(kind=0, level=4)
        for foot in range(4):
            set_foot(env, foot, 2.98, 0., support=True)
        self.assertEqual(float(tick(env, term, cfg)[0]), 0.)
        for foot in range(4):
            set_foot(env, foot, 2.99, .01)
        self.assertEqual(float(tick(env, term, cfg)[0]), 0.)

    def test_wall_side_contact_does_not_count_as_support_or_landing(self):
        env, term, cfg = clearance_env(kind=0, level=4)
        tick(env, term, cfg)
        set_foot(env, 0, 2.98, .03, side=True)
        self.assertGreater(float(tick(env, term, cfg)[0]), 0.)
        self.assertNotIn("Metrics/swing_peak_height", env.extras["log"])
        self.assertEqual(float(term.swing_count[0]), 0.)

    def test_world_height_offset_does_not_change_the_riser_cost(self):
        values = []
        for offset in (0., 1.5):
            env, term, cfg = clearance_env(kind=0, level=4)
            env._elevation["corner"][:, 2] += offset
            env.scene["robot"].data.root_link_pos_w[:, 2] += offset
            for foot in range(4):
                set_foot(env, foot, 2.8, 0., support=True)
            tick(env, term, cfg)
            set_foot(env, 0, 2.98, .03)
            values.append(float(tick(env, term, cfg)[0]))
        self.assertAlmostEqual(values[0], values[1], places=6)

    def test_initial_fall_or_no_forward_stride_does_not_create_a_swing_event(self):
        env, term, cfg = clearance_env()
        for foot in range(4):
            set_foot(env, foot, 1.4, .1)
        tick(env, term, cfg)
        for foot in range(4):
            set_foot(env, foot, 1.4, 0., support=True)
        self.assertEqual(float(tick(env, term, cfg)[0]), 0.)
        set_foot(env, 0, 1.4, .01)
        tick(env, term, cfg)
        set_foot(env, 0, 1.4, 0., support=True)
        self.assertEqual(float(tick(env, term, cfg)[0]), 0.)
        self.assertEqual(float(term.swing_count[0]), 0.)

    def test_event_cost_is_independent_of_control_period(self):
        actual = []
        for dt in (.01, .02):
            _, _, cfg, cost = completed_swing(.01, dt)
            actual.append(float(cost[0]) * cfg.weight * dt)
        self.assertAlmostEqual(actual[0], actual[1], places=6)

    def test_same_tick_partial_reset_and_invalid_values_do_not_leak_state(self):
        env, term, cfg = clearance_env(count=2)
        tick(env, term, cfg)
        set_foot(env, 0, 1.6, .01)
        tick(env, term, cfg)
        set_foot(env, 0, 1.7, 0., support=True)
        cost = tick(env, term, cfg)
        before = term.swing_count.clone()
        torch.testing.assert_close(term(env, **cfg.params), cost)
        torch.testing.assert_close(term.swing_count, before)
        term.reset(torch.tensor([0]))
        self.assertEqual(float(term.swing_count[0]), 0.)
        self.assertEqual(float(term.swing_count[1]), 1.)
        env.scene["robot"].data.site_pos_w[0, 0, 2] = torch.nan
        env.scene["elevation_contacts"].data.force[0, 0] = torch.nan
        self.assertTrue(torch.isfinite(tick(env, term, cfg)).all())


if __name__ == "__main__":
    unittest.main()
