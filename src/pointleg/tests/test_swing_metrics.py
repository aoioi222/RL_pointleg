"""Check support-relative swing peaks and event-only TensorBoard logging."""

import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.swing_metrics import FootSwingPeakHeightMetric
from pointleg.tests.test_obstacle_recovery import recovery_env


def metric_env():
    env, _, _ = recovery_env()
    env.extras = {"log": {}}
    cfg = load_env_cfg("PointLeg-RoughMixed-v0").metrics["swing_peak_height"]
    term = FootSwingPeakHeightMetric(cfg, env)
    return env, term, cfg


def tick(env, term, cfg):
    env.common_step_counter += 1
    env.extras["log"] = {}
    return term(env, **cfg.params).clone()


def set_foot(env, term, index, bottom, support=False, side=False):
    env.scene["robot"].data.site_pos_w[:, index, 2] = bottom + .025
    channel = term.contact_ids[index]
    data = env.scene["obstacle_foot_contact"].data
    data.found[:, channel] = float(support or side)
    data.force[:, channel] = torch.tensor([20., 0., 0.]) if side else torch.tensor([0., 0., 40. if support else 0.])


class SwingMetricTests(unittest.TestCase):
    def test_metric_is_registered_without_adding_a_height_reward_or_sensor(self):
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        self.assertIs(cfg.metrics["swing_peak_height"].func, FootSwingPeakHeightMetric)
        self.assertEqual(cfg.metrics["swing_peak_height"].reduce, "last")
        self.assertNotIn("foot_swing_height", cfg.rewards)
        self.assertEqual(len(cfg.rewards), 40)
        self.assertNotIn("swing_peak_height", load_env_cfg("PointLeg-Rough-v0").metrics)

    def test_peak_is_measured_at_physical_foot_bottom_and_logged_on_landing(self):
        env, term, cfg = metric_env()
        tick(env, term, cfg)
        for _ in range(4):
            set_foot(env, term, 0, .18)
            self.assertTrue((tick(env, term, cfg) == 0.).all())
            self.assertNotIn("Metrics/swing_peak_height", env.extras["log"])
        set_foot(env, term, 0, 0., support=True)
        result = tick(env, term, cfg)
        torch.testing.assert_close(result, torch.full((3,), .18))
        torch.testing.assert_close(env.extras["log"]["Metrics/swing_peak_height"], torch.full((3,), .18))
        self.assertIn("Metrics/swing_peak_height_fl", env.extras["log"])
        self.assertNotIn("Metrics/swing_peak_height_fr", env.extras["log"])
        tick(env, term, cfg)
        self.assertNotIn("Metrics/swing_peak_height", env.extras["log"])

    def test_all_feet_are_mapped_by_name_and_samples_are_kept_for_event_weighting(self):
        for index, leg in enumerate(("fl", "fr", "rl", "rr")):
            env, term, cfg = metric_env()
            tick(env, term, cfg)
            heights = torch.tensor([.10, .20, .30])
            for _ in range(4):
                set_foot(env, term, index, heights)
                tick(env, term, cfg)
            set_foot(env, term, index, 0., support=True)
            tick(env, term, cfg)
            logged = env.extras["log"][f"Metrics/swing_peak_height_{leg}"]
            torch.testing.assert_close(logged, heights)
            self.assertEqual(logged.numel(), 3)

    def test_raised_takeoff_surface_and_landing_height_are_handled(self):
        env, term, cfg = metric_env()
        env.scene["robot"].data.root_link_pos_w[:, 2] += 1.5
        env.scene["robot"].data.site_pos_w[:, :, 2] += 1.5
        tick(env, term, cfg)
        for _ in range(4):
            set_foot(env, term, 0, 1.60)
            tick(env, term, cfg)
        set_foot(env, term, 0, 1.65, support=True)
        tick(env, term, cfg)
        torch.testing.assert_close(env.extras["log"]["Metrics/swing_peak_height"], torch.full((3,), .15))
        for _ in range(4):
            set_foot(env, term, 0, 1.72)
            tick(env, term, cfg)
        set_foot(env, term, 0, 1.65, support=True)
        tick(env, term, cfg)
        torch.testing.assert_close(env.extras["log"]["Metrics/swing_peak_height"], torch.full((3,), .07))

    def test_body_lifting_is_visible_separately_from_relative_leg_lift(self):
        env, term, cfg = metric_env()
        tick(env, term, cfg)
        env.scene["robot"].data.root_link_pos_w[:, 2] += .10
        for _ in range(4):
            set_foot(env, term, 0, .10)
            tick(env, term, cfg)
        set_foot(env, term, 0, .10, support=True)
        tick(env, term, cfg)
        torch.testing.assert_close(env.extras["log"]["Metrics/swing_peak_height"], torch.full((3,), .10))
        torch.testing.assert_close(env.extras["log"]["Metrics/swing_peak_height_relative_body"], torch.zeros(3))

    def test_side_wall_contact_does_not_count_as_landing(self):
        env, term, cfg = metric_env()
        tick(env, term, cfg)
        for _ in range(5):
            set_foot(env, term, 0, .12, side=True)
            tick(env, term, cfg)
            self.assertNotIn("Metrics/swing_peak_height", env.extras["log"])
        set_foot(env, term, 0, .10, support=True)
        tick(env, term, cfg)
        torch.testing.assert_close(env.extras["log"]["Metrics/swing_peak_height"], torch.full((3,), .12))

    def test_initial_drop_and_short_contact_chatter_do_not_create_swings(self):
        env, term, cfg = metric_env()
        for _ in range(5):
            set_foot(env, term, 0, .20)
            tick(env, term, cfg)
        set_foot(env, term, 0, 0., support=True)
        tick(env, term, cfg)
        self.assertNotIn("Metrics/swing_peak_height", env.extras["log"])
        set_foot(env, term, 0, .05)
        tick(env, term, cfg)
        set_foot(env, term, 0, 0., support=True)
        tick(env, term, cfg)
        self.assertNotIn("Metrics/swing_peak_height", env.extras["log"])

    def test_partial_reset_same_tick_calls_and_invalid_samples_are_isolated(self):
        env, term, cfg = metric_env()
        tick(env, term, cfg)
        set_foot(env, term, 0, .18)
        tick(env, term, cfg)
        time = term.air_time.clone()
        term(env, **cfg.params)
        torch.testing.assert_close(term.air_time, time)
        term.reset(torch.tensor([1]))
        self.assertFalse(term.anchored[1].any())
        torch.testing.assert_close(term.peak[[0, 2], 0], torch.full((2,), .18))
        env.scene["robot"].data.site_pos_w[0, 0, 2] = torch.nan
        result = tick(env, term, cfg)
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(term.swinging[0, 0])
        term.reset()
        self.assertTrue((term.swing_count == 0).all())


if __name__ == "__main__":
    unittest.main()
