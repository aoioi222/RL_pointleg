"""Check cadence budgets, height discrimination and compatible phase limits."""

import unittest

import torch

from legacy_high_step_config import load_env_cfg
from test_high_step import cycle_env, run_cycle


def cycle_return(stance=.36, air=.24, height=.15, **overrides):
    env, sensor, cfg, term = cycle_env()
    cfg.params.update(overrides)
    settled = run_cycle(
        env, sensor, cfg, term, stance_time=stance, air_time=air, height=height,
    )[1]
    return float(settled[0] * env.step_dt)


class HighStepPeriodTests(unittest.TestCase):
    def test_cycle_budget_removes_rate_advantage_when_other_scores_are_equal(self):
        fast = cycle_return(stance=.18, air=.12, timing_weight=0.) / .3
        target = cycle_return(timing_weight=0.) / .6
        self.assertAlmostEqual(fast, target, places=5)
        delayed = cycle_return(stance=.96, air=.24, timing_weight=0.) / 1.2
        self.assertLess(delayed, target)

    def test_six_tenths_second_target_beats_fast_lower_steps_per_second(self):
        target = cycle_return() / .6
        fast = cycle_return(stance=.188, air=.138, height=.13) / .326
        self.assertGreater(target, 1.5 * fast)
        weaker_fast = cycle_return(
            stance=.188, air=.138, height=.13, timing_weight=.25,
        ) / .326
        self.assertLess(fast, weaker_fast)

    def test_thirteen_and_fifteen_centimeters_keep_a_clear_score_gap(self):
        target = cycle_return()
        lower = cycle_return(height=.13)
        old_lower = cycle_return(height=.13, height_std=.05)
        self.assertLess(lower / target, .8)
        self.assertAlmostEqual(lower, old_lower, places=6)

    def test_phase_limits_and_landing_distance_allow_the_new_target(self):
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        cycle = cfg.rewards["high_step_cycle"].params
        phase = cfg.rewards["high_step_phase"]
        self.assertAlmostEqual(cycle["target_cycle"], .6)
        self.assertGreaterEqual(phase.params["max_air_time"], cycle["max_air_time"])
        maximum_stride = cfg.commands["twist"].ranges.lin_vel_x[1] * cycle["target_cycle"]
        self.assertGreaterEqual(cycle["max_target_step_length"], maximum_stride)
        self.assertGreater(cycle["max_step_length"], maximum_stride)
        env, sensor, cycle_cfg, term = cycle_env()
        env.commands[:, 0] = .25
        settled = run_cycle(env, sensor, cycle_cfg, term, stride=.15)[1]
        torch.testing.assert_close(settled * env.step_dt, torch.full((3,), 2.))
        sensor.data.current_contact_time[:] = .36
        sensor.data.current_contact_time[:, 0] = 0.
        sensor.data.current_air_time[:, 0] = .24
        torch.testing.assert_close(phase.func(env, **phase.params), torch.zeros(3))

    def test_invalid_timing_weights_and_step_caps_are_rejected(self):
        for overrides in ({"timing_weight": -.1}, {"timing_weight": .8},
                          {"max_target_step_length": .19}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    cycle_return(**overrides)


if __name__ == "__main__":
    unittest.main()
