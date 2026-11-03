"""Check contact-point slip and forward-distance lift credit without physics."""

from copy import deepcopy
import math
import unittest

import torch

from mjlab.tasks.registry import load_rl_cfg
from legacy_high_step_config import load_env_cfg
from pointleg.mdp.high_step_support import high_step_contact_slip_penalty, high_step_short_stance_penalty
from test_high_step import cycle_env, run_cycle
from test_high_step_correction import correction_env


def support_tick(bundle, distance=.004):
    env, sensor, _, _, cfg, support = bundle
    env.common_step_counter += 1
    env.scene["robot"].data.root_link_pos_w[:, 0] += distance
    for foot, contact in enumerate(support.ids):
        sensor.data.pos[:, contact] = env.scene["robot"].data.site_pos_w[:, foot] + torch.tensor([0., 0., -.025])
    return support(env, **cfg.params)


def lift_env(height=.15, step_dt=.02, body_bounce=False, foot_progress=.04):
    env, sensor, cfg, term = cycle_env(step_dt=step_dt)
    feet = [0, 3]
    sensor.data.found[:, feet] = 0.
    sensor.data.force[:, feet] = 0.
    sensor.data.current_contact_time[:, feet] = 0.
    sensor.data.current_air_time[:, feet] = env.step_dt
    term(env, **cfg.params)
    env.scene["robot"].data.root_link_pos_w[:, 0] += .012
    term(env, **cfg.params)
    env.scene["robot"].data.site_pos_w[:, feet, 0] += foot_progress
    env.scene["robot"].data.site_pos_w[:, feet, 2] += height
    if body_bounce:
        env.scene["robot"].data.root_link_pos_w[:, 2] += height
    sensor.data.current_air_time[:, feet] = .12
    term(env, **cfg.params)
    return env, sensor, cfg, term


def lift_distance(bundle, distance):
    env, _, cfg, term = bundle
    env.scene["robot"].data.root_link_pos_w[:, 0] += distance
    term(env, **cfg.params)
    return term.feedback["lift"].clone() * env.step_dt


class ContactPointSlipTests(unittest.TestCase):
    def test_rolling_cancels_reference_velocity_without_penalty_or_drift(self):
        bundle = correction_env()
        env, _, _, _, _, support = bundle
        robot = env.scene["robot"].data
        robot.site_lin_vel_w[:, 0, 0] = .1
        robot.site_ang_vel_w[:, 0, 1] = 4.
        for _ in range(20):
            robot.site_pos_w[:, 0, 0] += .1 * env.step_dt
            support_tick(bundle)
        self.assertLess(support.slip_speed[:, 0].max().item(), 1.e-6)
        self.assertLess(support.drift[:, 0].max().item(), 1.e-6)
        self.assertEqual(high_step_contact_slip_penalty(env).sum().item(), 0.)
        self.assertGreater(env.extras["log"]["Metrics/slip_velocity_mean"].item(), .02)

    def test_dragging_has_a_bounded_cost_and_tolerance(self):
        bundle = correction_env()
        env, _, _, _, _, support = bundle
        env.scene["robot"].data.site_lin_vel_w[:, 0, 0] = .12
        for _ in range(4):
            support_tick(bundle)
        value = high_step_contact_slip_penalty(env)
        torch.testing.assert_close(value, torch.full((3,), .625), atol=1.e-5, rtol=0.)
        env.scene["robot"].data.site_lin_vel_w[:, 0, 0] = .015
        support_tick(bundle)
        self.assertEqual(high_step_contact_slip_penalty(env).sum().item(), 0.)

    def test_light_contacts_free_feet_and_initial_grace_are_not_slip(self):
        bundle = correction_env()
        env, sensor, _, _, _, support = bundle
        env.scene["robot"].data.site_lin_vel_w[:, 0, 0] = .5
        for _ in range(2):
            support_tick(bundle)
            self.assertEqual(support.slip_cost.sum().item(), 0.)
        support_tick(bundle)
        self.assertGreater(support.slip_cost.min().item(), .6)
        sensor.data.force[:, 0, 2] = 7.9
        support_tick(bundle)
        self.assertEqual(support.slip_cost.sum().item(), 0.)
        sensor.data.found[:, 0] = 0.
        support_tick(bundle)
        self.assertEqual(support.slip_cost.sum().item(), 0.)

    def test_normal_velocity_is_removed_and_contact_order_is_respected(self):
        for reverse in (False, True):
            bundle = correction_env(reverse)
            env, _, _, _, _, support = bundle
            robot = env.scene["robot"].data
            robot.site_lin_vel_w[:, 0, 2] = -.4
            for _ in range(5):
                support_tick(bundle)
            self.assertEqual(support.slip_speed.sum().item(), 0.)
            robot.site_lin_vel_w[:, 0, 0] = .12
            support_tick(bundle)
            self.assertAlmostEqual(support.slip_speed[0, 0].item(), .12, places=6)
            self.assertEqual(support.slip_speed[:, 1:].sum().item(), 0.)

    def test_returning_to_old_foot_position_does_not_erase_slip(self):
        bundle = correction_env()
        env, sensor, _, _, _, support = bundle
        robot = env.scene["robot"].data
        start = robot.site_pos_w[:, 0, 0].clone()
        for frame in range(20):
            velocity = .2 if frame < 10 else -.2
            robot.site_lin_vel_w[:, 0, 0] = velocity
            robot.site_pos_w[:, 0, 0] += velocity * env.step_dt
            support_tick(bundle)
        torch.testing.assert_close(robot.site_pos_w[:, 0, 0], start, atol=1.e-6, rtol=0.)
        self.assertGreater(support.drift[:, 0].min().item(), .06)
        sensor.data.found[:, 0] = 0.
        support_tick(bundle)
        self.assertEqual(support.swing_quality[:, 0].sum().item(), 0.)

    def test_same_step_partial_reset_preserves_other_env_slip_and_end_events(self):
        bundle = correction_env()
        env, sensor, _, _, cfg, support = bundle
        env.scene["robot"].data.site_lin_vel_w[:, 0, 0] = .12
        for _ in range(9):
            support_tick(bundle)
        slip = support.drift.clone()
        high_step_contact_slip_penalty(env)
        torch.testing.assert_close(support.drift, slip)
        sensor.data.found[:, 0] = 0.
        support_tick(bundle)
        short_cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["short_stance"]
        cost = high_step_short_stance_penalty(env, **short_cfg.params).clone()
        self.assertGreater(cost.min().item(), 0.)
        support.reset(torch.tensor([0]))
        support(env, **cfg.params)
        after = high_step_short_stance_penalty(env, **short_cfg.params)
        self.assertEqual(after[0].item(), 0.)
        torch.testing.assert_close(after[1:], cost[1:])

    def test_nonfinite_contact_data_is_filtered_and_world_frame_is_required(self):
        bundle = correction_env()
        env, sensor, _, _, _, support = bundle
        env.scene["robot"].data.site_lin_vel_w[0, 0, 0] = torch.nan
        sensor.data.pos[1, 0, 0] = torch.nan
        env.common_step_counter += 1
        support(env, **support.params)
        self.assertFalse(support.active[:2].any())
        self.assertTrue(torch.isfinite(support.slip_cost).all())
        self.assertTrue(torch.isfinite(support.drift).all())
        sensor.cfg = deepcopy(sensor.cfg)
        sensor.cfg.global_frame = False
        env.common_step_counter += 1
        with self.assertRaisesRegex(ValueError, "world-frame"):
            support(env, **support.params)

    def test_long_stance_has_less_soft_penalty_without_raising_cycle_gate(self):
        values = []
        short_cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["short_stance"]
        for frames in (9, 12, 18):
            bundle = correction_env()
            env, sensor, _, _, _, _ = bundle
            for _ in range(frames):
                support_tick(bundle)
            sensor.data.found[:, 0] = 0.
            support_tick(bundle)
            values.append(high_step_short_stance_penalty(env, **short_cfg.params)[0].item())
        self.assertGreater(values[0], 0.)
        self.assertAlmostEqual(values[1], 0., places=6)
        self.assertEqual(values[2], 0.)
        self.assertEqual(load_env_cfg("PointLeg-HighStep-v0").rewards["high_step_cycle"].params["min_stance_time"], .16)


class DistanceLiftTests(unittest.TestCase):
    def test_squared_height_scores_have_the_specified_values(self):
        for height in (.08, .12, .15):
            env, sensor, cfg, term = cycle_env()
            run_cycle(env, sensor, cfg, term, height=height)
            self.assertAlmostEqual(env.extras["log"]["Metrics/high_step_height_score"].mean().item(), (height / .15) ** 2, places=5)

    def test_overshoot_decay_is_preserved(self):
        env, sensor, cfg, term = cycle_env()
        run_cycle(env, sensor, cfg, term, height=.20)
        self.assertAlmostEqual(env.extras["log"]["Metrics/high_step_height_score"].mean().item(), math.exp(-(.05 / .03) ** 2), places=5)

    def test_lift_uses_new_distance_and_matches_the_payment_formula(self):
        bundle = lift_env()
        reward = lift_distance(bundle, .004)
        torch.testing.assert_close(reward, torch.full((3,), .02 * .004 / .12 * 2), atol=1.e-6, rtol=0.)
        self.assertEqual(lift_distance(bundle, 0.).sum().item(), 0.)
        self.assertEqual(lift_distance(bundle, -.004).sum().item(), 0.)
        self.assertEqual(lift_distance(bundle, .004).sum().item(), 0.)

    def test_overspeed_cannot_increase_the_distance_budget(self):
        normal = lift_distance(lift_env(), .004)
        excessive = lift_distance(lift_env(), .1)
        torch.testing.assert_close(normal, excessive)

    def test_same_quality_and_distance_have_same_reward_at_different_control_rates(self):
        totals = []
        for dt in (.01, .02):
            bundle = lift_env(step_dt=dt)
            total = torch.zeros(3)
            for _ in range(round(.20 / dt)):
                total += lift_distance(bundle, .2 * dt)
            totals.append(total)
        torch.testing.assert_close(totals[0], totals[1], atol=1.e-6, rtol=0.)

    def test_8_12_and_15_centimeter_lifts_are_ordered_at_equal_progress(self):
        values = [lift_distance(lift_env(height=h), .004)[0].item() for h in (.08, .12, .15)]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertAlmostEqual(values[0] / values[2], (.08 / .15) ** 2, places=5)

    def test_stationary_foot_body_bounce_flight_failure_and_long_air_do_not_pay(self):
        for mode in ("stationary_foot", "body_bounce", "flight", "failure", "long_air"):
            bundle = lift_env(body_bounce=mode == "body_bounce", foot_progress=0. if mode == "stationary_foot" else .04)
            env, sensor, cfg, term = bundle
            if mode == "flight":
                sensor.data.found.zero_()
                sensor.data.force.zero_()
            if mode == "failure":
                from types import SimpleNamespace
                env.termination_manager = SimpleNamespace(terminated=torch.ones(3, dtype=torch.bool))
            if mode == "long_air":
                sensor.data.current_air_time[:, [0, 3]] = .4
            self.assertEqual(lift_distance(bundle, .004).sum().item(), 0., mode)

    def test_invalid_swing_credit_cannot_be_banked_for_later(self):
        bundle = lift_env(foot_progress=0.)
        self.assertEqual(lift_distance(bundle, .03).sum().item(), 0.)
        bundle[0].scene["robot"].data.site_pos_w[:, [0, 3], 0] += .04
        self.assertEqual(lift_distance(bundle, 0.).sum().item(), 0.)

    def test_extra_swing_resets_cannot_repay_the_same_forward_ground(self):
        bundle = lift_env()
        env, sensor, cfg, term = bundle
        self.assertGreater(lift_distance(bundle, .004).min().item(), 0.)
        robot = env.scene["robot"].data
        feet = [0, 3]
        for _ in range(3):
            self.assertEqual(lift_distance(bundle, -.02).sum().item(), 0.)
            robot.site_pos_w[:, feet, 2] -= .15
            sensor.data.found[:, feet] = 1.
            sensor.data.force[:, feet, 2] = 30.
            sensor.data.current_contact_time[:, feet] = .36
            sensor.data.current_air_time[:, feet] = 0.
            term(env, **cfg.params)
            sensor.data.found[:, feet] = 0.
            sensor.data.force[:, feet] = 0.
            sensor.data.current_contact_time[:, feet] = 0.
            sensor.data.current_air_time[:, feet] = env.step_dt
            term(env, **cfg.params)
            robot.site_pos_w[:, feet, 0] += .04
            robot.site_pos_w[:, feet, 2] += .15
            sensor.data.current_air_time[:, feet] = .12
            self.assertEqual(lift_distance(bundle, .02).sum().item(), 0.)

    def test_partial_reset_and_nonfinite_root_do_not_create_free_credit(self):
        bundle = lift_env()
        env, _, cfg, term = bundle
        lift_distance(bundle, .004)
        term.reset(torch.tensor([0]))
        value = lift_distance(bundle, .004)
        self.assertEqual(value[0].item(), 0.)
        self.assertGreater(value[1:].min().item(), 0.)
        env.scene["robot"].data.root_link_pos_w[1, 0] = torch.nan
        value = lift_distance(bundle, .004)
        self.assertEqual(value[1].item(), 0.)
        self.assertTrue(torch.isfinite(value).all())


class SlipConfigTests(unittest.TestCase):
    def test_interfaces_other_tasks_and_unrelated_weights_are_preserved(self):
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        sensor = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
        self.assertIn("pos", sensor.fields)
        self.assertEqual(cfg.rewards["feet_slip"].weight, -.2)
        self.assertEqual(cfg.rewards["short_stance"].weight, -.1)
        self.assertEqual(cfg.rewards["high_step_cycle"].params["timing_weight"], .5)
        self.assertEqual(cfg.rewards["high_step_cycle"].params["minimum_support_force"], 8.)
        self.assertEqual(cfg.rewards["high_step_swing_retraction"].params["minimum_support_force"], 8.)
        self.assertEqual(cfg.rewards["high_step_yaw_rate"].weight, -1.)
        self.assertEqual(cfg.rewards["high_step_contact_force"].weight, -.5)
        self.assertEqual(cfg.rewards["high_step_cycle"].weight, .38)
        flat = load_env_cfg("PointLeg-Flat-v0")
        self.assertEqual(cfg.actions, flat.actions)
        self.assertEqual(cfg.observations, flat.observations)
        self.assertEqual(load_rl_cfg("PointLeg-HighStep-v0").algorithm, load_rl_cfg("PointLeg-Flat-v0").algorithm)
        for other in ("PointLeg-Flat-v0", "PointLeg-Elevation-v0", "PointLeg-Jump-v0"):
            self.assertNotEqual(load_env_cfg(other).rewards.get("feet_slip"), cfg.rewards["feet_slip"])


if __name__ == "__main__":
    unittest.main()
