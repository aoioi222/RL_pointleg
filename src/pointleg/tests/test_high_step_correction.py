"""Check the flat correction stage with loaded-foot tensor trajectories."""

import math
from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import load_rl_cfg
from legacy_high_step_config import load_env_cfg
from pointleg.mdp.high_step_support import high_step_velocity_reward
from test_high_step import cycle_env


def correction_env(reverse=False):
    env, sensor, cycle_cfg, cycle = cycle_env(reverse_contacts=reverse)
    env.common_step_counter = 0
    env.scene["robot"].data.root_link_lin_vel_b = torch.zeros(3, 3)
    env.termination_manager = SimpleNamespace(terminated=torch.zeros(3, dtype=torch.bool))
    sensor.data.current_contact_time.zero_()
    sensor.data.current_air_time.zero_()
    env._correction_previous_feet = env.scene["robot"].data.site_pos_w.clone()
    cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["high_step_support"]
    support = cfg.func(cfg, env)
    return env, sensor, cycle_cfg, cycle, cfg, support


def correction_tick(bundle, distance=0., lateral=0., vertical=0.):
    env, sensor, cycle_cfg, cycle, cfg, support = bundle
    env.common_step_counter += 1
    env.episode_length_buf += 1
    robot = env.scene["robot"].data
    robot.root_link_pos_w[:, 0] += distance
    robot.root_link_pos_w[:, 1] += lateral
    robot.root_link_pos_w[:, 2] += vertical
    robot.root_link_lin_vel_b[:, 0] = math.hypot(distance, lateral) / env.step_dt
    robot.site_lin_vel_w[:] = (robot.site_pos_w - env._correction_previous_feet) / env.step_dt
    env._correction_previous_feet.copy_(robot.site_pos_w)
    data = sensor.data
    for foot, contact in enumerate(support.ids):
        data.pos[:, contact] = robot.site_pos_w[:, foot] + torch.tensor([0., 0., -.025])
    found = data.found > 0
    data.last_contact_time = torch.where(~found & (data.current_contact_time > 0.), data.current_contact_time, data.last_contact_time)
    data.last_air_time = torch.where(found & (data.current_air_time > 0.), data.current_air_time, data.last_air_time)
    data.current_contact_time = torch.where(found, data.current_contact_time + env.step_dt, 0.)
    data.current_air_time = torch.where(~found, data.current_air_time + env.step_dt, 0.)
    env.extras["log"] = {}
    cost = support(env, **cfg.params).clone()
    reward = cycle(env, **cycle_cfg.params).clone()
    return cost, reward


def contacts(bundle, feet, loaded):
    _, sensor, _, _, _, support = bundle
    ids = [support.ids[i] for i in feet]
    sensor.data.found[:, ids] = float(loaded)
    sensor.data.force[:, ids] = 0.
    if loaded:
        sensor.data.force[:, ids, 2] = 30.


def push_then_lift(duration=.36, distance=.08, drift=0., rise=0., reverse=False, yaw=False):
    bundle = correction_env(reverse)
    env, _, _, _, _, support = bundle
    robot = env.scene["robot"].data
    if yaw:
        robot.root_link_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    correction_tick(bundle)
    for _ in range(round(duration / env.step_dt)):
        robot.site_pos_w[:, 0, 0] += drift / round(duration / env.step_dt)
        correction_tick(bundle, distance=0. if yaw else distance / round(duration / env.step_dt),
                        lateral=distance / round(duration / env.step_dt) if yaw else 0.,
                        vertical=rise / round(duration / env.step_dt))
    contacts(bundle, (0,), False)
    correction_tick(bundle)
    return bundle


def corrected_cycle(duration=.36, push=.08):
    bundle = correction_env()
    env, sensor, cycle_cfg, cycle, cfg, support = bundle
    for _ in range(round(duration / env.step_dt)):
        correction_tick(bundle, distance=push / round(duration / env.step_dt))
    feet = (0, 3)
    contacts(bundle, feet, False)
    start = env.scene["robot"].data.site_pos_w.clone()
    for i in range(12):
        fraction = (i + 1) / 12
        env.scene["robot"].data.site_pos_w[:, feet, 0] = start[:, feet, 0] + .12 * fraction
        env.scene["robot"].data.site_pos_w[:, feet, 2] = start[:, feet, 2] + .15 * math.sin(math.pi * fraction)
        correction_tick(bundle, distance=.004)
    contacts(bundle, feet, True)
    total = torch.zeros(3)
    for _ in range(12):
        total += correction_tick(bundle)[1] * env.step_dt
    return bundle, total


class HighStepCorrectionTests(unittest.TestCase):
    def test_full_stance_and_backstroke_improve_actual_cycle_rewards(self):
        bundle, full = corrected_cycle()
        _, short = corrected_cycle(duration=.18, push=.01)
        self.assertTrue((full > short * 3.).all())
        self.assertTrue((full > 1.8).all())
        self.assertEqual(correction_tick(bundle)[1].sum().item(), 0.)

    def test_world_fixed_feet_are_valid_and_body_bounce_is_not_a_push(self):
        bundle = push_then_lift()
        self.assertGreater(bundle[-1].swing_quality[:, 0].min().item(), .99)
        for args in ({"duration": .08}, {"distance": .005}, {"distance": 0., "rise": .08}, {"drift": .08}):
            with self.subTest(args=args):
                self.assertLess(push_then_lift(**args)[-1].swing_quality[:, 0].max().item(), .15)

    def test_backward_sliding_cannot_amplify_the_push(self):
        fixed = push_then_lift(distance=.01)[-1].swing_quality[:, 0]
        moving = push_then_lift(distance=.01, drift=-.005)[-1].swing_quality[:, 0]
        self.assertTrue((moving < fixed).all())

    def test_contact_order_and_heading_rotation_do_not_change_push_quality(self):
        expected = push_then_lift()[-1].swing_quality
        torch.testing.assert_close(push_then_lift(reverse=True)[-1].swing_quality, expected)
        torch.testing.assert_close(push_then_lift(yaw=True)[-1].swing_quality, expected)

    def test_fixed_feet_expire_velocity_reward_and_get_overstay_cost(self):
        bundle = correction_env()
        env, _, _, _, cfg, support = bundle
        for _ in range(75):
            cost, _ = correction_tick(bundle)
        self.assertEqual(support.renewal.sum().item(), 0.)
        self.assertGreater(cost.min().item(), .99)
        correction_tick(bundle, distance=.004)
        self.assertEqual(high_step_velocity_reward(env, .2, "twist").sum().item(), 0.)
        support.record_cycles(torch.tensor([[True, False, False, True]]).expand(3, -1))
        self.assertTrue((support.renewal > 0.).all())
        self.assertTrue((support.renewal < 1.).all())

    def test_tracking_does_not_reward_static_poses_or_repay_old_ground(self):
        bundle = correction_env()
        env = bundle[0]
        correction_tick(bundle)
        self.assertEqual(high_step_velocity_reward(env, .2, "twist").sum().item(), 0.)
        correction_tick(bundle, distance=.004)
        self.assertGreater(high_step_velocity_reward(env, .2, "twist").min().item(), .99)
        correction_tick(bundle, distance=-.004)
        correction_tick(bundle, distance=.004)
        self.assertEqual(high_step_velocity_reward(env, .2, "twist").sum().item(), 0.)
        env.commands[:] = 0.
        correction_tick(bundle)
        torch.testing.assert_close(high_step_velocity_reward(env, .2, "twist"), torch.ones(3))

    def test_cache_and_partial_reset_do_not_age_other_environments_twice(self):
        bundle = correction_env()
        env, _, _, _, cfg, support = bundle
        for _ in range(20):
            correction_tick(bundle)
        age, duration = support.age.clone(), support.duration.clone()
        support(env, **cfg.params)
        torch.testing.assert_close(support.age, age)
        torch.testing.assert_close(support.duration, duration)
        support.reset(torch.tensor([0]))
        self.assertEqual(support.age[0].sum().item(), 0.)
        torch.testing.assert_close(support.age[1:], age[1:])
        self.assertEqual(support.renewal[0].item(), 1.)

    def test_failure_invalid_motion_and_flight_mask_positive_rewards(self):
        bundle = correction_env()
        env, sensor, cycle_cfg, cycle, cfg, support = bundle
        correction_tick(bundle)
        correction_tick(bundle, distance=.004)
        env.termination_manager.terminated[:] = True
        self.assertEqual(high_step_velocity_reward(env, .2, "twist").sum().item(), 0.)
        self.assertEqual(cycle(env, **cycle_cfg.params).sum().item(), 0.)
        env.termination_manager.terminated[:] = False
        env.scene["robot"].data.root_link_pos_w[0, 0] = torch.nan
        correction_tick(bundle)
        self.assertTrue(torch.isfinite(support.cached_cost).all())
        self.assertTrue(torch.isfinite(high_step_velocity_reward(env, .2, "twist")).all())
        bundle = correction_env()
        contacts(bundle, (0, 1, 2, 3), False)
        correction_tick(bundle, distance=.004)
        self.assertEqual(high_step_velocity_reward(bundle[0], .2, "twist").sum().item(), 0.)

    def test_invalid_support_intervals_are_rejected(self):
        env, _, _, _, cfg, support = correction_env()
        for values in ({"minimum_force": 0.}, {"target_stance": .1}, {"renewal_timeout": .5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                support(env, **{**cfg.params, **values})

    def test_only_highstep_gets_the_flat_correction_settings(self):
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        self.assertEqual(cfg.rewards["high_step_cycle"].params["target_height"], .15)
        self.assertEqual(cfg.rewards["high_step_cycle"].params["target_cycle"], .6)
        self.assertEqual(cfg.rewards["high_step_workspace"].params["nominal_forward"], -.02)
        self.assertEqual(cfg.rewards["high_step_workspace"].params["forward_margin"], .14)
        self.assertEqual(cfg.rewards["high_step_phase"].params["stance_weight"], 0.)
        self.assertEqual(cfg.rewards["high_step_mean_leg_symmetry"].params["front_rear_weight"], .75)
        flat = load_env_cfg("PointLeg-Flat-v0")
        self.assertEqual(cfg.actions, flat.actions)
        self.assertEqual(list(cfg.observations["actor"].terms), list(flat.observations["actor"].terms))
        for task in ("PointLeg-Flat-v0", "PointLeg-Rough-v0", "PointLeg-Elevation-v0", "PointLeg-Jump-v0"):
            self.assertNotIn("high_step_support", load_env_cfg(task).rewards)
        self.assertEqual(load_rl_cfg("PointLeg-HighStep-v0").algorithm.learning_rate, 8.e-4)


if __name__ == "__main__":
    unittest.main()
