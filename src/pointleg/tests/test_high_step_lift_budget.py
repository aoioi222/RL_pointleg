"""Validate distance-funded lift and excessive-height costs with synthetic motion."""

import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.high_step_basic import (
    basic_height_event, basic_height_overshoot, basic_height_progress,
)
from pointleg.mdp.high_step_hip_swing import HighStepHipSwingReward
from test_high_step_basic import basic_env, contact, tick
from test_high_step_coordination import coordination_env


def walk_cycle(bundle, distance=.07, height=.15, steps=20):
    env, _, _, term = bundle
    robot = env.scene["robot"].data
    base_z = robot.site_pos_w[:, 0, 2].clone()
    tick(bundle)
    for _ in range(steps):
        robot.root_link_pos_w[:, 0] += distance / steps
        tick(bundle)
    contact(bundle, 0, False)
    tick(bundle)
    event, progress, excess = (torch.zeros(env.num_envs) for _ in range(3))
    for _ in range(6):
        robot.site_pos_w[:, 0, 2] = base_z + height
        tick(bundle)
        progress += basic_height_progress(env) * env.step_dt
        excess += basic_height_overshoot(env) * env.step_dt
    robot.site_pos_w[:, 0, 2] = base_z
    robot.site_pos_w[:, 0, 0] += .05
    contact(bundle, 0, True)
    for _ in range(9):
        tick(bundle)
        event += basic_height_event(env) * env.step_dt
    return event, progress, excess


class HighStepLiftBudgetTests(unittest.TestCase):
    def test_dividing_equal_distance_into_more_steps_does_not_add_lift_reward(self):
        one = walk_cycle(basic_env(), .07)
        bundle = basic_env()
        short_a = walk_cycle(bundle, .035)
        short_b = walk_cycle(bundle, .035)
        for index in (0, 1):
            torch.testing.assert_close(one[index], short_a[index] + short_b[index], atol=1.e-5, rtol=0.)
            torch.testing.assert_close(one[index], torch.ones(3), atol=1.e-5, rtol=0.)

    def test_height_saturates_at_target_and_excessive_lift_has_increasing_cost(self):
        values = [walk_cycle(basic_env(), height=h) for h in (.08, .12, .15, .17, .20)]
        for index in (0, 1):
            self.assertTrue((values[0][index] < values[1][index]).all())
            self.assertTrue((values[1][index] < values[2][index]).all())
            torch.testing.assert_close(values[2][index], values[3][index])
            torch.testing.assert_close(values[3][index], values[4][index])
        self.assertLess(values[2][2].max().item(), 1.e-8)
        self.assertTrue((values[3][2] > 0.).all())
        self.assertTrue((values[4][2] > values[3][2]).all())

    def test_no_stance_progress_means_no_lift_credit(self):
        event, progress, _ = walk_cycle(basic_env(), distance=0.)
        self.assertEqual(event.sum().item(), 0.)
        self.assertEqual(progress.sum().item(), 0.)

    def test_retracing_old_ground_cannot_fund_another_step(self):
        bundle = basic_env()
        walk_cycle(bundle)
        env = bundle[0]
        robot = env.scene["robot"].data
        robot.root_link_pos_w[:, 0] -= .07
        tick(bundle)
        event, progress, _ = walk_cycle(bundle)
        self.assertLess(event.max().item(), 1.e-5)
        self.assertLess(progress.max().item(), 1.e-5)

    def test_overspeed_distance_is_not_banked(self):
        bundle = basic_env()
        env, _, _, term = bundle
        robot = env.scene["robot"].data
        tick(bundle)
        robot.root_link_pos_w[:, 0] += .5
        tick(bundle)
        torch.testing.assert_close(term.stance_progress[:, 0], torch.full((3,), .004), atol=1.e-6, rtol=0.)
        tick(bundle)
        torch.testing.assert_close(term.stance_progress[:, 0], torch.full((3,), .004), atol=1.e-6, rtol=0.)

    def test_cache_partial_reset_and_invalid_states_preserve_budget_integrity(self):
        bundle = basic_env()
        env, _, cfg, term = bundle
        walk_cycle(bundle)
        before = term.stance_progress.clone()
        term(env, **cfg.params)
        torch.testing.assert_close(term.stance_progress, before)
        frontier = term.progress_frontier.clone()
        term.reset(torch.tensor([0]))
        term(env, **cfg.params)
        torch.testing.assert_close(term.progress_frontier[1:], frontier[1:])
        self.assertEqual(term.swing_distance_credit[0].sum().item(), 0.)
        env.scene["robot"].data.root_link_pos_w[1, 0] = torch.nan
        env.termination_manager.terminated[2] = True
        tick(bundle)
        for value in (basic_height_event(env), basic_height_progress(env), basic_height_overshoot(env)):
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value[1:].sum().item(), 0.)

    def test_flight_does_not_receive_lift_progress_and_short_target_preserves_hard_gate(self):
        bundle = basic_env()
        env, _, cfg, term = bundle
        self.assertEqual(cfg.params["min_stance_time"], .16)
        self.assertEqual(cfg.params["short_stance_target"], .24)
        for _ in range(20):
            tick(bundle)
        for foot in range(4):
            contact(bundle, foot, False)
        tick(bundle)
        env.scene["robot"].data.site_pos_w[:, :, 2] += .20
        tick(bundle)
        self.assertEqual(basic_height_progress(env).sum().item(), 0.)
        self.assertEqual(basic_height_event(env).sum().item(), 0.)
        self.assertGreater(basic_height_overshoot(env).min().item(), 0.)
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        self.assertEqual(cfg.rewards["active_lift_overshoot"].weight, -.5)

    def test_hip_swing_event_uses_the_same_distance_budget(self):
        rewards = []
        for distance in (.035, .07):
            bundle = basic_env()
            env = bundle[0]
            robot = env.scene["robot"]
            reference = coordination_env()[0].scene["robot"]
            robot.joint_names = reference.joint_names
            robot.find_bodies = reference.find_bodies
            robot.data.joint_pos = reference.data.joint_pos.clone()
            robot.data.body_link_pos_w = reference.data.body_link_pos_w.clone()
            cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["hip_swing_coordination"]
            term = HighStepHipSwingReward(cfg, env)
            total = torch.zeros(3)

            def step():
                nonlocal total
                tick(bundle)
                total += term(env, **cfg.params) * env.step_dt

            step()
            for _ in range(20):
                robot.data.root_link_pos_w[:, 0] += distance / 20
                step()
            contact(bundle, 0, False)
            step()
            for _ in range(6):
                robot.data.site_pos_w[:, 0, 2] = .175
                robot.data.root_link_pos_w[:, 0] += .0035
                robot.data.joint_pos[:, term.joints[0]] += torch.pi / 54
                robot.data.body_link_pos_w[:, term.knees[0], 2] += .005
                step()
            robot.data.site_pos_w[:, 0, 2] = .025
            robot.data.site_pos_w[:, 0, 0] += .05
            contact(bundle, 0, True)
            for _ in range(10):
                step()
            rewards.append(total)
        self.assertGreater(rewards[1].min().item(), .9)
        torch.testing.assert_close(rewards[0] * 2., rewards[1], atol=1.e-5, rtol=0.)


if __name__ == "__main__":
    unittest.main()
