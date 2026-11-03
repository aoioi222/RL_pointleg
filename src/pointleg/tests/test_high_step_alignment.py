"""Check straight walking and lateral leg placement without constraining lift."""

import math
import unittest

import torch

from legacy_high_step_config import load_env_cfg
from pointleg.robot_cfg import LEG_NAMES
from pointleg.tests.test_turn_rewards import turn_env


class HighStepAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_env_cfg("PointLeg-HighStep-v0")

    def test_sideways_cost_is_symmetric_and_stronger_outside_the_deadband(self):
        env = turn_env(4)
        env.command[:] = torch.tensor([.2, 0., 0.])
        env.scene["robot"].data.root_link_lin_vel_b[:, 1] = torch.tensor([.01, .05, -.05, .05])
        env.command[3] = 0.
        cfg = self.cfg.rewards["command_orthogonal_drift"]
        cost = cfg.func(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([0., .1225, .1225, 0.]))
        old = load_env_cfg("PointLeg-Flat-v0").rewards["command_orthogonal_drift"]
        self.assertGreater((cost * abs(cfg.weight))[1], (old.func(env, **old.params) * abs(old.weight))[1])

    def test_yaw_cost_uses_command_error_and_logs_signed_and_absolute_drift(self):
        env = turn_env(4)
        env.command[:] = torch.tensor([.2, 0., 0.])
        env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.deg2rad(torch.tensor([1., 12., -12., 12.]))
        env.command[3, 2] = math.radians(12.)
        env.scene["robot"].data.root_link_lin_vel_b[:, 1] = torch.tensor([.05, -.05, .01, 0.])
        cfg = self.cfg.rewards["high_step_yaw_rate"]
        cost = cfg.func(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([0., .25, .25, 0.]))
        log = env.extras["log"]
        torch.testing.assert_close(log["Metrics/high_step_lateral_speed_abs_m_s"], torch.tensor([.05, .05, .01, 0.]))
        torch.testing.assert_close(log["Metrics/high_step_yaw_rate_error_deg_s"], torch.tensor([1., 12., -12., 0.]))

    def test_workspace_detects_single_rear_splay_and_front_shift_but_allows_high_lift(self):
        env = turn_env(4)
        env.command[:] = torch.tensor([.2, 0., 0.])
        feet = env.scene["robot"].data.site_pos_w
        feet[1, 2, 1] += .08
        feet[2, 3, 1] -= .08
        feet[3, :2, 1] -= .08
        feet[:, :, 2] += .17
        cfg = self.cfg.rewards["high_step_workspace"]
        cost = cfg.func(cfg, env)(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([0., .0625, .0625, .125]))

    def test_front_and_rear_centers_cannot_cancel_each_other(self):
        env = turn_env(2)
        env.command[:] = torch.tensor([.2, 0., 0.])
        robot = env.scene["robot"].data
        robot.root_link_pos_w = torch.zeros(2, 3)
        robot.site_pos_w[0, :2, 1] += .04
        robot.site_pos_w[0, 2:, 1] -= .04
        robot.site_pos_w[1, :2, 1] -= .04
        robot.site_pos_w[1, 2:, 1] += .04
        env.scene["feet_ground_contact"].data.current_contact_time = torch.ones(2, 4)
        cfg = self.cfg.rewards["foot_lateral_placement"]
        cost = cfg.func(cfg, env)(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([.25, .25]), atol=1.e-6, rtol=1.e-5)

    def test_sustained_hip_bias_catches_each_leg_without_synchronizing_motion(self):
        env = turn_env(4)
        env.common_step_counter = 0
        robot = env.scene["robot"]
        names = tuple(f"{leg}_hip_abduction_joint" for leg in LEG_NAMES)
        robot.find_joints = lambda name: ([names.index(name)], [name])
        robot.data.joint_pos = torch.eye(4) * math.radians(12.)
        robot.data.default_joint_pos = torch.zeros(4, 4)
        cfg = self.cfg.rewards["hip_abduction_bias"]
        term = cfg.func(cfg, env)
        for _ in range(150):
            env.common_step_counter += 1
            cost = term(env, **cfg.params)
        self.assertTrue((cost > .3).all())
        torch.testing.assert_close(cost, cost[0].expand(4))
        term.reset()
        for step in range(150):
            env.common_step_counter += 1
            robot.data.joint_pos[:] = math.radians(12.) * math.sin(2. * math.pi * 2.5 * step * env.step_dt)
            cost = term(env, **cfg.params)
            self.assertEqual(cost.max().item(), 0.)
        term.reset(torch.tensor([0]))
        self.assertEqual(term.elapsed[0].item(), 0.)
        self.assertGreater(term.elapsed[1].item(), 2.)

    def test_only_high_step_changes_and_policy_terms_keep_the_same_interface(self):
        flat = load_env_cfg("PointLeg-Flat-v0")
        self.assertEqual(flat.rewards["command_orthogonal_drift"].weight, -1.5)
        self.assertEqual(flat.rewards["hip_abduction_excess"].params["tolerance"], .12)
        self.assertNotIn("hip_abduction_bias", flat.rewards)
        self.assertNotIn("high_step_yaw_rate", flat.rewards)
        self.assertNotIn("high_step_drift_bias", flat.rewards)
        self.assertEqual(list(self.cfg.observations["actor"].terms), list(flat.observations["actor"].terms))
        self.assertEqual(list(self.cfg.actions), list(flat.actions))
        self.assertGreater(self.cfg.rewards["high_step_cycle"].weight, 0.)
        self.assertEqual(self.cfg.rewards["high_step_cycle"].params["target_height"], .15)
        self.assertAlmostEqual(self.cfg.rewards["hip_abduction_excess"].params["tolerance"], math.radians(4.))

    def test_opposite_environment_drift_does_not_cancel_the_bias_penalty(self):
        env = turn_env(2)
        env.common_step_counter = 0
        env.command[:] = torch.tensor([.2, 0., 0.])
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_b[:, 1] = torch.tensor([.03, -.03])
        robot.root_link_ang_vel_b[:, 2] = torch.deg2rad(torch.tensor([6., -6.]))
        cfg = self.cfg.rewards["high_step_drift_bias"]
        term = cfg.func(cfg, env)
        for _ in range(150):
            env.common_step_counter += 1
            cost = term(env, **cfg.params)
        self.assertTrue((cost > .6).all())
        torch.testing.assert_close(cost, cost[0].expand(2))
        self.assertEqual(term.mean.mean(dim=0).abs().sum().item(), 0.)
        self.assertGreater(env.extras["log"]["Metrics/high_step_lateral_bias_abs_m_s"].min().item(), .029)

    def test_bias_allows_alternating_gait_sway_and_deadband_motion(self):
        env = turn_env(2)
        env.common_step_counter = 0
        env.command[:] = torch.tensor([.2, 0., 0.])
        robot = env.scene["robot"].data
        cfg = self.cfg.rewards["high_step_drift_bias"]
        term = cfg.func(cfg, env)
        for step in range(200):
            env.common_step_counter += 1
            sway = math.sin(2. * math.pi * 2.5 * step * env.step_dt)
            robot.root_link_lin_vel_b[:, 1] = torch.tensor([.05 * sway, .004])
            robot.root_link_ang_vel_b[:, 2] = torch.tensor([math.radians(8.) * sway, math.radians(.8)])
            cost = term(env, **cfg.params)
            if step > 60:
                self.assertLess(cost[0].item(), .01)
                self.assertEqual(cost[1].item(), 0.)

    def test_yaw_strength_increases_without_increasing_lateral_cost(self):
        env = turn_env(4)
        env.common_step_counter = 0
        env.command[:] = torch.tensor([.2, 0., 0.])
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_b[0, 1] = .03
        robot.root_link_ang_vel_b[1:3, 2] = torch.deg2rad(torch.tensor([6., -6.]))
        cfg = self.cfg.rewards["high_step_drift_bias"]
        new, old = cfg.func(cfg, env), cfg.func(cfg, env)
        old_params = {**cfg.params, "yaw_scale_degrees": 10.}
        for _ in range(150):
            env.common_step_counter += 1
            new_cost = new(env, **cfg.params).clone()
            old_cost = old(env, **old_params).clone()
        torch.testing.assert_close(new_cost[0], old_cost[0])
        torch.testing.assert_close(new_cost[1:3], old_cost[1:3] * (10. / 9.) ** 2)
        self.assertEqual(new_cost[3].item(), 0.)
        instant = self.cfg.rewards["high_step_yaw_rate"]
        raw = instant.func(env, **instant.params)
        self.assertEqual(instant.weight, -1.)
        self.assertLess((raw * instant.weight)[1], (raw * -.75)[1])

    def test_bias_updates_once_per_step_and_resets_per_environment(self):
        env = turn_env(2)
        env.common_step_counter = 0
        env.command[:] = torch.tensor([.2, 0., 0.])
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_b[:, 1] = .03
        cfg = self.cfg.rewards["high_step_drift_bias"]
        term = cfg.func(cfg, env)
        for _ in range(100):
            env.common_step_counter += 1
            term(env, **cfg.params)
        mean, age = term.mean.clone(), term.age.clone()
        term(env, **cfg.params)
        torch.testing.assert_close(term.mean, mean)
        torch.testing.assert_close(term.age, age)
        term.reset(torch.tensor([0]))
        self.assertEqual(term.mean[0].abs().sum().item(), 0.)
        torch.testing.assert_close(term.mean[1], mean[1])
        env.command[1] = 0.
        robot.root_link_lin_vel_b[0, 1] = torch.nan
        env.common_step_counter += 1
        cost = term(env, **cfg.params)
        torch.testing.assert_close(cost, torch.zeros(2))
        torch.testing.assert_close(term.mean, torch.zeros(2, 2))
        torch.testing.assert_close(term.age, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
