"""Verify straight-path costs and sustained hip bias without locking gait."""

from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.elevation import elevation_alignment_metric, elevation_cost
from pointleg.mdp.gait_bias import HipAbductionBiasPenalty
from pointleg.robot_cfg import LEG_NAMES
from pointleg.tests.test_elevation import fake_env


def bias_env():
    cfg = load_env_cfg("PointLeg-Elevation-v0").rewards["hip_abduction_bias"]
    names = tuple(f"{leg}_hip_abduction_joint" for leg in LEG_NAMES)
    robot = SimpleNamespace(
        find_joints=lambda name: ([names.index(name)], [name]),
        data=SimpleNamespace(joint_pos=torch.zeros(2, 4), default_joint_pos=torch.zeros(2, 4)),
    )
    env = SimpleNamespace(
        scene={"robot": robot}, num_envs=2, device="cpu", step_dt=.02,
        common_step_counter=0, extras={"log": {}},
    )
    return env, cfg.func(cfg, env), cfg


def bias_tick(env, term, cfg):
    env.common_step_counter += 1
    return term(env, **cfg.params).clone()


class ElevationAlignmentTests(unittest.TestCase):
    def test_changes_are_only_enabled_for_elevation_walking(self):
        walking = load_env_cfg("PointLeg-Elevation-v0")
        jumping = load_env_cfg("PointLeg-Jump-v0")
        self.assertEqual(walking.rewards["lateral"].weight, -3.)
        self.assertEqual(walking.rewards["center"].weight, 0.)
        self.assertEqual(walking.rewards["center"].params["tolerance"], .01)
        self.assertEqual(walking.rewards["center"].params["scale"], .10)
        self.assertEqual(walking.rewards["heading"].weight, 0.)
        self.assertEqual(walking.rewards["yaw_rate"].weight, -.75)
        self.assertEqual(jumping.rewards["lateral"].weight, -.5)
        self.assertEqual(jumping.rewards["lateral"].func.__module__, "pointleg.mdp.jump")
        for task in ("PointLeg-Jump-v0", "PointLeg-Rough-v0", "PointLeg-RoughMixed-v0", "PointLeg-Flat-v0"):
            self.assertNotIn("hip_abduction_bias", load_env_cfg(task).rewards)

    def test_lateral_cost_uses_body_velocity_and_keeps_world_metrics_for_navigation(self):
        cfg = load_env_cfg("PointLeg-Elevation-v0").rewards["lateral"]
        env = fake_env(count=3)
        env._elevation["last_step"][:] = env.common_step_counter
        env._elevation["updated_tick"] = env.common_step_counter
        env.scene["robot"].data.root_link_lin_vel_w[:, 1] = torch.tensor([.05, -.05, .01])
        env.scene["robot"].data.root_link_lin_vel_b[:, 1] = torch.tensor([.05, -.05, .01])
        new = elevation_cost(env, **cfg.params) * cfg.weight
        old = elevation_cost(env, key="lateral") * -.5
        self.assertAlmostEqual(new[0].item(), -.3675, places=5)
        self.assertAlmostEqual(new[0].item(), new[1].item(), places=6)
        self.assertEqual(new[2].item(), 0.)
        self.assertLess(new[0].item(), old[0].item())
        torch.testing.assert_close(elevation_alignment_metric(env, "lateral_velocity"), torch.tensor([.05, -.05, .01]))
        torch.testing.assert_close(elevation_alignment_metric(env, "lateral_speed_abs"), torch.tensor([.05, .05, .01]))

    def test_center_offsets_are_logged_without_an_unobservable_control_reward(self):
        env = fake_env(count=3)
        env._elevation["updated_tick"] = env.common_step_counter
        env.scene["robot"].data.root_link_pos_w[:, 1] += torch.tensor([.02, -.02, .005])
        cfg = load_env_cfg("PointLeg-Elevation-v0")
        term = cfg.rewards["center"]
        cost = elevation_cost(env, **term.params) * term.weight
        self.assertEqual(cost[0].item(), 0.)
        self.assertAlmostEqual(cost[0].item(), cost[1].item(), places=6)
        self.assertEqual(cost[2].item(), 0.)
        self.assertEqual(cfg.rewards["hip_abduction_bias"].weight, -.5)
        self.assertEqual(cfg.rewards["hip_abduction_bias"].params["tolerance_degrees"], 4.)

    def test_center_and_heading_are_navigation_metrics_when_drift_has_stopped(self):
        env = fake_env(count=2)
        env._elevation["last_step"][:] = env.common_step_counter
        env._elevation["updated_tick"] = env.common_step_counter
        env.scene["robot"].data.root_link_pos_w[:, 1] += torch.tensor([.10, -.10])
        env.scene["robot"].data.heading_w[:] = torch.deg2rad(torch.tensor([8., -8.]))
        cfg = load_env_cfg("PointLeg-Elevation-v0")
        for key in ("center", "heading"):
            reward = cfg.rewards[key]
            values = elevation_cost(env, **reward.params) * reward.weight
            self.assertTrue((values == 0.).all())
            self.assertAlmostEqual(values[0].item(), values[1].item(), places=5)
        torch.testing.assert_close(elevation_alignment_metric(env, "center_error_abs"), torch.full((2,), .1))
        torch.testing.assert_close(elevation_alignment_metric(env, "heading_error_deg"), torch.full((2,), 8.))

    def test_all_four_hips_receive_the_same_persistent_bias_penalty(self):
        for joint in range(4):
            env, term, cfg = bias_env()
            env.scene["robot"].data.joint_pos[:, joint] = torch.deg2rad(torch.tensor([12., -12.]))
            for _ in range(150):
                result = bias_tick(env, term, cfg)
            self.assertGreater(result.min().item(), .24)
            self.assertAlmostEqual(result[0].item(), result[1].item(), places=6)
            self.assertLess((result * cfg.weight).max().item(), 0.)
            self.assertIn(f"Metrics/hip_abduction_bias_abs_{LEG_NAMES[joint]}_deg", env.extras["log"])

    def test_short_or_alternating_hip_motion_is_not_treated_as_a_fixed_offset(self):
        env, term, cfg = bias_env()
        for step in range(150):
            env.scene["robot"].data.joint_pos[:] = torch.deg2rad(torch.tensor(12. if step % 2 else -12.))
            result = bias_tick(env, term, cfg)
        self.assertTrue((result == 0.).all())
        self.assertLess(torch.rad2deg(term.bias.abs()).max().item(), 4.)

    def test_bias_state_reset_same_tick_and_invalid_values_are_safe(self):
        env, term, cfg = bias_env()
        env.scene["robot"].data.joint_pos[:] = torch.deg2rad(torch.tensor(12.))
        for _ in range(40):
            bias_tick(env, term, cfg)
        before, time = term.bias.clone(), term.elapsed.clone()
        term(env, **cfg.params)
        torch.testing.assert_close(term.bias, before)
        torch.testing.assert_close(term.elapsed, time)
        term.reset(torch.tensor([0]))
        self.assertTrue((term.bias[0] == 0.).all())
        torch.testing.assert_close(term.bias[1], before[1])
        env.scene["robot"].data.joint_pos[0, 0] = torch.nan
        self.assertTrue(torch.isfinite(bias_tick(env, term, cfg)).all())
