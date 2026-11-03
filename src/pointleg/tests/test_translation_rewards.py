"""Check lateral tracking, backward stepping and low-speed reward gates."""

import math
from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.rewards import (
    TranslationFootWorkspacePenalty,
    TranslationStanceOverstayPenalty,
    command_orthogonal_drift_penalty,
    feet_contact_without_cmd,
)
from pointleg.tests.test_turn_rewards import turn_env, set_yaw


class TranslationRewardTests(unittest.TestCase):
    def test_drift_penalty_is_enabled_for_flat_and_rough(self):
        rough, flat = load_env_cfg("PointLeg-Rough-v0"), load_env_cfg("PointLeg-Flat-v0")
        for name in ("translation_stance_overstay", "translation_foot_workspace"):
            self.assertLess(rough.rewards[name].weight, 0.0)
            self.assertNotIn(name, flat.rewards)
        self.assertLess(rough.rewards["command_orthogonal_drift"].weight, 0.0)
        self.assertLess(flat.rewards["command_orthogonal_drift"].weight, 0.0)
        self.assertEqual(
            flat.rewards["command_orthogonal_drift"].params,
            rough.rewards["command_orthogonal_drift"].params,
        )
        for name in ("turn_stance_twist", "turn_foot_workspace", "feet_slip", "knee_extension"):
            self.assertIn(name, rough.rewards)
        self.assertEqual(list(rough.observations["actor"].terms), list(flat.observations["actor"].terms))

    def test_lateral_forward_drift_cost_is_symmetric_in_both_directions(self):
        env = turn_env(4)
        env.command[:] = torch.tensor([[0., 0.3, 0.], [0., -0.3, 0.], [0., 0.3, 0.], [0., -0.3, 0.]])
        env.scene["robot"].data.root_link_lin_vel_b[:] = torch.tensor([
            [0.19, 0.3, 0.], [0.19, -0.3, 0.], [-0.19, 0.3, 0.], [-0.19, -0.3, 0.],
        ])
        torch.testing.assert_close(command_orthogonal_drift_penalty(env, "twist"), torch.ones(4))

    def test_perpendicular_cost_allows_correct_diagonal_and_backward_commands(self):
        env = turn_env(4)
        env.command[:] = torch.tensor([[0.3, 0.4, 0.], [-0.3, 0.4, 0.], [-0.3, 0., 0.], [0., 0.3, 0.]])
        env.scene["robot"].data.root_link_lin_vel_b[:] = env.command
        env.scene["robot"].data.root_link_lin_vel_b[3, 0] = 0.03
        torch.testing.assert_close(command_orthogonal_drift_penalty(env, "twist"), torch.zeros(4))
        env.scene["robot"].data.root_link_lin_vel_b[2, 1] = 0.19
        torch.testing.assert_close(command_orthogonal_drift_penalty(env, "twist"), torch.tensor([0., 0., 1., 0.]))

    def test_perpendicular_cost_ignores_standing_pure_turns_and_tiny_commands(self):
        env = turn_env(4)
        env.command[0] = 0.0
        env.command[2] = torch.tensor([0.0, 0.04, 0.0])
        env.command[3] = torch.tensor([0.0, 0.06, 0.0])
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 20.0
        cost = command_orthogonal_drift_penalty(env, "twist")
        torch.testing.assert_close(cost, torch.tensor([0., 0., 0., 4.]))
        self.assertTrue(torch.isfinite(cost).all())

    def test_low_speed_progress_cost_covers_backward_and_lateral_stalls(self):
        env = turn_env(6)
        env.command[:] = torch.tensor([
            [-0.08, 0., 0.], [0., 0.08, 0.], [0., -0.08, 0.],
            [-0.08, 0., 0.], [0., 0., 0.], [0., 0.04, 0.],
        ])
        env.scene["robot"].data.root_link_lin_vel_b[3, 0] = -0.06
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["command_progress_deficit"]
        torch.testing.assert_close(cfg.func(env, **cfg.params), torch.tensor([1., 1., 1., 0., 0., 0.]))

    def test_low_speed_movement_is_not_rewarded_as_standing(self):
        env = turn_env(3)
        env.command[:] = torch.tensor([[0., 0., 0.], [-0.08, 0., 0.], [0., 0.08, 0.]])
        robot = env.scene["robot"]
        robot.data.joint_pos = torch.ones(3, 12)
        robot.data.default_joint_pos = torch.zeros(3, 12)
        robot.data.projected_gravity_b = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["stand_still"]
        params = {**cfg.params, "asset_cfg": SimpleNamespace(name="robot", joint_ids=list(range(12)))}
        torch.testing.assert_close(cfg.func(env, **params), torch.tensor([1., 0., 0.]))
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["feet_contact_without_cmd"]
        torch.testing.assert_close(cfg.func(env, **cfg.params), torch.tensor([4., 0., 0.]))
        torch.testing.assert_close(feet_contact_without_cmd(env, "twist", "feet_ground_contact"), torch.full((3,), 4.0))

    def test_overstay_penalizes_back_and_both_lateral_directions_not_forward_or_turn(self):
        env = turn_env(6)
        env.command[:] = torch.tensor([
            [-0.3, 0., 0.], [0., 0.3, 0.], [0., -0.3, 0.],
            [0.3, 0., 0.], [0., 0., 0.5], [0., 0., 0.],
        ])
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_stance_overstay"]
        term = TranslationStanceOverstayPenalty(cfg, env)
        for _ in range(30):
            cost = term(env, **cfg.params)
        self.assertLess(cost.abs().max().item(), 1.0e-6)
        for _ in range(15):
            cost = term(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([1., 1., 1., 0., 0., 0.]), atol=1.0e-5, rtol=1.0e-5)

    def test_stance_limit_adapts_to_speed_and_is_bounded(self):
        env = turn_env(3)
        env.command[:] = torch.tensor([[-0.06, 0., 0.], [-0.3, 0., 0.], [-1., 0., 0.]])
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_stance_overstay"]
        term = TranslationStanceOverstayPenalty(cfg, env)
        for _ in range(20):
            cost = term(env, **cfg.params)
        self.assertEqual(cost[0].item(), 0.0)
        self.assertEqual(cost[1].item(), 0.0)
        self.assertGreater(cost[2].item(), 0.0)
        for _ in range(55):
            cost = term(env, **cfg.params)
        self.assertTrue((cost > 0.0).all())
        self.assertLessEqual(cost.max().item(), 4.0)

    def test_normal_alternating_support_is_not_penalized(self):
        env = turn_env(1)
        env.command[:] = torch.tensor([[-0.3, 0., 0.]])
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_stance_overstay"]
        term = TranslationStanceOverstayPenalty(cfg, env)
        sensor = env.scene["feet_ground_contact"]
        for step in range(200):
            contact = torch.tensor([[1., 0., 0., 1.]]) if (step // 15) % 2 else torch.tensor([[0., 1., 1., 0.]])
            sensor.data.found[:] = contact
            sensor.data.current_air_time[:] = torch.where(contact > 0, 0.0, sensor.data.current_air_time + env.step_dt)
            self.assertLess(term(env, **cfg.params).abs().max().item(), 1.0e-6)

    def test_overstay_history_survives_contact_flicker_and_resets_per_foot(self):
        env = turn_env(2)
        env.command[:] = torch.tensor([[-0.3, 0., 0.]])
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_stance_overstay"]
        term = TranslationStanceOverstayPenalty(cfg, env)
        for _ in range(45):
            term(env, **cfg.params)
        sensor = env.scene["feet_ground_contact"]
        sensor.data.found[0, 3] = 0.0
        sensor.data.current_air_time[0, 3] = 0.02
        term(env, **cfg.params)
        self.assertGreater(term.stance_age[0, 0].item(), 0.8)
        sensor.data.current_air_time[0, 3] = 0.10
        term(env, **cfg.params)
        self.assertEqual(term.stance_age[0, 0].item(), 0.0)
        self.assertGreater(term.stance_age[0, 1].item(), 0.8)
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.stance_age[0], torch.zeros(4))
        self.assertGreater(term.stance_age[1].min().item(), 0.8)
        term.reset()
        torch.testing.assert_close(term.stance_age, torch.zeros(2, 4))

    def test_standing_or_forward_command_clears_overstay_history(self):
        env = turn_env(2)
        env.command[:] = torch.tensor([[-0.3, 0., 0.]])
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_stance_overstay"]
        term = TranslationStanceOverstayPenalty(cfg, env)
        for _ in range(45):
            term(env, **cfg.params)
        env.command[0] = 0.0
        env.command[1] = torch.tensor([0.3, 0., 0.])
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))
        torch.testing.assert_close(term.stance_age, torch.zeros(2, 4))

    def test_translation_workspace_penalizes_all_four_legs_and_mirrors_lateral_limits(self):
        env = turn_env(6)
        env.command[:] = torch.tensor([[-0.3, 0., 0.]])
        env.command[4] = torch.tensor([0., 0.3, 0.])
        env.command[5] = torch.tensor([0., -0.3, 0.])
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_foot_workspace"]
        term = TranslationFootWorkspacePenalty(cfg, env)
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(6))
        for leg in range(4):
            env.scene["robot"].data.site_pos_w[leg, leg, 0] += 0.32
        env.scene["robot"].data.site_pos_w[4, 0, 1] += 0.20
        env.scene["robot"].data.site_pos_w[5, 1, 1] -= 0.20
        env.scene["feet_ground_contact"].data.found[1] = 0.0
        torch.testing.assert_close(term(env, **cfg.params), torch.full((6,), 0.25))

    def test_translation_workspace_is_world_transform_invariant_and_excludes_pure_turns(self):
        env = turn_env(4)
        env.command[:] = torch.tensor([[-0.3, 0., 0.], [0.3, 0., 0.], [0., 0., 0.5], [0., 0., 0.]])
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_foot_workspace"]
        term = TranslationFootWorkspacePenalty(cfg, env)
        robot = env.scene["robot"]
        robot.data.site_pos_w[:, 0, 0] += 0.32
        expected = torch.tensor([0.25, 0., 0., 0.])
        torch.testing.assert_close(term(env, **cfg.params), expected)
        set_yaw(env, 90.0)
        for positions in (robot.data.site_pos_w, robot.data.body_link_pos_w):
            old = positions.clone()
            positions[:] = torch.stack((-old[:, :, 1], old[:, :, 0], old[:, :, 2]), dim=-1) + torch.tensor([2., -3., 0.5])
        torch.testing.assert_close(term(env, **cfg.params), expected)

    def test_invalid_scales_and_timing_raise_clear_errors(self):
        env = turn_env(1)
        with self.assertRaises(ValueError):
            command_orthogonal_drift_penalty(env, "twist", deviation_scale=0.0)
        with self.assertRaises(ValueError):
            command_orthogonal_drift_penalty(env, "twist", minimum_command_speed=math.nan)
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["translation_stance_overstay"]
        term = TranslationStanceOverstayPenalty(cfg, env)
        with self.assertRaises(ValueError):
            term(env, **{**cfg.params, "minimum_stance_limit": 2.0})
        env.scene["feet_ground_contact"].data.current_air_time = None
        with self.assertRaises(RuntimeError):
            term(env, **cfg.params)


if __name__ == "__main__":
    unittest.main()
