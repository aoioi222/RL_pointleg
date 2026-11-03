"""Check in-place turning costs, per-foot history and left/right symmetry."""

import math
from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.rewards import (
    TurnFootWorkspacePenalty,
    TurnStanceTwistPenalty,
    turn_linear_drift_penalty,
    turn_progress_deficit_penalty,
    turn_yaw_error_degrees,
)
from pointleg.robot_cfg import FOOT_BODY_NAMES, FOOT_SITE_NAMES


def turn_env(count=4):
    command = torch.zeros(count, 3)
    command[:, 2] = math.radians(30.0)
    hips = torch.tensor([
        [0.392, 0.039, 0.47], [0.392, -0.039, 0.47],
        [-0.089, 0.039, 0.47], [-0.089, -0.039, 0.47],
    ]).repeat(count, 1, 1)
    feet = hips + torch.tensor([
        [0.035, 0.105, -0.4], [0.035, -0.105, -0.4],
        [0.035, 0.105, -0.4], [0.035, -0.105, -0.4],
    ])
    hip_names = [f"{leg}_hip_pitch_Link" for leg in ("fl", "fr", "rl", "rr")]
    robot = SimpleNamespace(
        find_sites=lambda name: ([FOOT_SITE_NAMES.index(name)], [name]),
        find_bodies=lambda name: ([hip_names.index(name)], [name]),
        data=SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            root_link_lin_vel_b=torch.zeros(count, 3),
            root_link_ang_vel_b=torch.zeros(count, 3),
            body_link_pos_w=hips, site_pos_w=feet,
        ),
    )
    sensor = SimpleNamespace(
        primary_names=list(reversed(FOOT_BODY_NAMES)),
        data=SimpleNamespace(found=torch.ones(count, 4), current_air_time=torch.zeros(count, 4)),
    )
    env = SimpleNamespace(
        num_envs=count, device="cpu", step_dt=0.02, extras={"log": {}},
        scene={"robot": robot, "feet_ground_contact": sensor}, command=command,
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    return env


def set_yaw(env, degrees):
    angle = torch.as_tensor(degrees, dtype=torch.float32).expand(env.num_envs)
    half = torch.deg2rad(angle) / 2.0
    env.scene["robot"].data.root_link_quat_w[:] = torch.stack(
        (half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()), dim=1
    )


class TurnRewardTests(unittest.TestCase):
    def test_terms_are_rough_only_negative_and_keep_policy_interface(self):
        rough, flat = load_env_cfg("PointLeg-Rough-v0"), load_env_cfg("PointLeg-Flat-v0")
        for name in ("turn_stance_twist", "turn_foot_workspace", "turn_progress_deficit", "turn_linear_drift"):
            self.assertIn(name, rough.rewards)
            self.assertLess(rough.rewards[name].weight, 0.0)
            self.assertNotIn(name, flat.rewards)
        self.assertLess(rough.rewards["feet_slip"].weight, 0.0)
        self.assertEqual(list(rough.observations["actor"].terms), list(flat.observations["actor"].terms))
        self.assertEqual(list(rough.actions), list(flat.actions))

    def test_signed_yaw_deficit_penalizes_stall_and_wrong_direction(self):
        env = turn_env(6)
        env.command[:4, 2] = torch.deg2rad(torch.tensor([30.0, -30.0, 30.0, -30.0]))
        env.scene["robot"].data.root_link_ang_vel_b[:4, 2] = torch.deg2rad(
            torch.tensor([18.0, -18.0, 0.0, 18.0])
        )
        env.command[4] = 0.0
        env.command[5, 0] = 0.4
        torch.testing.assert_close(turn_progress_deficit_penalty(env, "twist"), torch.tensor([0., 0., 1., 4., 0., 0.]))

    def test_drift_uses_tolerance_and_leaves_translation_commands_unaffected(self):
        env = turn_env(4)
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0.04, 0.2, 0.2, 0.2])
        env.command[2] = 0.0
        env.command[3, 0] = 0.3
        torch.testing.assert_close(turn_linear_drift_penalty(env, "twist"), torch.tensor([0., 1., 0., 0.]))

    def test_twist_cost_is_symmetric_and_only_charges_beyond_tolerance(self):
        env = turn_env(4)
        env.command[1, 2] *= -1.0
        env.command[2] = 0.0
        env.command[3, 0] = 0.3
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["turn_stance_twist"]
        term = TurnStanceTwistPenalty(cfg, env)
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(4))
        set_yaw(env, [10.0, -10.0, 10.0, 10.0])
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(4))
        set_yaw(env, [22.0, -22.0, 22.0, 22.0])
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([1., 1., 0., 0.]))
        self.assertAlmostEqual(env.extras["log"]["Metrics/turn_stance_twist_max_deg"].item(), 22.0, places=4)

    def test_twist_history_survives_contact_flicker_and_clears_per_foot_after_swing(self):
        env = turn_env(1)
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["turn_stance_twist"]
        term = TurnStanceTwistPenalty(cfg, env)
        term(env, **cfg.params)
        set_yaw(env, 22.0)
        term(env, **cfg.params)
        sensor = env.scene["feet_ground_contact"]
        sensor.data.found[0, 3] = 0.0
        sensor.data.current_air_time[0, 3] = 0.02
        term(env, **cfg.params)
        sensor.data.found[0, 3] = 1.0
        sensor.data.current_air_time[0, 3] = 0.0
        torch.testing.assert_close(term(env, **cfg.params), torch.ones(1))
        sensor.data.found[0, 3] = 0.0
        sensor.data.current_air_time[0, 3] = 0.10
        term(env, **cfg.params)
        self.assertEqual(term.twist[0, 0].item(), 0.0)
        self.assertGreater(term.twist[0, 1].item(), 0.0)
        sensor.data.found[0, 3] = 1.0
        sensor.data.current_air_time[0, 3] = 0.0
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0.75]))

    def test_twist_handles_yaw_wrap_and_partial_and_full_reset(self):
        env = turn_env(2)
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["turn_stance_twist"]
        term = TurnStanceTwistPenalty(cfg, env)
        set_yaw(env, 179.0)
        term(env, **cfg.params)
        set_yaw(env, -179.0)
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))
        set_yaw(env, -159.0)
        torch.testing.assert_close(term(env, **cfg.params), torch.ones(2))
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0., 1.]))
        term.reset()
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))

    def test_turn_stop_and_restart_does_not_keep_previous_stance_twist(self):
        env = turn_env(1)
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["turn_stance_twist"]
        term = TurnStanceTwistPenalty(cfg, env)
        term(env, **cfg.params)
        set_yaw(env, 25.0)
        self.assertGreater(term(env, **cfg.params).item(), 0.0)
        env.command[:] = 0.0
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(1))
        env.command[:, 2] = math.radians(30.0)
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(1))

    def test_workspace_covers_all_legs_in_both_stance_and_swing(self):
        env = turn_env(6)
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["turn_foot_workspace"]
        term = TurnFootWorkspacePenalty(cfg, env)
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(6))
        for leg in range(4):
            env.scene["robot"].data.site_pos_w[leg, leg, 0] += 0.32
        env.scene["feet_ground_contact"].data.found[1] = 0.0
        env.command[2, 2] *= -1.0
        env.command[4] = 0.0
        env.command[5, 0] = 0.3
        env.scene["robot"].data.site_pos_w[4:, :, 0] += 0.32
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0.25, 0.25, 0.25, 0.25, 0., 0.]))

    def test_workspace_is_mirrored_and_invariant_to_world_yaw_and_translation(self):
        env = turn_env(2)
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["turn_foot_workspace"]
        term = TurnFootWorkspacePenalty(cfg, env)
        robot = env.scene["robot"]
        robot.data.site_pos_w[0, 0, 1] += 0.20
        robot.data.site_pos_w[1, 1, 1] -= 0.20
        env.command[1, 2] *= -1.0
        expected = torch.tensor([0.25, 0.25])
        torch.testing.assert_close(term(env, **cfg.params), expected)
        set_yaw(env, 90.0)
        for positions in (robot.data.site_pos_w, robot.data.body_link_pos_w):
            original = positions.clone()
            positions[:] = torch.stack((-original[:, :, 1], original[:, :, 0], original[:, :, 2]), dim=-1)
            positions += torch.tensor([3.0, -2.0, 0.5])
        torch.testing.assert_close(term(env, **cfg.params), expected)

    def test_left_and_right_error_metrics_use_degrees_and_no_sample_returns_zero(self):
        env = turn_env(4)
        env.command[:, 2] = torch.deg2rad(torch.tensor([30., -30., -30., 30.]))
        env.command[3, 0] = 0.3
        env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.deg2rad(torch.tensor([20., -10., -20., 0.]))
        torch.testing.assert_close(turn_yaw_error_degrees(env, "twist", 1), torch.full((4,), 10.0))
        torch.testing.assert_close(turn_yaw_error_degrees(env, "twist", -1), torch.full((4,), 15.0))
        env.command[:] = 0.0
        torch.testing.assert_close(turn_yaw_error_degrees(env, "twist", -1), torch.zeros(4))


if __name__ == "__main__":
    unittest.main()
