"""Check stance timing and left/right duty without prescribing joint phases."""

from types import SimpleNamespace
import math
import re
import unittest

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from pointleg import robot_cfg
from pointleg.mdp import rewards
from test_pointleg import make_env


def gait_env():
    env = make_env()
    env.step_dt = 0.02
    env.extras = {"log": {}}
    env.commands = torch.tensor([[0.4, 0.0, 0.0]]).repeat(3, 1)
    env.command_manager = SimpleNamespace(get_command=lambda name: env.commands)
    data = SimpleNamespace(
        found=torch.ones(3, 4), current_contact_time=torch.zeros(3, 4),
        last_contact_time=torch.zeros(3, 4), last_air_time=torch.full((3, 4), 0.34),
    )
    sensor = SimpleNamespace(
        primary_names=list(robot_cfg.FOOT_BODY_NAMES), data=data,
        compute_first_contact=lambda dt: data.found.bool(),
        compute_first_air=lambda dt: data.found == 0,
    )
    env.scene["feet_ground_contact"] = sensor
    robot = env.scene["robot"]
    robot.find_sites = lambda name: ([robot_cfg.FOOT_SITE_NAMES.index(name)], [name])
    robot.data.site_pos_w = torch.zeros(3, 4, 3)
    robot.data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)
    ground = torch.zeros(3, 1, 3)
    ground[:, 0, 2] = env.scene.env_origins[:, 2]
    normals = torch.zeros_like(ground)
    normals[:, :, 2] = 1.0
    env.scene["base_height_sensor"] = SimpleNamespace(data=SimpleNamespace(
        hit_pos_w=ground, normals_w=normals, distances=torch.ones(3, 1),
    ))
    return env, sensor


class FlatGaitRewardTests(unittest.TestCase):
    def test_static_abduction_mirror_uses_opposite_sign_without_changing_sagittal_pairs(self):
        env, _ = gait_env()
        env.commands[:] = 0.0
        robot = env.scene["robot"]
        names = robot_cfg.LEG_JOINT_NAMES
        robot.find_joints = lambda pattern: (
            [i for i, name in enumerate(names) if re.fullmatch(pattern, name)],
            [name for name in names if re.fullmatch(pattern, name)],
        )
        q = robot.data.joint_pos
        q[:, [1, 4, 7, 10]] = 0.55
        q[:, [2, 5, 8, 11]] = -1.125
        q[0, [0, 3, 6, 9]] = torch.tensor([0.1, 0.1, 0.1, 0.1])
        q[1, [0, 3, 6, 9]] = torch.tensor([0.1, -0.1, 0.1, -0.1])
        q[2, [0, 3, 6, 9]] = torch.tensor([0.1, 0.1, -0.1, -0.1])
        cfg = load_env_cfg("PointLeg-Flat-v0")
        abduction = cfg.rewards["static_abduction_symmetry"]
        torch.testing.assert_close(
            abduction.func(env, **abduction.params), torch.tensor([0.04, 0.0, 0.04])
        )
        sagittal = cfg.rewards["static_leg_symmetry"]
        self.assertEqual(sagittal.func(env, **sagittal.params).tolist(), [0.0, 0.0, 0.0])
        same_sign = {**abduction.params, "mirror_sign": 1.0}
        torch.testing.assert_close(abduction.func(env, **same_sign), torch.tensor([0.0, 0.04, 0.0]))
        env.commands[:, 0] = 0.4
        self.assertEqual(abduction.func(env, **abduction.params).tolist(), [0.0, 0.0, 0.0])

    def placement_env(self):
        env, sensor = gait_env()
        robot = env.scene["robot"]
        local = torch.tensor([
            [0.4, 0.14, -0.4], [0.4, -0.14, -0.4],
            [-0.05, 0.14, -0.4], [-0.05, -0.14, -0.4],
        ]).repeat(3, 1, 1)
        local[1, :, 1] += 0.06
        local[2, :2, 1] += 0.06
        local[2, 2:, 1] -= 0.06
        robot.data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
        robot.data.site_pos_w = local + robot.data.root_link_pos_w[:, None, :]
        sensor.data.current_contact_time[:] = 0.2
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["foot_lateral_placement"]
        return env, sensor, cfg

    def test_lateral_placement_is_body_relative_and_does_not_cancel_front_rear_bias(self):
        env, _, cfg = self.placement_env()
        robot = env.scene["robot"]
        robot.data.root_link_pos_w[2] += torch.tensor([3.0, -5.0, 0.0])
        local = robot.data.site_pos_w[2] - torch.tensor([0.0, 0.0, robot_cfg.STANDING_BASE_HEIGHT])
        rotated = torch.stack((-local[:, 1], local[:, 0], local[:, 2]), dim=-1)
        robot.data.root_link_quat_w[2] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
        robot.data.site_pos_w[2] = rotated + robot.data.root_link_pos_w[2]
        term = cfg.func(cfg, env)
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0.0, 0.64, 0.64]), atol=1e-5, rtol=1e-5)

    def test_lateral_placement_waits_for_both_sides_and_ignores_airborne_sites(self):
        env, sensor, cfg = self.placement_env()
        sensor.primary_names = list(reversed(robot_cfg.FOOT_BODY_NAMES))
        term = cfg.func(cfg, env)
        sensor.data.found[:] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        sensor.data.found[:] = torch.tensor([0.0, 1.0, 1.0, 0.0])
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0.0, 0.64, 0.64]))
        env.scene["robot"].data.site_pos_w[0, [0, 3], 1] = 10.0
        self.assertEqual(term(env, **cfg.params)[0].item(), 0.0)
        term.reset(torch.tensor([1]))
        self.assertFalse(term.seen[1].any())
        self.assertTrue(term.seen[2].all())
        term.reset()
        self.assertFalse(term.seen.any())

    def test_lateral_placement_disables_for_lateral_turning_and_static_commands(self):
        env, _, cfg = self.placement_env()
        term = cfg.func(cfg, env)
        term(env, **cfg.params)
        env.commands[0] = torch.tensor([0.0, 0.3, 0.0])
        env.commands[1] = torch.tensor([0.3, 0.0, 0.3])
        env.commands[2] = 0.0
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        self.assertFalse(term.seen.any())

    def test_brief_touch_does_not_receive_deferred_height_reward(self):
        env, sensor = gait_env()
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["foot_swing_height"]
        term = cfg.func(cfg, env)
        sensor.data.found[:, :2] = 0.0
        env.scene["robot"].data.site_pos_w[:, :2, 2] = env.scene.env_origins[:, 2, None] + 0.1
        term(env, **cfg.params)
        sensor.data.found[:] = 1.0
        sensor.data.current_contact_time[:] = 0.04
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(term.pending_valid[:, :2].all())
        sensor.data.found[:, :2] = 0.0
        sensor.data.current_contact_time[:, :2] = 0.0
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        self.assertFalse(term.pending_valid.any())

    def test_pending_height_is_cleared_by_static_command_and_partial_reset(self):
        env, sensor = gait_env()
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["foot_swing_height"]
        term = cfg.func(cfg, env)
        sensor.data.found[:, :2] = 0.0
        term(env, **cfg.params)
        sensor.data.found[:] = 1.0
        term(env, **cfg.params)
        term.reset(torch.tensor([0]))
        self.assertFalse(term.pending_valid[0].any())
        self.assertTrue(term.pending_valid[1:].any())
        env.commands[1] = 0.0
        sensor.data.current_contact_time[:] = 0.2
        term(env, **cfg.params)
        self.assertFalse(term.pending_valid[1].any())

    def test_short_cycles_have_bounded_height_reward_per_second(self):
        for step_dt in (0.01, 0.02):
            env, sensor = gait_env()
            env.step_dt = step_dt
            cfg = load_env_cfg("PointLeg-Flat-v0").rewards["foot_swing_height"]
            term = cfg.func(cfg, env)
            sensor.data.found[:, :2] = 0.0
            env.scene["robot"].data.site_pos_w[:, :2, 2] = env.scene.env_origins[:, 2, None] + 0.1
            term(env, **cfg.params)
            sensor.data.found[:] = 1.0
            sensor.data.last_air_time[:] = 0.08
            sensor.data.current_contact_time[:] = 0.16
            integrated = term(env, **cfg.params) * step_dt
            torch.testing.assert_close(integrated, torch.full((3,), 2.0 * 0.24 / 0.50))
            self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])

    def test_short_stance_cost_ignores_long_stance_and_static_commands(self):
        env, sensor = gait_env()
        sensor.data.found[:, 0] = 0.0
        sensor.data.last_contact_time[:, 0] = torch.tensor([0.04, 0.16, 0.24])
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["short_stance"]
        torch.testing.assert_close(cfg.func(env, **cfg.params), torch.tensor([28.125, 0.0, 0.0]))
        env.commands[:] = 0.0
        self.assertEqual(cfg.func(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])

    def test_alternating_diagonal_stance_is_allowed_but_one_sided_stance_is_penalized(self):
        env, sensor = gait_env()
        sensor.primary_names = list(reversed(robot_cfg.FOOT_BODY_NAMES))
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["stance_balance"]
        term = cfg.func(cfg, env)
        for step in range(300):
            diagonal = [1.0, 0.0, 0.0, 1.0] if (step // 10) % 2 == 0 else [0.0, 1.0, 1.0, 0.0]
            sensor.data.found[0] = torch.tensor(diagonal)
            sensor.data.found[1:] = torch.tensor([0.0, 1.0, 0.0, 1.0])
            result = term(env, **cfg.params)
        self.assertLess(result[0].item(), 1.0e-3)
        self.assertGreater(result[1].item(), 10.0)
        self.assertGreater(term.duty[1, 0].item(), 0.99)
        self.assertLess(term.duty[1, 1].item(), 0.01)
        term.reset(torch.tensor([1]))
        torch.testing.assert_close(term.duty[1], torch.full((4,), 0.5))
        self.assertEqual(term.age[1].item(), 0.0)
        self.assertGreater(term.age[2].item(), 1.0)
        term.reset()
        self.assertEqual(term.age.sum().item(), 0.0)

    def test_balance_disables_for_lateral_and_turning_then_warms_up_again(self):
        env, sensor = gait_env()
        sensor.data.found[:] = torch.tensor([1.0, 0.0, 1.0, 0.0])
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["stance_balance"]
        term = cfg.func(cfg, env)
        env.commands[0, 1] = 0.3
        env.commands[1, 2] = 0.3
        env.commands[2, 0] = -0.3
        for _ in range(80):
            result = term(env, **cfg.params)
        self.assertEqual(result[:2].tolist(), [0.0, 0.0])
        self.assertGreater(result[2].item(), 0.0)
        env.commands[:, 1:] = 0.0
        self.assertEqual(term(env, **cfg.params)[:2].tolist(), [0.0, 0.0])

    def test_slip_penalizes_only_grounded_foot_velocity(self):
        env, sensor = gait_env()
        env.scene["robot"].data.site_lin_vel_w = torch.zeros(3, 4, 3)
        env.scene["robot"].data.site_lin_vel_w[:, 0, 0] = 1.0
        sensor.data.found[0, 0] = 0.0
        env.commands[2] = 0.0
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["feet_slip"]
        params = {**cfg.params, "asset_cfg": SceneEntityCfg("robot", site_ids=[0, 1, 2, 3])}
        torch.testing.assert_close(cfg.func(env, **params), torch.tensor([0.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
