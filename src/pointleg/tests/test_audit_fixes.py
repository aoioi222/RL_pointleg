"""Exercise the runtime behaviors identified in audit section A."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import pointleg
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.commands import UniformThresholdVelocityCommand
from pointleg.mdp.only_positive_rewards import configure_positive_rewards
from pointleg.mdp.rewards import upright_roll_only
from pointleg.mdp.terrain_utils import terrain_type_mask
from pointleg.rsl_rl.modules.him_estimator import HIMEstimator


class CommandScene(dict):
    def __init__(self, robot, terrain=None):
        super().__init__(robot=robot)
        self.terrain = terrain


def command_env(cfg, count=256, terrain_types=None):
    robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(count)))
    terrain = SimpleNamespace(
        cfg=cfg.scene.terrain,
        terrain_types=torch.zeros(count, dtype=torch.long) if terrain_types is None else terrain_types,
    )
    return SimpleNamespace(num_envs=count, device="cpu", step_dt=0.02, scene=CommandScene(robot, terrain))


class AuditFixTests(unittest.TestCase):
    def test_rough_full_range_samples_all_directions_on_both_terrain_types(self):
        torch.manual_seed(24)
        cfg = load_env_cfg("PointLeg-Rough-v0")
        count = 8192
        terrain_types = torch.arange(count) % 2
        env = command_env(cfg, count=count, terrain_types=terrain_types)
        command = cfg.commands["twist"].build(env)
        command._resample_command(torch.arange(count))
        command._update_command()
        env.command_manager = SimpleNamespace(get_command=lambda name: command.command)
        active = ~command.is_standing_env
        self.assertAlmostEqual(command.is_backward_env.sum().item() / active.sum().item(), 0.2, delta=0.025)
        for terrain_id in (0, 1):
            for axis in range(3):
                values = command.command[terrain_types == terrain_id, axis]
                self.assertGreater(int((values > 0).sum()), 100)
                self.assertGreater(int((values < 0).sum()), 100)
        lateral = command.command[command.is_lateral_env, 1]
        self.assertGreater((lateral.abs() > 0).float().mean().item(), 0.90)
        self.assertEqual(torch.count_nonzero(command.command[command.is_standing_env]).item(), 0)
        for direction, axis, sign in (
            ("forward", 0, 1), ("backward", 0, -1), ("left", 1, 1),
            ("right", 1, -1), ("turn_left", 2, 1), ("turn_right", 2, -1),
        ):
            metric = cfg.metrics[f"cmd_{direction}_active"]
            torch.testing.assert_close(
                metric.func(env, **metric.params),
                (command.command[:, axis] * sign > 1.0e-6).float(),
            )

    def test_rough_deadbands_preserve_slow_motion_and_clear_tiny_commands(self):
        import math
        cfg = load_env_cfg("PointLeg-Rough-v0")
        command = cfg.commands["twist"].build(command_env(cfg, count=3))
        command.vel_command_b[:] = torch.tensor([
            [0.0, 0.05, math.radians(3.0)],
            [-0.05, 0.0, -math.radians(3.0)],
            [0.005, 0.005, math.radians(1.0)],
        ])
        expected = command.command.clone()
        expected[2] = 0.0
        command._apply_deadbands()
        torch.testing.assert_close(command.command, expected)

    def test_registered_commands_build_the_custom_type(self):
        for task in ("Flat", "Rough", "Crawl"):
            cfg = load_env_cfg(f"PointLeg-{task}-v0")
            command = cfg.commands["twist"].build(command_env(cfg))
            self.assertIsInstance(command, UniformThresholdVelocityCommand)
            command._resample_command(torch.arange(256))
            command._update_command()
            self.assertTrue(torch.isfinite(command.command).all())

    def test_forward_standing_and_axis_groups_remain_exclusive(self):
        torch.manual_seed(11)
        cfg = load_env_cfg("PointLeg-Rough-v0")
        env = command_env(cfg, count=2048)
        command = cfg.commands["twist"].build(env)
        command._resample_command(torch.arange(2048))
        command._update_command()
        self.assertFalse((command.is_lateral_env & command.is_forward_env).any())
        self.assertFalse((command.is_yaw_env & command.is_forward_env).any())
        self.assertFalse((command.is_lateral_env & command.is_yaw_env).any())
        self.assertFalse((command.is_standing_env & command.is_lateral_env).any())
        self.assertEqual(torch.count_nonzero(command.command[command.is_standing_env]).item(), 0)
        forward = command.is_forward_env & ~command.is_standing_env
        self.assertEqual(torch.count_nonzero(command.command[forward, 1:]).item(), 0)
        self.assertTrue(command.is_yaw_env.any())
        self.assertTrue(command.is_lateral_env.any())
        candidates = ~command.is_forward_env & ~command.is_standing_env & ~command.is_backward_env
        self.assertAlmostEqual(command.is_lateral_env.sum().item() / candidates.sum().item(), 0.35, delta=0.05)
        self.assertAlmostEqual(command.is_yaw_env.sum().item() / candidates.sum().item(), 0.35, delta=0.05)

    def test_deadband_preserves_rotation_when_linear_command_is_small(self):
        cfg = load_env_cfg("PointLeg-Flat-v0")
        command = cfg.commands["twist"].build(command_env(cfg, count=1))
        command.vel_command_b[:] = torch.tensor([[0.01, 0.01, 0.5]])
        command._apply_deadbands()
        torch.testing.assert_close(command.command, torch.tensor([[0.0, 0.0, 0.5]]))

    def test_backward_group_is_exclusive_and_tracks_sampled_heading(self):
        torch.manual_seed(18)
        cfg = load_env_cfg("PointLeg-Flat-v0")
        env = command_env(cfg, count=8192)
        env.scene["robot"].data.heading_w[:] = 0.7
        command = cfg.commands["twist"].build(env)
        command._resample_command(torch.arange(8192))
        backward = command.is_backward_env
        active = ~command.is_standing_env
        self.assertAlmostEqual(backward.sum().item() / active.sum().item(), 0.2, delta=0.025)
        for group in (command.is_standing_env, command.is_forward_env,
                      command.is_lateral_env, command.is_yaw_env):
            self.assertFalse((backward & group).any())
        vx = command.command[backward, 0].clone()
        self.assertTrue(((vx >= -0.4) & (vx <= -0.2)).all())
        self.assertEqual(torch.count_nonzero(command.command[backward, 1:]).item(), 0)
        torch.testing.assert_close(command.heading_target[backward], torch.full_like(vx, 0.7))
        command._update_command()
        torch.testing.assert_close(command.command[backward, 0], vx)
        self.assertEqual(torch.count_nonzero(command.command[backward, 1:]).item(), 0)
        env.scene["robot"].data.heading_w[backward] += 0.3
        command._update_command()
        self.assertTrue((command.command[backward, 2] < 0.0).all())
        torch.testing.assert_close(command.command[backward, 0], vx)

    def test_backward_partial_resampling_clears_only_selected_flags(self):
        cfg = load_env_cfg("PointLeg-Flat-v0")
        cfg.commands["twist"] = replace(
            cfg.commands["twist"], rel_forward_envs=0.0,
            rel_backward_envs=1.0, rel_standing_envs=0.0,
        )
        command = cfg.commands["twist"].build(command_env(cfg, count=8))
        command._resample_command(torch.arange(8))
        self.assertTrue(command.is_backward_env.all())
        previous = command.command.clone()
        command.cfg.rel_backward_envs = 0.0
        command._resample_command(torch.tensor([0]))
        self.assertFalse(command.is_backward_env[0])
        self.assertTrue(command.is_backward_env[1:].all())
        torch.testing.assert_close(command.command[1:], previous[1:])

    def test_backward_config_validation_and_generic_rough_sampling(self):
        cfg = load_env_cfg("PointLeg-Flat-v0")
        with self.assertRaises(ValueError):
            replace(cfg.commands["twist"], rel_backward_envs=0.7)
        with self.assertRaises(ValueError):
            replace(cfg.commands["twist"], backward_velocity_range=(-0.3, -0.1))
        self.assertEqual(load_env_cfg("PointLeg-Crawl-v0").commands["twist"].rel_backward_envs, 0.0)
        self.assertEqual(load_env_cfg("PointLeg-Rough-v0").commands["twist"].rel_backward_envs, 0.2)
        cfg = load_env_cfg("PointLeg-Rough-v0")
        cfg.commands["twist"] = replace(cfg.commands["twist"], rel_backward_envs=0.2)
        env = command_env(cfg, terrain_types=torch.ones(256, dtype=torch.long))
        command = cfg.commands["twist"].build(env)
        command._resample_command(torch.arange(256))
        command._update_command()
        self.assertTrue(command.is_backward_env.any())
        terrain_names = set(cfg.scene.terrain.terrain_generator.sub_terrains)
        self.assertFalse(set(command.cfg.climbing_terrain_names) & terrain_names)

    def test_rough_commands_record_world_progress_direction(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        env = command_env(cfg, terrain_types=torch.ones(256, dtype=torch.long))
        env.scene["robot"].data.heading_w[:] = 0.5
        command = cfg.commands["twist"].build(env)
        command._resample_command(torch.arange(256))
        command._update_command()
        speed = torch.linalg.vector_norm(command.command[:, :2], dim=1)
        world_speed = torch.linalg.vector_norm(command.progress_direction_w, dim=1)
        torch.testing.assert_close(world_speed, speed)
        self.assertEqual(torch.count_nonzero(command.command[command.is_standing_env]).item(), 0)

    def test_pd_event_uses_actuator_gain_randomization(self):
        for task in ("Flat", "Rough", "Crawl"):
            cfg = load_env_cfg(f"PointLeg-{task}-v0")
            self.assertNotIn("actuator_stiffness", cfg.events)
            self.assertNotIn("actuator_damping", cfg.events)
            event = cfg.events["actuator_pd_gains"]
            self.assertIs(event.func, dr.pd_gains)
            self.assertEqual(event.params["kp_range"], (0.9, 1.1))
            self.assertEqual(event.params["kd_range"], (0.9, 1.1))

    def test_reward_floor_is_local_and_idempotent(self):
        self.assertEqual(RewardManager.compute.__module__, "mjlab.managers.reward_manager")
        env_a = SimpleNamespace(num_envs=1, device="cpu")
        env_b = SimpleNamespace(num_envs=1, device="cpu")
        rewards = {"negative": RewardTermCfg(func=lambda env: torch.ones(1), weight=-1.0)}
        env_a.reward_manager = RewardManager(rewards, env_a)
        env_b.reward_manager = RewardManager(rewards, env_b)
        configure_positive_rewards(env_a, None)
        configure_positive_rewards(env_a, None)
        self.assertEqual(env_a.reward_manager.compute(0.02).item(), 0.0)
        self.assertAlmostEqual(env_b.reward_manager.compute(0.02).item(), -0.02)
        self.assertEqual(env_a.reward_manager._reward_buf.item(), 0.0)
        self.assertAlmostEqual(env_a.reward_manager._episode_sums["negative"].item(), -0.02)
        self.assertNotIn("positive_rewards", load_env_cfg("PointLeg-Flat-v0").events)
        self.assertNotIn("positive_rewards", load_env_cfg("PointLeg-Rough-v0").events)
        self.assertEqual(load_env_cfg("PointLeg-Crawl-v0").events["positive_rewards"].mode, "startup")

    def test_rough_removes_mirror_posture_constraints(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        self.assertNotIn("joint_mirror", cfg.rewards)
        self.assertNotIn("abduction_mirror", cfg.rewards)
        self.assertNotIn("joint_pos_penalty_ab", cfg.rewards)
        self.assertNotIn("joint_pos_penalty_sagittal", cfg.rewards)

    def test_all_default_poses_are_inside_soft_limits(self):
        for task in ("Flat", "Rough", "Crawl"):
            cfg = load_env_cfg(f"PointLeg-{task}-v0")
            robot_cfg = cfg.scene.entities["robot"]
            model = Entity(robot_cfg).spec.compile()
            ranges = model.jnt_range[1:]
            centers = ranges.mean(axis=1)
            half = (ranges[:, 1] - ranges[:, 0]) * robot_cfg.articulation.soft_joint_pos_limit_factor / 2
            qpos = model.key_qpos[0, 7:]
            self.assertTrue((qpos >= centers - half).all(), task)
            self.assertTrue((qpos <= centers + half).all(), task)

    def test_playback_keeps_terrain_column_identity(self):
        for task in ("Rough", "Crawl"):
            cfg = load_env_cfg(f"PointLeg-{task}-v0", play=True)
            generator = cfg.scene.terrain.terrain_generator
            self.assertTrue(generator.curriculum)
            self.assertEqual(generator.num_cols, len(generator.sub_terrains))
            self.assertFalse(cfg.curriculum)
        cfg = load_env_cfg("PointLeg-Flat-v0", play=True)
        env = command_env(cfg, count=3, terrain_types=torch.tensor([0, 3, 9]))
        self.assertEqual(terrain_type_mask(env, ("flat",)).tolist(), [True, True, True])
        cfg.scene.terrain.terrain_generator.sub_terrains["other"] = cfg.scene.terrain.terrain_generator.sub_terrains["flat"]
        cfg.scene.terrain.terrain_generator.curriculum = False
        with self.assertRaisesRegex(ValueError, "type-column"):
            terrain_type_mask(env, ("flat",))

    def test_roll_reward_default_argument(self):
        robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([[0.0, 0.5, -0.866]])))
        env = SimpleNamespace(scene={"robot": robot})
        torch.testing.assert_close(upright_roll_only(env), torch.tensor([0.25]))

    def test_him_uses_actor_prefix_and_velocity_suffix(self):
        estimator = HIMEstimator(2, 45, enc_hidden_dims=[16, 8], tar_hidden_dims=[16])
        history = torch.randn(4, 90)
        critic = torch.randn(4, 325)
        captured = []
        handle = estimator.target.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        with torch.no_grad():
            expected_loss = torch.nn.functional.mse_loss(estimator.encoder(history)[:, :3], critic[:, 45:48]).item()
        try:
            velocity_loss, contrast_loss = estimator.update(history, critic)
        finally:
            handle.remove()
        torch.testing.assert_close(captured[0], critic[:, :45])
        self.assertAlmostEqual(velocity_loss, expected_loss, places=5)
        self.assertTrue(np.isfinite(contrast_loss))
        with self.assertRaisesRegex(ValueError, "actor prefix"):
            estimator.update(history, critic[:, :45])


if __name__ == "__main__":
    unittest.main()
