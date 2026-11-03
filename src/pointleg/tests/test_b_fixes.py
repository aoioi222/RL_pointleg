"""Regression tests for height, evaluation and asynchronous curricula."""

from dataclasses import asdict
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pointleg import robot_cfg
from pointleg.config.debug import configure_debug_training
from pointleg.config.evaluation import evaluation_env_cfg
from pointleg.mdp.curriculums import command_levels_adaptive
from pointleg.mdp.rewards import crawl_height_reward, upward
from pointleg.mdp.terminations import root_height_out_of_range
from pointleg.run import run_debug_training


def height_env():
    ground = torch.tensor([0.0, 1.5, -0.7])
    heights = torch.tensor([0.20, 0.20, 0.20])
    root = torch.zeros(3, 3)
    root[:, 2] = ground + heights
    hits = torch.zeros(3, 1, 3)
    hits[:, 0, 2] = ground
    normals = torch.zeros(3, 1, 3)
    normals[:, 0, 2] = 1.0
    return SimpleNamespace(scene={
        "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root)),
        "ground": SimpleNamespace(data=SimpleNamespace(
            hit_pos_w=hits, normals_w=normals, distances=heights[:, None].clone(),
        )),
    })


def command_env():
    ranges = SimpleNamespace(lin_vel_x=(-1.0, 1.0))
    command = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
    return SimpleNamespace(
        cfg=SimpleNamespace(episode_length_s=1.0), device="cpu", step_dt=0.1,
        common_step_counter=0, episode_length_buf=torch.zeros(3, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        reward_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(weight=2.0),
            _episode_sums={"track": torch.zeros(3)},
        ),
    )


def make_curriculum(env, **kwargs):
    params = dict(command_name="twist", reward_term_name="track", axis="x",
                  initial_range=(-0.2, 0.2), delta_command=0.1,
                  ema_alpha=1.0, target_ratio=0.8)
    params.update(kwargs)
    return command_levels_adaptive(CurriculumTermCfg(func=command_levels_adaptive, params=params), env)


class BusinessConfigurationTests(unittest.TestCase):
    def test_flat_height_target_uses_standing_configuration(self):
        cfg = load_env_cfg("PointLeg-Flat-v0")
        target = cfg.rewards["base_height_l2"].params["target_height"]
        self.assertEqual(target, robot_cfg.STANDING_BASE_HEIGHT)
        self.assertEqual(target, cfg.scene.entities["robot"].init_state.pos[2])

    def test_crawl_height_is_relative_to_local_ground(self):
        env = height_env()
        reward = crawl_height_reward(env, sensor_name="ground")
        torch.testing.assert_close(reward, torch.ones(3))
        self.assertFalse(root_height_out_of_range(env, 0.1, 0.38, sensor_name="ground").any())
        env.scene["robot"].data.root_link_pos_w[:, 2] += 0.1
        reward = crawl_height_reward(env, sensor_name="ground")
        torch.testing.assert_close(reward, torch.full((3,), math.exp(-0.08 ** 2 / 0.05 ** 2)))
        cfg = load_env_cfg("PointLeg-Crawl-v0")
        self.assertEqual(cfg.rewards["crawl_height_reward"].params["sensor_name"],
                         cfg.terminations["base_height_out_of_range"].params["sensor_name"])

    def test_invalid_ground_never_gets_crawl_height_reward(self):
        env = height_env()
        data = env.scene["ground"].data
        data.distances[0] = -1.0
        data.normals_w[1, 0, 2] = -1.0
        env.scene["robot"].data.root_link_pos_w[2, 2] = float("nan")
        torch.testing.assert_close(crawl_height_reward(env, sensor_name="ground"), torch.zeros(3))
        self.assertTrue(root_height_out_of_range(env, 0.1, 0.38, sensor_name="ground").all())

    def test_nominal_play_removes_physics_randomization_and_delays(self):
        for task in ("Flat", "Rough", "Crawl"):
            with self.subTest(task=task):
                cfg = load_env_cfg(f"PointLeg-{task}-v0", play=True)
                allowed_events = {"reset_scene", "positive_rewards"}
                if task == "Rough":
                    allowed_events.add("randomize_terrain")
                self.assertLessEqual(set(cfg.events), allowed_events)
                self.assertIn("reset_scene", cfg.events)
                self.assertFalse(cfg.curriculum)
                self.assertTrue(all(not group.enable_corruption for group in cfg.observations.values()))
                action = cfg.actions["leg_joint_pos"]
                self.assertEqual((action.min_delay, action.max_delay), (0, 0))
                self.assertEqual(cfg.seed, 42)

    def test_robust_evaluation_keeps_randomization_not_curricula(self):
        cfg = evaluation_env_cfg("PointLeg-Rough-v0", mode="robust")
        self.assertIn("base_inertia", cfg.events)
        self.assertIn("reset_base", cfg.events)
        self.assertIn("push_robot", cfg.events)
        self.assertTrue(cfg.observations["actor"].enable_corruption)
        self.assertEqual(cfg.actions["leg_joint_pos"].max_delay, 2)
        self.assertFalse(cfg.curriculum)
        self.assertNotEqual(cfg.episode_length_s, load_env_cfg("PointLeg-Rough-v0").episode_length_s)
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            evaluation_env_cfg("PointLeg-Rough-v0", mode="other")

    def test_mass_and_inertia_randomize_as_one_density_model(self):
        for task in ("Flat", "Rough", "Crawl"):
            cfg = load_env_cfg(f"PointLeg-{task}-v0")
            event = cfg.events["base_inertia"]
            self.assertIs(event.func, dr.pseudo_inertia)
            self.assertNotIn("body_mass_base", cfg.events)
            self.assertNotIn("base_com", cfg.events)
            low, high = event.params["alpha_range"]
            self.assertAlmostEqual(math.exp(2 * low), 0.85)
            self.assertAlmostEqual(math.exp(2 * high), 1.45)

    def test_upward_is_a_positive_upright_reward(self):
        robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([
            [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])))
        torch.testing.assert_close(upward(SimpleNamespace(scene={"robot": robot})), torch.tensor([4.0, 1.0, 0.0]))

    def test_debug_profile_keeps_policy_and_default_configs_unchanged(self):
        for task in ("Flat", "Rough", "Crawl"):
            task_id = f"PointLeg-{task}-v0"
            cfg, agent = load_env_cfg(task_id), load_rl_cfg(task_id)
            actor, critic = asdict(agent.actor), asdict(agent.critic)
            configure_debug_training(cfg, agent)
            self.assertEqual(cfg.scene.num_envs, 16)
            self.assertEqual((agent.num_steps_per_env, agent.algorithm.num_mini_batches), (8, 1))
            self.assertEqual((agent.max_iterations, agent.logger, agent.upload_model), (2, "tensorboard", False))
            self.assertEqual(asdict(agent.actor), actor)
            self.assertEqual(asdict(agent.critic), critic)
            self.assertEqual(load_env_cfg(task_id).scene.num_envs, 2048)

    def test_debug_cli_respects_explicit_native_overrides(self):
        with patch("mjlab.scripts.train.launch_training") as launch:
            run_debug_training("PointLeg-Flat-v0", [
                "--env.scene.num-envs", "4", "--agent.max-iterations", "1",
            ])
        task_id, cfg = launch.call_args.args
        self.assertEqual(task_id, "PointLeg-Flat-v0")
        self.assertEqual(cfg.env.scene.num_envs, 4)
        self.assertEqual(cfg.agent.max_iterations, 1)
        self.assertEqual(cfg.agent.num_steps_per_env, 8)
        self.assertFalse(cfg.agent.upload_model)


class AsynchronousCurriculumTests(unittest.TestCase):
    def test_update_after_crossing_period_without_exact_modulo(self):
        env = command_env()
        curriculum = make_curriculum(env)
        curriculum(env, torch.tensor([0]))
        self.assertEqual(curriculum._window_episode_count, 0)
        env.episode_length_buf[:] = 9
        env.reward_manager._episode_sums["track"][:] = 1.8
        env.common_step_counter = 9
        curriculum(env, torch.tensor([0]))
        self.assertEqual(curriculum._current_range, [-0.2, 0.2])
        env.common_step_counter = 11
        curriculum(env, torch.tensor([1]))
        self.assertAlmostEqual(curriculum._current_range[1], 0.3)
        env.common_step_counter = 12
        curriculum(env, torch.tensor([2]))
        self.assertAlmostEqual(curriculum._current_range[1], 0.3)

    def test_episode_count_weighting_and_short_failures(self):
        env = command_env()
        curriculum = make_curriculum(env)
        env.episode_length_buf[:] = torch.tensor([10, 1, 1])
        env.reward_manager._episode_sums["track"][:] = torch.tensor([2.0, 0.2, 0.2])
        env.common_step_counter = 9
        curriculum(env, torch.tensor([0]))
        env.common_step_counter = 11
        curriculum(env, torch.tensor([1, 2]))
        self.assertAlmostEqual(curriculum._running_mean, 0.4)
        self.assertEqual(curriculum._current_range, [-0.2, 0.2])

    def test_large_gaps_expand_once_and_clamp_to_full_range(self):
        env = command_env()
        curriculum = make_curriculum(env, delta_command=2.0)
        env.episode_length_buf[:] = 10
        env.reward_manager._episode_sums["track"][:] = 2.0
        env.common_step_counter = 101
        curriculum(env, None)
        self.assertEqual(curriculum._current_range, [-1.0, 1.0])
        self.assertEqual(curriculum._window_episode_count, 0)


if __name__ == "__main__":
    unittest.main()
