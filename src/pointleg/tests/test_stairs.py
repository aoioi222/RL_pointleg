"""Check fixed-height stair completion and clearance without physics stepping."""

import unittest
import tempfile
import math
from pathlib import Path
from unittest.mock import patch

import torch

from mjlab.managers.termination_manager import TerminationManager
from mjlab.managers.reward_manager import RewardManager
from pointleg.mdp.elevation import ElevationTerrainCurriculum, collect_elevation_contacts, stairs_exit_supported_feet, update_elevation_state
from test_elevation import bounds, contacts
from test_elevation_paired import paired_env, runner_for


def destination_contacts(env):
    state = env._elevation
    contacts(env, [state["exit_x"][0] + .2] * 4)
    sensor = env.scene["elevation_contacts"].data
    platform = state["step_surfaces"][:, 2]
    sensor.pos[:, [0, 4, 8, 12], 0] = platform[:, 0, None] + .52
    height = platform[:, 2]
    sensor.pos[:, :, 2] = height[:, None]
    sensor.pos += state["corner"][:, None, :]
    sensor.force[:] = 0.
    sensor.force[:, [0, 4, 8, 12], 2] = 10.
    env.scene["robot"].data.root_link_pos_w[:, 0] = state["corner"][:, 0] + platform[:, 0] + .52
    env.scene["robot"].data.root_link_pos_w[:, 1] = state["corner"][:, 1] + 4.
    env.episode_length_buf[:] = 1


class StairsTests(unittest.TestCase):
    def test_right_front_hip_limit_penalizes_only_forward_excess_in_body_frame(self):
        env = paired_env("PointLeg-Stairs-v0", count=5)
        state = env._elevation
        state["initialized"][:] = True
        robot = env.scene["robot"]
        robot.find_bodies = lambda name: ([0 if "hip" in name else 1], [name])
        robot.data.body_link_pos_w = torch.zeros(5, 2, 3)
        angles = torch.deg2rad(torch.tensor([-60., 20., 30., 40., 40.]))
        robot.data.body_link_pos_w[:, 1, 0] = .25 * torch.sin(angles)
        robot.data.body_link_pos_w[:, 1, 2] = -.25 * torch.cos(angles)
        state["failed"][4] = True
        cfg = env.cfg.rewards["fr_hip_forward_limit"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0., 0., 1., 4., 0.]), atol=1.e-5, rtol=0.)
            angle = math.radians(30.)
            robot.data.root_link_quat_w[0] = torch.tensor([math.cos(angle / 2), 0., math.sin(angle / 2), 0.])
            robot.data.body_link_pos_w[0, 1] = torch.tensor([-.25 * math.sin(angle), 0., -.25 * math.cos(angle)])
            self.assertLess(float(term(env, **cfg.params)[0]), 1.e-10)

    def test_right_front_knee_limit_requires_normal_flexion_not_reverse_bending(self):
        env = paired_env("PointLeg-Stairs-v0", count=6)
        state = env._elevation
        state["initialized"][:] = True
        robot = env.scene["robot"]
        robot.find_bodies = lambda name: ([0 if "hip" in name else 1], [name])
        robot.find_sites = lambda names, preserve_order=False: ([1], list(names))
        robot.data.body_link_pos_w = torch.zeros(6, 2, 3)
        robot.data.body_link_pos_w[:, 1, 2] = -.25
        angles = torch.deg2rad(torch.tensor([60., 30., 20., 0., -40., 0.]))
        robot.data.site_pos_w[:, 1, 0] = .28 * torch.sin(angles)
        robot.data.site_pos_w[:, 1, 2] = -.25 - .28 * torch.cos(angles)
        state["failed"][5] = True
        cfg = env.cfg.rewards["fr_knee_min_flexion"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0., 0., 1., 4., 4., 0.]), atol=1.e-5, rtol=0.)
            robot.data.site_pos_w[0, 1, 0] = float("nan")
            self.assertTrue(torch.isfinite(term(env, **cfg.params)).all())

    def test_rear_thigh_above_body_uses_signed_body_frame(self):
        env = paired_env("PointLeg-Stairs-v0", count=4)
        state = env._elevation
        state["initialized"][:] = True
        robot = env.scene["robot"]
        indices = {"rl_hip_pitch_Link": 0, "rl_knee_Link": 1, "rr_hip_pitch_Link": 2, "rr_knee_Link": 3}
        robot.find_bodies = lambda name: ([indices[name]], [name])
        robot.data.body_link_pos_w = torch.zeros(4, 4, 3)
        robot.data.body_link_pos_w[:, 1, 0] = -.25
        robot.data.body_link_pos_w[:, 3, 0] = -.25
        robot.data.body_link_pos_w[:, 1, 2] = torch.tensor([-.1, 0., .1, .1])
        state["failed"][3] = True
        cfg = env.cfg.rewards["rear_thigh_above_body"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            cost = term(env)
            self.assertEqual(float(cost[0]), 0.)
            self.assertEqual(float(cost[1]), 0.)
            self.assertGreater(float(cost[2]), .4)
            self.assertEqual(float(cost[3]), 0.)
            robot.data.body_link_pos_w[2, 1, 2] = .25 * math.tan(math.radians(5.))
            self.assertAlmostEqual(float(term(env, **cfg.params)[2]), .125, places=5)
            angle = math.radians(30.)
            robot.data.root_link_quat_w[1] = torch.tensor([math.cos(angle / 2), 0., math.sin(angle / 2), 0.])
            robot.data.body_link_pos_w[1, 1, 0] = -.25 * math.cos(angle)
            robot.data.body_link_pos_w[1, 1, 2] = .25 * math.sin(angle)
            robot.data.body_link_pos_w[1, 3] = robot.data.body_link_pos_w[1, 1]
            self.assertLess(float(term(env)[1]), 1.e-10)

    def test_front_support_loss_is_immediate_and_resets_when_one_foot_loads(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        env._elevation["initialized"][:] = True
        contacts(env, [2.] * 4)
        sensor = env.scene["elevation_contacts"].data
        sensor.found[:, 0] = 0.
        sensor.found[0, 4] = 0.
        cfg = env.cfg.rewards["front_support_loss"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            env.common_step_counter += 1
            torch.testing.assert_close(term(env), torch.tensor([7. / 12., 0.]))
            torch.testing.assert_close(term(env), torch.tensor([7. / 12., 0.]))
            for _ in range(5):
                env.common_step_counter += 1
                term(env)
            self.assertAlmostEqual(float(term(env)[0]), 1., places=6)
            sensor.found[0, 0] = 1.
            env.common_step_counter += 1
            torch.testing.assert_close(term(env), torch.zeros(2))

    def test_front_step_sequence_penalizes_takeoff_before_other_stabilizes(self):
        env = paired_env("PointLeg-Stairs-v0", count=3)
        env._elevation["initialized"][:] = True
        contacts(env, [2.] * 4)
        sensor = env.scene["elevation_contacts"].data
        sensor.found[:, 4] = 0.
        cfg = env.cfg.rewards["front_premature_takeoff"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            for _ in range(6):
                env.common_step_counter += 1
                torch.testing.assert_close(term(env), torch.zeros(3))
            sensor.found[0, 0] = 0.
            sensor.found[1, 4] = 1.
            sensor.found[2, 4] = 1.
            env.common_step_counter += 1
            torch.testing.assert_close(term(env), torch.tensor([5., 0., 0.]))
            torch.testing.assert_close(term(env), torch.tensor([5., 0., 0.]))
            sensor.found[1, 0] = 0.
            env.common_step_counter += 1
            torch.testing.assert_close(term(env), torch.tensor([0., 5., 0.]))
            for _ in range(10):
                env.common_step_counter += 1
                term(env, **cfg.params)
            sensor.found[2, 0] = 0.
            env.common_step_counter += 1
            self.assertEqual(float(term(env, **cfg.params)[2]), 0.)
            term.reset(torch.tensor([1]))
            self.assertFalse(bool(term.anchored[1].any()))
            self.assertTrue(bool(term.anchored[2].all()))

    def test_front_step_sequence_uses_configured_twenty_centiseconds(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        env._elevation["initialized"][:] = True
        contacts(env, [2.] * 4)
        sensor = env.scene["elevation_contacts"].data
        sensor.found[:, 4] = 0.
        cfg = env.cfg.rewards["front_premature_takeoff"]
        self.assertEqual(cfg.params["stable_time"], .20)
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            env.common_step_counter += 1
            term(env, **cfg.params)
            sensor.found[:, 4] = 1.
            for _ in range(5):
                env.common_step_counter += 1
                term(env, **cfg.params)
            sensor.found[0, 0] = 0.
            env.common_step_counter += 1
            self.assertEqual(float(term(env, **cfg.params)[0]), 5.)
            for _ in range(4):
                env.common_step_counter += 1
                term(env, **cfg.params)
            sensor.found[1, 0] = 0.
            env.common_step_counter += 1
            self.assertEqual(float(term(env, **cfg.params)[1]), 0.)

    def test_front_reach_is_directional_and_only_on_ascent(self):
        env = paired_env("PointLeg-Stairs-v0", count=3)
        state = env._elevation
        state["initialized"][:] = True
        env.scene.terrain.terrain_types[:] = torch.tensor([0, 0, 1])
        robot = env.scene["robot"]
        robot.find_sites = lambda names, preserve_order=False: ([0, 1], list(names))
        robot.find_bodies = lambda name: ([0 if name.startswith("fl") else 1], [name])
        robot.data.body_link_pos_w = torch.zeros(3, 2, 3)
        robot.data.root_link_pos_w[:, 0] = state["corner"][:, 0] + state["entry_x"] - 1.
        robot.data.site_pos_w[:, 0, 0] = torch.tensor([.24, -.34, .34])
        cfg = env.cfg.rewards["front_overreach"]
        self.assertEqual(cfg.params["tolerance"], .18)
        self.assertEqual(cfg.weight, -1.5)
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            torch.testing.assert_close(term(env, **cfg.params), torch.tensor([.125, 0., 0.]), atol=1.e-5, rtol=0.)
            robot.data.site_pos_w[0, 0, 0] = .18
            self.assertEqual(float(term(env, **cfg.params)[0]), 0.)

    def test_down_shin_contact_cost_is_specific_to_shin_force_and_descent(self):
        env = paired_env("PointLeg-Stairs-v0", count=4)
        env._elevation["initialized"][:] = True
        env.scene.terrain.terrain_types[:] = torch.tensor([1, 1, 0, 1])
        sensor = env.scene["elevation_body_contacts"]
        sensor.primary_names = ["robot/base_link"] + [f"robot/{leg}_knee_Link" for leg in ("fl", "fr", "rl", "rr")]
        sensor.data.found = torch.zeros(4, 10)
        sensor.data.force = torch.zeros(4, 10, 3)
        sensor.data.found[:, 4] = 1.
        sensor.data.force[:, 4, 2] = torch.tensor([20., 2., 20., 20.])
        env._elevation["failed"][3] = True
        cfg = env.cfg.rewards["down_shin_contact"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            torch.testing.assert_close(term(env), torch.tensor([1., 0., 0., 0.]))
            self.assertEqual(float(env.extras["log"]["Metrics/stairs_down_shin_force_fr_n"][0]), 20.)
            sensor.data.force[0, 4, 2] = float("nan")
            self.assertTrue(torch.isfinite(term(env)).all())
            self.assertEqual(float(term(env)[0]), 0.)

    def test_swing_speed_penalizes_only_excess_in_unloaded_legs(self):
        env = paired_env("PointLeg-Stairs-v0", count=4)
        state = env._elevation
        state["initialized"][:] = True
        contacts(env, [2.] * 4)
        sensor = env.scene["elevation_contacts"].data
        sensor.found[[0, 1, 3], 4] = 0.
        robot = env.scene["robot"].data
        robot.joint_vel = torch.zeros(4, 12)
        robot.joint_vel[:, 4:6] = torch.tensor([8., -10.])
        robot.joint_vel[1, 4:6] = torch.tensor([4., 6.])
        state["failed"][3] = True
        cfg = env.cfg.rewards["swing_joint_speed"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            torch.testing.assert_close(term(env), torch.tensor([.25, 0., 0., 0.]))
            self.assertIn("Metrics/stairs_swing_hip_speed_fr_rad_s", env.extras["log"])
            robot.joint_vel[0, 4] = float("nan")
            self.assertTrue(torch.isfinite(term(env)).all())
            self.assertEqual(float(term(env)[0]), 0.)

    def test_stuck_recovery_requires_unloaded_stalled_foot_and_rewards_progress_once(self):
        env = paired_env("PointLeg-Stairs-v0", count=4)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(4, 1)})()
        env.scene.terrain.terrain_types[:] = torch.tensor([0, 0, 0, 1])
        state = env._elevation
        state["initialized"][:] = True
        contacts(env, [2.] * 4)
        sensor = env.scene["elevation_contacts"].data
        sensor.found[[0, 2, 3], 0] = 0.
        robot = env.scene["robot"].data
        edge = state["step_surfaces"][:, 0, 0]
        top = state["step_surfaces"][:, 0, 2]
        robot.site_pos_w[:, 0, 0] = state["corner"][:, 0] + edge - .02
        robot.site_pos_w[:, 0, 2] = state["corner"][:, 2] + top - .04
        robot.site_pos_w[2, 0, 0] -= .3
        cfg = env.cfg.rewards["stuck_foot_recovery"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state), patch("pointleg.mdp.stairs.elevation_foot_bottom", side_effect=lambda env, ids: env.scene["robot"].data.site_pos_w[:, ids]):
            for _ in range(20):
                env.common_step_counter += 1
                torch.testing.assert_close(term(env), torch.zeros(4))
            self.assertEqual(term.recovering[:, 0].tolist(), [True, False, False, False])
            robot.site_pos_w[0, 0, 2] = state["corner"][0, 2] + top[0] + .021
            env.common_step_counter += 1
            self.assertGreater(float(term(env)[0]), 0.)
            env.common_step_counter += 1
            self.assertEqual(float(term(env)[0]), 0.)
            robot.site_pos_w[0, 0, 0] = state["corner"][0, 0] + edge[0] + .05
            env.common_step_counter += 1
            self.assertGreater(float(term(env)[0]), 0.)
            sensor.found[0, 0] = 1.
            sensor.pos[0, 0, 0] = state["corner"][0, 0] + edge[0] + .05
            sensor.pos[0, 0, 2] = state["corner"][0, 2] + top[0]
            env.common_step_counter += 1
            self.assertGreater(float(term(env)[0]), 0.)
            self.assertFalse(bool(term.recovering[0, 0]))
            term.reset(torch.tensor([0]))
            self.assertFalse(bool(term.used[0].any()))

    def test_resume_old_completion_clears_success_windows_but_retains_caps(self):
        runner = runner_for("PointLeg-Stairs-v0")
        course = runner.env._elevation_curriculum
        course.level_caps[0] = 2
        course.windows[0].extend([True, False])
        course.windows[1].append(True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "prior.pt"
            runner.save(str(checkpoint))
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            data["infos"]["elevation_walking"]["completion"] = "all_treads_covered_and_four_loaded_feet_on_exit_platform"
            torch.save(data, checkpoint)
            restored = runner_for("PointLeg-Stairs-v0")
            with self.assertWarns(UserWarning):
                restored.load(str(checkpoint))
            self.assertEqual(int(restored.env._elevation_curriculum.level_caps[0]), 2)
            self.assertEqual(len(restored.env._elevation_curriculum.windows[0]), 0)
            self.assertEqual(len(restored.env._elevation_curriculum.windows[1]), 0)

    def test_knee_flexion_rewards_only_lifted_moving_swing_with_support(self):
        env = paired_env("PointLeg-Stairs-v0", count=4)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(4, 1)})()
        contacts(env, [2.] * 4)
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .15
        cfg = env.cfg.rewards["swing_knee_flexion"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            env.common_step_counter += 1
            torch.testing.assert_close(term(env), torch.zeros(4))
            env.scene["elevation_contacts"].data.found[:3, 0] = 0.
            env.scene["robot"].data.site_pos_w[:3, 0, 2] = .03
            env.scene["robot"].data.joint_pos[:, 2] = torch.tensor([-1.6, -1.125, -1.6, -1.6])
            env.scene["robot"].data.root_link_lin_vel_b[2, 0] = 0.
            env.common_step_counter += 1
            score = term(env)
            torch.testing.assert_close(score, torch.tensor([.25, 0., 0., 0.]), atol=1.e-6, rtol=0.)
            torch.testing.assert_close(term(env), score)
            env.scene["robot"].data.joint_pos[0, 2] = -2.2
            env.common_step_counter += 1
            self.assertAlmostEqual(float(term(env)[0]), .25, places=6)

    def test_thigh_target_changes_with_stair_height(self):
        env = paired_env("PointLeg-Stairs-v0", count=5)
        env.scene.terrain.terrain_types[:] = 0
        state = env._elevation
        heights = torch.tensor([.06, .075, .09, .105, .12])
        state["top_height"][:] = heights * 6.
        robot = env.scene["robot"]
        names = [f"{leg}_{suffix}" for leg in ("fl", "fr", "rl", "rr") for suffix in ("hip_pitch_Link", "knee_Link")]
        robot.find_bodies = lambda name: ([names.index(name)], [name])
        robot.data.body_link_pos_w = torch.zeros(5, 8, 3)
        targets = torch.atan(heights / .32) + math.radians(15.)
        robot.data.body_link_pos_w[:, [1, 3, 5, 7], 0] = (.25 * targets.cos())[:, None]
        robot.data.body_link_pos_w[:, [1, 3, 5, 7], 2] = (-.25 * targets.sin())[:, None]
        robot.data.root_link_pos_w[:, 0] = state["corner"][:, 0] + state["entry_x"]
        cfg = env.cfg.rewards["thigh_body_alignment"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            torch.testing.assert_close(term(env, **cfg.params), torch.zeros(5))
            torch.testing.assert_close(env.extras["log"]["Metrics/stairs_thigh_target_angle_deg"], torch.rad2deg(targets))

    def test_left_front_tracking_penalizes_mismatch_but_relaxes_step_height(self):
        env = paired_env("PointLeg-Stairs-v0", count=3)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(3, 1)})()
        contacts(env, [2.] * 4)
        cfg = env.cfg.rewards["left_front_rear_tracking"]
        term = cfg.func(cfg, env)
        env.scene["robot"].data.joint_pos[:, 1] = 1.0
        env.scene["robot"].data.site_pos_w[1, 0, 2] = .12
        env.scene["elevation_contacts"].data.found[2, 0] = 0.
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            cost = term(env, **cfg.params)
        self.assertGreater(float(cost[0]), .4)
        self.assertAlmostEqual(float(cost[1]), float(cost[0]) * .25, places=5)
        self.assertEqual(float(cost[2]), 0.)
        env.scene["robot"].data.joint_pos[:, 10] = 1.0
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))

    def test_thigh_alignment_uses_stair_gradient_and_penalizes_flat_and_vertical(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        robot = env.scene["robot"]
        names = [f"{leg}_{suffix}" for leg in ("fl", "fr", "rl", "rr") for suffix in ("hip_pitch_Link", "knee_Link")]
        robot.find_bodies = lambda name: ([names.index(name)], [name])
        robot.data.body_link_pos_w = torch.zeros(2, 8, 3)
        robot.data.body_link_pos_w[:, [1, 3, 5, 7], 0] = .25
        robot.data.body_link_pos_w[:, [1, 3, 5, 7], 2] = -.12
        state = env._elevation
        robot.data.root_link_pos_w[:, 0] = state["corner"][:, 0] + state["entry_x"]
        cfg = env.cfg.rewards["thigh_body_alignment"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))
            robot.data.body_link_pos_w[:, 1, 2] = 0.
            flat = term(env, **cfg.params)
            self.assertGreater(float(flat[0]), 0.)
            self.assertEqual(float(flat[1]), 0.)
            robot.data.body_link_pos_w[:, 1, 0] = 0.
            robot.data.body_link_pos_w[:, 1, 2] = -.25
            cost = term(env, **cfg.params)
            torch.testing.assert_close(cost, torch.tensor([.25, 0.]))
            robot.data.root_link_pos_w[0, 0] = state["corner"][0, 0] + state["entry_x"][0] - 1.
            torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))

    def test_old_fixed_height_checkpoint_requires_weights_only_transfer(self):
        runner = runner_for("PointLeg-Stairs-v0")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old.pt"
            runner.save(str(checkpoint))
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for contract in data["infos"]["elevation_layout"].values():
                contract.pop("step_height_range_m")
                contract.update(revision=3, fixed_step_height_m=.12)
            torch.save(data, checkpoint)
            with self.assertRaisesRegex(ValueError, "Stair geometry changed"):
                runner_for("PointLeg-Stairs-v0").load(str(checkpoint))
            runner_for("PointLeg-Stairs-v0").load(str(checkpoint), load_cfg={"actor": True, "critic": True})

    def test_stair_height_curriculum_and_independent_direction_promotion(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        terrain = env.cfg.scene.terrain.terrain_generator
        self.assertIsNone(terrain.layout_shape)
        for kind in ("stairs_up", "stairs_down"):
            sub = terrain.sub_terrains[kind]
            self.assertEqual(sub.stair_height_range, (.06, .12))
            self.assertIsNone(sub.stair_height_override)
        for level, row in enumerate(env._elevation["profiles"]):
            for kind in (0, 1):
                self.assertAlmostEqual(row[kind].top_height / 6., .06 + .015 * level)
        cfg = env.cfg.curriculum["elevation_levels"]
        cfg.params.update(window_size=2, threshold=.8)
        course = ElevationTerrainCurriculum(cfg, env)
        env.episode_length_buf[:] = 5
        env._elevation["complete"][:] = torch.tensor([True, False])
        for _ in range(2):
            course(env, None)
        self.assertEqual(course.level_caps[:2].tolist(), [1, 0])
        self.assertEqual(env.scene.terrain.terrain_levels[:2].tolist(), [1, 0])

    def test_swing_lift_penalty_is_per_foot_and_does_not_terminate(self):
        env = paired_env("PointLeg-Stairs-v0", count=3)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(3, 1)})()
        contacts(env, [2.] * 4)
        cfg = env.cfg.rewards["swing_lift_deficit"]
        term = cfg.func(cfg, env)
        state = env._elevation
        sensor = env.scene["elevation_contacts"].data
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            env.common_step_counter += 1
            torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))
            sensor.found[:, 0] = 0.
            env.scene["robot"].data.site_pos_w[:, 0, 2] = torch.tensor([.02, .04, .08])
            for _ in range(5):
                env.common_step_counter += 1
                term(env, **cfg.params)
            sensor.found[:, 0] = 1.
            env.common_step_counter += 1
            cost = term(env, **cfg.params).clone()
            self.assertGreater(float(cost[0]), float(cost[1]))
            self.assertGreater(float(cost[1]), 0.)
            self.assertEqual(float(cost[2]), 0.)
            self.assertFalse(state["failed"].any())
            torch.testing.assert_close(term(env, **cfg.params), cost)
            env.common_step_counter += 1
            torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))
            term.reset(torch.tensor([0]))
            self.assertFalse(bool(term.anchored[0].any()))
            self.assertTrue(bool(term.anchored[1].all()))

    def test_lift_target_tracks_each_environment_stair_height(self):
        env = paired_env("PointLeg-Stairs-v0", count=10)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(10, 1)})()
        contacts(env, [2.] * 4)
        cfg = env.cfg.rewards["swing_lift_deficit"]
        term = cfg.func(cfg, env)
        state = env._elevation
        heights = torch.tensor([.06, .075, .09, .105, .12]).repeat_interleave(2)
        state["top_height"][:] = heights * 6.
        sensor = env.scene["elevation_contacts"].data
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            env.common_step_counter += 1
            term(env, **cfg.params)
            sensor.found[:, 0] = 0.
            env.scene["robot"].data.site_pos_w[:, 0, 2] = heights * .8
            for _ in range(5):
                env.common_step_counter += 1
                term(env, **cfg.params)
            sensor.found[:, 0] = 1.
            env.common_step_counter += 1
            cost = term(env, **cfg.params).clone()
            torch.testing.assert_close(cost, torch.full((10,), .05), atol=1.e-6, rtol=0.)
            torch.testing.assert_close(env.extras["log"]["Metrics/stairs_target_lift_m"], heights)
            self.assertFalse(state["failed"].any())

    def test_placement_rewards_each_foot_interior_more_than_edge_only_once(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        state = env._elevation
        cfg = env.cfg.rewards["tread_foot_placement"]
        term = cfg.func(cfg, env)
        contacts(env, [3.01] * 4, zs=[.12] * 4)
        sensor = env.scene["elevation_contacts"].data
        sensor.found[:, :] = 0.
        sensor.found[:, [0, 4]] = 1.
        tread = state["step_surfaces"][:, 0]
        sensor.pos[:, [0, 4], 0] = tread[:, 0, None] + torch.tensor([[.005], [.10]])
        sensor.pos[:, [0, 4], 2] = tread[:, 2, None]
        sensor.pos += state["corner"][:, None, :]
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            for _ in range(4):
                env.common_step_counter += 1
                torch.testing.assert_close(term(env), torch.zeros(2))
            env.common_step_counter += 1
            reward = term(env).clone()
            self.assertGreater(float(reward[1]), float(reward[0]) * 10.)
            self.assertEqual(int(term.paid[1, :, 0].sum()), 2)
            env.common_step_counter += 1
            torch.testing.assert_close(term(env), torch.zeros(2))

    def test_four_foot_standstill_penalty_resets_on_walking_or_one_foot_lift(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(2, 1)})()
        contacts(env, [2.] * 4)
        cfg = env.cfg.rewards["four_foot_standstill"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=env._elevation):
            for _ in range(85):
                env.common_step_counter += 1
                cost = term(env).clone()
            self.assertGreater(float(cost[0]), .9)
            env.scene["elevation_contacts"].data.found[0, 0] = 0.
            env.scene["robot"].data.root_link_pos_w[1, 0] += .03
            env.common_step_counter += 1
            torch.testing.assert_close(term(env), torch.zeros(2))

    def test_progress_reward_requires_forward_progress_and_stable_support(self):
        env = paired_env("PointLeg-Stairs-v0", count=1)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]])})()
        contacts(env, [2., 2., 2., 2.])
        cfg = env.cfg.rewards["stairs_progress"]
        term = cfg.func(cfg, env)
        state = env._elevation
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            for _ in range(20):
                env.common_step_counter += 1
                self.assertEqual(float(term(env)[0]), 0.)
            env.scene["robot"].data.root_link_pos_w[:, 0] += .07
            env.common_step_counter += 1
            term(env)
            env.scene["elevation_contacts"].data.found[:, 0] = 0.
            env.common_step_counter += 1
            self.assertGreater(float(term(env)[0]), 0.)
            env.common_step_counter += 1
            self.assertEqual(float(term(env)[0]), 0.)

    def test_config_copies_slope_stability_constraints(self):
        env = paired_env("PointLeg-Stairs-v0")
        self.assertEqual(env.cfg.episode_length_s, 120.)
        self.assertEqual(env.cfg.commands["twist"].ranges.lin_vel_x, (.10, .20))
        self.assertEqual(env.cfg.rewards["swing_lift_deficit"].weight, -2.)
        self.assertEqual(env.cfg.rewards["hip_abduction_zero"].weight, -7.2)
        self.assertEqual(env.cfg.rewards["step_timing"].params["target_period"], .80)
        self.assertEqual(env.cfg.rewards["step_timing"].params["minimum_stance"], .30)

    def test_first_three_treads_and_four_loaded_middle_feet_are_required(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        destination_contacts(env)
        collect_elevation_contacts(env)
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            state = update_elevation_state(env)
        self.assertFalse(state["complete"].any())
        env._elevation["covered"][:, :3] = True
        env.common_step_counter += 1
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            state = update_elevation_state(env)
        self.assertTrue(stairs_exit_supported_feet(env).all())
        self.assertTrue(state["complete"].all())
        self.assertFalse(state["failed"].any())

    def test_middle_success_rejects_missing_tread_unloaded_foot_and_wrong_platform(self):
        env = paired_env("PointLeg-Stairs-v0", count=6)
        destination_contacts(env)
        state = env._elevation
        state["covered"][:, :3] = True
        state["covered"][0, 0] = False
        sensor = env.scene["elevation_contacts"].data
        sensor.found[1, 0] = 0.
        sensor.force[2, 0, 2] = 7.
        sensor.pos[3, 0, 2] += .02
        sensor.pos[4, 0, 0] = state["corner"][4, 0] + state["step_surfaces"][4, 2, 0] + .1
        sensor.pos[5, 0, 0] = state["corner"][5, 0] + state["step_surfaces"][5, 2, 1] + .1
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            result = update_elevation_state(env)
        self.assertFalse(result["complete"].any())
        self.assertTrue(stairs_exit_supported_feet(env)[0].all())
        self.assertTrue((stairs_exit_supported_feet(env)[1:].sum(dim=1) == 3).all())

    def test_middle_success_pays_once_and_failure_has_priority(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        destination_contacts(env)
        env._elevation["covered"][:, :3] = True
        env._elevation["outside"][1] = True
        termination = TerminationManager(env.cfg.terminations, env)
        reward = RewardManager({name: env.cfg.rewards[name] for name in ("task_completed", "task_failed")}, env)
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            done = termination.compute()
            values = reward.compute(env.step_dt)
        self.assertTrue(done.all())
        torch.testing.assert_close(values, torch.tensor([20., -50.]))
        self.assertTrue(bool(env._elevation["complete"][0]))
        self.assertFalse(bool(env._elevation["complete"][1]))


    def test_upward_tread_contact_without_120mm_lift_is_diagnostic_only(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        state = env._elevation
        state["foot_initialized"][:, 0] = True
        state["foot_loaded"][:, 0] = False
        state["swing_active"][:, 0] = True
        state["swing_origin_z"][:, 0] = 0.
        state["swing_peak_z"][:, 0] = .119
        first = state["step_surfaces"][:, 0]
        contacts(env, [3.05, 3.05, 3.05, 3.05], zs=[.12] * 4)
        sensor = env.scene["elevation_contacts"].data
        sensor.pos[:, [0, 4, 8, 12], 0] = first[:, 0, None] + .02
        sensor.pos[:, [0, 4, 8, 12], 2] = first[:, 2, None]
        collect_elevation_contacts(env)
        self.assertTrue(bool(state["stairs_clearance_failed"][0]))
        self.assertFalse(bool(state["stairs_clearance_failed"][1]))
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            failed = update_elevation_state(env)["failed"]
        self.assertFalse(bool(failed[0]))
        self.assertFalse(bool(failed[1]))


if __name__ == "__main__":
    unittest.main()
