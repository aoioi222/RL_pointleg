"""Check fixed-height stair completion and clearance without physics stepping."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from mjlab.managers.termination_manager import TerminationManager
from pointleg.mdp.elevation import ElevationTerrainCurriculum, collect_elevation_contacts, stairs_exit_supported_feet, update_elevation_state
from test_elevation import bounds, contacts
from test_elevation_paired import paired_env, runner_for


def destination_contacts(env):
    state = env._elevation
    contacts(env, [state["exit_x"][0] + .2] * 4)
    sensor = env.scene["elevation_contacts"].data
    sensor.pos[:, [0, 4, 8, 12], 0] = state["exit_x"][:, None] + .2
    height = torch.where(env.scene.terrain.terrain_types == 0, state["top_height"], 0.)
    sensor.pos[:, :, 2] = height[:, None]
    sensor.pos += state["corner"][:, None, :]
    sensor.force[:] = 0.
    sensor.force[:, [0, 4, 8, 12], 2] = 10.
    env.scene["robot"].data.root_link_pos_w[:, 0] = state["corner"][:, 0] + state["exit_x"] + .2
    env.scene["robot"].data.root_link_pos_w[:, 1] = state["corner"][:, 1] + 4.
    env.episode_length_buf[:] = 1


class StairsTests(unittest.TestCase):
    def test_thigh_alignment_is_body_relative_soft_and_upstairs_only(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        robot = env.scene["robot"]
        names = [f"{leg}_{suffix}" for leg in ("fl", "fr", "rl", "rr") for suffix in ("hip_pitch_Link", "knee_Link")]
        robot.find_bodies = lambda name: ([names.index(name)], [name])
        robot.data.body_link_pos_w = torch.zeros(2, 8, 3)
        robot.data.body_link_pos_w[:, [1, 3, 5, 7], 0] = .25
        state = env._elevation
        robot.data.root_link_pos_w[:, 0] = state["corner"][:, 0] + state["entry_x"]
        cfg = env.cfg.rewards["thigh_body_alignment"]
        term = cfg.func(cfg, env)
        with patch("pointleg.mdp.stairs.update_elevation_state", return_value=state):
            torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))
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
        self.assertEqual(env.cfg.commands["twist"].ranges.lin_vel_x, (.15, .30))
        self.assertEqual(env.cfg.rewards["hip_abduction_zero"].weight, -7.2)
        self.assertEqual(env.cfg.rewards["step_timing"].params["target_period"], .70)
        self.assertEqual(env.cfg.rewards["step_timing"].params["minimum_stance"], .30)

    def test_all_treads_and_four_loaded_destination_feet_are_required(self):
        env = paired_env("PointLeg-Stairs-v0", count=2)
        destination_contacts(env)
        collect_elevation_contacts(env)
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            state = update_elevation_state(env)
        self.assertFalse(state["complete"].any())
        env._elevation["covered"][:] = True
        env.common_step_counter += 1
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            state = update_elevation_state(env)
        self.assertTrue(stairs_exit_supported_feet(env).all())
        self.assertTrue(state["complete"].all())
        self.assertFalse(state["failed"].any())

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
