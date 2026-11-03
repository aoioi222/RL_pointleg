"""Regression tests for the generic rough-terrain training stage."""

from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains import TerrainGenerator
from mjlab.terrains.terrain_entity import TerrainEntity

from pointleg.config.evaluation import evaluation_env_cfg
from pointleg.evaluate import EpisodeResult, build_case_report
from pointleg.mdp.curriculums import rough_terrain_levels
from pointleg.robot_cfg import FOOT_BODY_NAMES, FOOT_SITE_NAMES
from pointleg.mdp.rewards import (
    base_clearance_outside_range_l2,
    command_aligned_progress_reward,
    command_progress_deficit_penalty,
    knee_extension_penalty,
    pitch_excess_penalty,
    ForwardFootTouchdownPenalty,
    ForwardFootSwingPenalty,
)


class FakeScene(dict):
    def __init__(self, robot, terrain, env_origins):
        super().__init__(robot=robot)
        self.terrain = terrain
        self.env_origins = env_origins


def terrain_curriculum_env(
    progress_x,
    progress_y=None,
    level=0,
    failed=None,
    integrated_progress=None,
    integrated_command_speed=None,
):
    progress_x = torch.as_tensor(progress_x, dtype=torch.float32)
    count = len(progress_x)
    if progress_y is None:
        progress_y = torch.zeros(count)
    if failed is None:
        failed = torch.zeros(count, dtype=torch.bool)
    terrain_origins = torch.zeros(10, 2, 3)
    generator = SimpleNamespace(
        num_rows=10,
        size=(8.0, 8.0),
        sub_terrains={name: object() for name in (
            "random_rough", "low_random_bumps"
        )},
    )
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=generator),
        terrain_origins=terrain_origins,
        env_origins=torch.zeros(count, 3),
        terrain_levels=torch.full((count,), level, dtype=torch.long),
        terrain_types=torch.zeros(count, dtype=torch.long),
    )
    position = torch.stack((progress_x, torch.as_tensor(progress_y), torch.zeros(count)), dim=1)
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=position))
    direction = torch.tensor([[0.4, 0.0]]).repeat(count, 1)
    command_term = SimpleNamespace(progress_direction_w=direction)
    command = torch.tensor([[0.4, 0.0, 0.0]]).repeat(count, 1)
    scene = FakeScene(robot, terrain, terrain.env_origins)
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        scene=scene,
        episode_length_buf=torch.full((count,), 1000, dtype=torch.long),
        reset_terminated=failed,
        reset_time_outs=~failed,
        command_manager=SimpleNamespace(
            get_term=lambda name: command_term,
            get_command=lambda name: command,
        ),
    )
    if integrated_progress is not None and integrated_command_speed is not None:
        env.metrics_manager = SimpleNamespace(_episode_sums={
            "effective_progress": torch.as_tensor(integrated_progress, dtype=torch.float32),
            "cmd_lin_vel": torch.as_tensor(integrated_command_speed, dtype=torch.float32),
        })
    return env


def make_terrain_curriculum(env, window_size=100, initial_level_cap=0):
    cfg = CurriculumTermCfg(
        func=rough_terrain_levels,
        params={
            "command_name": "twist",
            "initial_level_cap": initial_level_cap,
            "window_size": window_size,
            "success_threshold": 0.8,
            "success_progress_ratio": 0.6,
            "demote_progress_ratio": 0.4,
        },
    )
    return rough_terrain_levels(cfg, env)


class RoughTerrainConfigurationTests(unittest.TestCase):
    def test_terrain_mix_and_final_difficulty_limits(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        terrain = cfg.scene.terrain
        generator = terrain.terrain_generator
        self.assertEqual(
            list(generator.sub_terrains),
            ["random_rough", "low_random_bumps"],
        )
        proportions = [sub.proportion for sub in generator.sub_terrains.values()]
        np.testing.assert_allclose(proportions, [0.52, 0.48])
        self.assertAlmostEqual(sum(proportions), 1.0)
        self.assertEqual(terrain.max_init_terrain_level, 2)
        self.assertEqual(cfg.curriculum["terrain_levels"].params["initial_level_cap"], 2)
        self.assertEqual(generator.difficulty_range, (0.15, 1.0))
        self.assertEqual(generator.border_width, 2.0)
        self.assertTrue(generator.shuffle_curriculum_columns)
        self.assertEqual(generator.sub_terrains["random_rough"].noise_range, (-0.085, 0.085))
        self.assertTrue(generator.sub_terrains["random_rough"].scale_with_difficulty)
        self.assertEqual(generator.sub_terrains["random_rough"].horizontal_scale, 0.10)
        self.assertEqual(generator.sub_terrains["random_rough"].border_width, 0.0)
        self.assertEqual(generator.sub_terrains["low_random_bumps"].grid_height_range, (0.0, 0.085))
        bumps = generator.sub_terrains["low_random_bumps"]
        self.assertAlmostEqual(bumps.platform_width, 3.0 * bumps.grid_width)

    def test_terrain_level_zero_has_finite_supported_origins_and_compiles(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        generator = TerrainGenerator(cfg.scene.terrain.terrain_generator)
        spec = mujoco.MjSpec()
        generator.compile(spec)
        model = spec.compile()
        self.assertTrue(np.isfinite(generator.terrain_origins).all())
        self.assertGreater(np.ptp(model.hfield_data[:model.hfield_nrow[0] * model.hfield_ncol[0]]), 0.0)
        self.assertEqual(generator.terrain_type_columns.shape, (10, 2))
        for terrain_columns in generator.terrain_type_columns:
            np.testing.assert_array_equal(np.sort(terrain_columns), np.arange(2))
        self.assertEqual(generator.grid_shape, (4, 4))
        self.assertEqual(np.unique(generator.terrain_origins[:, :, :2].reshape(-1, 2), axis=0).shape[0], 16)

    def test_random_rough_physical_surface_respects_seventeen_centimeter_total_drop(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        rough = cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"]
        for difficulty in (0.15, 1.0):
            for seed in (42, 123):
                spec = mujoco.MjSpec()
                spec.worldbody.add_body(name="terrain")
                output = rough.function(difficulty, spec, np.random.default_rng(seed))
                geometry = output.geometries[0]
                self.assertEqual(geometry.hfield.nrow, 80)
                self.assertEqual(geometry.hfield.ncol, 80)
                heights = (
                    geometry.geom.pos[2]
                    + np.asarray(geometry.hfield.userdata) * geometry.hfield.size[2]
                )
                limit = 0.085 * difficulty
                self.assertGreaterEqual(float(heights.min()), -limit - 1.0e-6)
                self.assertLessEqual(float(heights.max()), limit + 1.0e-6)
                self.assertGreater(float(heights.max()), 0.0)
                self.assertLess(float(heights.min()), 0.0)
                self.assertLessEqual(float(np.ptp(heights)), 0.17 * difficulty + 1.0e-6)
                if difficulty == 1.0:
                    self.assertGreaterEqual(float(np.ptp(heights)), 0.165 - 1.0e-6)
                spec.compile()

    def test_bump_physical_tops_respect_seventeen_centimeter_total_drop(self):
        bumps = load_env_cfg("PointLeg-Rough-v0").scene.terrain.terrain_generator.sub_terrains["low_random_bumps"]
        for difficulty in (0.15, 0.5, 1.0):
            for seed in (42, 123):
                spec = mujoco.MjSpec()
                spec.worldbody.add_body(name="terrain")
                output = bumps.function(difficulty, spec, np.random.default_rng(seed))
                tops = np.asarray([geometry.geom.pos[2] + geometry.geom.size[2] for geometry in output.geometries])
                limit = 0.085 * difficulty
                self.assertGreaterEqual(float(tops.min()), -limit - 1.0e-6)
                self.assertLessEqual(float(tops.max()), limit + 1.0e-6)
                self.assertLessEqual(float(np.ptp(tops)), 0.17 * difficulty + 1.0e-6)
                self.assertAlmostEqual(output.origin[2], limit)
                if difficulty == 1.0:
                    self.assertGreaterEqual(float(np.ptp(tops)), 0.165)
                spec.compile()

    def test_bump_edges_fit_ten_centimeter_seams_without_platform_holes(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        bumps = cfg.scene.terrain.terrain_generator.sub_terrains["low_random_bumps"]
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        output = bumps.function(0.5, spec, np.random.default_rng(42))
        self.assertAlmostEqual(output.geometries[0].geom.size[1] * 4.0, 0.10)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for x in np.linspace(0.01, 7.99, 161):
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                model, data, np.array([x, 4.1, 1.0]), np.array([0.0, 0.0, -1.0]),
                None, 1, -1, geom_id,
            )
            self.assertGreaterEqual(distance, 0.0, f"Missing collision surface at x={x}")

    def test_play_resamples_tiles_before_reset_while_fixed_evaluation_keeps_cases(self):
        cfg = load_env_cfg("PointLeg-Rough-v0", play=True)
        self.assertIsNone(cfg.scene.terrain.max_init_terrain_level)
        self.assertEqual(list(cfg.events)[0], "randomize_terrain")
        self.assertIn("reset_scene", cfg.events)
        self.assertEqual(cfg.events["randomize_terrain"].mode, "reset")
        for profile in ("nominal", "robust"):
            fixed_cfg = evaluation_env_cfg("PointLeg-Rough-v0", profile)
            self.assertNotIn("randomize_terrain", fixed_cfg.events)

    def test_shuffled_tiles_preserve_semantic_terrain_type_origins(self):
        terrain = object.__new__(TerrainEntity)
        terrain.terrain_origins = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        terrain.terrain_type_columns = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
        origins = terrain.get_terrain_origins(
            torch.tensor([0, 0, 1, 1]), torch.tensor([0, 3, 1, 2])
        )
        torch.testing.assert_close(origins, torch.tensor([
            [6.0, 7.0, 8.0], [3.0, 4.0, 5.0],
            [21.0, 22.0, 23.0], [12.0, 13.0, 14.0],
        ]))

    def test_commands_rewards_and_metrics_follow_generic_rough_contract(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        command = cfg.commands["twist"]
        flat = load_env_cfg("PointLeg-Flat-v0").commands["twist"]
        self.assertEqual(command.ranges, flat.ranges)
        for name in ("rel_standing_envs", "rel_forward_envs", "rel_backward_envs",
                     "backward_velocity_range", "resampling_time_range", "heading_control_stiffness"):
            self.assertEqual(getattr(command, name), getattr(flat, name))
        self.assertEqual(command.rel_lateral_envs, flat.flat_lateral_probability)
        self.assertEqual(command.rel_yaw_envs, flat.flat_yaw_probability)
        self.assertEqual(set(cfg.curriculum), {"terrain_levels"})
        self.assertLess(cfg.rewards["is_terminated"].weight, 0.0)
        for name in ("command_aligned_progress", "command_progress_deficit", "feet_slip",
                     "support_deficit", "long_foot_air_time", "knee_extension",
                     "forward_foot_touchdown", "forward_foot_swing", "stance_balance",
                     "contact_forces", "local_ground_clearance", "joint_power"):
            self.assertIn(name, cfg.rewards)
        self.assertEqual(cfg.rewards["knee_extension"].weight, -0.5)
        self.assertAlmostEqual(
            cfg.rewards["knee_extension"].params["maximum_angle"], np.deg2rad(-25.0)
        )
        self.assertEqual(cfg.rewards["feet_air_time"].weight, 0.05)
        self.assertEqual(cfg.rewards["long_foot_air_time"].params["max_air_time"], 0.80)
        self.assertEqual(cfg.rewards["local_ground_clearance"].params["minimum_height"], 0.28)
        self.assertEqual(cfg.rewards["local_ground_clearance"].params["maximum_height"], 0.44)
        for name in ("terrain_type", "terrain_level", "effective_progress", "feet_slip",
                     "contact_overload", "joint_power", "failure_bad_orientation"):
            self.assertIn(name, cfg.metrics)
        for name in ("stair_lateral_yaw_drift", "joint_mirror", "abduction_mirror",
                     "undesired_contacts"):
            self.assertNotIn(name, cfg.rewards)


class RoughTerrainCurriculumTests(unittest.TestCase):
    def test_first_reset_preserves_distributed_initial_levels(self):
        env = terrain_curriculum_env([0.0, 0.0, 0.0])
        env.scene.terrain.terrain_levels[:] = torch.tensor([0, 1, 2])
        env.episode_length_buf.zero_()
        curriculum = make_terrain_curriculum(env, initial_level_cap=2)
        curriculum(env, torch.arange(3))
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [0, 1, 2])
        self.assertEqual(curriculum._level_cap, 2)
        self.assertEqual(len(curriculum._success_window), 0)

    def test_failure_demotes_one_level_from_initial_cap(self):
        env = terrain_curriculum_env([0.0], level=2, failed=torch.tensor([True]))
        curriculum = make_terrain_curriculum(env, initial_level_cap=2)
        curriculum(env, torch.tensor([0]))
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [1])

    def test_eighty_percent_of_one_hundred_unlocks_one_level(self):
        failed = torch.zeros(100, dtype=torch.bool)
        failed[80:] = True
        progress = torch.full((100,), 3.0)
        progress[80:] = 0.0
        env = terrain_curriculum_env(progress, failed=failed)
        curriculum = make_terrain_curriculum(env)
        result = curriculum(env, torch.arange(100))
        self.assertEqual(curriculum._level_cap, 1)
        self.assertAlmostEqual(result["success_rate"].item(), 0.8)
        self.assertTrue((env.scene.terrain.terrain_levels[:80] == 1).all())

    def test_seventy_nine_percent_does_not_unlock(self):
        failed = torch.zeros(100, dtype=torch.bool)
        failed[79:] = True
        progress = torch.full((100,), 3.0)
        progress[79:] = 0.0
        env = terrain_curriculum_env(progress, failed=failed)
        curriculum = make_terrain_curriculum(env)
        curriculum(env, torch.arange(100))
        self.assertEqual(curriculum._level_cap, 0)

    def test_reverse_and_lateral_motion_demote_while_middle_band_holds(self):
        env = terrain_curriculum_env([-1.0, 0.0, 1.8], progress_y=[0.0, 3.0, 0.0], level=2)
        curriculum = make_terrain_curriculum(env)
        curriculum._level_cap = 3
        curriculum(env, torch.arange(3))
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [1, 1, 2])

    def test_integrated_metrics_follow_commands_that_change_during_episode(self):
        env = terrain_curriculum_env(
            progress_x=[-2.0, 4.0],
            level=1,
            integrated_progress=[280.0, -50.0],
            integrated_command_speed=[400.0, 400.0],
        )
        curriculum = make_terrain_curriculum(env, window_size=10)
        curriculum._level_cap = 2
        result = curriculum(env, torch.arange(2))
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [2, 0])
        self.assertAlmostEqual(result["effective_progress"].item(), 2.3)


class RoughRewardAndReportTests(unittest.TestCase):
    def test_rough_air_time_reward_requires_support_and_a_motion_command(self):
        found = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ])
        sensor = SimpleNamespace(data=SimpleNamespace(
            found=found,
            current_air_time=torch.where(found > 0, 0.0, 0.2),
        ))
        commands = torch.tensor([[0.3, 0.0, 0.0]]).repeat(4, 1)
        commands[3] = 0.0
        env = SimpleNamespace(
            scene={"feet_ground_contact": sensor}, extras={"log": {}},
            command_manager=SimpleNamespace(get_command=lambda name: commands),
        )
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["feet_air_time"]
        torch.testing.assert_close(
            cfg.func(env, **cfg.params) * cfg.weight,
            torch.tensor([0.10, 0.0, 0.0, 0.0]),
        )

    def test_rough_short_stance_cost_requires_brief_stance_and_actual_takeoff(self):
        duration = torch.zeros(4, 4)
        duration[:, 0] = torch.tensor([0.04, 0.12, 0.04, 0.04])
        airborne = torch.zeros(4, 4, dtype=torch.bool)
        airborne[:3, 0] = True
        sensor = SimpleNamespace(
            data=SimpleNamespace(last_contact_time=duration),
            compute_first_air=lambda dt: airborne,
        )
        commands = torch.tensor([[0.3, 0.0, 0.0]]).repeat(4, 1)
        commands[2] = 0.0
        env = SimpleNamespace(
            scene={"feet_ground_contact": sensor}, step_dt=0.02,
            command_manager=SimpleNamespace(get_command=lambda name: commands),
        )
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["short_stance"]
        torch.testing.assert_close(
            cfg.func(env, **cfg.params) * cfg.weight * env.step_dt,
            torch.tensor([-0.0108, 0.0, 0.0, 0.0]),
        )

    def test_rough_hip_cost_allows_twelve_degrees_and_does_not_penalize_other_joints(self):
        q = torch.zeros(3, 12)
        q[:, [1, 2, 4, 5, 7, 8, 10, 11]] = 2.0
        q[0, [0, 3, 6, 9]] = torch.deg2rad(torch.tensor([10.0, -10.0, 10.0, -10.0]))
        q[1, [0, 3, 6, 9]] = torch.deg2rad(torch.tensor([20.0, -20.0, 20.0, -20.0]))
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos=q, default_joint_pos=torch.zeros_like(q)))
        env = SimpleNamespace(scene={"robot": robot}, extras={"log": {}})
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["hip_abduction_excess"]
        self.assertEqual(cfg.params["asset_cfg"].joint_names, (".*_hip_abduction_joint",))
        params = {**cfg.params, "asset_cfg": SimpleNamespace(name="robot", joint_ids=[0, 3, 6, 9])}
        expected = torch.tensor([0.0, -0.2 * 4 * (np.deg2rad(8.0) / 0.25) ** 2, 0.0], dtype=torch.float32)
        torch.testing.assert_close(cfg.func(env, **params) * cfg.weight, expected)

    def test_swing_overreach_penalizes_every_leg_and_turning_without_penalizing_stance(self):
        count = 8
        sensor = SimpleNamespace(
            primary_names=list(reversed(FOOT_BODY_NAMES)),
            data=SimpleNamespace(found=torch.zeros(count, 4)),
        )
        sensor.data.found[4] = 1.0
        positions = torch.zeros(count, 4, 3)
        positions[:, :, 0] = torch.tensor([0.43, 0.43, -0.05, -0.05])
        for leg in range(4):
            positions[leg, leg, 0] += 0.28
        positions[4:7, 1, 0] += 0.28
        robot = SimpleNamespace(
            find_sites=lambda name: ([FOOT_SITE_NAMES.index(name)], [name]),
            data=SimpleNamespace(
                root_link_pos_w=torch.zeros(count, 3),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
                site_pos_w=positions,
            ),
        )
        commands = torch.tensor([[0.3, 0.0, 0.0]]).repeat(count, 1)
        commands[5] = 0.0
        commands[6] = torch.tensor([0.0, 0.0, 0.4])
        env = SimpleNamespace(
            num_envs=count,
            scene={"robot": robot, "feet_ground_contact": sensor},
            command_manager=SimpleNamespace(get_command=lambda name: commands),
        )
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["forward_foot_swing"]
        self.assertIs(cfg.func, ForwardFootSwingPenalty)
        term = cfg.func(cfg, env)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        torch.testing.assert_close(term(env, **cfg.params), expected)
        torch.testing.assert_close(term(env, **cfg.params), expected)
        sensor.data.found[0, 3] = 1.0
        expected[0] = 0.0
        torch.testing.assert_close(term(env, **cfg.params), expected)
        local = positions.clone()
        shift = torch.tensor([3.0, -2.0, 1.0])
        robot.data.root_link_pos_w[:] = shift
        robot.data.root_link_quat_w[:] = torch.tensor([2.0 ** -0.5, 0.0, 0.0, 2.0 ** -0.5])
        robot.data.site_pos_w[:] = torch.stack((-local[:, :, 1], local[:, :, 0], local[:, :, 2]), dim=-1) + shift
        torch.testing.assert_close(term(env, **cfg.params), expected)

    def test_rough_stance_balance_allows_alternation_but_penalizes_one_unused_foot(self):
        contact = torch.ones(3, 4)
        sensor = SimpleNamespace(primary_names=FOOT_BODY_NAMES, data=SimpleNamespace(found=contact))
        commands = torch.tensor([[0.3, 0.0, 0.0]]).repeat(3, 1)
        commands[2, 2] = 0.3
        env = SimpleNamespace(
            num_envs=3, device="cpu", step_dt=0.02, extras={"log": {}},
            scene={"feet_ground_contact": sensor},
            command_manager=SimpleNamespace(get_command=lambda name: commands),
        )
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["stance_balance"]
        term = cfg.func(cfg, env)
        for step in range(200):
            contact[0] = torch.tensor([1.0, 0.0, 0.0, 1.0]) if (step // 15) % 2 else torch.tensor([0.0, 1.0, 1.0, 0.0])
            contact[1:] = torch.tensor([1.0, 0.0, 1.0, 1.0])
            cost = term(env, **cfg.params)
        self.assertLess(cost[0].item(), 0.01)
        self.assertGreater(cost[1].item(), 1.0)
        self.assertEqual(cost[2].item(), 0.0)
        previous = term.duty[1].clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.duty[1], previous)
        torch.testing.assert_close(term.duty[0], torch.full((4,), 0.5))
        term.reset()
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))

    def test_pitch_cost_penalizes_constant_lean_symmetrically_and_ignores_roll(self):
        angles = torch.deg2rad(torch.tensor([0.0, 4.0, 10.0, -10.0, 15.0, 20.0]))
        gravity = torch.stack((angles.sin(), torch.zeros_like(angles), -angles.cos()), dim=1)
        gravity = torch.cat((gravity, torch.tensor([[0.0, 1.0, 0.0]])))
        env = SimpleNamespace(scene={
            "robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity)),
        })
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["pitch_excess"]
        self.assertIs(cfg.func, pitch_excess_penalty)
        cost = cfg.func(env, **cfg.params)
        torch.testing.assert_close(
            cost * cfg.weight,
            torch.tensor([0.0, 0.0, -0.125, -0.125, -0.5, -1.125, 0.0]),
        )
        torch.testing.assert_close(cfg.func(env, **cfg.params), cost)
        self.assertNotIn("pitch_excess", load_env_cfg("PointLeg-Flat-v0").rewards)

    def test_aligned_progress_is_signed_and_back_and_forth_cancels(self):
        velocity = torch.tensor([[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0]])
        robot = SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))
        command = torch.tensor([[0.4, 0.0, 0.0]]).repeat(3, 1)
        env = SimpleNamespace(
            scene={"robot": robot},
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
        reward = command_aligned_progress_reward(env, "twist")
        torch.testing.assert_close(reward, torch.tensor([1.0, -1.0, 0.0]))
        self.assertEqual((reward[0] + reward[1]).item(), 0.0)

    def test_progress_deficit_penalizes_stall_and_reverse_but_not_valid_progress(self):
        velocity = torch.tensor([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0],
            [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0],
        ])
        robot = SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))
        command = torch.tensor([
            [0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0], [0.0, 0.0, 0.0],
        ])
        env = SimpleNamespace(
            scene={"robot": robot},
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
        penalty = command_progress_deficit_penalty(env, "twist")
        torch.testing.assert_close(penalty, torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]))

    def test_knee_extension_penalty_is_one_sided(self):
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([
            [np.deg2rad(-50.0), np.deg2rad(-25.0)],
            [np.deg2rad(-15.0), np.deg2rad(0.0)],
        ], dtype=torch.float32)))
        asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])
        env = SimpleNamespace(scene={"robot": robot})
        penalty = knee_extension_penalty(
            env,
            maximum_angle=np.deg2rad(-25.0),
            transition_width=np.deg2rad(10.0),
            asset_cfg=asset_cfg,
        )
        torch.testing.assert_close(penalty[0], torch.tensor(0.0))
        self.assertGreater(penalty[1].item(), 1.0)

    def test_forward_touchdown_penalty_only_applies_to_overreach_in_straight_forward_motion(self):
        class Robot:
            def __init__(self):
                self.data = SimpleNamespace(
                    root_link_pos_w=torch.zeros(3, 3),
                    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
                    site_pos_w=torch.tensor([
                        [[0.43, 0.0, 0.0], [0.43, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
                        [[0.75, 0.0, 0.0], [0.43, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
                        [[0.75, 0.0, 0.0], [0.43, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
                    ]),
                )

            def find_sites(self, name):
                return [FOOT_SITE_NAMES.index(name)], []

        contacts = torch.tensor([[True, False, False, False]]).repeat(3, 1)
        sensor = SimpleNamespace(
            primary_names=FOOT_BODY_NAMES,
            data=SimpleNamespace(found=contacts),
            compute_first_contact=lambda dt: contacts,
        )
        command = torch.tensor([
            [0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.2],
        ])
        env = SimpleNamespace(
            num_envs=3,
            step_dt=0.02,
            scene={"robot": Robot(), "feet_ground_contact": sensor},
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
        cfg = SimpleNamespace(params={"sensor_name": "feet_ground_contact"})
        penalty = ForwardFootTouchdownPenalty(cfg, env)(
            env,
            sensor_name="feet_ground_contact",
            command_name="twist",
            nominal_foot_x=(0.43, 0.43, -0.05, -0.05),
        )
        torch.testing.assert_close(penalty[[0, 2]], torch.zeros(2))
        self.assertGreater(penalty[1].item(), 0.0)

    def test_local_clearance_penalty_has_safe_band_and_invalid_failure_cost(self):
        root = torch.tensor([[0.0, 0.0, 0.30], [0.0, 0.0, 0.15], [0.0, 0.0, 0.30]])
        hit = torch.zeros(3, 1, 3)
        normals = torch.zeros(3, 1, 3)
        normals[:, :, 2] = 1.0
        distances = root[:, 2, None].clone()
        distances[2] = -1.0
        env = SimpleNamespace(scene={
            "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root)),
            "ground": SimpleNamespace(data=SimpleNamespace(
                hit_pos_w=hit, normals_w=normals, distances=distances,
            )),
        })
        penalty = base_clearance_outside_range_l2(env, 0.22, 0.55, "ground")
        torch.testing.assert_close(penalty, torch.tensor([0.0, 0.07 ** 2, 1.0]))

    def test_lower_rough_clearance_band_accepts_lower_body_and_penalizes_high_body(self):
        heights = torch.tensor([0.27, 0.28, 0.32, 0.42, 0.44, 0.48])
        root = torch.zeros(6, 3)
        root[:, 2] = heights
        normals = torch.zeros(6, 1, 3)
        normals[:, :, 2] = 1.0
        env = SimpleNamespace(scene={
            "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root)),
            "base_height_sensor": SimpleNamespace(data=SimpleNamespace(
                hit_pos_w=torch.zeros(6, 1, 3), normals_w=normals,
                distances=heights[:, None],
            )),
        })
        cfg = load_env_cfg("PointLeg-Rough-v0").rewards["local_ground_clearance"]
        torch.testing.assert_close(
            cfg.func(env, **cfg.params) * cfg.weight,
            torch.tensor([-0.002, 0.0, 0.0, 0.0, 0.0, -0.032]),
        )

    def test_report_uses_complete_attempt_count_and_failure_reasons(self):
        results = [
            EpisodeResult(True, "time_out", 0.1, 3.0, 0.2, 5.0, 0.0, 11),
            EpisodeResult(False, "bad_orientation", 0.4, 1.0, 0.5, 7.0, 2.0, 12),
        ]
        report = build_case_report("random_rough", 9, results, 0.8)
        self.assertEqual((report["attempts"], report["successes"]), (2, 1))
        self.assertEqual(report["termination_reasons"], {"bad_orientation": 1, "time_out": 1})
        self.assertEqual(report["seeds"], [11, 12])
        self.assertFalse(report["accepted"])


if __name__ == "__main__":
    unittest.main()
