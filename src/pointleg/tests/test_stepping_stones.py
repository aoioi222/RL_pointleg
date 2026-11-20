"""Regression tests for fixed stepping-stone geometry and adjudication."""

import math
from dataclasses import replace
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import mujoco
import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator
from pointleg.config.stepping_stones import configure_competition_six
from pointleg.run import main as pointleg_main
from pointleg.mdp.stepping_stones import (
    EXIT_SETTLE_TIME_S,
    EXIT_SETTLE_TIMEOUT_S,
    NO_FORWARD_STEP_GRACE_S,
    RIGHT_REAR_FOLLOW_GRACE_S,
    SPAWN_X,
    SteppingStoneCurriculum,
    SteppingStonesVelocityCommand,
    _foot_bottom,
    collect_stepping_stones_substep,
    cover_reward,
    foot_contact_observation,
    forward_velocity_reward,
    reset_stepping_stones,
    route_alignment_reward,
    settle_progress_observation,
    stage_observation,
    stepping_bad_orientation,
    stepping_body_level_reward,
    stepping_failed,
    stepping_failure_reward,
    stepping_flight_penalty,
    stepping_foot_slip_penalty,
    stepping_hip_abduction_symmetry_reward,
    stepping_lateral_velocity_penalty,
    stepping_metric,
    stepping_shin_support_penalty,
    stepping_settle_timeout,
    stepping_stalled,
    stone_center_reward,
    success_reward,
)
from pointleg.robot_cfg import FOOT_BODY_NAMES, FOOT_SITE_NAMES, LEG_JOINT_NAMES, LEG_NAMES
from pointleg.terrains.stepping_stones import (
    MAX_STONES,
    PLATFORM_HEIGHT,
    PLATFORM_SIZE,
    STONE_SIZE,
    SteppingStonesTerrainCfg,
    right_platform_start,
    stone_centers,
)


class Scene(dict):
    pass


class Robot:
    def __init__(self, count, spawn_height):
        self.joint_names = list(LEG_JOINT_NAMES)
        root = torch.zeros(count, 13)
        root[:, 2] = spawn_height
        root[:, 3] = 1.0
        self.data = SimpleNamespace(
            default_root_state=root,
            root_link_pos_w=root[:, :3].clone(),
            root_link_quat_w=root[:, 3:7].clone(),
            root_link_lin_vel_b=torch.zeros(count, 3),
            root_link_lin_vel_w=torch.zeros(count, 3),
            root_link_ang_vel_w=torch.zeros(count, 3),
            joint_pos=torch.zeros(count, len(LEG_JOINT_NAMES)),
            default_joint_pos=torch.zeros(count, len(LEG_JOINT_NAMES)),
            encoder_bias=torch.zeros(count, len(LEG_JOINT_NAMES)),
            soft_joint_pos_limits=torch.tensor([-2.0, 2.0]).expand(count, len(LEG_JOINT_NAMES), 2),
            site_pos_w=torch.zeros(count, 4, 3),
            body_link_pos_w=torch.zeros(count, 4, 3),
            site_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 4, 1),
            heading_w=torch.zeros(count),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        )
        self.written = root.clone()

    def find_sites(self, name):
        return [FOOT_SITE_NAMES.index(name)], [name]

    def find_joints(self, name):
        return [self.joint_names.index(name)], [name]

    def find_bodies(self, name):
        return [[f"{leg}_knee_Link" for leg in LEG_NAMES].index(name)], [name]

    def find_joints_by_actuator_names(self, patterns):
        return list(range(len(self.joint_names))), self.joint_names

    def set_joint_position_target(self, target, joint_ids):
        self.position_target = target.clone()

    def write_root_state_to_sim(self, value, env_ids):
        self.written[env_ids] = value
        self.data.root_link_pos_w[env_ids] = value[:, :3]
        self.data.root_link_quat_w[env_ids] = value[:, 3:7]


def legacy_cfg():
    """Keep historical handoff tests paired with the legacy policy contract."""
    cfg = load_env_cfg("PointLeg-SteppingStones-v0")
    from pointleg.mdp import stepping_stones as legacy
    from mjlab.managers import RewardTermCfg

    class LegacyCompetition(SteppingStonesTerrainCfg):
        def count_and_gap(self, difficulty):
            count = round(difficulty * 6) + 2
            return count, .05 + .05 * min(count - 2, 4) / 4

    generator = cfg.scene.terrain.terrain_generator
    cfg.scene.terrain.terrain_generator = replace(generator, num_rows=7, num_cols=2, layout_shape=(4, 4), sub_terrains={
        "stepping_stones": SteppingStonesTerrainCfg(proportion=.5),
        "competition_stones": LegacyCompetition(proportion=.5, competition=True),
    })
    for group in cfg.observations.values():
        group.terms.pop("fixed_gait")
    cfg.rewards = {key: value for key, value in cfg.rewards.items()
                   if not key.startswith("stepping_gait_") and key not in ("stepping_early_failure", "stepping_backward_velocity", "stepping_body_stall")}
    cfg.terminations["stepping_stalled"].params["timeout_s"] = 4.0
    for name, func, weight, params in (
        ("foot_forward_step", "foot_forward_step_reward", .75, {}),
        ("right_rear_forward_step", "right_rear_forward_step_reward", 2., {}),
        ("right_rear_overdue", "right_rear_overdue_penalty", -2., {}),
        ("no_forward_step", "stepping_no_forward_step_penalty", -8., {}),
        ("left_front_support", "stepping_left_front_support_penalty", -12., {}),
        ("left_front_liftoff", "stepping_left_front_liftoff_penalty", -10., {}),
        ("left_front_overreach", "stepping_left_front_overreach_penalty", -4., {}),
        ("premature_support_handoff", "stepping_premature_support_handoff_penalty", -3., {}),
        ("startup_progress", "startup_progress_reward", 2., {}),
        ("startup_early_step", "startup_early_step_penalty", -1., {}),
        ("forward_velocity", "forward_velocity_reward", 2.5, {}),
        ("body_level", "stepping_body_level_reward", 1.5, {"std_degrees": 12.}),
        ("body_height", "stepping_body_height_reward", 1., {"target": .362, "tolerance": .03, "scale": .08}),
        ("route_alignment", "route_alignment_reward", .5, {"heading_scale_degrees": 12., "lateral_speed_scale": .08}),
    ):
        cfg.rewards[f"stepping_{name}"] = RewardTermCfg(func=getattr(legacy, func), weight=weight, params=params)
    for name, weight in (("cover", 20.), ("stone_center", 2.), ("failure", -12.), ("knee_min_angle", -8.), ("swing_height", -5.)):
        cfg.rewards[f"stepping_{name}"].weight = weight
    return cfg


def fake_env(count=2):
    cfg = legacy_cfg()
    scene = Scene(robot=Robot(count, cfg.scene.entities["robot"].init_state.pos[2]))
    scene.env_origins = torch.zeros(count, 3)
    scene.terrain = SimpleNamespace(
        cfg=cfg.scene.terrain,
        terrain_levels=torch.zeros(count, dtype=torch.long),
        terrain_types=torch.zeros(count, dtype=torch.long),
        env_origins=scene.env_origins,
        get_terrain_origins=lambda levels, types: torch.zeros(len(levels), 3),
        get_geometry_levels=lambda levels, types: levels,
    )
    scene["stepping_foot_contact"] = SimpleNamespace(
        primary_names=[f"robot/{name}" for name in FOOT_BODY_NAMES],
        data=SimpleNamespace(
            found=torch.zeros(count, 4),
            pos=torch.zeros(count, 4, 3),
            normal=torch.tensor([0.0, 0.0, -1.0]).repeat(count, 4, 1),
            force=torch.tensor([10.0, 0.0, 0.0]).repeat(count, 4, 1),
        ),
    )
    scene["stepping_shin_contact"] = SimpleNamespace(
        primary_names=[f"robot/{leg}_shin_collision" for leg in LEG_NAMES],
        data=SimpleNamespace(
            found=torch.zeros(count, 16), force=torch.zeros(count, 16, 3), pos=torch.zeros(count, 16, 3),
        ),
    )
    return SimpleNamespace(
        cfg=cfg,
        scene=scene,
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        physics_dt=0.005,
        common_step_counter=0,
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        max_episode_length=3000,
        extras={"log": {}},
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.25, 0.0, 0.0]]).repeat(count, 1)
        ),
    )


def set_contacts(env, points):
    sensor = env.scene["stepping_foot_contact"].data
    sensor.found[:] = 1.0
    sensor.pos[:] = torch.as_tensor(points, dtype=torch.float32)[None]


def mounted_env(count=1):
    env = fake_env(count)
    reset_stepping_stones(env, None)
    set_contacts(env, [[.4, 0.0, .2]] * 4)
    env.scene["robot"].data.site_pos_w[:] = env.scene["stepping_foot_contact"].data.pos
    collect_stepping_stones_substep(env)
    return env


def finish_first_right_rear_step(env):
    feet = env.scene["stepping_foot_contact"].data
    for _ in range(4):
        collect_stepping_stones_substep(env)
    feet.found[:, 3] = 0
    for _ in range(4):
        collect_stepping_stones_substep(env)
    feet.found[:, 3] = 1
    feet.pos[:, 3] = env._stepping_stones["stone_centers"][:, 1]
    for _ in range(16):
        collect_stepping_stones_substep(env)


def set_shin_load(env, leg, force, position=(0.40, 0.0, 0.20)):
    shin = env.scene["stepping_shin_contact"].data
    slot = leg * 4
    shin.found[:, slot] = force > 0
    shin.force[:, slot, 2] = -force
    shin.pos[:, slot] = torch.as_tensor(position)


class SteppingStoneGeometryTests(unittest.TestCase):
    def test_play_reset_randomizes_terrain_and_updates_origins(self):
        from pointleg.config.stepping_stones import make_stepping_stones_env_cfg

        cfg = make_stepping_stones_env_cfg(play=True)
        self.assertTrue(cfg.events["reset_stepping_stones"].params["randomize_levels"])
        self.assertEqual(make_stepping_stones_env_cfg(play=False).events["reset_stepping_stones"].params, {})
        env = fake_env(64)
        env.scene.terrain.get_terrain_origins = lambda levels, types: torch.stack(
            (levels.float(), types.float(), torch.zeros_like(levels).float()), dim=1
        )
        torch.manual_seed(123)
        reset_stepping_stones(env, None, randomize_levels=True)
        levels = env.scene.terrain.terrain_levels
        types = env.scene.terrain.terrain_types
        self.assertEqual(set(levels.tolist()), set(range(7)))
        self.assertEqual(set(types.tolist()), {0, 1})
        torch.testing.assert_close(env.scene.env_origins[:, 0], levels.float())
        torch.testing.assert_close(env.scene.env_origins[:, 1], types.float())
        torch.testing.assert_close(env.scene["robot"].written[:, 0], levels.float() + SPAWN_X)
        torch.testing.assert_close(env._stepping_stones["active_count"], levels + 2)
        index = ((levels == 6) & (types == 0)).nonzero()[0, 0]
        np.testing.assert_allclose(env._stepping_stones["stone_centers"][index].numpy(), stone_centers(8, .1), atol=1.e-6)

    def test_curriculum_geometry_and_runtime_targets_match_all_levels(self):
        env = fake_env(7)
        env.scene.terrain.terrain_levels[:] = torch.arange(7)
        reset_stepping_stones(env, None)
        for level in range(7):
            count = level + 2
            gap = .05 + .05 * level / 6
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name="terrain")
            output = SteppingStonesTerrainCfg(size=(8., 8.)).function(level / 6, spec, np.random.default_rng(7))
            centers = env._stepping_stones["stone_centers"][level, :count].numpy()
            np.testing.assert_allclose(centers, stone_centers(count, gap), atol=1.e-6)
            for geom, center in zip(output.geometries[2:2 + count], centers, strict=True):
                np.testing.assert_allclose(geom.geom.pos[:2] - output.origin[:2], center[:2], atol=1.e-6)
            self.assertAlmostEqual(output.geometries[-1].geom.pos[0] - output.origin[0] - .4, right_platform_start(count, gap))
            np.testing.assert_allclose(np.diff(centers[:, 0]) - STONE_SIZE, gap, atol=1.e-6)

    def test_dimensions_and_progression_match_the_drawing(self):
        self.assertEqual(MAX_STONES, 8)
        self.assertAlmostEqual(STONE_SIZE, 0.15)
        self.assertAlmostEqual(PLATFORM_SIZE, 0.80)
        self.assertAlmostEqual(PLATFORM_HEIGHT, 0.20)
        np.testing.assert_allclose(stone_centers(2)[:, :2], [[.925, .2], [1.125, -.2]])
        self.assertAlmostEqual(right_platform_start(2), 1.25)
        self.assertAlmostEqual(right_platform_start(8, .10), 2.90)

    def test_active_blocks_compile_as_one_hundred_fifty_mm_collision_boxes(self):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        terrain = SteppingStonesTerrainCfg(size=(8.0, 8.0))
        output = terrain.function(1.0, spec, np.random.default_rng(7))
        model = spec.compile()
        self.assertEqual(len(output.geometries), 11)
        stones = output.geometries[2:10]
        self.assertEqual(len(stones), 8)
        for terrain_geom, center in zip(stones, stone_centers(8, .10), strict=True):
            geom = terrain_geom.geom
            np.testing.assert_allclose(geom.size, [.075, .075, .1])
            np.testing.assert_allclose(geom.pos[:2], output.origin[:2] + center[:2])
            self.assertAlmostEqual(geom.pos[2] + geom.size[2], PLATFORM_HEIGHT)
        self.assertEqual(model.ngeom, 11)

    def test_competition_six_training_preset_fixes_geometry_and_keeps_native_overrides(self):
        with (
            patch("sys.argv", [
                "pointleg", "train", "PointLeg-SteppingStones-v0", "--competition-six",
                "--env.scene.num-envs", "512", "--agent.max-iterations", "1501",
            ]),
            patch("mjlab.scripts.train.launch_training") as launch,
        ):
            pointleg_main()
        task, train = launch.call_args.args
        self.assertEqual(task, "PointLeg-SteppingStones-v0")
        self.assertEqual(train.env.scene.num_envs, 512)
        self.assertEqual(train.agent.max_iterations, 1501)
        self.assertEqual(train.env.curriculum, {})
        terrain_cfg = train.env.scene.terrain
        self.assertEqual(terrain_cfg.max_init_terrain_level, 0)
        generator = TerrainGenerator(terrain_cfg.terrain_generator)
        spec = mujoco.MjSpec()
        generator.compile(spec)
        spec.compile()
        self.assertEqual(generator.terrain_origins.shape, (1, 1, 3))
        self.assertEqual(set(generator.cfg.sub_terrains), {"competition_stones"})

        env = fake_env(4)
        env.scene.terrain.cfg = terrain_cfg
        reset_stepping_stones(env, None)
        state = env._stepping_stones
        torch.testing.assert_close(state["active_count"], torch.full((4,), 6))
        torch.testing.assert_close(state["stone_half_size"], torch.full((4,), .10))
        torch.testing.assert_close(state["right_platform_start"], torch.full((4,), 2.7))
        torch.testing.assert_close(torch.diff(state["stone_centers"][:, :6, 0]), torch.full((4, 5), .30))
        mixed = load_env_cfg(task).scene.terrain.terrain_generator
        self.assertEqual((mixed.num_rows, len(mixed.sub_terrains)), (4, 1))

    def test_competition_six_gap_curriculum_changes_spacing_with_six_active_stones(self):
        with (
            patch("sys.argv", [
                "pointleg", "train", "PointLeg-SteppingStones-v0", "--competition-six-curriculum",
                "--env.scene.num-envs", "512",
            ]),
            patch("mjlab.scripts.train.launch_training") as launch,
        ):
            pointleg_main()
        _, train = launch.call_args.args
        term = train.env.curriculum["stepping_stone_levels"]
        self.assertEqual(term.params["window_size"], 256)
        self.assertTrue(term.params["promote_at_current_cap"])
        env = fake_env(4)
        terrain = env.scene.terrain
        terrain.cfg = train.env.scene.terrain
        terrain.terrain_levels[:] = torch.arange(4)
        generator = TerrainGenerator(terrain.cfg.terrain_generator)
        spec = mujoco.MjSpec()
        generator.compile(spec)
        model = spec.compile()
        terrain.env_origins[:] = torch.as_tensor(generator.terrain_origins[:, 0])
        reset_stepping_stones(env, None)
        state = env._stepping_stones
        expected_gaps = torch.full((4,), .10)
        torch.testing.assert_close(state["active_count"], torch.arange(3, 7))
        torch.testing.assert_close(stepping_metric(env, "stone_gap_m"), expected_gaps)
        for i, origin in enumerate(terrain.env_origins):
            center = state["stone_centers"][i, state["active_count"][i] - 1] + origin
            center[2] -= .10
            matches = np.isclose(model.geom_pos, center.numpy(), atol=2.e-6).all(axis=1)
            np.testing.assert_allclose(model.geom_size[matches], [[.10, .10, .10]])
        torch.testing.assert_close(state["right_platform_start"][-1], torch.tensor(2.7))

    def test_competition_six_gap_promotes_only_after_successes_at_current_cap(self):
        env = fake_env(4)
        configure_competition_six(env.cfg, gap_curriculum=True)
        reset_stepping_stones(env, None)
        term = env.cfg.curriculum["stepping_stone_levels"]
        term.params.update(initial_level_cap=1, window_size=4)
        curriculum = SteppingStoneCurriculum(term, env)
        ids = torch.arange(4)
        env._stepping_stones["success"][:] = True
        env.episode_length_buf[:] = 1
        curriculum(env, ids)
        self.assertEqual(curriculum.level_cap, 1)
        self.assertEqual(len(curriculum.window), 0)
        torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.ones(4, dtype=torch.long))
        curriculum(env, ids[:3])
        self.assertEqual(curriculum.level_cap, 1)
        curriculum(env, ids[3:])
        self.assertEqual(curriculum.level_cap, 2)

    def test_competition_six_stones_match_sim2sim_mesh_and_runtime_geometry(self):
        env = fake_env(14)
        terrain = env.scene.terrain
        cfg = terrain.cfg.terrain_generator
        generator = TerrainGenerator(cfg)
        spec = mujoco.MjSpec()
        generator.compile(spec)
        model = spec.compile()
        np.testing.assert_array_equal(generator.terrain_level_indices, np.repeat(np.arange(7)[:, None], 2, axis=1))
        terrain.terrain_levels[:] = torch.arange(7).repeat_interleave(2)
        terrain.terrain_types[:] = torch.arange(2).repeat(7)
        terrain.get_geometry_levels = lambda levels, types: torch.as_tensor(generator.terrain_level_indices)[levels, types]
        terrain.env_origins[:] = torch.tensor(
            generator.terrain_origins[terrain.terrain_levels.numpy(), terrain.terrain_types.numpy()]
        )
        reset_stepping_stones(env, None)
        state = env._stepping_stones
        for origin, count, centers, half_size, exit_start in zip(
            terrain.env_origins, state["active_count"], state["stone_centers"],
            state["stone_half_size"], state["right_platform_start"], strict=True,
        ):
            for center in centers[:count]:
                box_center = (origin + center - torch.tensor([0., 0., .1])).numpy()
                match = np.isclose(model.geom_pos, box_center, atol=2.e-6).all(axis=1)
                np.testing.assert_allclose(model.geom_size[match], [[half_size, half_size, .1]], atol=1.e-6)
            exit_center = (origin + torch.tensor([exit_start + .4, 0., .1])).numpy()
            match = np.isclose(model.geom_pos, exit_center, atol=2.e-6).all(axis=1)
            np.testing.assert_allclose(model.geom_size[match], [[.4, .4, .1]], atol=1.e-6)

        root = Path(__file__).resolve().parents[3]
        course = ElementTree.parse(root / "sim2sim/terrain/scene_terrain.xml").find(".//body[@name='stepping_stones']")
        centers = sorted(
            tuple(map(float, geom.attrib["pos"].split())) for geom in course.findall("geom")
            if geom.attrib["name"].startswith("stone_")
        )
        six_stone_env = 9
        np.testing.assert_allclose(
            state["stone_centers"][six_stone_env, :6].numpy(), np.asarray(centers) + [0, 0, .1], atol=1.e-6
        )
        self.assertAlmostEqual(state["right_platform_start"][six_stone_env].item(), 2.7, places=6)


class SteppingStoneLogicTests(unittest.TestCase):
    def test_left_front_overreach_and_landing_rewards_follow_the_next_same_side_stone(self):
        env = mounted_env(3)
        reset_stepping_stones(env, None, terrain_level=4)
        state = env._stepping_stones
        feet = env.scene["stepping_foot_contact"].data
        sites = env.scene["robot"].data.site_pos_w
        env.scene["robot"].data.joint_pos[:, state["knee_ids"][0]] = -.6
        for _ in range(4):
            collect_stepping_stones_substep(env)
        target_front = state["stone_centers"][:, 0, 0] + state["stone_half_size"]

        env.common_step_counter += 1
        feet.found[:, 0] = 0
        sites[:, 0, 0] = target_front + torch.tensor([.01, .10, .10])
        for _ in range(4):
            collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_left_front_overreach"]
        torch.testing.assert_close(term.func(env) * term.weight, torch.tensor([0., -4., -4.]))

        env.common_step_counter += 1
        feet.found[:, 0] = 1
        feet.pos[0, 0] = state["stone_centers"][0, 0]
        feet.pos[1:, 0] = state["stone_centers"][1:, 2]
        feet.pos[2, 1] = feet.pos[2, 0]
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertTrue(state["visited"][0, 0].item())
        self.assertEqual(state["visited"][:, 2].tolist(), [False, False, True])
        torch.testing.assert_close(state["stone_center_pulse"], torch.tensor([1., 0., 1.]))
        torch.testing.assert_close(state["foot_step_pulse"], torch.tensor([1., 0., 1.]))
        self.assertEqual(state["left_front_handoff"].last_surface[:, 0].tolist(), [1, 3, 3])
        self.assertEqual(state["left_front_handoff"].left_front_route_surface.tolist(), [1, 0, 0])
        feet.pos[1:, 0] = state["stone_centers"][1:, 0]
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertEqual(state["left_front_handoff"].left_front_route_surface.tolist(), [1, 1, 1])

    def test_left_front_penalty_charges_missed_descents_without_contact(self):
        env = mounted_env(7)
        state = env._stepping_stones
        feet = env.scene["stepping_foot_contact"].data
        sites = env.scene["robot"].data.site_pos_w
        env.scene["robot"].data.joint_pos[:, state["knee_ids"][0]] = -.6
        for _ in range(4):
            collect_stepping_stones_substep(env)
        sites[:, 0] = state["stone_centers"][:, 0]
        sites[:, 0, 0] += torch.tensor([0., -.15, 0., -.15, -.15, .20, -.15])
        sites[2, 0, 1] += .15
        sites[:, 0, 2] += torch.tensor([.01, .01, .01, .11, 0., .01, .01])
        feet.found[:6, 0] = 0
        collect_stepping_stones_substep(env)

        env.common_step_counter += 1
        sites[:, 0, 2] += torch.tensor([-.01, -.01, -.01, -.01, .01, -.01, -.01])
        collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_left_front_overreach"]
        torch.testing.assert_close(term.func(env) * term.weight, torch.tensor([0., -1., -1., 0., 0., -1., 0.]))
        self.assertFalse(state["visited"].any().item())
        reset_stepping_stones(env, torch.tensor([1]))
        self.assertTrue(torch.isnan(state["left_front_previous_height"][1]).item())

    def test_left_front_missed_descent_uses_exit_top_after_the_last_left_stone(self):
        env = mounted_env(4)
        state = env._stepping_stones
        feet = env.scene["stepping_foot_contact"].data
        sites = env.scene["robot"].data.site_pos_w
        env.scene["robot"].data.joint_pos[:, state["knee_ids"][0]] = -.6
        for _ in range(4):
            collect_stepping_stones_substep(env)
        feet.pos[:, 0] = state["stone_centers"][:, 0]
        for _ in range(16):
            collect_stepping_stones_substep(env)

        feet.pos[3, 0, 0] = state["right_platform_start"][3] + .10
        feet.pos[3, 0, 1] = 0.
        for _ in range(16):
            collect_stepping_stones_substep(env)

        feet.found[:, 0] = 0
        sites[:, 0, 0] = state["right_platform_start"] + torch.tensor([-.03, .10, .40, .40])
        sites[:, 0, 1] = 0.
        sites[:, 0, 2] = PLATFORM_HEIGHT + .01
        collect_stepping_stones_substep(env)

        env.common_step_counter += 1
        sites[:, 0, 2] = PLATFORM_HEIGHT
        collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_left_front_overreach"]
        torch.testing.assert_close(term.func(env) * term.weight, torch.tensor([-1., 0., -1., 0.]))

    def test_left_front_swing_penalizes_nearly_straight_knee_before_touchdown(self):
        env = mounted_env(3)
        state = env._stepping_stones
        for _ in range(4):
            collect_stepping_stones_substep(env)
        robot = env.scene["robot"].data
        robot.site_pos_w[:, 0] = state["stone_centers"][:, 0]
        robot.site_pos_w[:, 0, 2] += .09
        robot.joint_pos[:, state["knee_ids"][0]] = -torch.deg2rad(torch.tensor([5., 25., 5.]))
        env.scene["stepping_foot_contact"].data.found[:2, 0] = 0
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_left_front_overreach"]
        torch.testing.assert_close(term.func(env) * term.weight, torch.tensor([-1., 0., 0.]))

    def test_no_forward_step_penalty_charges_rocking_more_than_standing_after_grace(self):
        env = mounted_env(4)
        state = env._stepping_stones
        state["startup_closed"][:3] = True
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_w[[0, 3], 0] = .15
        robot.root_link_ang_vel_w[1, 2] = .75
        grace = math.ceil(NO_FORWARD_STEP_GRACE_S / env.physics_dt)
        for _ in range(grace):
            collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_no_forward_step"]
        torch.testing.assert_close(term.func(env), torch.zeros(4))

        env.common_step_counter += 1
        for _ in range(4):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env) * term.weight, torch.tensor([-8., -8., -1.6, 0.]))
        torch.testing.assert_close(stepping_metric(env, "no_forward_step_time_s"), torch.tensor([.02, .02, .02, 0.]))

    def test_no_forward_step_penalty_requires_forward_progress_to_clear_and_exempts_exit(self):
        env = mounted_env(2)
        state = env._stepping_stones
        state["startup_closed"][:] = True
        grace = math.ceil(NO_FORWARD_STEP_GRACE_S / env.physics_dt)
        state["no_forward_step_substeps"][:] = grace
        sensor = env.scene["stepping_foot_contact"].data
        env.scene["robot"].data.root_link_lin_vel_w[:, 0] = .15
        term = env.cfg.rewards["stepping_no_forward_step"]
        for _ in range(4):
            collect_stepping_stones_substep(env)

        env.common_step_counter += 1
        sensor.found[0, 1] = 0
        for _ in range(4):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.tensor([.2, 1.]))

        env.common_step_counter += 1
        sensor.found[0, 1] = 1
        for _ in range(4):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.ones(2))

        sensor.pos[0, 1] = state["stone_centers"][0, 1]
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertEqual(state["no_forward_step_substeps"][0].item(), 0)

        env.common_step_counter += 1
        state["stage"][1] = 3
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(2))
        reset_stepping_stones(env, torch.tensor([0]))
        self.assertEqual(stepping_metric(env, "no_forward_step_time_s")[0].item(), 0.)

    def test_foot_support_progress_requires_clean_load_and_resets_per_environment(self):
        env = mounted_env(2)
        feet = env.scene["stepping_foot_contact"].data
        feet.force[0, 3, 0] = 6.0
        set_shin_load(env, 1, 8.0)
        for _ in range(3):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(foot_contact_observation(env), torch.ones(2, 4))
        torch.testing.assert_close(foot_contact_observation(env, support_progress=True), torch.tensor([
            [.25, 0., .25, 0.],
            [.25, 0., .25, .25],
        ]))
        feet.force[0, 3, 0] = 8.0
        set_shin_load(env, 1, 0.0)
        for _ in range(20):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(foot_contact_observation(env, support_progress=True), torch.ones(2, 4))
        reset_stepping_stones(env, torch.tensor([0]))
        torch.testing.assert_close(foot_contact_observation(env, support_progress=True), torch.tensor([
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
        ]))

    def test_config_is_forward_only_and_has_all_task_adjudicators(self):
        cfg = load_env_cfg("PointLeg-SteppingStones-v0")
        self.assertEqual(cfg.episode_length_s, 60.0)
        self.assertEqual(cfg.scene.terrain.terrain_generator.num_rows, 4)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (0.3, 0.4))
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_y, (0.0, 0.0))
        self.assertEqual(cfg.commands["twist"].ranges.ang_vel_z, (0.0, 0.0))
        contact = next(sensor for sensor in cfg.scene.sensors if sensor.name == "stepping_foot_contact")
        self.assertEqual(contact.fields, ("found", "pos", "normal", "force"))
        self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 0)
        self.assertIn("stepping_failed", cfg.terminations)
        self.assertIn("stepping_succeeded", cfg.terminations)
        self.assertNotIn("stepping_setup_timeout", cfg.terminations)
        self.assertIn("stepping_timeout", cfg.terminations)
        self.assertIn("stepping_contact_latch", cfg.metrics)
        self.assertEqual(
            {key: reward.weight for key, reward in cfg.rewards.items()},
            {
                "stepping_cover": 12.0,
                "stepping_stone_center": 0.5,
                "stepping_stone_edge": -3.5,
                "stepping_low_support": -12.0,
                "stepping_startup_body_rise": -6.0,
                "stepping_success": 40.0,
                "stepping_exit_stability": 4.0,
                "stepping_failure": -25.0,
                "stepping_early_failure": 0.0,
                "stepping_forward_velocity": 5.0,
                "stepping_body_stall": -12.0,
                "stepping_backward_velocity": -3.0,
                "stepping_gait_progress": 2.0,
                "stepping_gait_complete": 4.0,
                "stepping_gait_wrong_lift": -2.0,
                "stepping_gait_support": -8.0,
                "stepping_gait_target": -3.0,
                "stepping_gait_platform_edge": -5.0,
                "stepping_gait_clearance": -1.5,
                "stepping_gait_delay": -2.0,
                "stepping_gait_backward": -4.0,
                "stepping_lateral_velocity": -3.0,
                "stepping_hip_abduction_symmetry": 0.5,
                "stepping_hip_abduction_target": -1.0,
                "stepping_knee_min_angle": -2.0,
                "stepping_leg_clearance": -4.0,
                "stepping_body_tilt": -2.0,
                "stepping_flight": -2.0,
                "stepping_shin_stone": -8.0,
                "stepping_shin_support": -1.0,
                "stepping_swing_height": -2.0,
                "stepping_foot_slip": -0.5,
                "joint_pos_limits": -5.0,
                "action_rate": -0.01,
            },
        )
        self.assertIn("stepping_forward_speed_mps", cfg.metrics)
        self.assertEqual(cfg.metrics["stepping_peak_foot_clearance_m"].reduce, "last")
        runner = load_rl_cfg("PointLeg-SteppingStones-v0")
        self.assertEqual(runner.algorithm.gamma, 0.9995)
        self.assertEqual(runner.algorithm.learning_rate, 3.0e-4)

    def test_reset_samples_the_configured_forward_speed_range(self):
        env = fake_env(128)
        reset_stepping_stones(env, None)
        speed = env._stepping_stones["episode_speed"]
        self.assertTrue(torch.all(speed >= 0.3))
        self.assertTrue(torch.all(speed <= 0.4))

    def test_spawn_on_left_platform_and_fail_on_ground_from_first_substep(self):
        env = fake_env(1)
        env.scene.env_origins[:] = torch.tensor([[2.0, 3.0, 4.0]])
        reset_stepping_stones(env, None)
        root = env.scene["robot"].written
        torch.testing.assert_close(root[:, :3], torch.tensor([[2.0 + SPAWN_X, 3.0, 4.0 + .362 + PLATFORM_HEIGHT]]))
        self.assertEqual(env._stepping_stones["stage"].item(), 2)
        set_contacts(env, [[2.4, 3.0, 4.0]] * 4)
        collect_stepping_stones_substep(env)
        self.assertTrue(stepping_failed(env).item())

    def test_spawn_has_rear_support_margin_and_feet_aligned_to_stone_rows(self):
        task = load_env_cfg("PointLeg-SteppingStones-v0")
        cfg = task.scene.entities["robot"]
        model = cfg.spec_fn().compile()
        data = mujoco.MjData(model)
        data.qpos[:3] = (SPAWN_X, 0.0, cfg.init_state.pos[2] + PLATFORM_HEIGHT)
        for name in LEG_JOINT_NAMES:
            value = next(value for pattern, value in cfg.init_state.joint_pos.items() if re.fullmatch(pattern, name))
            data.qpos[model.joint(name).qposadr.item()] = value
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        foot_geoms = [f"{leg}_foot_Link_collision" for leg in LEG_NAMES]
        feet = np.stack([data.geom(name).xpos for name in foot_geoms])
        radii = np.array([model.geom(name).size[0] for name in foot_geoms])
        center_of_mass_x = data.subtree_com[0, 0]
        self.assertGreater(center_of_mass_x - feet[2:, 0].max(), .23)
        self.assertGreater(feet[:2, 0].min() - center_of_mass_x, .25)
        np.testing.assert_allclose(feet[:, 1], [.20, -.20, .20, -.20], atol=.01)
        self.assertTrue(np.all((feet[:, 0] > radii) & (feet[:, 0] < PLATFORM_SIZE - radii)))
        front_hip_x = np.array([data.body(f"{leg}_hip_pitch_Link").xpos[0] for leg in LEG_NAMES[:2]])
        np.testing.assert_allclose(feet[:2, 0], front_hip_x, atol=.001)
        np.testing.assert_allclose(stone_centers(2)[0][0] - feet[:2, 0], .203, atol=.005)
        gap = feet[:, 2] - radii - PLATFORM_HEIGHT
        self.assertTrue(np.all((0.0 <= gap) & (gap < .001)))
        hips = [f"{leg}_hip_abduction_Link" for leg in LEG_NAMES]
        hip_z = np.array([data.body(name).xpos[2] for name in hips])
        self.assertLess(abs(hip_z[2:].mean() - hip_z[:2].mean()), .003)

    def test_velocity_starts_on_reset_and_stops_at_the_finish(self):
        env = fake_env(1)
        reset_stepping_stones(env, None)
        command = object.__new__(SteppingStonesVelocityCommand)
        command._env = env
        command.robot = env.scene["robot"]
        command.vel_command_b = torch.zeros(1, 3)
        command.vel_command_w = torch.zeros(1, 3)
        command.progress_direction_w = torch.zeros(1, 3)
        for name in ("is_standing_env", "is_forward_env", "is_heading_env", "is_world_env", "is_lateral_env", "is_yaw_env", "is_backward_env"):
            setattr(command, name, torch.zeros(1, dtype=torch.bool))
        command._apply_task_command()
        state = env._stepping_stones
        speed = state["episode_speed"].item()
        self.assertEqual(stage_observation(env).tolist(), [[0.0, 0.0, 1.0, 0.0]])
        self.assertTrue(command.is_forward_env.item())
        self.assertFalse(command.is_standing_env.item())
        torch.testing.assert_close(command.vel_command_b, torch.tensor([[speed, 0.0, 0.0]]))
        self.assertEqual(settle_progress_observation(env).item(), 0.0)
        state["visited"][:, :2] = True
        set_contacts(env, [[1.8, 0.0, .2]] * 4)
        for _ in range(4):
            collect_stepping_stones_substep(env)
        command._apply_task_command()
        self.assertEqual(command.vel_command_b[0, 0].item(), 0.0)
        self.assertEqual(stage_observation(env).tolist(), [[0.0, 0.0, 0.0, 1.0]])
        self.assertGreater(settle_progress_observation(env).item(), 0.0)
        self.assertFalse(state["success"].item())

    def test_policy_actions_apply_at_spawn_without_waiting_for_foot_contacts(self):
        env = fake_env(1)
        reset_stepping_stones(env, None)
        robot = env.scene["robot"]
        robot.data.default_joint_pos[:] = 0.3
        action_cfg = env.cfg.actions["leg_joint_pos"]
        action_cfg.min_delay = action_cfg.max_delay = 0
        action = action_cfg.build(env)
        action.process_actions(torch.ones(1, action.action_dim))
        action.apply_actions()
        self.assertTrue(torch.all(robot.position_target > robot.data.default_joint_pos).item())

    def test_every_leg_can_cover_first_stone_after_twenty_ms(self):
        env = mounted_env(4)
        sensor = env.scene["stepping_foot_contact"].data
        feet = torch.arange(4)
        sensor.pos[feet, feet] = torch.tensor(stone_centers(2)[0], dtype=torch.float32)
        env.common_step_counter += 1
        for _ in range(3):
            collect_stepping_stones_substep(env)
        self.assertFalse(env._stepping_stones["visited"].any().item())
        collect_stepping_stones_substep(env)
        self.assertTrue(env._stepping_stones["visited"][:, 0].all().item())
        term = env.cfg.rewards["stepping_cover"]
        torch.testing.assert_close(cover_reward(env) * term.weight * env.step_dt, torch.full((4,), 10.0))

    def test_contact_must_be_continuous_for_the_same_foot_and_stone(self):
        env = mounted_env()
        sensor = env.scene["stepping_foot_contact"].data
        first, second = torch.tensor(stone_centers(2), dtype=torch.float32)
        sensor.pos[:, 0] = first
        for _ in range(3):
            collect_stepping_stones_substep(env)
        sensor.pos[:, 0] = second
        sensor.pos[:, 1] = first
        collect_stepping_stones_substep(env)
        self.assertFalse(env._stepping_stones["visited"].any().item())
        sensor.pos[:, 1] = second
        for _ in range(3):
            collect_stepping_stones_substep(env)
        self.assertEqual(env._stepping_stones["visited"][0, :2].tolist(), [False, True])

    def test_side_contact_and_inactive_stone_do_not_earn_coverage(self):
        env = mounted_env(2)
        sensor = env.scene["stepping_foot_contact"].data
        sensor.pos[0, 0] = torch.tensor(stone_centers(2)[0], dtype=torch.float32)
        sensor.pos[0, 0, 1] += STONE_SIZE / 2
        sensor.normal[0, 0] = torch.tensor([0.0, -1.0, 0.0])
        sensor.pos[1, 0] = torch.tensor(stone_centers(3)[2], dtype=torch.float32)
        for _ in range(8):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(cover_reward(env), torch.zeros(2))
        self.assertFalse(env._stepping_stones["visited"].any().item())

    def test_two_stones_can_score_out_of_order_and_each_scores_once(self):
        env = mounted_env()
        sensor = env.scene["stepping_foot_contact"].data
        first, second = torch.tensor(stone_centers(2), dtype=torch.float32)
        term = env.cfg.rewards["stepping_cover"]
        for point, expected in ((second, 10.0), (first, 10.0), (second, 0.0)):
            env.common_step_counter += 1
            sensor.pos[:, 2] = point
            for _ in range(4):
                collect_stepping_stones_substep(env)
            self.assertAlmostEqual(cover_reward(env).item() * term.weight * env.step_dt, expected)
        self.assertEqual(stepping_metric(env, "covered_fraction").item(), 1.0)

    def test_ground_later_in_the_same_control_step_cancels_coverage_points(self):
        env = mounted_env()
        sensor = env.scene["stepping_foot_contact"].data
        sensor.pos[:, 0] = torch.tensor(stone_centers(2)[0], dtype=torch.float32)
        for _ in range(2):
            collect_stepping_stones_substep(env)
        env.common_step_counter += 1
        for _ in range(2):
            collect_stepping_stones_substep(env)
        self.assertGreater(cover_reward(env).item(), 0.0)
        sensor.pos[:, 1, 2] = 0.0
        collect_stepping_stones_substep(env)
        sensor.found[:, 1] = 0.0
        collect_stepping_stones_substep(env)
        self.assertTrue(stepping_failed(env).item())
        self.assertEqual(cover_reward(env).item(), 0.0)
        cfg = env.cfg.rewards["stepping_failure"]
        self.assertAlmostEqual(stepping_failure_reward(env).item() * cfg.weight * env.step_dt, -12.0)

    def test_external_failure_also_suppresses_coverage_points(self):
        env = mounted_env()
        env.scene["stepping_foot_contact"].data.pos[:, 3] = torch.tensor(stone_centers(2)[0], dtype=torch.float32)
        for _ in range(4):
            collect_stepping_stones_substep(env)
        self.assertGreater(cover_reward(env).item(), 0.0)
        env.reset_terminated = torch.tensor([True])
        self.assertEqual(cover_reward(env).item(), 0.0)
        self.assertEqual(stepping_metric(env, "failure").item(), 1.0)

    def test_tilt_still_fails_in_stone_array(self):
        env = mounted_env()
        env.scene["robot"].data.projected_gravity_b[:, 2] = -0.4
        self.assertTrue(stepping_bad_orientation(env).item())

    def test_ground_contact_still_fails_during_exit_transition(self):
        env = mounted_env()
        state = env._stepping_stones
        state["visited"][:, :2] = True
        env.scene["robot"].data.root_link_pos_w[:, 0] = 1.1
        env.common_step_counter += 1
        set_contacts(env, [[.4, 0.0, .2]] * 3 + [[1.4, 0.0, 0.0]])
        collect_stepping_stones_substep(env)
        self.assertTrue(state["platform_transition"].item())
        self.assertTrue(stepping_failed(env).item())

    def test_finish_requires_all_stones_and_each_foot_visiting_then_standing_on_exit(self):
        env = mounted_env()
        state = env._stepping_stones
        sensor = env.scene["stepping_foot_contact"].data
        exit_x = right_platform_start(2) + PLATFORM_SIZE / 2
        sensor.found.zero_()
        for foot in range(4):
            sensor.found.zero_()
            sensor.found[:, foot] = 1
            sensor.pos[:, foot] = torch.tensor([exit_x, 0., .2])
            for _ in range(3):
                collect_stepping_stones_substep(env)
            self.assertFalse(state["exit_visited"][0, foot].item())
            collect_stepping_stones_substep(env)
            self.assertTrue(state["exit_visited"][0, foot].item())
        self.assertFalse(state["success"].item())
        self.assertEqual(settle_progress_observation(env).item(), 1.0)
        state["visited"][:, :2] = True
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        self.assertEqual(stage_observation(env).tolist(), [[0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(settle_progress_observation(env).item(), 0.0)
        self.assertFalse(state["success"].item())
        set_contacts(env, [[exit_x, 0.0, .2]] * 4)
        required = math.ceil(EXIT_SETTLE_TIME_S / env.physics_dt)
        for _ in range(required - 1):
            collect_stepping_stones_substep(env)
        self.assertFalse(state["success"].item())
        sensor.force[0, 0, 0] = 7.99
        collect_stepping_stones_substep(env)
        self.assertEqual(settle_progress_observation(env).item(), 0.0)
        sensor.force[0, 0, 0] = 8.0
        for _ in range(required):
            collect_stepping_stones_substep(env)
        self.assertTrue(state["success"].item())
        self.assertEqual(settle_progress_observation(env).item(), 1.0)
        cfg = env.cfg.rewards["stepping_success"]
        self.assertAlmostEqual(success_reward(env).item() * cfg.weight * env.step_dt, 40.0)
        reset_stepping_stones(env, None)
        self.assertFalse(state["exit_visited"].any().item())

    def test_exit_requires_quiet_level_nominal_posture_and_clean_support(self):
        env = mounted_env(6)
        state = env._stepping_stones
        state["visited"][:, :2] = True
        set_contacts(env, [[1.65, 0.0, .2]] * 4)
        robot = env.scene["robot"].data
        robot.root_link_pos_w[1, 2] -= .06
        robot.projected_gravity_b[2] = torch.tensor([.2, 0., -.98])
        robot.root_link_lin_vel_w[3, 0] = .06
        robot.root_link_ang_vel_w[4, 2] = .16
        robot.joint_pos[5, 0] = .50
        for _ in range(4):
            collect_stepping_stones_substep(env)
        self.assertTrue((state["stage"] == 3).all().item())
        self.assertEqual(state["exit_stable_substeps"].tolist(), [1, 0, 0, 0, 0, 0])
        term = env.cfg.rewards["stepping_exit_stability"]
        score = term.func(env) * term.weight * env.step_dt
        self.assertAlmostEqual(score[0].item(), 4.0)
        self.assertTrue((score[1:] < score[0]).all().item())

        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(6))
        state["exit_settle_substeps"][:] = math.ceil(EXIT_SETTLE_TIMEOUT_S / env.physics_dt) - 1
        self.assertFalse(stepping_settle_timeout(env).any().item())
        collect_stepping_stones_substep(env)
        self.assertTrue(stepping_settle_timeout(env).all().item())

    def test_ground_beyond_the_exit_still_fails(self):
        env = mounted_env()
        state = env._stepping_stones
        state["visited"][:, :2] = True
        set_contacts(env, [[1.8, 0.0, .2]] * 4)
        collect_stepping_stones_substep(env)
        set_contacts(env, [[2.35, 0.0, 0.0]] * 4)
        collect_stepping_stones_substep(env)
        self.assertTrue(stepping_failed(env).item())
        self.assertFalse(state["success"].item())

    def test_exit_approach_exempts_tilt_until_all_feet_arrive(self):
        env = mounted_env()
        state = env._stepping_stones
        state["visited"][:, :2] = True
        env.scene["robot"].data.root_link_pos_w[:, 0] = 1.1
        env.scene["robot"].data.projected_gravity_b[:, 2] = -0.4
        env.scene["stepping_foot_contact"].data.found[:] = 0
        env.common_step_counter += 1
        for _ in range(25):
            collect_stepping_stones_substep(env)
        self.assertFalse(stepping_failed(env).item())
        self.assertFalse(stepping_bad_orientation(env).item())
        set_contacts(env, [[1.8, 0.0, .2]] * 4)
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        self.assertTrue(stepping_bad_orientation(env).item())

    def test_stall_terminates_after_four_seconds_without_new_forward_progress(self):
        env = mounted_env()
        cfg = env.cfg.terminations["stepping_stalled"]
        root = env.scene["robot"].data.root_link_pos_w
        for _ in range(199):
            root[:, 1] -= .01
            env.common_step_counter += 1
            self.assertFalse(stepping_stalled(env, **cfg.params).item())
        env.common_step_counter += 1
        self.assertTrue(stepping_stalled(env, **cfg.params).item())
        self.assertTrue(stepping_stalled(env, **cfg.params).item())

        root[:, 0] += .101
        env.common_step_counter += 1
        self.assertFalse(stepping_stalled(env, **cfg.params).item())
        self.assertEqual(env._stepping_stones["stall_age"].item(), 0.0)

        env._stepping_stones["stage"][:] = 3
        env.common_step_counter += 1
        self.assertFalse(stepping_stalled(env, **cfg.params).item())

    def test_metrics_report_forward_speed_and_hold_peak_height_until_reset(self):
        env = mounted_env(2)
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_w[:, :2] = torch.tensor([[.32, -.12], [-.06, .03]])
        robot.site_pos_w[:, 0, 2] = torch.tensor([.32, .27])
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(stepping_metric(env, "forward_speed_mps"), torch.tensor([.32, -.06]))
        robot.site_pos_w[:, :, 2] = .2
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(stepping_metric(env, "peak_foot_clearance_m"), torch.tensor([.12, .07]))
        reset_stepping_stones(env, torch.tensor([1]))
        torch.testing.assert_close(stepping_metric(env, "peak_foot_clearance_m"), torch.tensor([.12, 0.0]))

    def test_partial_reset_only_clears_selected_contact_history(self):
        env = mounted_env(2)
        sensor = env.scene["stepping_foot_contact"].data
        sensor.pos[:, 0] = torch.tensor(stone_centers(2)[0], dtype=torch.float32)
        for _ in range(3):
            collect_stepping_stones_substep(env)
        reset_stepping_stones(env, torch.tensor([0]))
        collect_stepping_stones_substep(env)
        self.assertEqual(env._stepping_stones["visited"][:, 0].tolist(), [False, True])

    def test_task_observations_keep_their_order_and_dimensions(self):
        from pointleg.config.env_cfgs import flat_env_cfg

        env = mounted_env()
        base_groups = flat_env_cfg().observations
        for group, expected_size in (("actor", 60), ("critic", 56)):
            base = base_groups[group].terms
            terms = env.cfg.observations[group].terms
            self.assertEqual(list(terms)[:len(base)], list(base))
            added = [term.func(env).shape[1] for key, term in terms.items() if key not in base]
            self.assertEqual(sum(added), expected_size)


class SteppingStoneRewardTests(unittest.TestCase):
    def test_shin_force_penalizes_stones_and_platforms_but_not_ground(self):
        env = mounted_env(5)
        sensor = env.scene["stepping_shin_contact"].data
        sensor.found[:, 0] = 1
        sensor.force[:, 0, 0] = torch.tensor([20., 20., 20., 20., 2.])
        sensor.pos[:, 0] = env._stepping_stones["stone_centers"][:, 0]
        sensor.pos[1, 0, 0] = .4
        sensor.pos[2, 0, 0] = right_platform_start(2) + .4
        sensor.pos[3, 0] = torch.tensor([.4, 0., 0.])
        env.common_step_counter += 1
        for _ in range(2):
            collect_stepping_stones_substep(env)
        sensor.found.zero_()
        for _ in range(2):
            collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_shin_stone"]
        torch.testing.assert_close(term.func(env), torch.tensor([.5, .5, .5, 0., 0.]))
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(5))

    def test_shin_support_confirms_continuous_load_and_rearms_after_release(self):
        env = mounted_env()
        term = env.cfg.rewards["stepping_shin_support"]

        def pulse():
            return (stepping_shin_support_penalty(env) * term.weight * env.step_dt).item()

        set_shin_load(env, 3, 8.)
        for _ in range(3):
            collect_stepping_stones_substep(env)
        self.assertEqual(pulse(), 0.0)
        set_shin_load(env, 3, 0.)
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        set_shin_load(env, 3, 8.)
        for _ in range(3):
            collect_stepping_stones_substep(env)
        self.assertEqual(pulse(), 0.0)
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        self.assertAlmostEqual(pulse(), -1.0)
        self.assertFalse(stepping_failed(env).item())

        env.common_step_counter += 1
        set_shin_load(env, 3, 0.)
        for _ in range(3):
            collect_stepping_stones_substep(env)
        set_shin_load(env, 3, 25.)
        collect_stepping_stones_substep(env)
        self.assertEqual(pulse(), 0.0)
        env.common_step_counter += 1
        set_shin_load(env, 3, 0.)
        for _ in range(4):
            collect_stepping_stones_substep(env)
        env.common_step_counter += 1
        set_shin_load(env, 3, 25.)
        collect_stepping_stones_substep(env)
        self.assertAlmostEqual(pulse(), -1.0)

    def test_only_loaded_clean_feet_score_coverage_and_center(self):
        env = mounted_env()
        first, second = stone_centers(2)
        set_contacts(env, [[.4, 0, .2], first.tolist(), [.4, 0, .2], second.tolist()])
        feet = env.scene["stepping_foot_contact"].data
        feet.force[0, 3, 0] = 8.
        set_shin_load(env, 3, 25., second)
        env.common_step_counter += 1
        for _ in range(4):
            collect_stepping_stones_substep(env)
        self.assertEqual(env._stepping_stones["visited"][0, :2].tolist(), [True, False])
        self.assertAlmostEqual((stone_center_reward(env) * env.step_dt).item(), 1.0)

        env.common_step_counter += 1
        set_shin_load(env, 3, 0.)
        feet.force[0, 3, 0] = 7.99
        for _ in range(4):
            collect_stepping_stones_substep(env)
        self.assertFalse(env._stepping_stones["visited"][0, 1].item())
        feet.force[0, 3, 0] = 8.
        for _ in range(3):
            collect_stepping_stones_substep(env)
        self.assertFalse(env._stepping_stones["visited"][0, 1].item())
        collect_stepping_stones_substep(env)
        self.assertTrue(env._stepping_stones["visited"][0, 1].item())
        self.assertAlmostEqual((stone_center_reward(env) * env.step_dt).item(), 1.0)

    def test_shin_support_cancels_only_positive_velocity_for_the_control_step(self):
        env = mounted_env(2)
        env.scene["robot"].data.root_link_lin_vel_w[:, 0] = torch.tensor([.25, -.25])
        set_shin_load(env, 3, 25.)
        collect_stepping_stones_substep(env)
        set_shin_load(env, 3, 0.)
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(forward_velocity_reward(env), torch.tensor([0., -3.]))
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(forward_velocity_reward(env), torch.tensor([1., -3.]))

    def test_stone_edge_penalty_applies_on_all_four_sides(self):
        env = mounted_env(4)
        sensor = env.scene["stepping_foot_contact"].data
        sensor.pos[:, 0] = env._stepping_stones["stone_centers"][:, 0].clone()
        sensor.pos[:, 0, :2] += torch.tensor([[.046, 0.], [-.046, 0.], [0., .046], [0., -.046]])
        for _ in range(4):
            collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_stone_edge"]
        torch.testing.assert_close(term.func(env) * term.weight * env.step_dt, torch.full((4,), -3.5))

    def test_stone_contact_and_edge_use_the_selected_surface_size(self):
        env = fake_env(3)
        env.scene.terrain.terrain_types[:] = torch.tensor([0, 1, 1])
        reset_stepping_stones(env, None)
        set_contacts(env, [[.4, 0.0, .2]] * 4)
        sensor = env.scene["stepping_foot_contact"].data
        sensor.pos[:, 1] = env._stepping_stones["stone_centers"][:, 1]
        sensor.pos[:, 1, 0] -= torch.tensor([.085, .085, .060])
        for _ in range(4):
            collect_stepping_stones_substep(env)
        self.assertEqual(env._stepping_stones["visited"][:, 1].tolist(), [False, True, True])
        term = env.cfg.rewards["stepping_stone_edge"]
        torch.testing.assert_close(term.func(env) * term.weight * env.step_dt, torch.tensor([0., -3.5, 0.]))

    def test_stone_center_and_edge_pay_on_confirmed_touchdown_without_repeat_bonus(self):
        env = mounted_env(3)
        sensor = env.scene["stepping_foot_contact"].data
        centers = env._stepping_stones["stone_centers"][:, 0].clone()
        sensor.pos[:, 0] = centers
        sensor.pos[:, 0, 0] += torch.tensor([0., .044, .046])
        for _ in range(3):
            collect_stepping_stones_substep(env)
        center_term = env.cfg.rewards["stepping_stone_center"]
        edge_term = env.cfg.rewards["stepping_stone_edge"]
        torch.testing.assert_close(center_term.func(env), torch.zeros(3))
        collect_stepping_stones_substep(env)
        scores = center_term.func(env) * center_term.weight * env.step_dt
        self.assertAlmostEqual(scores[0].item(), 2.0)
        self.assertGreater(scores[1].item(), 0.0)
        self.assertEqual(scores[2].item(), 0.0)
        torch.testing.assert_close(edge_term.func(env) * edge_term.weight * env.step_dt, torch.tensor([0., 0., -3.5]))
        env.common_step_counter += 1
        sensor.found[:, 0] = 0
        collect_stepping_stones_substep(env)
        sensor.found[:, 0] = 1
        for _ in range(4):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(center_term.func(env), torch.zeros(3))
        self.assertAlmostEqual((edge_term.func(env) * edge_term.weight * env.step_dt)[2].item(), -3.5)
        reset_stepping_stones(env, torch.tensor([0]))
        self.assertFalse(env._stepping_stones["stone_center_awarded"][0].any().item())

    def test_swing_height_penalty_uses_foot_bottom_and_airborne_mask(self):
        env = mounted_env(5)
        sensor = env.scene["stepping_foot_contact"].data
        sensor.found[:, 0] = 0
        robot = env.scene["robot"].data
        robot.site_pos_w[:, 0, 2] = PLATFORM_HEIGHT + torch.tensor([.07, .105, .14, .14, .14])
        sensor.found[3, 0] = 1
        env._stepping_stones["stage"][4] = 3
        term = env.cfg.rewards["stepping_swing_height"]
        torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0., .25, 1., 0., 0.]))

    def test_leg_clearance_detects_foot_and_opposite_shin_proximity(self):
        env = mounted_env(3)
        robot = env.scene["robot"].data
        feet = torch.tensor([[0., 0., 0.], [0., 2., 0.], [1., 0., 0.], [1., 2., 0.]])
        robot.site_pos_w[:] = feet
        robot.body_link_pos_w[:] = feet + torch.tensor([0., 0., .30])
        robot.site_pos_w[1, 2] = torch.tensor([.04, 0., 0.])
        robot.body_link_pos_w[1, 2] = torch.tensor([.04, 0., .30])
        robot.site_pos_w[2, 2] = torch.tensor([.04, 0., .15])
        robot.body_link_pos_w[2, 0] = torch.tensor([0., 0., .30])
        term = env.cfg.rewards["stepping_leg_clearance"]
        cost = term.func(env, **term.params)
        self.assertEqual(cost[0].item(), 0.0)
        self.assertGreater(cost[1].item(), 0.0)
        self.assertGreater(cost[2].item(), 0.0)

    def test_startup_progress_is_one_time_and_early_step_penalized(self):
        env = mounted_env(2)
        robot = env.scene["robot"].data
        env.common_step_counter += 1
        robot.root_link_pos_w[:, 0] += torch.tensor([.05, .02])
        collect_stepping_stones_substep(env)
        term = env.cfg.rewards["stepping_startup_progress"]
        torch.testing.assert_close(term.func(env) * term.weight * env.step_dt, torch.tensor([2., .8]), atol=1.e-5, rtol=0)
        env.common_step_counter += 1
        env.scene["stepping_foot_contact"].data.found[:, 0] = 0
        for _ in range(16):
            collect_stepping_stones_substep(env)
        penalty = env.cfg.rewards["stepping_startup_early_step"]
        torch.testing.assert_close(penalty.func(env) * penalty.weight * env.step_dt, torch.tensor([0., -1.]))
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(penalty.func(env), torch.zeros(2))
        reset_stepping_stones(env, None)
        self.assertFalse(env._stepping_stones["startup_closed"].any().item())

    def test_startup_body_rise_charges_height_and_upward_speed_without_forward_motion(self):
        env = mounted_env(5)
        robot = env.scene["robot"].data
        term = env.cfg.rewards["stepping_startup_body_rise"]
        env.common_step_counter += 1
        robot.root_link_pos_w[:, 2] += torch.tensor([.04, 0., .01, .07, -.03])
        robot.root_link_lin_vel_w[:, 2] = torch.tensor([0., .25, .09, -.30, -.30])
        collect_stepping_stones_substep(env)
        robot.root_link_pos_w[:, 2] = env._stepping_stones["startup_spawn_z"]
        robot.root_link_lin_vel_w.zero_()
        for _ in range(3):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.tensor([.0625, .0625, 0., .25, 0.]))
        self.assertEqual(env.cfg.metrics["stepping_startup_peak_body_rise_m"].reduce, "last")
        torch.testing.assert_close(
            stepping_metric(env, "startup_peak_body_rise_m"), torch.tensor([.04, 0., .01, .07, 0.])
        )

    def test_startup_body_rise_detects_nose_up_with_a_stationary_root(self):
        env = mounted_env(3)
        robot = env.scene["robot"].data
        env.common_step_counter += 1
        robot.root_link_quat_w[0] = torch.tensor([math.cos(.1), 0., -math.sin(.1), 0.])
        robot.root_link_ang_vel_w[1, 1] = -1.0
        robot.root_link_ang_vel_w[2, 1] = 1.0
        collect_stepping_stones_substep(env)

        term = env.cfg.rewards["stepping_startup_body_rise"]
        self.assertTrue((term.func(env)[:2] > 0).all())
        self.assertEqual(term.func(env)[2].item(), 0.0)
        torch.testing.assert_close(stepping_metric(env, "startup_peak_body_rise_m"), torch.zeros(3))
        self.assertGreater(stepping_metric(env, "startup_peak_front_rise_m")[0].item(), .06)

    def test_startup_body_rise_expires_and_restarts_from_each_reset_height(self):
        env = mounted_env(2)
        robot = env.scene["robot"].data
        term = env.cfg.rewards["stepping_startup_body_rise"]
        env.episode_length_buf[:] = 2500
        for _ in range(198):
            collect_stepping_stones_substep(env)
        env.common_step_counter += 1
        robot.root_link_pos_w[:, 2] += .05
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.full((2,), .140625))
        env.common_step_counter += 1
        robot.root_link_pos_w[:, 2] += .1
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(2))
        torch.testing.assert_close(stepping_metric(env, "startup_peak_body_rise_m"), torch.full((2,), .05))

        env.scene.env_origins[1, 2] = 2.
        reset_stepping_stones(env, torch.tensor([1]))
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(2))
        robot.root_link_pos_w[1, 2] += .05
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.tensor([0., .140625]))

    def test_tilt_penalty_applies_to_roll_and_pitch_even_when_stationary(self):
        env = mounted_env(3)
        angle = math.radians(23)
        env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([
            [0., 0., -1.], [math.sin(angle), 0., -math.cos(angle)],
            [0., math.sin(angle), -math.cos(angle)],
        ])
        term = env.cfg.rewards["stepping_body_tilt"]
        torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0., 1., 1.]))

    def test_knee_minimum_angle_applies_to_every_leg_and_rejects_reverse_bending(self):
        env = mounted_env(6)
        ids = env._stepping_stones["knee_ids"]
        env.scene["robot"].data.joint_pos[:, ids] = torch.tensor([
            [0., 0., 0., 0.],
            [0., -.3, -.3, -.3],
            [-.3, 0., -.3, -.3],
            [-.3, -.3, 0., -.3],
            [-.3, -.3, -.3, 0.],
            [-math.radians(5), -math.radians(5), -.3, -.3],
        ])
        term = env.cfg.rewards["stepping_knee_min_angle"]
        torch.testing.assert_close(term.func(env, **term.params), torch.tensor([1., .25, .25, .25, .25, 0.]))
        env.scene["robot"].data.joint_pos[5, ids] = torch.tensor([
            0., math.radians(5), math.radians(10), math.radians(20),
        ])
        torch.testing.assert_close(term.func(env, **term.params)[5], torch.tensor(1.))

    def test_body_height_reward_tracks_platform_height_only_when_advancing(self):
        env = mounted_env(6)
        robot = env.scene["robot"].data
        env.scene.env_origins[:, 2] = torch.tensor([0., 1., 0., 0., 0., 0.])
        robot.root_link_pos_w[:, 2] = env.scene.env_origins[:, 2] + PLATFORM_HEIGHT + torch.tensor(
            [.362, .392, .472, .252, .362, float("nan")]
        )
        robot.root_link_lin_vel_w[:, 0] = torch.tensor([.25, .25, .25, .25, 0., .25])
        term = env.cfg.rewards["stepping_body_height"]
        value = term.func(env, **term.params)
        torch.testing.assert_close(value[:4], torch.tensor([1., 1., math.exp(-1), math.exp(-1)]))
        torch.testing.assert_close(value[4:], torch.zeros(2))

    def test_hip_abduction_target_uses_mirrored_angles_and_deadband(self):
        env = mounted_env(4)
        ids = env._stepping_stones["hip_abduction_ids"]
        env.scene["robot"].data.joint_pos[:, ids] = torch.tensor([
            [0.15, -0.15, 0.15, -0.15],
            [0.12, -0.12, 0.12, -0.12],
            [0.0, 0.0, 0.0, 0.0],
            [-0.15, 0.15, -0.15, 0.15],
        ])
        term = env.cfg.rewards["stepping_hip_abduction_target"]
        cost = term.func(env, **term.params)
        torch.testing.assert_close(cost[:2], torch.zeros(2))
        self.assertGreater(cost[2].item(), 0.0)
        self.assertGreater(cost[3].item(), cost[2].item())

    def test_all_feet_can_step_first_and_only_left_front_needs_a_right_rear_step_to_repeat(self):
        env = mounted_env(4)
        sensor = env.scene["stepping_foot_contact"].data
        term = env.cfg.rewards["stepping_foot_forward_step"]
        state = env._stepping_stones
        leg = torch.arange(4)
        entry = sensor.pos[leg, leg].clone()
        first, second = state["stone_centers"][:, :2].unbind(dim=1)
        exit_top = entry.clone()
        exit_top[:, 0] = state["right_platform_start"] + .02
        for _ in range(3):
            collect_stepping_stones_substep(env)

        far_entry = entry.clone()
        far_entry[:, 0] += .20
        for destination, rewards in (
            (far_entry, [0.] * 4), (first, [.75] * 4), (second, [0., .75, .75, .75]),
            (second, [0.] * 4), (first, [0.] * 4), (second, [0.] * 4), (exit_top, [0., .75, .75, .75]),
        ):
            env.common_step_counter += 1
            sensor.found[leg, leg] = 0
            collect_stepping_stones_substep(env)
            sensor.pos[leg, leg] = destination
            sensor.found[leg, leg] = 1
            for _ in range(16):
                collect_stepping_stones_substep(env)
            torch.testing.assert_close(term.func(env) * term.weight * env.step_dt, torch.tensor(rewards))

    def test_right_rear_must_reach_a_new_surface_before_left_front_can_step_again(self):
        env = mounted_env(3)
        reset_stepping_stones(env, None, terrain_level=4)
        state = env._stepping_stones
        handoff = state["left_front_handoff"]
        feet = env.scene["stepping_foot_contact"].data
        stones = state["stone_centers"]

        def collect(count):
            env.common_step_counter += 1
            for _ in range(count):
                collect_stepping_stones_substep(env)

        collect(4)
        feet.pos[1, 3] = stones[1, 1]
        collect(16)
        rear = env.cfg.rewards["stepping_right_rear_forward_step"]
        torch.testing.assert_close(rear.func(env) * rear.weight * env.step_dt, torch.tensor([0., 2., 0.]))

        feet.pos[:, 0] = stones[:, 0]
        collect(16)
        self.assertEqual(handoff.right_rear_follow_pending.tolist(), [True, False, True])
        feet.pos[:, 1] = stones[:, 1]
        feet.pos[:, 2] = stones[:, 0]
        collect(16)
        torch.testing.assert_close(state["foot_step_pulse"], torch.full((3,), 2.))
        overdue = env.cfg.rewards["stepping_right_rear_overdue"]
        torch.testing.assert_close(overdue.func(env), torch.zeros(3))

        state["right_rear_follow_substeps"][0] = math.ceil(RIGHT_REAR_FOLLOW_GRACE_S / env.physics_dt)
        feet.found[0, 3] = 0
        collect(4)
        feet.found[0, 3] = 1
        collect(16)
        self.assertEqual(handoff.right_rear_follow_pending.tolist(), [True, False, True])
        torch.testing.assert_close(overdue.func(env) * overdue.weight, torch.tensor([-2., 0., 0.]))
        torch.testing.assert_close(rear.func(env), torch.zeros(3))

        feet.found[:, 0] = 0
        collect(1)
        feet.pos[2, 3] = stones[2, 1]
        collect(16)
        self.assertEqual(handoff.right_rear_follow_pending.tolist(), [True, False, False])
        torch.testing.assert_close(stepping_metric(env, "right_rear_forward_steps"), torch.tensor([0., 1., 1.]))
        feet.found[:, 0] = 1
        feet.pos[:, 0] = stones[:, 2]
        collect(16)
        torch.testing.assert_close(state["foot_step_pulse"], torch.tensor([0., 1., 0.]))
        self.assertEqual(state["visited"][:, 2].tolist(), [False, True, False])

        feet.pos[0, 3] = stones[0, 1]
        feet.pos[[0, 2], 0] = stones[[0, 2], 0]
        collect(16)
        feet.pos[[0, 2], 0] = stones[[0, 2], 2]
        collect(16)
        torch.testing.assert_close(state["foot_step_pulse"], torch.tensor([1., 0., 1.]))
        self.assertEqual(state["visited"][:, 2].tolist(), [True, True, True])

    def test_left_front_support_penalty_tracks_effective_contact(self):
        env = mounted_env(6)
        finish_first_right_rear_step(env)
        reset_stepping_stones(env, torch.tensor([5]))
        sensor = env.scene["stepping_foot_contact"].data
        term = env.cfg.rewards["stepping_left_front_support"]
        env.common_step_counter += 1
        sensor.found[:, 1] = 0
        sensor.found[[0, 1, 3, 4, 5], 3] = 0
        sensor.found[[1, 2, 5], 0] = 0
        sensor.force[3, 0, 0] = 7.99
        collect_stepping_stones_substep(env)
        self.assertFalse(stepping_failed(env).any().item())

        sensor.found[[1, 3], 3] = 1
        set_shin_load(env, 0, 8.)
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.tensor([.25, .5, 0., .5, .25, 0.]))

        sensor.found[:, 3] = 1
        for _ in range(16):
            collect_stepping_stones_substep(env)
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(6))

    def test_left_front_hold_requires_stable_right_rear_landing_and_charges_brief_release(self):
        env = mounted_env()
        finish_first_right_rear_step(env)
        sensor = env.scene["stepping_foot_contact"].data
        terms = env.cfg.rewards
        hold = terms["stepping_left_front_support"]
        liftoff = terms["stepping_left_front_liftoff"]
        low_support = terms["stepping_low_support"]
        sensor.found[:, 3] = 0
        collect_stepping_stones_substep(env)
        env.common_step_counter += 1
        sensor.found[:, :3] = 0
        sensor.found[:, 3] = 1
        sensor.force[:, 3, 0] = 7.99
        collect_stepping_stones_substep(env)
        self.assertAlmostEqual((liftoff.func(env) * liftoff.weight * env.step_dt).item(), -10.0)
        for _ in range(3):
            collect_stepping_stones_substep(env)
        self.assertAlmostEqual((liftoff.func(env) * liftoff.weight * env.step_dt).item(), -10.0)

        sensor.force[:, 3, 0] = 8.
        for _ in range(15):
            collect_stepping_stones_substep(env)
        sensor.force[:, 3, 0] = 7.99
        collect_stepping_stones_substep(env)
        sensor.force[:, 3, 0] = 8.
        for _ in range(14):
            collect_stepping_stones_substep(env)

        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        self.assertAlmostEqual(hold.func(env).item(), .25)
        self.assertEqual(liftoff.func(env).item(), 0.0)
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        self.assertEqual(hold.func(env).item(), 0.0)
        self.assertAlmostEqual(low_support.func(env).item(), .25)
        self.assertFalse(stepping_failed(env).item())

        env.common_step_counter += 1
        sensor.found[:, 0] = 1
        collect_stepping_stones_substep(env)
        self.assertEqual(low_support.func(env).item(), 0.0)

    def test_only_the_opening_step_is_exempt_from_the_left_front_restriction(self):
        env = mounted_env()
        feet = env.scene["stepping_foot_contact"].data
        state = env._stepping_stones
        handoff = state["left_front_handoff"]
        for _ in range(3):
            collect_stepping_stones_substep(env)
        feet.found[:, 3] = 0
        collect_stepping_stones_substep(env)
        feet.found[:, 3] = 1
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertFalse(handoff.first_right_rear_step_complete.item())
        feet.found[:, [0, 3]] = 0
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertFalse(handoff.first_right_rear_step_complete.item())
        self.assertEqual(stepping_metric(env, "left_front_overlap_count").item(), 0.)

        feet.found[:, 0] = 1
        feet.pos[:, 0] = state["stone_centers"][:, 0]
        for _ in range(4):
            collect_stepping_stones_substep(env)
        self.assertTrue(state["visited"][0, 0].item())
        feet.found[:, 3] = 1
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertFalse(handoff.first_right_rear_step_complete.item())
        feet.force[:, 3, 0] = 6.
        for _ in range(4):
            collect_stepping_stones_substep(env)
        feet.force[:, 3, 0] = 10.
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertFalse(handoff.first_right_rear_step_complete.item())
        feet.pos[:, 3] = state["stone_centers"][:, 1]
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertTrue(handoff.first_right_rear_step_complete.item())

        feet.found[:, [0, 3]] = 0
        collect_stepping_stones_substep(env)
        self.assertEqual(stepping_metric(env, "left_front_overlap_count").item(), 1.)
        reset_stepping_stones(env, None)
        self.assertFalse(handoff.first_right_rear_step_complete.item())

    def test_left_front_overlap_is_immediate_when_right_rear_unloads_on_top(self):
        env = mounted_env(3)
        finish_first_right_rear_step(env)
        feet = env.scene["stepping_foot_contact"].data
        feet.found[:, 0] = 0
        for _ in range(4):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(stepping_metric(env, "left_front_overlap_count"), torch.zeros(3))

        env.common_step_counter += 1
        feet.force[:2, 3, 0] = 6.0
        feet.found[2, 3] = 0
        collect_stepping_stones_substep(env)
        torch.testing.assert_close(stepping_metric(env, "left_front_overlap_count"), torch.ones(3))
        feet.force[1, 3, 0] = 8.0
        for _ in range(3):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(stepping_metric(env, "left_front_overlap_count"), torch.ones(3))
        hold = env.cfg.rewards["stepping_left_front_support"]
        torch.testing.assert_close(hold.func(env), torch.ones(3))
        reset_stepping_stones(env, torch.tensor([0]))
        torch.testing.assert_close(stepping_metric(env, "left_front_overlap_count"), torch.tensor([0., 1., 1.]))
        self.assertFalse(env._stepping_stones["left_front_handoff"].left_front_blocked[0].item())

    def test_left_front_overlap_blocks_old_landing_but_other_feet_and_later_steps_score(self):
        env = mounted_env(2)
        reset_stepping_stones(env, None, terrain_level=4)
        set_contacts(env, [[.4, 0., .2]] * 4)
        feet = env.scene["stepping_foot_contact"].data
        robot = env.scene["robot"].data
        state = env._stepping_stones
        robot.site_pos_w[:] = feet.pos
        finish_first_right_rear_step(env)

        feet.found[:, [0, 3]] = 0
        feet.found[1, 1] = 0
        for _ in range(16):
            collect_stepping_stones_substep(env)
        feet.found[:, 3] = 1
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertTrue(state["left_front_handoff"].first_right_rear_step_complete.all().item())

        env.common_step_counter += 1
        feet.found[:, 0] = 1
        feet.found[1, 1] = 1
        feet.pos[:, 0] = state["stone_centers"][:, 0]
        feet.pos[1, 1] = feet.pos[1, 0]
        robot.site_pos_w[:] = feet.pos
        for _ in range(16):
            collect_stepping_stones_substep(env)
        for key, score in (("stepping_cover", 20. / 6), ("stepping_stone_center", 2.), ("stepping_foot_forward_step", .75)):
            term = env.cfg.rewards[key]
            torch.testing.assert_close(term.func(env) * term.weight * env.step_dt, torch.tensor([0., score]))
        self.assertEqual(state["visited"][:, 0].tolist(), [False, True])
        self.assertFalse(state["stone_center_awarded"][:, 0, 0].any().item())
        self.assertFalse(state["left_front_handoff"].left_front_blocked.any().item())
        for _ in range(8):
            collect_stepping_stones_substep(env)
        self.assertFalse(state["visited"][0, 0].item())

        env.common_step_counter += 1
        feet.found[:, 0] = 0
        for _ in range(16):
            collect_stepping_stones_substep(env)
        feet.found[:, 0] = 1
        feet.pos[1, 0] = state["stone_centers"][1, 2]
        robot.site_pos_w[:] = feet.pos
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertTrue(state["visited"][0, 0].item())
        self.assertTrue(state["visited"][1, 2].item())
        term = env.cfg.rewards["stepping_foot_forward_step"]
        torch.testing.assert_close(term.func(env) * term.weight * env.step_dt, torch.tensor([0., .75]))
        self.assertFalse(stepping_failed(env).any().item())

    def test_premature_support_handoff_requires_real_left_front_load_before_pair_release(self):
        env = mounted_env(6)
        feet = env.scene["stepping_foot_contact"].data
        feet.found[[0, 4, 5], 0] = 0
        feet.force[1, 0, 0] = 7.99
        shin = env.scene["stepping_shin_contact"].data
        shin.found[2, 0] = 1
        shin.force[2, 0, 2] = -8.
        shin.pos[2, 0] = torch.tensor([.4, 0., .2])
        for _ in range(16):
            collect_stepping_stones_substep(env)
        feet.found[5, 0] = 1
        for _ in range(13):
            collect_stepping_stones_substep(env)

        env.common_step_counter += 1
        feet.found[:, 1:3] = 0
        feet.found[4, 2] = 1
        term = env.cfg.rewards["stepping_premature_support_handoff"]
        for _ in range(3):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(6))
        collect_stepping_stones_substep(env)
        expected = torch.tensor([1., 1., 1., 0., 0., 1.])
        torch.testing.assert_close(term.func(env) * term.weight * env.step_dt, -3.0 * expected)
        self.assertFalse(stepping_failed(env).any().item())

        env.common_step_counter += 1
        for _ in range(8):
            collect_stepping_stones_substep(env)
        torch.testing.assert_close(term.func(env), torch.zeros(6))
        torch.testing.assert_close(stepping_metric(env, "premature_support_handoffs"), expected)

        feet.found[0, 2] = 1
        collect_stepping_stones_substep(env)
        feet.found[0, 2] = 0
        for _ in range(4):
            collect_stepping_stones_substep(env)
        expected[0] = 2.
        torch.testing.assert_close(stepping_metric(env, "premature_support_handoffs"), expected)
        reset_stepping_stones(env, torch.tensor([0]))
        expected[0] = 0.
        torch.testing.assert_close(stepping_metric(env, "premature_support_handoffs"), expected)

    def test_forward_step_requires_eighty_continuous_ms_of_clean_loading(self):
        env = mounted_env()
        sensor = env.scene["stepping_foot_contact"].data
        term = env.cfg.rewards["stepping_foot_forward_step"]
        for _ in range(3):
            collect_stepping_stones_substep(env)
        sensor.found[:, 3] = 0
        for _ in range(16):
            collect_stepping_stones_substep(env)
        sensor.pos[:, 3] = env._stepping_stones["stone_centers"][:, 0]
        sensor.found[:, 3] = 1
        for _ in range(15):
            collect_stepping_stones_substep(env)
        set_shin_load(env, 3, 8., position=sensor.pos[0, 3].tolist())
        collect_stepping_stones_substep(env)
        self.assertEqual(term.func(env).item(), 0.0)
        set_shin_load(env, 3, 0.)
        for _ in range(15):
            collect_stepping_stones_substep(env)
        self.assertEqual(term.func(env).item(), 0.0)
        collect_stepping_stones_substep(env)
        self.assertAlmostEqual((term.func(env) * term.weight * env.step_dt).item(), .75)

    def test_diagonal_feet_can_land_separately_and_only_the_supported_foot_scores(self):
        env = mounted_env()
        sensor = env.scene["stepping_foot_contact"].data
        term = env.cfg.rewards["stepping_foot_forward_step"]
        for _ in range(3):
            collect_stepping_stones_substep(env)
        sensor.found[:, [0, 3]] = 0
        for _ in range(16):
            collect_stepping_stones_substep(env)
        sensor.pos[:, [0, 3]] = env._stepping_stones["stone_centers"][:, 0:1]
        sensor.found[:, 0] = 1
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertAlmostEqual((term.func(env) * term.weight * env.step_dt).item(), .75)
        handoff = env._stepping_stones["left_front_handoff"]
        self.assertEqual(handoff.loaded_substeps[0, 3].item(), 0)

        env.common_step_counter += 1
        sensor.found[:, 3] = 1
        for _ in range(16):
            collect_stepping_stones_substep(env)
        self.assertAlmostEqual((term.func(env) * term.weight * env.step_dt).item(), .75)
        torch.testing.assert_close(handoff.last_rewarded_surface[0, [0, 3]], torch.ones(2, dtype=torch.long))

    def test_body_level_scales_with_forward_speed_and_rejects_tilt(self):
        env = mounted_env(8)
        robot = env.scene["robot"].data
        sin_tilt, cos_tilt = math.sin(math.radians(12)), math.cos(math.radians(12))
        robot.projected_gravity_b[1:4] = torch.tensor(
            [[sin_tilt, 0, -cos_tilt], [0, -sin_tilt, -cos_tilt], [0, 0, 1]]
        )
        robot.root_link_lin_vel_w[:, 0] = torch.tensor([.25, .25, .25, .25, .125, 0.0, -.25, .25])
        env._stepping_stones["stage"][7] = 3
        expected = torch.tensor([1.0, math.exp(-1), math.exp(-1), 0.0, .5, 0.0, 0.0, 0.0])
        torch.testing.assert_close(stepping_body_level_reward(env, 12.0), expected)

    def test_flight_is_charged_by_time_without_a_120_ms_termination(self):
        env = mounted_env()
        sensor = env.scene["stepping_foot_contact"].data
        sensor.found.zero_()
        total = 0.0
        term = env.cfg.rewards["stepping_flight"]
        for _ in range(10):
            env.common_step_counter += 1
            for _ in range(4):
                collect_stepping_stones_substep(env)
            self.assertFalse(stepping_failed(env).item())
            total += term.func(env).item() * term.weight * env.step_dt
        self.assertAlmostEqual(total, -0.4)
        sensor.pos[:, 2, 2] = 0.0
        sensor.found[:, 2] = 1.0
        collect_stepping_stones_substep(env)
        self.assertTrue(stepping_failed(env).item())

    def test_flight_cost_integrates_physical_substeps_and_resets(self):
        env = fake_env(1)
        reset_stepping_stones(env, None)
        collect_stepping_stones_substep(env)
        collect_stepping_stones_substep(env)
        set_contacts(env, [[.4, 0.0, .2]] * 4)
        collect_stepping_stones_substep(env)
        collect_stepping_stones_substep(env)
        self.assertAlmostEqual(stepping_flight_penalty(env).item(), .5)
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        self.assertEqual(stepping_flight_penalty(env).item(), 0.0)

    def test_legal_foot_slip_is_penalized_at_low_command_speed(self):
        env = mounted_env()
        env.command_manager.get_command = lambda name: torch.tensor([[.075, 0., 0.]])
        self.assertNotIn("feet_slip", env.cfg.rewards)
        robot = env.scene["robot"].data
        robot.site_lin_vel_w = torch.tensor([[[-.14, 0., 0.], [.5, 0., 0.], [.01, 0., 0.], [0., 0., .3]]])
        env.scene["stepping_foot_contact"].data.found[:, 1] = 0.0
        collect_stepping_stones_substep(env)
        cfg = env.cfg.rewards["stepping_foot_slip"]
        self.assertAlmostEqual(stepping_foot_slip_penalty(env, **cfg.params).item() * cfg.weight, -0.5)

    def test_forward_speed_reward_prefers_the_command_and_ignores_leg_only_motion(self):
        env = mounted_env(8)
        robot = env.scene["robot"].data
        env.command_manager.get_command = lambda name: torch.tensor([[.1, 0., 0.]]).repeat(8, 1)
        robot.root_link_lin_vel_w[:, 0] = torch.tensor([0., -.1, .05, .1, .2, .1, .1, float("nan")])
        robot.root_link_lin_vel_b[:, 0] = .1
        robot.root_link_lin_vel_w[0, 1:] = .3
        env._stepping_stones["stage"][5] = 3
        env._stepping_stones["failure"][6] = True
        cfg = env.cfg.rewards["stepping_forward_velocity"]
        self.assertEqual(cfg.weight, 2.5)
        reward = forward_velocity_reward(env, **cfg.params)
        torch.testing.assert_close(reward[[0, 4, 5, 6, 7]], torch.zeros(5))
        self.assertEqual(reward[1].item(), -3.0)
        self.assertEqual(reward[3].item(), 1.0)
        self.assertGreater(reward[2].item(), reward[4].item())
        self.assertAlmostEqual(reward[2].item(), 0.75)
        self.assertLess((reward[1] + reward[3]).item(), 0.0)

    def test_hip_abduction_symmetry_allows_mirrored_nonzero_angles(self):
        env = mounted_env(5)
        state = env._stepping_stones
        robot = env.scene["robot"].data
        angles = torch.tensor([0.2, -0.2, 0.3, -0.3])
        robot.joint_pos[:, state["hip_abduction_ids"]] = angles
        robot.root_link_lin_vel_w[:, 0] = 0.25
        robot.joint_pos[1, state["hip_abduction_ids"][1]] = 0.2
        robot.root_link_lin_vel_w[2, 0] = 0.0
        state["stage"][3] = 3
        state["failure"][4] = True
        cfg = env.cfg.rewards["stepping_hip_abduction_symmetry"]
        result = stepping_hip_abduction_symmetry_reward(env, **cfg.params)
        self.assertEqual(result[0].item(), 1.0)
        self.assertLess(result[1].item(), 1.0)
        torch.testing.assert_close(result[2:], torch.zeros(3))

    def test_lateral_velocity_penalty_matches_slope_deadband_and_saturation(self):
        env = mounted_env(6)
        env.scene["robot"].data.root_link_lin_vel_b[:, 1] = torch.tensor(
            [0.0, 0.01, -0.01, 0.06, -0.06, 0.20]
        )
        cfg = env.cfg.rewards["stepping_lateral_velocity"]
        self.assertEqual(cfg.params, {"tolerance": 0.01, "scale": 0.10})
        cost = stepping_lateral_velocity_penalty(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 1.0]))

    def test_route_alignment_rewards_forward_motion_without_heading_error_or_sideways_speed(self):
        env = mounted_env(11)
        state = env._stepping_stones
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_w[:, 0] = .25
        robot.heading_w[1:3] = torch.tensor([math.radians(12), -math.radians(12)])
        robot.root_link_lin_vel_w[3:5, 1] = torch.tensor([.08, -.08])
        robot.root_link_lin_vel_w[5, :2] = torch.tensor([0.0, .08])
        robot.root_link_lin_vel_w[6, 0] = -.25
        robot.heading_w[7] = math.pi
        robot.heading_w[8] = 2.0 * math.pi
        state["stage"][9] = 3
        state["failure"][10] = True
        cfg = env.cfg.rewards["stepping_route_alignment"]
        decay = math.exp(-1.0)
        expected = torch.tensor([1.0, decay, decay, decay, decay, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(route_alignment_reward(env, **cfg.params), expected)
        self.assertEqual(cfg.weight, 0.5)

    def test_real_ball_bottom_is_used_when_the_foot_rotates(self):
        env = mounted_env()
        env.scene["robot"].data.site_pos_w[:, :, 2] = .24
        env.scene["robot"].data.site_quat_w[:] = torch.tensor([0., 1., 0., 0.])
        np.testing.assert_allclose(_foot_bottom(env, env._stepping_stones)[:, :, 2].numpy(), .19, atol=1.e-6)


if __name__ == "__main__":
    unittest.main()
