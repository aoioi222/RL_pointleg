"""Geometry, actual-contact task logic and reward-scale regression cases."""

from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pointleg.mdp.elevation import (
    ElevationSettings, ElevationTerrainCurriculum, collect_elevation_contacts,
    elevation_cost, elevation_event, reset_elevation, select_elevation_terrain,
    update_elevation_state,
)
from pointleg.robot_cfg import FOOT_SITE_NAMES, LEG_JOINT_NAMES
from pointleg.terrains.elevation import (
    CENTER_Y, EXIT_PLATFORM_LENGTH, FLAT_NAME, SPAWN_X, STAIR_PLATFORM_RANGE, STAIR_TREAD_COUNT,
    STEP_DEPTH, TERRAIN_NAMES, TILE_SIZE, WALL_CENTERS, ElevationTerrainCfg, geometry_profile,
)
from pointleg.transfer import _TransferEnv, prepare_transfer
from pointleg.rsl_rl.runners.elevation_runner import ElevationOnPolicyRunner
from pointleg.evaluate_elevation import evaluation_cases


class Scene(dict):
    pass


class Robot:
    def __init__(self, count):
        self.joint_names = list(LEG_JOINT_NAMES)
        pose = torch.zeros(count, 13)
        pose[:, 2], pose[:, 3] = 0.42, 1
        self.data = SimpleNamespace(
            default_root_state=pose, root_link_pos_w=pose[:, :3].clone(),
            root_link_quat_w=pose[:, 3:7].clone(),
            root_link_lin_vel_b=torch.zeros(count, 3), root_link_lin_vel_w=torch.zeros(count, 3),
            root_link_ang_vel_b=torch.zeros(count, 3), joint_pos=torch.zeros(count, 12),
            default_joint_pos=torch.zeros(count, 12),
            joint_pos_limits=torch.tensor([-3., 3.]).repeat(count, 12, 1),
            soft_joint_pos_limits=torch.tensor([-2.9, 2.9]).repeat(count, 12, 1),
            site_pos_w=torch.zeros(count, 4, 3), site_lin_vel_w=torch.zeros(count, 4, 3),
            site_ang_vel_w=torch.zeros(count, 4, 3),
            site_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(count, 4, 1),
            heading_w=torch.zeros(count), qfrc_actuator=torch.zeros(count, 12),
        )

    def find_sites(self, names, preserve_order=False):
        return list(range(4)), list(FOOT_SITE_NAMES)

    def write_root_state_to_sim(self, pose, env_ids):
        self.data.root_link_pos_w[env_ids] = pose[:, :3]
        self.data.root_link_quat_w[env_ids] = pose[:, 3:7]
        self.data.root_link_lin_vel_b[env_ids] = pose[:, 7:10]
        self.data.root_link_lin_vel_w[env_ids] = pose[:, 7:10]


def fake_env(jump=False, count=4, stair_height_override=None, stair_height_range=None):
    """Build legacy or fixed-height tensor fixtures without integrating physics."""
    from pointleg.config.elevation import make_elevation_env_cfg

    cfg = make_elevation_env_cfg(jump=jump)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    rng = np.random.default_rng(42)
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    for name, sub in sub_terrains.items():
        if name.startswith("stairs"):
            sub.stair_height_override = stair_height_override
            sub.stair_height_range = stair_height_range
    for level in range(5):
        for sub in sub_terrains.values():
            sub.function(level / 4, spec, rng)
    table = torch.tensor(np.array([[sub.generated_profile(level / 4).spawn for sub in sub_terrains.values()]
                                   for level in range(5)]), dtype=torch.float32)
    terrain = SimpleNamespace(
        cfg=cfg.scene.terrain, terrain_types=torch.full((count,), 4),
        terrain_levels=torch.zeros(count, dtype=torch.long), env_origins=torch.zeros(count, 3),
        get_terrain_origins=lambda levels, types: table[levels, types].clone(),
    )
    sensor = SimpleNamespace(cfg=SimpleNamespace(num_slots=4), data=SimpleNamespace(
        found=torch.zeros(count, 16), force=torch.zeros(count, 16, 3),
        pos=torch.zeros(count, 16, 3), normal=torch.zeros(count, 16, 3)))
    body = SimpleNamespace(cfg=SimpleNamespace(num_slots=2), primary_names=["robot/base_link"],
                           data=SimpleNamespace(found=torch.zeros(count, 2), force=torch.zeros(count, 2, 3), pos=torch.zeros(count, 2, 3), normal=torch.zeros(count, 2, 3)))
    scene = Scene(robot=Robot(count), elevation_contacts=sensor, elevation_body_contacts=body,
                  base_ground_contact=SimpleNamespace(data=SimpleNamespace(found=torch.zeros(count, 1))))
    scene.terrain = terrain
    env = SimpleNamespace(
        cfg=cfg, scene=scene, num_envs=count, device="cpu", step_dt=0.02, physics_dt=0.005,
        common_step_counter=0, episode_length_buf=torch.zeros(count, dtype=torch.long),
        action_manager=SimpleNamespace(action=torch.zeros(count, 12), prev_action=torch.zeros(count, 12)),
    )
    settings = ElevationSettings(jump=jump)
    select_elevation_terrain(env, None, settings)
    reset_elevation(env, None, settings, randomize=False)
    return env


def contacts(env, xs, zs=None, outside=False, effective=True):
    sensor = env.scene["elevation_contacts"].data
    sensor.found[:] = 0
    sensor.force[:] = 0
    sensor.pos[:] = 0
    sensor.normal[:] = 0
    for foot in range(4):
        slot = foot * 4
        sensor.found[:, slot] = float(effective)
        sensor.force[:, slot, 0] = 10 if effective else 0
        sensor.normal[:, slot, 2] = -1
        sensor.pos[:, slot, 0] = xs[foot]
        sensor.pos[:, slot, 1] = CENTER_Y + (0.45 if outside and foot == 3 else 0)
        sensor.pos[:, slot, 2] = 0 if zs is None else zs[foot]


def bounds(env):
    root = env.scene["robot"].data.root_link_pos_w[:, None]
    half = torch.tensor([0.3, 0.25, 0.1])
    return root - half, root + half


class ElevationGeometryTests(unittest.TestCase):
    def test_all_profile_surfaces_match_physical_rays(self):
        for kind in (*TERRAIN_NAMES, FLAT_NAME):
            for level in range(5):
                with self.subTest(kind=kind, level=level):
                    spec = mujoco.MjSpec()
                    spec.worldbody.add_body(name="terrain")
                    cfg = ElevationTerrainCfg(kind=kind, size=TILE_SIZE)
                    output = cfg.function(level / 4, spec, np.random.default_rng(42))
                    profile = cfg.generated_profile(level / 4)
                    model = spec.compile()
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    np.testing.assert_allclose(output.origin, profile.spawn)
                    probes = (1.2, 2.8, 2.999, 3.001, 3.16, 3.48, 3.80, 4.2, 4.99, 5.1)
                    if kind == "wall_sequence":
                        probes += WALL_CENTERS
                    if kind.startswith("stairs"):
                        probes += tuple(x for left, right, _ in profile.step_surfaces
                                        for x in (left - .001, left + .001, (left + right) / 2))
                        probes += (profile.landing_end_x - .001, profile.landing_end_x + .001)
                    for x in probes:
                        distance = mujoco.mj_ray(model, data, np.array([x, CENTER_Y, 2.]), np.array([0., 0., -1.]), None, 1, -1, np.array([-1], np.int32))
                        self.assertAlmostEqual(2 - distance, float(profile.surface_height(x)), places=5)

    def test_two_three_step_flights_have_seeded_random_platform_and_short_exit(self):
        lengths = []
        for seed in (12, 34, 12):
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name="terrain")
            cfg = ElevationTerrainCfg(kind="stairs_up", size=TILE_SIZE)
            cfg.function(1.0, spec, np.random.default_rng(seed))
            profile = cfg.generated_profile(1.0)
            self.assertEqual(len(profile.step_surfaces), STAIR_TREAD_COUNT)
            first_end = profile.step_surfaces[2][0] + STEP_DEPTH
            second_start = profile.step_surfaces[3][0]
            gap = second_start - first_end
            self.assertGreaterEqual(gap, STAIR_PLATFORM_RANGE[0])
            self.assertLessEqual(gap, STAIR_PLATFORM_RANGE[1])
            self.assertAlmostEqual(gap, profile.stair_platform_length)
            self.assertAlmostEqual(profile.step_surfaces[2][2], .36)
            self.assertAlmostEqual(profile.step_surfaces[-1][2], .72)
            self.assertAlmostEqual(profile.landing_end_x - profile.exit_x, EXIT_PLATFORM_LENGTH)
            self.assertLess(profile.landing_end_x, TILE_SIZE[0])
            lengths.append(gap)
        self.assertEqual(lengths[0], lengths[2])
        self.assertNotEqual(lengths[0], lengths[1])

    def test_downstairs_spawn_and_six_landing_surfaces_are_distinct(self):
        profile = geometry_profile("stairs_down", 1.0)
        self.assertAlmostEqual(profile.spawn[2], .72)
        np.testing.assert_allclose([surface[2] for surface in profile.step_surfaces], [.60, .48, .36, .24, .12, 0])
        self.assertGreater(profile.step_surfaces[0][0], SPAWN_X + .3)
        self.assertAlmostEqual(float(profile.surface_height(profile.step_surfaces[3][0] - .001)), .36)

    def test_profile_lookup_requires_generated_physical_geometry(self):
        with self.assertRaisesRegex(RuntimeError, "after terrain generation"):
            ElevationTerrainCfg(kind="stairs_up").generated_profile(0)

    def test_wall_dimensions_and_lane_width(self):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        out = ElevationTerrainCfg(kind="wall_sequence", size=TILE_SIZE).function(1, spec, np.random.default_rng(1))
        walls = [item.geom for item in out.geometries[1:]]
        self.assertEqual(len(walls), 3)
        for wall in walls:
            np.testing.assert_allclose(wall.size, [0.04, 0.4, 0.17])
        self.assertAlmostEqual(walls[1].pos[0] - walls[0].pos[0] - 0.08, 2.5)

    def test_task_contract_and_evaluation_events(self):
        rough = load_env_cfg("PointLeg-Rough-v0")
        for task in ("PointLeg-Elevation-v0",):
            cfg = load_env_cfg(task)
            self.assertEqual(list(cfg.observations["actor"].terms), list(rough.observations["actor"].terms))
            self.assertEqual(list(cfg.observations["critic"].terms), list(rough.observations["critic"].terms))
            jump = task == "PointLeg-Jump-v0"
            names = list(TERRAIN_NAMES) if jump else [*TERRAIN_NAMES, FLAT_NAME]
            self.assertEqual((cfg.scene.terrain.terrain_generator.num_rows, cfg.scene.terrain.terrain_generator.num_cols), (5, len(names)))
            self.assertEqual(list(cfg.scene.terrain.terrain_generator.sub_terrains), names)
            self.assertNotIn("lin_vel_z", cfg.rewards)
            self.assertEqual("support_deficit" in cfg.rewards, not jump)
            play = load_env_cfg(task, play=True)
            self.assertIn("reset_elevation", play.events)
            self.assertFalse(play.events["reset_elevation"].params["randomize"])
            self.assertEqual(play.events["reset_elevation"].params["randomize_approach"], task == "PointLeg-Elevation-v0")

    def test_fixed_cases_cover_allowed_types_five_levels_and_target_wall(self):
        walking = evaluation_cases("PointLeg-Elevation-v0")
        self.assertEqual(len(walking), 21)
        self.assertEqual(sum(case["terrain_type"] == FLAT_NAME for case in walking), 1)
        self.assertNotIn("wall_sequence", {case["terrain_type"] for case in walking})


class ElevationLogicTests(unittest.TestCase):
    def test_walking_approach_resamples_within_point_three_meters_on_all_four_types(self):
        env = fake_env(count=128)
        settings = env._elevation_settings
        reset_elevation(env, None, settings, randomize=True)
        first = env._elevation["start_x"].clone()
        for kind in range(4):
            mask = env.scene.terrain.terrain_types == kind
            offsets = first[mask] - (env._elevation["entry_x"][mask] - settings.walking_approach_distance)
            self.assertTrue((offsets >= -.3 - 1e-6).all())
            self.assertTrue((offsets <= .3 + 1e-6).all())
            self.assertTrue((offsets < 0).any() and (offsets > 0).any())
        reset_elevation(env, None, settings, randomize=True)
        self.assertFalse(torch.equal(first, env._elevation["start_x"]))
        reset_elevation(env, None, settings, randomize=False)
        nominal = torch.where(env.scene.terrain.terrain_types < 4,
                              env._elevation["entry_x"] - settings.walking_approach_distance, SPAWN_X)
        torch.testing.assert_close(env._elevation["start_x"], nominal)
        reset_elevation(env, None, settings, randomize=False, randomize_approach=True)
        self.assertFalse(torch.equal(env._elevation["start_x"], nominal))
        torch.testing.assert_close(env._elevation["speed"], torch.full((128,), .35))

    def test_reset_metadata_matches_sampled_stair_geometry(self):
        env = fake_env()
        for index, kind in enumerate(TERRAIN_NAMES[:2]):
            profile = env.scene.terrain.cfg.terrain_generator.sub_terrains[kind].generated_profile(0)
            torch.testing.assert_close(env._elevation["step_surfaces"][index], torch.tensor(profile.step_surfaces))
            self.assertAlmostEqual(float(env._elevation["exit_x"][index]), profile.exit_x, places=5)
            self.assertAlmostEqual(float(env._elevation["stair_platform_length"][index]), profile.stair_platform_length, places=6)
        self.assertFalse(env._elevation["covered"].any())

    def test_jump_trigger_distance_is_not_changed_by_walking_approach_randomization(self):
        env = fake_env(jump=True, count=64)
        reset_elevation(env, None, env._elevation_settings, randomize=True, randomize_approach=True)
        distances = env._elevation["wall_x"] - env._elevation["start_x"]
        self.assertTrue((distances >= .9 - 1e-6).all())
        self.assertTrue((distances <= 1.1 + 1e-6).all())

    def test_first_flight_is_not_completion_and_six_tread_reward_stays_bounded(self):
        env = fake_env(count=1)
        state = env._elevation
        robot = env.scene["robot"].data
        robot.root_link_pos_w[:, 0] = 4.5
        robot.root_link_lin_vel_b[:, 0] = .35
        state["covered"][:, :3] = True
        state["support_count"][:] = 4
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            for tick in range(1, 28):
                env.common_step_counter = tick
                env.episode_length_buf[:] = tick
                state = update_elevation_state(env)
                self.assertFalse(state["complete"].any())
                self.assertFalse(state["failed"].any())
            state["new_events"][:] = STAIR_TREAD_COUNT
            weight = env.cfg.rewards["first_contact_events"].weight
            self.assertAlmostEqual(float(elevation_event(env, "new_events")[0]) * env.step_dt * weight, .6)
            state["covered"][:] = True
            robot.root_link_pos_w[:, 0] = state["exit_x"] + .6
            for tick in range(28, 54):
                env.common_step_counter = tick
                env.episode_length_buf[:] = tick
                state = update_elevation_state(env)
            self.assertTrue(state["complete"].all())
            self.assertFalse(state["failed"].any())

    def test_forward_only_viser_panel_does_not_create_zero_range_sliders(self):
        for jump in (False, True):
            with self.subTest(jump=jump):
                env = fake_env(jump=jump)
                command = env.cfg.commands["twist"].build(env)
                gui = Mock()
                gui.add_folder.return_value = nullcontext()
                gui.add_slider.side_effect = AssertionError("Zero-range joystick axis")
                command.create_gui(
                    "twist", SimpleNamespace(gui=gui), lambda: 0,
                    on_change=Mock(), request_action=Mock(),
                )
                gui.add_markdown.assert_called_once()
                gui.add_slider.assert_not_called()
                self.assertIsNone(command._joystick_enabled)
                command.compute(env.step_dt)
                torch.testing.assert_close(command.command[:, 0], env._elevation["speed"])
                self.assertTrue((command.command[:, 1:] == 0).all())

    def test_spawn_type_is_restricted_even_if_initial_allocation_was_wrong(self):
        walk = fake_env(count=12)
        self.assertEqual(set(walk.scene.terrain.terrain_types.tolist()), {0, 1, 2, 3, 5})
        jump = fake_env(jump=True)
        self.assertTrue((jump.scene.terrain.terrain_types == 4).all())
        self.assertTrue((walk.scene["robot"].data.root_link_pos_w[walk.scene.terrain.terrain_types == 1, 2] > 0.42).all())

    def test_true_tread_contacts_deduplicate_and_side_faces_do_not_count(self):
        env = fake_env(count=1)
        contacts(env, [3.16, 3.48, 3.80, 1.2], [.04, .08, .12, 0])
        env.scene["elevation_contacts"].data.normal[:, :, 2] = 0
        collect_elevation_contacts(env)
        self.assertEqual(int(env._elevation["covered"].sum()), 0)
        env.scene["elevation_contacts"].data.normal[:, :, 2] = -1
        collect_elevation_contacts(env)
        self.assertEqual(int(env._elevation["covered"].sum()), 3)
        self.assertEqual(float(env._elevation["pending_new"][0]), 3)
        env.common_step_counter += 1
        collect_elevation_contacts(env)
        self.assertEqual(float(env._elevation["pending_new"][0]), 0)
        second = env._elevation["step_surfaces"][0, 3:]
        contacts(env, [float((left + right) / 2) for left, right, _ in second] + [1.2],
                 [float(height) for _, _, height in second] + [0])
        collect_elevation_contacts(env)
        self.assertTrue(env._elevation["covered"].all())
        self.assertEqual(float(env._elevation["pending_new"][0]), 3)

    def test_airborne_outside_foot_is_not_support_failure(self):
        env = fake_env(count=1)
        contacts(env, [1.2] * 4, outside=True, effective=False)
        collect_elevation_contacts(env)
        self.assertFalse(env._elevation["outside"].any())
        contacts(env, [1.2] * 4, outside=True)
        collect_elevation_contacts(env)
        self.assertTrue(env._elevation["outside"].all())

    def test_light_wall_side_contact_differs_from_wall_support(self):
        env = fake_env(jump=True, count=1)
        body = env.scene["elevation_body_contacts"].data
        body.found[0, 0] = 1
        body.force[0, 0, 0] = 2
        body.pos[0, 0] = torch.tensor([3., CENTER_Y, .1])
        body.normal[0, 0, 0] = 1
        collect_elevation_contacts(env)
        self.assertFalse(env._elevation["bad_wall"].any())
        self.assertFalse(env._elevation["bad_ground"].any())
        self.assertGreater(float(env._elevation["contact_cost"][0]), 0)
        body.normal[0, 0] = torch.tensor([0., 0., -1.])
        collect_elevation_contacts(env)
        self.assertTrue(env._elevation["bad_wall"].all())

    def test_friction_support_on_vertical_wall_is_not_a_free_jump(self):
        env = fake_env(jump=True, count=1)
        sensor = env.scene["elevation_contacts"].data
        sensor.found[0, 0] = 1
        sensor.force[0, 0] = torch.tensor([20., 0., 8.])
        sensor.normal[0, 0] = torch.tensor([1., 0., 0.])
        sensor.pos[0, 0] = torch.tensor([2.96, CENTER_Y, .08])
        collect_elevation_contacts(env)
        self.assertTrue(env._elevation["bad_wall"].all())

    def test_failure_masks_same_step_positive_events_and_terminal_phi(self):
        env = fake_env(count=1)
        contacts(env, [3.16, 3.48, 3.80, 1.2], [.04, .08, .12, 0], outside=True)
        collect_elevation_contacts(env)
        env.common_step_counter, env.episode_length_buf[:] = 1, 1
        env._elevation["phi"][:] = .4
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            state = update_elevation_state(env)
            self.assertTrue(state["failed"].all())
            self.assertEqual(float(state["new_events"][0]), 0)
            self.assertEqual(float(state["phi"][0]), 0)
            self.assertAlmostEqual(float(elevation_event(env, "failed")[0]) * env.step_dt * -50, -50)

    def test_missed_tread_cannot_become_exit_success(self):
        env = fake_env(count=1)
        env._elevation["covered"][:, :3] = True
        env.scene["robot"].data.root_link_pos_w[:, 0] = env._elevation["exit_x"] + .6
        env.common_step_counter = 1
        env.episode_length_buf[:] = 1
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            self.assertTrue(update_elevation_state(env)["failed"].all())
            self.assertFalse(env._elevation["complete"].any())

    def test_curriculum_is_per_type_and_roundtrips(self):
        env = fake_env(count=1)
        cfg = CurriculumTermCfg(func=ElevationTerrainCurriculum, params={"window_size": 2, "threshold": .8})
        course = ElevationTerrainCurriculum(cfg, env)
        env._elevation["complete"][:] = True
        env.episode_length_buf[:] = 1
        course(env, None)
        course(env, None)
        self.assertEqual(course.level_caps.tolist(), [1, 0, 0, 0, 0, 0])
        restored = ElevationTerrainCurriculum(cfg, env)
        restored.load_state_dict(course.state_dict())
        self.assertEqual(restored.state_dict(), course.state_dict())

    def test_jump_does_not_inherit_walking_velocity_cost_during_flight(self):
        env = fake_env(jump=True, count=1)
        env.common_step_counter = 1
        env._elevation["last_step"][:] = 1
        env._elevation["updated_tick"] = 1
        self.assertEqual(float(elevation_cost(env, "velocity")[0]), 0)
        env._elevation["landed"][:] = True
        self.assertGreater(float(elevation_cost(env, "velocity")[0]), 0)

    def test_jump_can_complete_only_after_flight_all_feet_cross_and_landing(self):
        env = fake_env(jump=True, count=1)
        robot = env.scene["robot"].data
        offsets = torch.tensor([.4, .4, -.3, -.3])

        def place(x, foot_height, supported):
            robot.root_link_pos_w[:, 0] = x
            robot.root_link_lin_vel_w[:, 0] = .35
            robot.root_link_lin_vel_b[:, 0] = .35
            robot.site_pos_w[:, :, 0] = x + offsets
            robot.site_pos_w[:, :, 1] = CENTER_Y
            robot.site_pos_w[:, :, 2] = foot_height
            contacts(env, (x + offsets).tolist(), [foot_height] * 4, effective=supported)
            collect_elevation_contacts(env)

        def control():
            env.common_step_counter += 1
            env.episode_length_buf += 1
            with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
                return update_elevation_state(env)

        place(2.0, 0, True)
        place(2.2, .5, False)
        place(2.3, .5, False)
        state = control()
        self.assertTrue(state["airborne"].all())
        self.assertFalse(state["complete"].any())
        place(2.8, .5, False)
        place(3.05, .5, False)
        place(3.5, .5, False)
        state = control()
        self.assertTrue(state["foot_passed"].all())
        self.assertTrue(state["crossed"].all())
        self.assertFalse(state["complete"].any())
        for _ in range(6):
            place(3.7, 0, True)
            state = control()
        self.assertTrue(state["complete"].all())
        self.assertFalse(state["failed"].any())

    def test_deadline_is_failure_not_successful_truncation(self):
        env = fake_env(count=1)
        env.common_step_counter = 1
        env.episode_length_buf[:] = int(env.cfg.episode_length_s / env.step_dt)
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            state = update_elevation_state(env)
        self.assertTrue(state["failed"].all())
        self.assertFalse(state["complete"].any())
        self.assertEqual(int(state["reason"][0]), 7)


class ElevationWalkingTests(unittest.TestCase):
    def test_progress_rewards_prefer_walking_over_freezing(self):
        from pointleg.tests.test_turn_rewards import turn_env

        env = fake_env(count=3)
        env._elevation["speed"][:] = .35
        env._elevation["updated_tick"] = env.common_step_counter
        env._elevation["forward_progress"][:] = .35 * env.step_dt
        env._elevation["support_count"][:] = 4.
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0., .35, -.35])
        cfg = load_env_cfg("PointLeg-Elevation-v0")
        for name in ("command_aligned_progress", "command_progress_deficit", "center", "heading", "time"):
            self.assertEqual(cfg.rewards[name].weight, 0.)
        term = cfg.rewards["velocity"]
        scores = term.weight * term.func(env, **term.params)
        self.assertGreater(scores[1].item(), scores[0].item())
        self.assertEqual(scores[0].item(), 0.)
        self.assertEqual(scores[2].item(), 0.)

    def test_workspace_preserves_vertical_motion_and_replaces_legacy_stance_cost(self):
        from pointleg.tests.test_turn_rewards import turn_env
        from pointleg.tests.test_elevation_clearance import clearance_env

        env = turn_env(1)
        env.command[:] = torch.tensor([[.35, 0., 0.]])
        cfg = load_env_cfg("PointLeg-Elevation-v0")
        term_cfg = cfg.rewards["translation_stance_overstay"]
        self.assertEqual(term_cfg.weight, 0.)
        term = term_cfg.func(term_cfg, env)
        for _ in range(60):
            cost = term(env, **term_cfg.params)
        self.assertGreater(cost.item(), 0.)
        env, _, _ = clearance_env()
        robot = env.scene["robot"]
        robot.data.body_link_pos_w = robot.data.site_pos_w.clone() - torch.tensor([
            [.035, .105, -.4], [.035, -.105, -.4], [.035, .105, -.4], [.035, -.105, -.4],
        ])
        robot.find_bodies = lambda name: ([['fl', 'fr', 'rl', 'rr'].index(name.split('_')[0])], [name])
        term_cfg = cfg.rewards["translation_foot_workspace"]
        term = term_cfg.func(term_cfg, env)
        self.assertEqual(term(env, **term_cfg.params).item(), 0.)
        env.scene["robot"].data.site_pos_w[:, :, 2] += .2
        self.assertEqual(term(env, **term_cfg.params).item(), 0.)
        env.scene["robot"].data.site_pos_w[:, 0, 0] += .5
        self.assertGreater(term(env, **term_cfg.params).item(), 0.)

    def test_stall_window_uses_net_world_progress_and_allows_slow_climbing(self):
        env = fake_env(count=3)
        start = env.scene["robot"].data.root_link_pos_w[:, 0].clone()
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .35
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            for tick in range(1, 201):
                env.common_step_counter = tick
                env.episode_length_buf[:] = tick
                env.scene["robot"].data.root_link_pos_w[1, 0] = start[1] + tick * env.step_dt * .1
                env.scene["robot"].data.root_link_pos_w[2, 0] = start[2] + (.03 if tick % 2 else -.03)
                state = update_elevation_state(env)
                if tick < 199:
                    self.assertFalse(state["stalled"].any())
                elapsed_before = state["stall_elapsed"].clone()
                update_elevation_state(env)
                torch.testing.assert_close(state["stall_elapsed"], elapsed_before)
        self.assertEqual(state["stalled"].tolist(), [True, False, True])
        self.assertEqual(state["reason"].tolist(), [9, 0, 9])
        self.assertEqual(state["failed"].tolist(), [True, False, True])
        reset_elevation(env, torch.tensor([0]), env._elevation_settings, randomize=False)
        self.assertFalse(state["stalled"][0])
        self.assertEqual(state["stall_elapsed"][0].item(), 0.)
        self.assertTrue(state["stalled"][2])

    def test_jump_does_not_receive_walking_stall_termination(self):
        env = fake_env(jump=True, count=1)
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            for tick in range(1, 220):
                env.common_step_counter = tick
                env.episode_length_buf[:] = tick
                state = update_elevation_state(env)
        self.assertFalse(state["stalled"].any())
        self.assertFalse(state["failed"].any())

    def test_flat_rehearsal_completes_without_stair_contact_events(self):
        env = fake_env(count=5)
        flat = env.scene.terrain.terrain_types == 5
        self.assertEqual(flat.sum().item(), 1)
        env.scene["robot"].data.root_link_pos_w[flat, 0] = env._elevation["exit_x"][flat] + .6
        env.scene["robot"].data.root_link_lin_vel_b[flat, 0] = .35
        env._elevation["support_count"][flat] = 4
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            for tick in range(1, 27):
                env.common_step_counter = tick
                env.episode_length_buf[:] = tick
                state = update_elevation_state(env)
        self.assertTrue(state["complete"][flat].all())
        self.assertFalse(state["covered"][flat].any())
        course = ElevationTerrainCurriculum(env.cfg.curriculum["elevation_levels"], env)
        course(env, None)
        self.assertEqual(env.scene.terrain.terrain_levels[flat].item(), 0)
        self.assertFalse(course.windows[5])

    def test_pitch_cost_allows_stair_transition_and_ramp_alignment(self):
        env = fake_env(count=5)
        robot = env.scene["robot"].data
        robot.root_link_pos_w[0, 0] = 3.2
        robot.root_link_pos_w[2, 0] = 4.0
        pitches = torch.tensor([math.radians(12), 0., -env._elevation["angle"][2], 0., math.radians(8)])
        robot.root_link_quat_w[:, 0] = torch.cos(pitches / 2)
        robot.root_link_quat_w[:, 2] = torch.sin(pitches / 2)
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            cost = elevation_cost(env, "attitude")
        self.assertEqual(cost[0].item(), 0.)
        self.assertEqual(cost[2].item(), 0.)
        self.assertGreater(cost[4].item(), 0.)

    def test_invalid_stall_settings_are_rejected(self):
        for kwargs in ({"stall_window_s": 0}, {"stall_grace_s": -1},
                       {"stall_min_distance_m": float("nan")}, {"stall_min_progress_ratio": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ElevationSettings(**kwargs)


class ElevationTransferTests(unittest.TestCase):
    def test_reward_version_changes_allow_resume_and_preserve_training_state(self):
        env = _TransferEnv("PointLeg-Elevation-v0")
        runner = ElevationOnPolicyRunner(env, asdict(load_rl_cfg("PointLeg-Elevation-v0")), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            runner.current_learning_iteration = 2499
            env.common_step_counter = 48000
            env._elevation_curriculum.level_caps[:] = 2
            runner.save(str(source))
            saved = torch.load(source, weights_only=False)
            saved["infos"]["elevation_walking"]["revision"] = 1
            saved["infos"]["elevation_walking"].pop("gait_style")
            torch.save(saved, source)
            runner.current_learning_iteration = 0
            env.common_step_counter = 0
            env._elevation_curriculum.level_caps[:] = 0
            with self.assertWarnsRegex(UserWarning, "Resuming network, optimizer, iteration and curriculum"):
                runner.load(str(source))
            self.assertEqual(runner.current_learning_iteration, 2499)
            self.assertEqual(env.common_step_counter, 48000)
            self.assertEqual(env._elevation_curriculum.level_caps.tolist(), [0, 0, 2, 2, 2, 2])
            saved["infos"].pop("elevation_walking")
            torch.save(saved, source)
            with self.assertWarns(UserWarning):
                runner.load(str(source))
            runner.load(str(source), load_cfg={"actor": True, "critic": True})

    def test_resume_still_rejects_a_different_curriculum_structure(self):
        env = _TransferEnv("PointLeg-Elevation-v0")
        runner = ElevationOnPolicyRunner(env, asdict(load_rl_cfg("PointLeg-Elevation-v0")), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            runner.save(str(source))
            saved = torch.load(source, weights_only=False)
            saved["infos"]["elevation_curriculum"]["level_caps"].pop()
            torch.save(saved, source)
            with self.assertRaisesRegex(ValueError, "Curriculum structure changed"):
                runner.load(str(source))

    def test_resume_rejects_old_or_changed_stair_layout_but_allows_weight_transfer(self):
        env = _TransferEnv("PointLeg-Elevation-v0")
        runner = ElevationOnPolicyRunner(env, asdict(load_rl_cfg("PointLeg-Elevation-v0")), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            runner.save(str(source))
            saved = torch.load(source, weights_only=False)
            saved["infos"].pop("elevation_layout")
            torch.save(saved, source)
            with self.assertRaisesRegex(ValueError, "Stair geometry changed"):
                runner.load(str(source))
            runner.load(str(source), load_cfg={"actor": True, "critic": True})
            runner.save(str(source))
            saved = torch.load(source, weights_only=False)
            for layout in saved["infos"]["elevation_layout"].values():
                layout.pop("fixed_step_height_m")
                layout["revision"] = 2
            torch.save(saved, source)
            with self.assertRaisesRegex(ValueError, "Stair geometry changed"):
                runner.load(str(source))
            runner.load(str(source), load_cfg={"actor": True, "critic": True})
            runner.save(str(source))
            env.cfg.scene.terrain.terrain_generator.sub_terrains["stairs_up"].stair_platform_range = (1.5, 2.0)
            with self.assertRaisesRegex(ValueError, "Stair geometry changed"):
                runner.load(str(source))

    def test_weights_only_transfer_resets_optimizer_counts_and_curriculum(self):
        env = _TransferEnv("PointLeg-Elevation-v0")
        runner = ElevationOnPolicyRunner(env, asdict(load_rl_cfg("PointLeg-Elevation-v0")), device="cpu")
        runner.current_learning_iteration = 321
        env.common_step_counter = 999
        env._elevation_curriculum.level_caps[:] = 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            runner.save(str(source))
            output = prepare_transfer("PointLeg-Elevation-v0", source, output_root=root)
            old = torch.load(source, weights_only=False)
            new = torch.load(output, weights_only=False)
            for key in ("actor_state_dict", "critic_state_dict"):
                for name in old[key]:
                    torch.testing.assert_close(new[key][name], old[key][name])
            self.assertEqual(new["iter"], 0)
            self.assertEqual(new["infos"]["env_state"]["common_step_counter"], 0)
            self.assertEqual(new["infos"]["elevation_curriculum"]["level_caps"], [0] * 6)
            self.assertEqual(new["optimizer_state_dict"]["state"], {})
            with self.assertRaises(FileExistsError):
                prepare_transfer("PointLeg-Elevation-v0", source, output_root=root)

    def test_actor_only_load_preserves_runtime_curriculum_and_counter(self):
        env = _TransferEnv("PointLeg-Elevation-v0")
        runner = ElevationOnPolicyRunner(env, asdict(load_rl_cfg("PointLeg-Elevation-v0")), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            runner.save(str(source))
            env.common_step_counter = 123
            env._elevation_curriculum.level_caps[:] = 2
            runner.load(str(source), load_cfg={"actor": True})
            self.assertEqual(env.common_step_counter, 123)
            self.assertEqual(env._elevation_curriculum.level_caps.tolist(), [2] * 6)
            runner.load(str(source))
            self.assertEqual(env.common_step_counter, 0)
            self.assertEqual(env._elevation_curriculum.level_caps.tolist(), [0] * 6)

    def test_retired_flat_warmup_is_rejected(self):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        out = ElevationTerrainCfg(kind="wall_sequence", size=TILE_SIZE, wall_height_override=0).function(1, spec, np.random.default_rng(42))
        self.assertEqual(len(out.geometries), 1)
        with self.assertRaisesRegex(ValueError, "retired"):
            _TransferEnv("PointLeg-Jump-v0", warmup=True)


if __name__ == "__main__":
    unittest.main()
