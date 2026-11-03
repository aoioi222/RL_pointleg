"""Regression tests for fixed stepping-stone geometry and adjudication."""

from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pointleg.mdp.stepping_stones import (
    _body_goal,
    _foot_bottom,
    collect_stepping_stones_substep,
    cover_reward,
    edge_penalty,
    goal_velocity_reward,
    landing_reward,
    lift_reward,
    mount_reward,
    reset_stepping_stones,
    route_progress_reward,
    stepping_failed,
    stepping_failure_reward,
    stepping_metric,
    success_reward,
)
from pointleg.robot_cfg import FOOT_BODY_NAMES, FOOT_SITE_NAMES
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
    def __init__(self, count):
        root = torch.zeros(count, 13)
        root[:, 2] = 0.42
        root[:, 3] = 1.0
        self.data = SimpleNamespace(
            default_root_state=root,
            root_link_pos_w=root[:, :3].clone(),
            root_link_lin_vel_b=torch.zeros(count, 3),
            root_link_lin_vel_w=torch.zeros(count, 3),
            root_link_ang_vel_w=torch.zeros(count, 3),
            site_pos_w=torch.zeros(count, 4, 3),
            site_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 4, 1),
            heading_w=torch.zeros(count),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        )
        self.written = root.clone()

    def find_sites(self, name):
        return [FOOT_SITE_NAMES.index(name)], [name]

    def write_root_state_to_sim(self, value, env_ids):
        self.written[env_ids] = value
        self.data.root_link_pos_w[env_ids] = value[:, :3]


def fake_env(count=2):
    cfg = load_env_cfg("PointLeg-SteppingStones-v0")
    scene = Scene(robot=Robot(count))
    scene.env_origins = torch.zeros(count, 3)
    scene.terrain = SimpleNamespace(
        cfg=cfg.scene.terrain,
        terrain_levels=torch.zeros(count, dtype=torch.long),
        terrain_types=torch.zeros(count, dtype=torch.long),
        env_origins=scene.env_origins,
        get_terrain_origins=lambda levels, types: torch.zeros(len(levels), 3),
    )
    scene["stepping_foot_contact"] = SimpleNamespace(
        primary_names=[f"robot/{name}" for name in FOOT_BODY_NAMES],
        data=SimpleNamespace(found=torch.zeros(count, 4), pos=torch.zeros(count, 4, 3)),
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


def swing_and_land(env, height, x=1.0, y=.2, steps=5):
    sensor = env.scene["stepping_foot_contact"].data
    sensor.found[:, 0] = 0
    env.scene["robot"].data.site_pos_w[:, 0, 2] = height
    for _ in range(steps):
        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
    sensor.found[:, 0] = 1
    sensor.pos[:, 0] = torch.tensor((x, y, .2))
    env.scene["robot"].data.site_pos_w[:, 0] = sensor.pos[:, 0]
    env.common_step_counter += 1
    collect_stepping_stones_substep(env)


class SteppingStoneGeometryTests(unittest.TestCase):
    def test_dimensions_and_progression_match_the_drawing(self):
        self.assertEqual(MAX_STONES, 8)
        self.assertAlmostEqual(STONE_SIZE, 0.20)
        self.assertAlmostEqual(PLATFORM_SIZE, 0.80)
        self.assertAlmostEqual(PLATFORM_HEIGHT, 0.20)
        np.testing.assert_allclose(stone_centers(2)[:, :2], [[1.0, .2], [1.3, -.2]])
        self.assertAlmostEqual(right_platform_start(2), 1.50)
        self.assertAlmostEqual(right_platform_start(8), 3.30)

    def test_active_blocks_compile_as_two_hundred_mm_collision_boxes(self):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        terrain = SteppingStonesTerrainCfg(size=(8.0, 8.0))
        output = terrain.function(1.0, spec, np.random.default_rng(7))
        model = spec.compile()
        self.assertEqual(len(output.geometries), 11)
        stones = output.geometries[2:10]
        self.assertEqual(len(stones), 8)
        for terrain_geom, center in zip(stones, stone_centers(8), strict=True):
            geom = terrain_geom.geom
            np.testing.assert_allclose(geom.size, [.1, .1, .1])
            np.testing.assert_allclose(geom.pos[:2], output.origin[:2] + center[:2])
            self.assertAlmostEqual(geom.pos[2] + geom.size[2], PLATFORM_HEIGHT)
        self.assertEqual(model.ngeom, 11)


class SteppingStoneLogicTests(unittest.TestCase):
    def test_config_is_forward_only_and_has_all_task_adjudicators(self):
        cfg = load_env_cfg("PointLeg-SteppingStones-v0")
        self.assertEqual(cfg.episode_length_s, 60.0)
        self.assertEqual(cfg.scene.terrain.terrain_generator.num_rows, 7)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (0.15, 0.35))
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_y, (0.0, 0.0))
        self.assertEqual(cfg.commands["twist"].ranges.ang_vel_z, (0.0, 0.0))
        self.assertIn("stepping_foot_contact", [sensor.name for sensor in cfg.scene.sensors])
        self.assertIn("stepping_failed", cfg.terminations)
        self.assertIn("stepping_succeeded", cfg.terminations)
        self.assertIn("stepping_timeout", cfg.terminations)
        self.assertIn("stepping_contact_latch", cfg.metrics)
        self.assertIn("stepping_goal_velocity", cfg.rewards)
        self.assertNotIn("stepping_forward_velocity", cfg.rewards)
        self.assertNotIn("feet_air_time", cfg.rewards)
        self.assertIn("stepping_low_clearance_fraction", cfg.metrics)
        runner = load_rl_cfg("PointLeg-SteppingStones-v0")
        self.assertEqual(runner.algorithm.gamma, 0.9995)
        self.assertEqual(runner.algorithm.learning_rate, 3.0e-4)

    def test_mount_requires_four_real_top_contacts_and_ground_ends_after_lock(self):
        env = fake_env()
        reset_stepping_stones(env, None)
        set_contacts(env, [[.4, 0.0, .2]] * 4)
        collect_stepping_stones_substep(env)
        state = env._stepping_stones
        self.assertTrue(state["strict_ground_check"].all())
        self.assertFalse(state["failure"].any())

        env.common_step_counter += 1
        set_contacts(env, [[.4, 0.0, .2], [.4, 0.0, .2], [.4, 0.0, .2], [.4, 0.0, 0.0]])
        collect_stepping_stones_substep(env)
        self.assertTrue(stepping_failed(env).all())

    def test_side_contact_and_repeated_contact_never_create_extra_coverage(self):
        env = fake_env()
        reset_stepping_stones(env, None)
        set_contacts(env, [[.4, 0.0, .2]] * 4)
        collect_stepping_stones_substep(env)

        env.common_step_counter += 1
        set_contacts(env, [[1.0, .31, .2], [.4, 0.0, .2], [.4, 0.0, .2], [.4, 0.0, .2]])
        collect_stepping_stones_substep(env)
        self.assertFalse(env._stepping_stones["visited"].any())

        env.common_step_counter += 1
        set_contacts(env, [[1.0, .2, .2], [.4, 0.0, .2], [.4, 0.0, .2], [.4, 0.0, .2]])
        collect_stepping_stones_substep(env)
        state = env._stepping_stones
        self.assertTrue(state["visited"][:, 0].all())
        first_pulse = state["cover_pulse"].clone()
        self.assertTrue((first_pulse > 0).all())

        env.common_step_counter += 1
        collect_stepping_stones_substep(env)
        self.assertTrue(state["visited"][:, 0].all())
        self.assertTrue((state["cover_pulse"] == 0).all())


class SteppingStoneRewardTests(unittest.TestCase):
    def test_center_landing_does_not_replace_clearance_quality(self):
        env = mounted_env()
        swing_and_land(env, .20)
        self.assertAlmostEqual(landing_reward(env).item() * env.step_dt, .5)
        self.assertEqual(lift_reward(env).item(), 0.0)
        self.assertEqual(stepping_metric(env, "low_clearance_fraction").item(), 1.0)

    def test_high_swing_does_not_replace_edge_landing_quality(self):
        env = mounted_env()
        swing_and_land(env, .24, x=1.099)
        self.assertAlmostEqual(lift_reward(env).item() * env.step_dt, 1.0, places=5)
        self.assertLess(landing_reward(env).item() * env.step_dt, .03)
        self.assertGreater(edge_penalty(env).item(), .8)
        self.assertEqual(stepping_metric(env, "edge_contact_fraction").item(), 1.0)

    def test_clearance_reward_saturates_and_is_independent_of_air_time(self):
        values = []
        for height, steps in ((.24, 5), (.35, 5), (.24, 20)):
            env = mounted_env()
            swing_and_land(env, height, steps=steps)
            values.append(lift_reward(env).item() * env.step_dt)
        np.testing.assert_allclose(values, [1.0, 1.0, 1.0], atol=1.e-5)

    def test_short_contact_dropout_cannot_earn_a_lift_reward(self):
        env = mounted_env()
        swing_and_land(env, .30, steps=1)
        self.assertEqual(lift_reward(env).item(), 0.0)
        self.assertTrue(env._stepping_stones["visited"][0, 0])

    def test_repeated_swing_on_visited_stone_cannot_repeat_event_rewards(self):
        env = mounted_env()
        swing_and_land(env, .24)
        self.assertGreater(lift_reward(env).item(), 0.0)
        swing_and_land(env, .30)
        for func in (cover_reward, landing_reward, lift_reward):
            self.assertEqual(func(env).item(), 0.0)

    def test_first_platform_contact_cannot_be_reused_for_mount_lift(self):
        env = fake_env(1)
        reset_stepping_stones(env, None)
        set_contacts(env, [[-.1, 0.0, 0.0]] * 4)
        collect_stepping_stones_substep(env)
        swing_and_land(env, .24, x=.4, y=0.0)
        self.assertAlmostEqual(lift_reward(env).item() * env.step_dt, .5, places=5)
        swing_and_land(env, .30, x=.4, y=0.0)
        self.assertEqual(lift_reward(env).item(), 0.0)

    def test_unobserved_takeoff_or_unsupported_swing_has_no_lift_reward(self):
        for invalid in ("unobserved", "unsupported"):
            env = mounted_env()
            if invalid == "unobserved":
                env._stepping_stones["previous_support"][:, 0] = False
            else:
                env.scene["stepping_foot_contact"].data.found[:, 1:3] = 0
            swing_and_land(env, .30)
            self.assertEqual(lift_reward(env).item(), 0.0)

    def test_ground_contact_later_in_same_control_step_removes_positive_events(self):
        env = mounted_env()
        env.common_step_counter += 1
        set_contacts(env, [[1.0, .2, .2]] + [[.4, 0.0, .2]] * 3)
        collect_stepping_stones_substep(env)
        self.assertGreater(cover_reward(env).item(), 0.0)
        env.scene["stepping_foot_contact"].data.pos[:, 3, 2] = 0.0
        collect_stepping_stones_substep(env)
        for func in (mount_reward, cover_reward, landing_reward, lift_reward, success_reward):
            self.assertEqual(func(env).item(), 0.0)
        self.assertAlmostEqual(stepping_failure_reward(env).item() * env.step_dt * -150., -150.)

    def test_external_failure_suppresses_positive_events(self):
        env = mounted_env()
        swing_and_land(env, .24)
        env.reset_terminated = torch.ones(1, dtype=torch.bool)
        for func in (mount_reward, cover_reward, landing_reward, lift_reward, success_reward):
            self.assertEqual(func(env).item(), 0.0)

    def test_real_ball_bottom_is_used_when_the_foot_rotates(self):
        env = mounted_env()
        env.scene["robot"].data.site_pos_w[:, :, 2] = .24
        env.scene["robot"].data.site_quat_w[:] = torch.tensor([0., 1., 0., 0.])
        np.testing.assert_allclose(_foot_bottom(env, env._stepping_stones)[:, :, 2].numpy(), .19, atol=1.e-6)

    def test_task_observation_size_remains_sixty_additional_fields(self):
        from pointleg.config.env_cfgs import flat_env_cfg
        env = mounted_env()
        base_terms = flat_env_cfg().observations["actor"].terms
        terms = env.cfg.observations["actor"].terms
        sizes = [term.func(env).shape[1] for key, term in terms.items() if key not in base_terms]
        self.assertEqual(sum(sizes), 60)


class SteppingStoneGoalTests(unittest.TestCase):
    def test_path_progress_stays_a_per_environment_tensor_after_reward_and_reset(self):
        env = mounted_env(2)
        self.assertEqual(route_progress_reward(env).tolist(), [0.0, 0.0])
        env.common_step_counter += 1
        env.scene["robot"].data.root_link_pos_w[:, 0] += .02
        self.assertTrue((route_progress_reward(env) > 0.0).all())
        self.assertEqual(env._stepping_stones["progress_ready"].shape, (2,))
        reset_stepping_stones(env, torch.tensor([0]))
        self.assertEqual(env._stepping_stones["progress_ready"].tolist(), [False, True])

    def test_body_targets_are_centered_between_pairs_not_on_each_foot_target(self):
        env = mounted_env()
        state = env._stepping_stones
        state["active_count"][:] = 4
        target, _ = _body_goal(env)
        np.testing.assert_allclose(target.numpy(), [[1.15, 0.0]], atol=1.e-6)
        state["visited"][:, 0] = True
        np.testing.assert_allclose(_body_goal(env)[0].numpy(), target.numpy())
        state["visited"][:, 1] = True
        np.testing.assert_allclose(_body_goal(env)[0].numpy(), [[1.75, 0.0]], atol=1.e-6)

    def test_inactive_stones_and_odd_tail_do_not_create_impossible_targets(self):
        env = mounted_env()
        state = env._stepping_stones
        state["active_count"][:] = 3
        state["visited"][:, :2] = True
        np.testing.assert_allclose(_body_goal(env)[0].numpy(), [[1.6, 0.0]], atol=1.e-6)
        state["visited"][:, 2] = True
        np.testing.assert_allclose(_body_goal(env)[0].numpy(), [[2.2, 0.0]], atol=1.e-6)
        self.assertFalse(state["success"].any())

    def test_goal_reward_is_bounded_idempotent_and_does_not_pay_for_retracing(self):
        env = mounted_env()
        robot = env.scene["robot"].data
        robot.root_link_pos_w[:, 0] = .4
        self.assertEqual(goal_velocity_reward(env).item(), 0.0)
        def move(x, speed):
            env.common_step_counter += 1
            robot.root_link_pos_w[:, 0] = x
            robot.root_link_lin_vel_w[:, 0] = speed
            return goal_velocity_reward(env).item()
        self.assertAlmostEqual(move(.41, .25), 1.0)
        self.assertAlmostEqual(goal_velocity_reward(env).item(), 1.0)
        self.assertLess(move(.4, -.5), -1.0)
        self.assertEqual(move(.41, .25), 0.0)
        self.assertEqual(move(.42, 2.0), 1.0)
        self.assertEqual(move(.42, .25), 0.0)

    def test_world_direction_is_used_instead_of_pitched_body_forward_speed(self):
        env = mounted_env()
        robot = env.scene["robot"].data
        robot.root_link_pos_w[:, 0] = .4
        goal_velocity_reward(env)
        env.common_step_counter += 1
        robot.root_link_pos_w[:, 0] += .01
        robot.root_link_lin_vel_b[:, 0] = 1.0
        robot.root_link_lin_vel_w[:, 2] = 1.0
        self.assertEqual(goal_velocity_reward(env).item(), 0.0)

    def test_goal_switch_has_no_bonus_and_settling_or_failure_has_no_speed_reward(self):
        env = mounted_env()
        robot = env.scene["robot"].data
        robot.root_link_pos_w[:, 0] = .4
        goal_velocity_reward(env)
        env._stepping_stones["visited"][:, :2] = True
        env.common_step_counter += 1
        robot.root_link_pos_w[:, 0] += .01
        robot.root_link_lin_vel_w[:, 0] = .25
        self.assertEqual(goal_velocity_reward(env).item(), 0.0)
        env._stepping_stones["stage"][:] = 3
        env.common_step_counter += 1
        robot.root_link_pos_w[:, 0] += .01
        self.assertEqual(goal_velocity_reward(env).item(), 0.0)
        env._stepping_stones["failure"][:] = True
        self.assertEqual(goal_velocity_reward(env).item(), 0.0)

    def test_partial_reset_clears_only_selected_environments_reward_history(self):
        env = mounted_env(2)
        goal_velocity_reward(env)
        env._stepping_stones["swing_landing_count"][:] = 4
        saved_goal = env._stepping_stones["goal_best_distance"][1].clone()
        reset_stepping_stones(env, torch.tensor([0]))
        self.assertTrue(torch.isinf(env._stepping_stones["goal_best_distance"][0]))
        self.assertEqual(env._stepping_stones["goal_best_distance"][1], saved_goal)
        self.assertEqual(env._stepping_stones["swing_landing_count"].tolist(), [0., 4.])


if __name__ == "__main__":
    unittest.main()
