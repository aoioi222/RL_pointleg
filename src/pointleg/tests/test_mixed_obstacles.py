"""Check free annulus geometry, independent resets and honest crossing outcomes."""

from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.mixed_obstacles import (
    MixedObstacleVelocityCommand, obstacle_completed, obstacle_failed,
    obstacle_state, reset_mixed_obstacles, ring_push_penalty, obstacle_stalled,
)
from pointleg.robot_cfg import FOOT_SITE_NAMES
from pointleg.terrains.mixed_obstacles import (
    RING_CENTER_Y, RING_COLUMNS, RING_FIRST_X, RING_HEIGHT,
    RING_NAMES, RING_OUTER_RADIUS, RING_ROWS, RING_SECTORS, RING_SPACING,
    START_X, FENCE_X, FENCE_APPROACH_DISTANCE, RING_APPROACH_DISTANCE,
    OBSTACLE_SPEED_RANGE, parking_centers, ring_centers, ring_spec, ring_inertia,
    ring_section_dimensions,
)


class Scene(dict):
    pass


class Body:
    def __init__(self, count, index):
        default = torch.zeros(count, 13)
        default[:, 2] = 0.42 if index == 0 else 0.0825
        default[:, 3] = 1.0
        self.data = SimpleNamespace(
            default_root_state=default, root_link_pos_w=default[:, :3].clone(),
            root_link_quat_w=default[:, 3:7].clone(),
            root_link_lin_vel_w=torch.zeros(count, 3),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            site_pos_w=torch.zeros(count, 4, 3), heading_w=torch.zeros(count),
        )
        self.indexing = SimpleNamespace(root_body_id=index, geom_ids=torch.arange((index - 1) * RING_SECTORS, index * RING_SECTORS, dtype=torch.int32))
        self.written = default.clone()

    def find_sites(self, name):
        return [FOOT_SITE_NAMES.index(name)], [name]

    def write_root_state_to_sim(self, value, env_ids):
        self.written[env_ids] = value
        self.data.root_link_pos_w[env_ids] = value[:, :3]
        self.data.root_link_quat_w[env_ids] = value[:, 3:7]


def mixed_env():
    cfg = load_env_cfg("PointLeg-RoughMixed-v0")
    scene = Scene(robot=Body(5, 0))
    scene.env_origins = torch.zeros(5, 3)
    scene.terrain = SimpleNamespace(
        cfg=cfg.scene.terrain, terrain_types=torch.arange(5),
        terrain_levels=torch.tensor([0, 1, 2, 2, 9]),
    )
    for index, name in enumerate(RING_NAMES):
        scene[name] = Body(5, index + 1)
    scene["feet_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=torch.ones(5, 4)))
    scene["base_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(5, 1)))
    radial, tangent, _ = ring_section_dimensions()
    body_count = 1 + len(RING_NAMES)
    geom_count = len(RING_NAMES) * RING_SECTORS
    model = SimpleNamespace(
        body_gravcomp=torch.ones(5, body_count),
        geom_size=torch.tensor([radial, tangent, .0825]).repeat(5, geom_count, 1),
        geom_rbound=torch.zeros(5, geom_count), geom_aabb=torch.zeros(5, geom_count, 2, 3),
        body_inertia=torch.zeros(5, body_count, 3), body_iquat=torch.zeros(5, body_count, 4),
        body_ipos=torch.zeros(5, body_count, 3),
    )
    return SimpleNamespace(
        scene=scene, num_envs=5, device="cpu", step_dt=0.02, common_step_counter=0,
        sim=SimpleNamespace(model=model),
    )


class MixedObstacleTests(unittest.TestCase):
    def test_array_has_two_centered_columns_and_seven_rows_along_positive_x(self):
        self.assertEqual((RING_COLUMNS, RING_ROWS), (2, 7))
        self.assertEqual(len(RING_NAMES), 14)
        centers = ring_centers().reshape(7, 2, 3)
        np.testing.assert_allclose(centers[:, :, 0], np.repeat((2.8 + np.arange(7) * .55)[:, None], 2, axis=1))
        np.testing.assert_allclose(centers[:, :, 1], np.tile([3.725, 4.275], (7, 1)))
        np.testing.assert_allclose(centers[:, :, 1].mean(axis=1), RING_CENTER_Y)
        self.assertAlmostEqual(np.ptp(centers[:, :, 0]) + 2 * RING_OUTER_RADIUS, 3.85)
        self.assertAlmostEqual(np.ptp(centers[:, :, 1]) + 2 * RING_OUTER_RADIUS, 1.10)
        self.assertGreater(centers[:, :, 0].min() - RING_OUTER_RADIUS, START_X + .3)
        self.assertLess(centers[:, :, 0].max() + RING_OUTER_RADIUS + .60 + .5, 8.)
        parked = parking_centers()
        distances = np.linalg.norm(parked[:, None, :2] - parked[None, :, :2], axis=-1)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(distances.min(), 2 * RING_OUTER_RADIUS)

    def test_full_two_by_seven_array_compiles_as_independent_free_rings(self):
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(8., 8., .1))
        for name, center in zip(RING_NAMES, ring_centers(), strict=True):
            frame = spec.worldbody.add_frame()
            spec.attach(ring_spec(position=tuple(center)), prefix=f"{name}/", frame=frame)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual((model.nq, model.nv), (14 * 7, 14 * 6))
        self.assertEqual(model.ngeom, 14 * RING_SECTORS + 1)
        self.assertAlmostEqual(model.body_mass.sum(), 14 * 6.2)
        np.testing.assert_allclose(data.xpos[1:], ring_centers())
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(all(contact.geom1 == 0 or contact.geom2 == 0 for contact in data.contact))

    def test_row_course_and_exit_follow_the_active_last_row(self):
        env = mixed_env()
        env.scene.terrain.terrain_types[:] = 4
        row_counts = []
        for level in range(10):
            env.scene.terrain.terrain_levels[:] = level
            reset_mixed_obstacles(env, None, ring_start_x_range=(0., 0.))
            state = obstacle_state(env)
            rows = 1 + (level * (RING_ROWS - 1) // 9)
            row_counts.append(rows)
            active = state["ring_active"].reshape(5, RING_ROWS, RING_COLUMNS)
            self.assertTrue(active[:, :rows].all())
            self.assertFalse(active[:, rows:].any())
            last_index = rows * RING_COLUMNS - 1
            torch.testing.assert_close(
                state["end_x"], state["ring_initial"][:, last_index, 0] + RING_OUTER_RADIUS,
            )
            torch.testing.assert_close(
                state["entry_x"], state["ring_initial"][:, 0, 0] - RING_OUTER_RADIUS,
            )
        self.assertEqual(row_counts, [1, 1, 2, 3, 3, 4, 5, 5, 6, 7])
        slowest_start = state["entry_x"] - RING_APPROACH_DISTANCE - .30
        slowest_completion_s = (state["end_x"] + .60 - slowest_start) / OBSTACLE_SPEED_RANGE[0] + .50
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        self.assertTrue((slowest_completion_s < cfg.episode_length_s).all())
        self.assertAlmostEqual(state["end_x"][0].item(), 6.375 - START_X, places=6)

    def test_crossing_four_rows_does_not_complete_the_full_seven_row_array(self):
        env = mixed_env()
        reset_mixed_obstacles(env, None, ring_start_x_range=(0., 0.))
        state = obstacle_state(env)
        data = env.scene["robot"].data
        old_exit = RING_FIRST_X - START_X + 3 * RING_SPACING + RING_OUTER_RADIUS
        data.root_link_pos_w[4, 0] = old_exit + .7
        data.site_pos_w[4, :, 0] = old_exit + .2
        for _ in range(30):
            env.common_step_counter += 1
            self.assertFalse(obstacle_completed(env)[4])
        data.root_link_pos_w[4, 0] = state["end_x"][4] + .7
        data.site_pos_w[4, :, 0] = state["end_x"][4] + .2
        for _ in range(26):
            env.common_step_counter += 1
            complete = obstacle_completed(env)
        self.assertTrue(complete[4])

    def test_ring_lane_limits_match_two_column_width_without_changing_fence_lane(self):
        env = mixed_env()
        reset_mixed_obstacles(env, None, forward_only=True, ring_start_x_range=(0., 0.))
        state = obstacle_state(env)
        data = env.scene["robot"].data
        data.root_link_pos_w[2, 1] = .60
        data.root_link_pos_w[4, 1] = .56
        failed = obstacle_failed(env)
        self.assertFalse(failed[2])
        self.assertTrue(failed[4])
        data.root_link_pos_w[4, 0] = state["end_x"][4] + .7
        data.site_pos_w[4, :, 0] = state["end_x"][4] + .2
        data.root_link_pos_w[4, 1] = .50
        for _ in range(30):
            env.common_step_counter += 1
            self.assertFalse(obstacle_completed(env)[4])
        data.root_link_pos_w[4, 1] = .40
        for _ in range(26):
            env.common_step_counter += 1
            complete = obstacle_completed(env)
        self.assertTrue(complete[4])

    def test_annulus_has_real_hole_flat_top_mass_and_free_joint(self):
        model = ring_spec().compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(model.nq, 7)
        self.assertEqual(model.nv, 6)
        self.assertAlmostEqual(model.body_mass.sum(), 6.2)
        self.assertEqual(model.ngeom, 24)
        for angle in np.linspace(0.01, 2 * np.pi + 0.01, 48, endpoint=False):
            for radius in (0.10, 0.18, 0.23, 0.27, 0.30):
                geom = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(model, data, np.array([radius * np.cos(angle), radius * np.sin(angle), 0.5]), np.array([0., 0., -1.]), None, 1, -1, geom)
                if 0.165 < radius < 0.275:
                    self.assertAlmostEqual(distance, 0.335, places=5)
                else:
                    self.assertEqual(distance, -1.0)

    def test_ring_really_moves_under_force_instead_of_being_welded(self):
        spec = ring_spec()
        spec.body("ring").gravcomp = 0.0
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(5., 5., .1))
        model = spec.compile()
        data = mujoco.MjData(model)
        model.opt.timestep = 0.002
        for _ in range(80):
            data.xfrc_applied[1, 0] = 200.0
            mujoco.mj_step(model, data)
        self.assertGreater(data.qpos[0], 0.02)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_mix_keeps_ordinary_task_unchanged_and_preserves_interface(self):
        original = load_env_cfg("PointLeg-Rough-v0")
        mixed = load_env_cfg("PointLeg-RoughMixed-v0")
        self.assertEqual(len(original.scene.entities), 1)
        self.assertEqual(len(original.scene.terrain.terrain_generator.sub_terrains), 2)
        terrains = mixed.scene.terrain.terrain_generator.sub_terrains
        np.testing.assert_allclose([t.proportion for t in terrains.values()], [.026, .024, .182, .168, .60])
        self.assertAlmostEqual(terrains["random_rough"].proportion + terrains["low_random_bumps"].proportion, .05)
        self.assertAlmostEqual(terrains["fenced_random_rough"].proportion + terrains["fenced_random_bumps"].proportion, .35)
        self.assertAlmostEqual(sum(t.proportion for t in terrains.values()), 1.0)
        np.testing.assert_allclose(
            [t.proportion for t in original.scene.terrain.terrain_generator.sub_terrains.values()], [.52, .48],
        )
        self.assertEqual(len(mixed.scene.entities), 15)
        self.assertEqual(list(mixed.observations["actor"].terms), list(original.observations["actor"].terms))
        self.assertEqual(list(mixed.observations["critic"].terms), list(original.observations["critic"].terms))
        self.assertEqual(list(mixed.actions), list(original.actions))
        self.assertTrue(mixed.terminations["obstacle_completed"].time_out)
        self.assertLess(mixed.rewards["ring_push"].weight, 0)

    def test_fence_is_fifty_mm_thick_and_final_height_is_152_mm_above_support(self):
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        for name in ("fenced_random_rough", "fenced_random_bumps"):
            terrain = cfg.scene.terrain.terrain_generator.sub_terrains[name]
            tops = []
            for difficulty in (1.0 / 3, 2.0 / 3, 1.0):
                spec = mujoco.MjSpec()
                spec.worldbody.add_body(name="terrain")
                output = terrain.function(difficulty, spec, np.random.default_rng(42))
                wall = output.geometries[-1].geom
                self.assertAlmostEqual(2 * wall.size[0], .050)
                self.assertAlmostEqual(2 * wall.size[2], .152 * difficulty)
                self.assertAlmostEqual(wall.pos[2] - wall.size[2], output.origin[2])
                tops.append(2 * wall.size[2])
                spec.compile()
            self.assertTrue(all(a < b for a, b in zip(tops, tops[1:])))

    def test_reset_restores_tires_and_zeroes_velocities_only_in_selected_worlds(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5))
        initial = env.scene[RING_NAMES[0]].written.clone()
        self.assertTrue((env.sim.model.body_gravcomp[:4, 1:] == 1).all())
        self.assertTrue((env.sim.model.body_gravcomp[4, 1:] == 0).all())
        for name in RING_NAMES:
            env.scene[name].written[:, :3] += .25
            env.scene[name].written[:, 7:] = 3.0
        reset_mixed_obstacles(env, torch.tensor([4]))
        torch.testing.assert_close(env.scene[RING_NAMES[0]].written[4], initial[4])
        self.assertTrue((env.scene[RING_NAMES[0]].written[0, 7:] == 3.0).all())
        self.assertTrue((env.scene[RING_NAMES[0]].written[4, 7:] == 0.0).all())

    def test_fence_direction_selects_matching_spawn_and_rings_always_face_positive_x(self):
        env = mixed_env()
        torch.manual_seed(42)
        seen = set()
        for _ in range(10):
            reset_mixed_obstacles(env, torch.arange(5))
            state = obstacle_state(env)
            seen.update(state["direction"][2:4].tolist())
            expected_x = FENCE_X - START_X - state["direction"][2:4] * FENCE_APPROACH_DISTANCE
            torch.testing.assert_close(env.scene["robot"].written[2:4, 0], expected_x)
            self.assertEqual(state["direction"][4].item(), 1.0)
        self.assertEqual(seen, {-1.0, 1.0})

    def test_ring_approach_randomization_is_bounded_varied_and_seed_reproducible(self):
        env = mixed_env()
        env.scene.terrain.terrain_types[:] = 4
        env.scene.env_origins[:, 0] = torch.arange(5) * 8.0
        torch.manual_seed(151)
        offsets = []
        for _ in range(32):
            reset_mixed_obstacles(env, torch.arange(5))
            root = env.scene["robot"].written
            nominal = RING_FIRST_X - RING_OUTER_RADIUS - START_X - RING_APPROACH_DISTANCE
            offset = root[:, 0] - env.scene.env_origins[:, 0] - nominal
            offsets.append(offset.clone())
            self.assertTrue((offset.abs() <= .30001).all())
            front = obstacle_state(env)["ring_initial"][:, 0, 0] - RING_OUTER_RADIUS
            distance = front - root[:, 0]
            self.assertTrue(((distance >= .59999) & (distance <= 1.20001)).all())
            torch.testing.assert_close(root[:, 1], env.scene.env_origins[:, 1])
            torch.testing.assert_close(root[:, 2], torch.full((5,), .42))
            torch.testing.assert_close(root[:, 3:7], torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1))
            self.assertTrue((root[:, 7:] == 0.).all())
        samples = torch.cat(offsets)
        self.assertLess(samples.min().item(), -.25)
        self.assertGreater(samples.max().item(), .25)
        torch.manual_seed(73)
        reset_mixed_obstacles(env, torch.arange(5))
        first = env.scene["robot"].written.clone()
        torch.manual_seed(73)
        reset_mixed_obstacles(env, torch.arange(5))
        torch.testing.assert_close(env.scene["robot"].written, first)

    def test_ring_approach_partial_reset_keeps_layout_goals_and_other_worlds(self):
        env = mixed_env()
        env.scene.terrain.terrain_types[:] = 4
        env.scene.env_origins[:, 0] = torch.arange(5) * 8.0
        reset_mixed_obstacles(env, torch.arange(5), ring_start_x_range=(0., 0.))
        state = obstacle_state(env)
        initial_robot = env.scene["robot"].written.clone()
        initial_rings = state["ring_initial"].clone()
        entry = state["entry_x"].clone()
        end = state["end_x"].clone()
        origin = env.scene.env_origins.clone()
        selected = torch.tensor([1, 4])
        for offset in (.3, -.3):
            reset_mixed_obstacles(env, selected, ring_start_x_range=(offset, offset))
            expected = initial_robot.clone()
            expected[selected, 0] += offset
            torch.testing.assert_close(env.scene["robot"].written, expected)
            torch.testing.assert_close(state["start_x"], expected[:, 0])
            torch.testing.assert_close(state["ring_initial"], initial_rings)
            for index, name in enumerate(RING_NAMES):
                torch.testing.assert_close(env.scene[name].written[:, :3], initial_rings[:, index])
            torch.testing.assert_close(state["entry_x"], entry)
            torch.testing.assert_close(state["end_x"], end)
            torch.testing.assert_close(env.scene.env_origins, origin)

    def test_ring_approach_does_not_change_fence_or_plain_spawns(self):
        env = mixed_env()
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        self.assertEqual(cfg.events["reset_mixed_obstacles"].params["ring_start_x_range"], (-.3, .3))
        reset_mixed_obstacles(env, torch.arange(5), forward_only=True, ring_start_x_range=(0., 0.))
        initial = env.scene["robot"].written.clone()
        reset_mixed_obstacles(env, torch.arange(5), forward_only=True, ring_start_x_range=(.3, .3))
        torch.testing.assert_close(env.scene["robot"].written[:4], initial[:4])
        self.assertAlmostEqual(env.scene["robot"].written[4, 0].item(), RING_FIRST_X - RING_OUTER_RADIUS - START_X - RING_APPROACH_DISTANCE + .3)
        before = env.scene["robot"].written.clone()
        reset_mixed_obstacles(env, torch.arange(4), forward_only=True)
        torch.testing.assert_close(env.scene["robot"].written, before)

    def test_ring_reverse_failure_uses_randomized_spawn(self):
        env = mixed_env()
        for offset in (-.3, .3):
            reset_mixed_obstacles(env, torch.arange(5), ring_start_x_range=(offset, offset))
            state = obstacle_state(env)
            root = env.scene["robot"].data.root_link_pos_w
            root[4, 0] = state["start_x"][4] - .34
            env.common_step_counter += 1
            self.assertFalse(obstacle_failed(env)[4])
            root[4, 0] = state["start_x"][4] - .36
            env.common_step_counter += 1
            self.assertTrue(obstacle_failed(env)[4])

    def test_direction_override_keeps_plain_commands_and_does_not_flip_mid_episode(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5))
        state = obstacle_state(env)
        state["direction"][2] = -1.0
        command = MixedObstacleVelocityCommand.__new__(MixedObstacleVelocityCommand)
        command._env = env
        command.robot = env.scene["robot"]
        for name in ("is_standing_env", "is_forward_env", "is_backward_env", "is_lateral_env", "is_yaw_env", "is_heading_env"):
            setattr(command, name, torch.ones(5, dtype=torch.bool))
        command.vel_command_b = torch.tensor([[.2, .3, -.4]]).repeat(5, 1)
        command.vel_command_w = torch.zeros(5, 3)
        command.progress_direction_w = torch.zeros(5, 2)
        for _ in range(3):
            command._apply_obstacle_commands()
            torch.testing.assert_close(command.vel_command_b[:2], torch.tensor([[.2, .3, -.4]]).repeat(2, 1))
            self.assertLess(command.vel_command_b[2, 0].item(), 0)
            self.assertGreater(command.vel_command_b[4, 0].item(), 0)
            torch.testing.assert_close(command.vel_command_b[2:, 1:], torch.zeros(3, 2))

    def test_stuck_obstacle_episode_ends_while_plain_rough_keeps_running(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5), forward_only=True)
        for _ in range(390):
            env.common_step_counter += 1
            self.assertFalse(obstacle_stalled(env).any())
        for _ in range(15):
            env.common_step_counter += 1
            stalled = obstacle_stalled(env)
        torch.testing.assert_close(stalled, torch.tensor([False, False, True, True, True]))
        reset_mixed_obstacles(env, torch.tensor([4]))
        env.common_step_counter += 1
        self.assertFalse(obstacle_stalled(env)[4])
        self.assertTrue(obstacle_stalled(env)[2])

    def test_actual_forward_progress_restarts_stall_timer_but_bobbing_does_not(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5), forward_only=True)
        for _ in range(300):
            env.common_step_counter += 1
            obstacle_stalled(env)
        robot = env.scene["robot"].data
        robot.root_link_pos_w[2:, 0] += .04
        env.common_step_counter += 1
        self.assertFalse(obstacle_stalled(env).any())
        self.assertTrue((obstacle_state(env)["stalled_s"][2:] == 0.).all())
        for index in range(405):
            robot.root_link_pos_w[2:, 0] = obstacle_state(env)["start_x"][2:] + (.04 if index % 2 else .02)
            env.common_step_counter += 1
            stalled = obstacle_stalled(env)
        self.assertTrue(stalled[2:].all())

    def test_close_fence_spawns_have_physical_support_and_clear_space_to_wall(self):
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        for name in ("fenced_random_rough", "fenced_random_bumps"):
            terrain = cfg.scene.terrain.terrain_generator.sub_terrains[name]
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name="terrain")
            output = terrain.function(1., spec, np.random.default_rng(42))
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for start_x in (FENCE_X - FENCE_APPROACH_DISTANCE, FENCE_X + FENCE_APPROACH_DISTANCE):
                for foot_x in (-.15, .45):
                    for foot_y in (-.25, .25):
                        point = np.asarray([start_x + foot_x, 4. + foot_y, 1.])
                        hit = np.asarray([-1], dtype=np.int32)
                        distance = mujoco.mj_ray(model, data, point, np.asarray([0., 0., -1.]), None, 1, -1, hit)
                        self.assertAlmostEqual(1. - distance, output.origin[2], places=5)
                        self.assertGreater(abs(point[0] - FENCE_X), terrain.fence_thickness / 2)

    def test_body_crossing_without_all_feet_does_not_count_and_stability_is_required(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5), forward_only=True)
        state = obstacle_state(env)
        data = env.scene["robot"].data
        data.root_link_pos_w[2, 0] = state["end_x"][2] + .7
        data.site_pos_w[2, :, 0] = state["end_x"][2] + .2
        data.site_pos_w[2, 3, 0] = state["end_x"][2] - .1
        for _ in range(30):
            env.common_step_counter += 1
            self.assertFalse(obstacle_completed(env)[2])
        data.site_pos_w[2, 3, 0] += .3
        for _ in range(24):
            env.common_step_counter += 1
            self.assertFalse(obstacle_completed(env)[2])
        env.common_step_counter += 1
        self.assertTrue(obstacle_completed(env)[2])

    def test_push_cost_has_small_tolerance_and_pushing_cannot_satisfy_success(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5))
        state = obstacle_state(env)
        self.assertEqual(ring_push_penalty(env)[4].item(), 0.0)
        env.scene[RING_NAMES[0]].data.root_link_pos_w[4, 0] += .08
        env.common_step_counter += 1
        self.assertGreater(ring_push_penalty(env)[4].item(), 0.0)
        data = env.scene["robot"].data
        data.root_link_pos_w[4, 0] = state["end_x"][4] + .7
        data.site_pos_w[4, :, 0] = state["end_x"][4] + .2
        for _ in range(30):
            env.common_step_counter += 1
            self.assertFalse(obstacle_completed(env)[4])

    def test_bypassing_lane_and_moving_opposite_to_assigned_direction_fail(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5), forward_only=True)
        env.scene["robot"].data.root_link_pos_w[2, 1] = .8
        env.scene["robot"].data.root_link_pos_w[3, 0] = -.4
        failed = obstacle_failed(env)
        self.assertTrue(failed[2])
        self.assertTrue(failed[3])
        self.assertFalse(failed[0])

    def test_nominal_play_and_evaluation_keep_required_reset(self):
        from pointleg.config.evaluation import evaluation_env_cfg
        for cfg in (load_env_cfg("PointLeg-RoughMixed-v0", play=True), evaluation_env_cfg("PointLeg-RoughMixed-v0")):
            self.assertIn("reset_mixed_obstacles", cfg.events)
            self.assertEqual(cfg.scene.terrain.terrain_generator.num_cols, 5)

    def test_backward_crossing_also_requires_all_feet_and_stable_completion(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5))
        state = obstacle_state(env)
        state["direction"][2] = -1.0
        data = env.scene["robot"].data
        data.root_link_pos_w[2, 0] = state["end_x"][2] - .7
        data.site_pos_w[2, :, 0] = state["end_x"][2] - .2
        for _ in range(26):
            env.common_step_counter += 1
            complete = obstacle_completed(env)
        self.assertTrue(complete[2])

    def test_tipping_a_ring_without_translation_is_not_a_valid_success(self):
        env = mixed_env()
        reset_mixed_obstacles(env, torch.arange(5))
        ring = env.scene[RING_NAMES[0]].data
        ring.root_link_quat_w[4] = torch.tensor([.9659258, .2588190, 0., 0.])
        self.assertGreater(ring_push_penalty(env)[4].item(), 0.)
        self.assertGreater(obstacle_state(env)["peak_tilt"][4].item(), 29.)

    def test_curriculum_does_not_upgrade_obstacles_on_velocity_progress_alone(self):
        from pointleg.tests.test_rough_training import terrain_curriculum_env
        from pointleg.mdp.curriculums import rough_terrain_levels
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        for complete in (False, True):
            env = terrain_curriculum_env([3.] * 100, level=1)
            env.scene.terrain.cfg.terrain_generator.sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
            env.scene.terrain.cfg.terrain_generator.curriculum = True
            env.scene.terrain.terrain_origins = torch.zeros(10, 5, 3)
            env.scene.terrain.terrain_types[:] = 2
            env._mixed_obstacles = {"complete": torch.full((100,), complete)}
            from copy import deepcopy
            term_cfg = deepcopy(cfg.curriculum["terrain_levels"])
            term_cfg.params["initial_level_cap"] = 2
            curriculum = rough_terrain_levels(term_cfg, env)
            curriculum(env, torch.arange(100))
            expected = 2 if complete else 0
            self.assertTrue((env.scene.terrain.terrain_levels == expected).all())

    def test_completed_short_approach_obstacle_is_not_rejected_by_long_tile_target(self):
        from pointleg.tests.test_rough_training import terrain_curriculum_env
        from pointleg.mdp.curriculums import rough_terrain_levels
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        env = terrain_curriculum_env([.1] * 100, level=0)
        terrain = env.scene.terrain
        terrain.cfg.terrain_generator.sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
        terrain.cfg.terrain_generator.curriculum = True
        terrain.terrain_origins = torch.zeros(10, 5, 3)
        terrain.terrain_types[:] = 2
        env._mixed_obstacles = {"complete": torch.ones(100, dtype=torch.bool)}
        curriculum = rough_terrain_levels(cfg.curriculum["terrain_levels"], env)
        result = curriculum(env, torch.arange(100))
        self.assertEqual(result["level_cap"].item(), 1.)
        self.assertTrue((terrain.terrain_levels == 1).all())

    def test_fixed_evaluation_uses_forward_spawn_and_reports_obstacle_completion(self):
        from pointleg.evaluate import _configure_case_env, build_case_report, EpisodeResult
        cfg = _configure_case_env("PointLeg-RoughMixed-v0", "nominal", 42, 5)
        self.assertTrue(cfg.events["reset_mixed_obstacles"].params["forward_only"])
        self.assertEqual(cfg.episode_length_s, 30.)
        result = EpisodeResult(True, "obstacle_completed", .1, 4., .0, 2., .0, 42, .03, True)
        report = build_case_report("movable_ring_array", 9, [result], .8, command=(.25, 0., 0.))
        self.assertEqual(report["obstacle_completions"], 1)
        self.assertAlmostEqual(report["maximum_ring_displacement_m"], .03)
        self.assertAlmostEqual(report["command"]["lin_vel_x"], .25)

    def test_height_course_starts_at_one_third_and_reaches_full_height(self):
        cfg = load_env_cfg("PointLeg-RoughMixed-v0")
        self.assertEqual(cfg.scene.terrain.terrain_generator.difficulty_range, (1.0 / 3, 1.0))
        self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 0)
        self.assertEqual(cfg.curriculum["terrain_levels"].params["initial_level_cap"], 0)
        self.assertGreaterEqual(cfg.sim.nconmax, 2048)
        self.assertGreaterEqual(cfg.sim.njmax, 6144)
        env = mixed_env()
        env.scene.terrain.terrain_types[:] = 4
        env.scene.terrain.terrain_levels[:] = torch.tensor([0, 2, 4, 6, 9])
        reset_mixed_obstacles(env, torch.arange(5))
        heights = obstacle_state(env)["ring_height"]
        self.assertAlmostEqual(heights[0].item(), .055, places=6)
        self.assertAlmostEqual(heights[-1].item(), .165, places=6)
        self.assertTrue((heights[1:] > heights[:-1]).all())
        torch.testing.assert_close(env.sim.model.geom_size[:, 0, 2] * 2, heights)
        torch.testing.assert_close(env.sim.model.geom_aabb[:, 0, 1, 2] * 2, heights)
        torch.testing.assert_close(env.scene[RING_NAMES[0]].written[:, 2] * 2, heights)
        self.assertTrue((env.sim.model.body_inertia[:, 1, 0][1:] > env.sim.model.body_inertia[:, 1, 0][:-1]).all())
        self.assertEqual(obstacle_state(env)["ring_active"][0].sum().item(), 2)
        self.assertEqual(obstacle_state(env)["ring_active"][-1].sum().item(), 14)
        before = env.sim.model.geom_size[4].clone()
        env.scene.terrain.terrain_levels[0] = 9
        reset_mixed_obstacles(env, torch.tensor([0]))
        torch.testing.assert_close(env.sim.model.geom_size[4], before)

    def test_scaled_ring_has_flat_collision_surface_and_consistent_inertia(self):
        for height in (.055, .11, .165):
            model = ring_spec(position=(0., 0., height / 2), height=height).compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            geom = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, np.array([.23, .001, .5]), np.array([0., 0., -1.]), None, 1, -1, geom)
            self.assertAlmostEqual(.5 - distance, height, places=6)
            self.assertAlmostEqual(model.body_mass.sum(), 6.2)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.body_iquat[1])
            rotation = rotation.reshape(3, 3)
            physical_inertia = rotation @ np.diag(model.body_inertia[1]) @ rotation.T
            np.testing.assert_allclose(physical_inertia, np.diag(ring_inertia(height)), rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
