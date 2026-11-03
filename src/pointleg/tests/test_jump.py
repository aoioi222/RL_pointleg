"""Physical geometry, strict contact outcomes, trigger timing and resume tests."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.entity import Entity
from pointleg.evaluate_jump import evaluation_cases
from pointleg.jump_trigger import JumpTrigger
from pointleg.mdp.jump import (
    JumpTerrainCurriculum, collect_jump_contacts, jump_state, reset_jump,
    select_jump_terrain, update_jump_state, jump_event,
)
from pointleg.rsl_rl.runners.jump_runner import JumpOnPolicyRunner
from pointleg.terrains.jump import (
    CENTER_Y, LAUNCH_EDGE, LEVEL_COUNT, TERRAIN_NAMES, TILE_SIZE,
    JumpTerrainCfg, jump_profile,
)
from pointleg.tests.test_elevation import Robot, Scene
from pointleg.transfer import _TransferEnv, prepare_transfer


def fake_jump(kind=0, level=24):
    cfg = load_env_cfg("PointLeg-Jump-v0")
    robot = Robot(1)
    robot.data.default_joint_pos = torch.zeros(1, 12)
    robot.data.default_joint_vel = torch.zeros(1, 12)
    robot.write_joint_state_to_sim = lambda *args, **kwargs: None
    feet = SimpleNamespace(cfg=SimpleNamespace(num_slots=4), data=SimpleNamespace(
        found=torch.zeros(1, 16), force=torch.zeros(1, 16, 3), pos=torch.zeros(1, 16, 3), normal=torch.zeros(1, 16, 3)))
    body = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 4), pos=torch.zeros(1, 4, 3)))
    scene = Scene(robot=robot, jump_feet=feet, jump_body=body)
    scene.terrain = SimpleNamespace(cfg=cfg.scene.terrain, terrain_types=torch.tensor([kind]),
                                    terrain_levels=torch.tensor([level]), env_origins=torch.zeros(1, 3),
                                    get_geometry_levels=lambda levels, kinds: levels,
                                    get_terrain_origins=lambda levels, kinds: torch.tensor([
                                        jump_profile(TERRAIN_NAMES[int(k)], int(l) // 5, int(l) % 5).spawn.tolist()
                                        for l, k in zip(levels, kinds)], dtype=torch.float))
    env = SimpleNamespace(cfg=cfg, scene=scene, num_envs=1, device="cpu", physics_dt=.005, step_dt=.02,
                          common_step_counter=0, episode_length_buf=torch.zeros(1, dtype=torch.long),
                          action_manager=SimpleNamespace(action=torch.zeros(1, 12), prev_action=torch.zeros(1, 12)),
                          _jump_fixed_cases=True)
    settings = cfg.events["reset_jump"].params["settings"]
    select_jump_terrain(env, None, settings)
    reset_jump(env, None, settings)
    put_contacts(env, [1.067, 1.066, .585, .584])
    robot.data.site_pos_w[:, :, 0] = torch.tensor([1.067, 1.066, .585, .584])
    robot.data.site_pos_w[:, :, 1] = CENTER_Y
    robot.data.site_pos_w[:, :, 2] = env._jump["height"][:, None]
    return env


def put_contacts(env, xs=None, z=None, count=4):
    data = env.scene["jump_feet"].data
    for value in vars(data).values():
        value.zero_()
    if xs is None:
        return
    for foot in range(count):
        slot = foot * 4
        data.found[:, slot] = 1
        data.force[:, slot, 2] = 10
        data.normal[:, slot, 2] = -1
        data.pos[:, slot, 0] = xs[foot]
        data.pos[:, slot, 1] = CENTER_Y
        data.pos[:, slot, 2] = float(env._jump["height"][0]) if z is None else z


def tick(env, substeps=4):
    for _ in range(substeps):
        collect_jump_contacts(env)
    env.common_step_counter += 1
    env.episode_length_buf += 1
    return update_jump_state(env)


def takeoff(env):
    tick(env)
    put_contacts(env)
    env.scene["robot"].data.root_link_lin_vel_w[:] = torch.tensor([1., 0., 1.])
    env.scene["robot"].data.site_pos_w[:, :, 2] += .3
    return tick(env)


def landing(env, span=.45, rear_delta=0.):
    front = env._jump["last_front"][0].max().item()
    rear = front + span
    put_contacts(env, [rear + .482, rear + .481, rear, rear + rear_delta])
    robot = env.scene["robot"].data
    robot.site_pos_w[:, :, 0] = torch.tensor([rear + .482, rear + .481, rear, rear + rear_delta])
    robot.site_pos_w[:, :, 2] = env._jump["height"][:, None]
    robot.root_link_pos_w[:, 0] = rear + .055
    robot.root_link_lin_vel_w[:] = 0
    return tick(env)


class JumpGeometryTests(unittest.TestCase):
    def test_rays_match_platform_gap_and_wall_for_every_axis_combination(self):
        for kind in TERRAIN_NAMES:
            for level in range(LEVEL_COUNT):
                with self.subTest(kind=kind, level=level):
                    profile = jump_profile(kind, level // 5, level % 5)
                    spec = mujoco.MjSpec()
                    spec.worldbody.add_body(name="terrain")
                    output = JumpTerrainCfg(kind=kind, size=TILE_SIZE).function(level / 24, spec, np.random.default_rng(42))
                    np.testing.assert_allclose(output.origin, profile.spawn)
                    model = spec.compile()
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    for x, expected in ((.8, profile.platform_height), (profile.landing_edge + .2, profile.platform_height),
                                        (profile.wall_center, profile.platform_height + profile.wall_rise if profile.wall_rise else 0.)):
                        distance = mujoco.mj_ray(model, data, np.array([x, CENTER_Y, 1.]), np.array([0., 0., -1.]),
                                                 None, 1, -1, np.array([-1], np.int32))
                        self.assertAlmostEqual(1 - distance, expected, places=6)
                    self.assertGreaterEqual(profile.gap, profile.wall_thickness)
        self.assertAlmostEqual(jump_profile(TERRAIN_NAMES[0], 0, 0).gap, .054)
        self.assertAlmostEqual(jump_profile(TERRAIN_NAMES[1], 0, 0).gap, .080)
        final = jump_profile(TERRAIN_NAMES[1], 4, 4)
        self.assertEqual((final.platform_height, final.gap, final.wall_rise, final.wall_thickness), (.2, .36, .3, .08))

    def test_task_uses_all_50_physical_cases_and_fixed_trigger_contract(self):
        cfg = load_env_cfg("PointLeg-Jump-v0")
        self.assertEqual(cfg.scene.terrain.terrain_generator.layout_shape, (10, 5))
        self.assertEqual(cfg.scene.terrain.terrain_generator.num_rows, 25)
        self.assertEqual(load_runner_cls("PointLeg-Jump-v0"), JumpOnPolicyRunner)
        self.assertNotIn("track_lin_vel", cfg.rewards)
        self.assertNotIn("support_deficit", cfg.rewards)
        self.assertEqual(len(evaluation_cases()), 50)
        self.assertEqual(len(evaluation_cases(final_only=True)), 2)
        play = load_env_cfg("PointLeg-Jump-v0", play=True)
        self.assertIn("reset_jump", play.events)
        self.assertNotIn("reset_base", play.events)
        self.assertIn("jump_distance_min", cfg.metrics)
        self.assertEqual(cfg.metrics["jump_complete"].reduce, "last")

    def test_reset_height_places_actual_ball_bottom_above_top_without_penetration(self):
        cfg = load_env_cfg("PointLeg-Jump-v0")
        entity = Entity(cfg.scene.entities["robot"])
        model = entity.spec.compile()
        data = mujoco.MjData(model)
        data.qpos[:] = model.key_qpos[0]
        data.qpos[2] += cfg.events["reset_jump"].params["sole_offset"]
        mujoco.mj_forward(model, data)
        heights = [data.geom_xpos[model.geom(name + "_collision").id, 2]
                   - model.geom_size[model.geom(name + "_collision").id, 0]
                   for name in ("fl_foot_Link", "fr_foot_Link", "rl_foot_Link", "rr_foot_Link")]
        self.assertAlmostEqual(min(heights), .0005, places=6)
        self.assertTrue(all(height >= .00049 for height in heights))


class JumpLogicTests(unittest.TestCase):
    def test_launch_flight_and_four_foot_half_second_stability_are_all_required(self):
        env = fake_jump()
        state = takeoff(env)
        self.assertTrue(state["airborne"][0])
        self.assertFalse(state["complete"][0])
        state = landing(env)
        self.assertAlmostEqual(float(state["distance_min"][0]), .45, places=5)
        for _ in range(23):
            state = tick(env)
        self.assertFalse(state["complete"][0])
        state = tick(env)
        self.assertTrue(state["complete"][0])
        self.assertFalse(state["failed"][0])

    def test_strict_range_checks_both_rear_feet_and_cannot_be_fixed_by_shuffling(self):
        for span, delta in ((.399, 0.), (.501, 0.), (.45, .06)):
            with self.subTest(span=span, delta=delta):
                env = fake_jump()
                takeoff(env)
                state = landing(env, span, delta)
                self.assertTrue(state["distance_failed"][0])
                self.assertFalse(state["complete"][0])
                state = landing(env, .45)
                self.assertTrue(state["failed"][0])

    def test_ground_wall_body_and_platform_side_contacts_latch_even_without_force(self):
        for mode in ("ground", "wall", "body", "side"):
            with self.subTest(mode=mode):
                env = fake_jump(kind=1)
                takeoff(env)
                if mode == "body":
                    env.scene["jump_body"].data.found[:, 0] = 1
                else:
                    x = 1.28 if mode in ("ground", "wall") else .8
                    z = .5 if mode == "wall" else (0. if mode == "ground" else .2)
                    put_contacts(env, [x] * 4, z=z, count=1)
                    env.scene["jump_feet"].data.force[:] = 0
                    if mode == "side":
                        env.scene["jump_feet"].data.normal[:] = 0
                collect_jump_contacts(env)
                put_contacts(env)
                env.scene["jump_body"].data.found[:] = 0
                state = tick(env)
                self.assertTrue(state["failed"][0])

    def test_grounded_crossing_cannot_replace_jump(self):
        env = fake_jump(level=0)
        tick(env)
        state = landing(env)
        self.assertTrue(state["bad_contact"][0])
        self.assertFalse(state["complete"][0])

    def test_delayed_launch_fails_and_standing_is_not_success(self):
        env = fake_jump()
        for _ in range(31):
            state = tick(env)
        self.assertTrue(state["late_launch"][0])
        self.assertTrue(state["failed"][0])

    def test_exact_distance_boundaries_and_post_landing_drift_fail(self):
        for distance in (.400, .500):
            env = fake_jump()
            takeoff(env)
            state = landing(env, distance)
            self.assertTrue(state["distance_failed"][0])
        env = fake_jump()
        takeoff(env)
        landing(env, .45)
        self.assertTrue(landing(env, .55)["distance_failed"][0])

    def test_substep_support_loss_interrupts_stability(self):
        env = fake_jump()
        takeoff(env)
        landing(env)
        for _ in range(20):
            tick(env)
        env.scene["jump_feet"].data.found[:, 12] = 0
        collect_jump_contacts(env)
        self.assertEqual(float(env._jump["stable_time"][0]), 0.)
        landing(env)
        self.assertFalse(env._jump["complete"][0])

    def test_three_feet_and_high_landing_velocity_do_not_accumulate_stability(self):
        env = fake_jump()
        takeoff(env)
        landing(env)
        env.scene["jump_feet"].data.found[:, 12] = 0
        for _ in range(26):
            state = tick(env)
        self.assertEqual(float(state["stable_time"][0]), 0.)
        landing(env)
        env.scene["robot"].data.root_link_lin_vel_w[:, 0] = .3
        self.assertEqual(float(tick(env)["stable_time"][0]), 0.)

    def test_second_jump_and_return_to_source_fail(self):
        for second in (True, False):
            env = fake_jump()
            takeoff(env)
            if second:
                landing(env)
                put_contacts(env)
            else:
                put_contacts(env, [1.067, 1.066, .585, .584])
            self.assertTrue(tick(env)["failed"][0])

    def test_single_events_and_dt_scaling_keep_event_rewards_bounded(self):
        env = fake_jump()
        takeoff(env)
        self.assertAlmostEqual(float(jump_event(env, "new_events")[0] * env.step_dt), 1.)
        self.assertEqual(float(tick(env)["new_events"][0]), 0.)
        landing(env)
        self.assertAlmostEqual(float(jump_event(env, "landing_events")[0] * env.step_dt), 1.)
        self.assertEqual(float(tick(env)["landing_events"][0]), 0.)

    def test_reset_is_fixed_and_clears_terminal_latches(self):
        env = fake_jump()
        state = takeoff(env)
        state["bad_contact"][:] = True
        reset_jump(env, None, env._jump_settings)
        first = env.scene["robot"].data.root_link_pos_w.clone()
        reset_jump(env, None, env._jump_settings)
        torch.testing.assert_close(first, env.scene["robot"].data.root_link_pos_w)
        self.assertFalse(state["airborne"].any() or state["bad_contact"].any())


class JumpCurriculumAndCheckpointTests(unittest.TestCase):
    def test_height_and_gap_unlock_only_from_their_own_frontier(self):
        env = fake_jump(level=0)
        cfg = env.cfg.curriculum["jump_levels"]
        cfg.params["window_size"] = 2
        curriculum = JumpTerrainCurriculum(cfg, env)
        env.episode_length_buf[:] = 1
        env._jump["complete"][:] = True
        for _ in range(2):
            curriculum(env, torch.tensor([0]))
        self.assertEqual(curriculum.level_caps.tolist(), [[1, 1], [0, 0]])
        env._jump["height_level"][:] = 1
        env._jump["gap_level"][:] = 0
        for _ in range(2):
            curriculum(env, torch.tensor([0]))
        self.assertEqual(curriculum.level_caps.tolist(), [[2, 1], [0, 0]])
        clone = JumpTerrainCurriculum(cfg, env)
        clone.load_state_dict(curriculum.state_dict())
        self.assertEqual(clone.state_dict(), curriculum.state_dict())

    def test_resume_roundtrip_and_old_contract_rejection_and_fresh_transfer(self):
        env = _TransferEnv("PointLeg-Jump-v0")
        runner = JumpOnPolicyRunner(env, asdict(load_rl_cfg("PointLeg-Jump-v0")), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            env._jump_curriculum.level_caps[:] = 2
            env.common_step_counter = 123
            runner.save(str(path))
            env.common_step_counter = 0
            env._jump_curriculum.level_caps[:] = 0
            runner.load(str(path))
            self.assertEqual(env.common_step_counter, 123)
            self.assertEqual(env._jump_curriculum.level_caps.tolist(), [[2, 2], [2, 2]])
            env.common_step_counter = 456
            runner.load(str(path), load_cfg={"actor": True})
            self.assertEqual(env.common_step_counter, 456)
            output = prepare_transfer("PointLeg-Jump-v0", path, output_root=directory)
            saved = torch.load(output, weights_only=False)
            self.assertEqual(saved["iter"], 0)
            self.assertEqual(saved["infos"]["jump_curriculum"]["level_caps"], [[0, 0], [0, 0]])
            old = torch.load(path, weights_only=False)
            old["infos"].pop("jump_contract")
            torch.save(old, path)
            with self.assertRaisesRegex(ValueError, "contract changed"):
                runner.load(str(path))
            with self.assertRaisesRegex(ValueError, "Inference needs"):
                runner.load(str(path), load_cfg={"actor": True})


class JumpTriggerTests(unittest.TestCase):
    def test_trigger_starts_immediately_and_repeated_signals_do_not_restart(self):
        clock = JumpTrigger()
        self.assertEqual(clock.update(False, 10.), (0., 0., 0.))
        self.assertEqual(clock.update(True, 11.), (1., 0., 0.))
        self.assertEqual(clock.update(True, 12.5), (1., .5, 0.))
        self.assertEqual(clock.update(True, 14.), (0., 0., 0.))
        self.assertEqual(clock.update(True, 15.), (0., 0., 0.))
        clock.update(False, 16.)
        self.assertEqual(clock.update(True, 17.), (1., 0., 0.))

    def test_training_command_uses_the_same_elapsed_time_as_deployment(self):
        cfg = load_env_cfg("PointLeg-Jump-v0").commands["twist"]
        env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02, episode_length_buf=torch.tensor([0]))
        command = cfg.build(env)
        clock = JumpTrigger()
        for tick_index in (0, 1, 25, 75, 149):
            env.episode_length_buf[:] = tick_index
            expected = torch.tensor([clock.update(True, tick_index * .02)])
            torch.testing.assert_close(command.command, expected)


if __name__ == "__main__":
    unittest.main()
