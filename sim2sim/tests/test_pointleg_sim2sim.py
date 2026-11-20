"""Check the point-foot CPU interface against training data and inference."""

from dataclasses import asdict
from pathlib import Path
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_rl_cfg
from pointleg import robot_cfg
from pointleg.mdp.stepping_stones import EXIT_SETTLE_TIME_S, EXIT_SETTLE_TIMEOUT_S, SUPPORT_PROGRESS_INFO_KEY
from rsl_rl.models import MLPModel
from sim2sim.pointleg_sim2sim import STEPPING_DEFAULT_SPEED, STEPPING_ENTRY_SLOT, STEPPING_TASK, PointLegSimulation, load_policy, stepping_policy_observation_dim
from sim2sim.stepping_stones_replay import LEFT_FRONT, RIGHT_FRONT, RIGHT_REAR, STEPPING_TRACE_FIELDS


TASK = "PointLeg-Flat-v0"
COURSE = Path(__file__).resolve().parents[1] / "terrain" / "scene_terrain.xml"


def make_actor(task=TASK, obs_dim=45, action_dim=12):
    cfg = asdict(load_rl_cfg(task).actor)
    for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        cfg.pop(key, None)
    return MLPModel(
        TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1]),
        {"actor": ["actor"]}, "actor", action_dim, **cfg,
    ).eval()


class PointLegSim2SimTests(unittest.TestCase):
    def test_fixed_gait_policy_selects_shared_replay_targets_and_observations(self):
        sim = PointLegSimulation(STEPPING_TASK, scene=COURSE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.pt"
            torch.save({"actor_state_dict": make_actor(STEPPING_TASK, 123).state_dict(),
                        "infos": {SUPPORT_PROGRESS_INFO_KEY: True}}, path)
            self.assertEqual(stepping_policy_observation_dim(path), 123)
            policy = load_policy(path, STEPPING_TASK, 123)
            sim.configure_stepping_policy(policy)
            self.assertTrue(sim.stepping.fixed_enabled)
            self.assertEqual(sim.obs_dim, 123)
            torch.testing.assert_close(sim.observe()[0, 117:121], torch.tensor([1., 0., 0., 1.]))
            replay = sim.stepping
            positions, stones, exit_start = replay._fixed_inputs()
            torch.testing.assert_close(stones[0, :6, 0], torch.tensor([1., 1.3, 1.6, 1.9, 2.2, 2.5]))
            self.assertAlmostEqual(exit_start.item(), 2.7, places=6)
            support = np.ones(4, dtype=bool)
            surface = np.zeros(4, dtype=np.int64)
            positions[:] = torch.tensor([[[.72, .2, .2], [.72, -.2, .2], [.1, .2, .2], [.1, -.2, .2]]])
            with patch.object(replay, "_fixed_inputs", return_value=(positions, stones, exit_start)):
                for _ in range(16):
                    replay._update_fixed_gait(surface, support, .005)
            replay.refresh_observations(60.)
            torch.testing.assert_close(sim.observe()[0, 117:121], torch.tensor([1., 0., 0., 1.]))
            self.assertEqual(tuple(policy(sim.observe()).shape), (1, 12))
            sim.reset()
            observation = sim.observe()
            self.assertTrue(replay.fixed_gait.ready.all())
            self.assertEqual(replay.fixed_gait.age.item(), 0)
            self.assertEqual(replay.fixed_gait.completed.item(), 0)
            torch.testing.assert_close(observation[0, 117:121], torch.tensor([1., 0., 0., 1.]))

    def test_stepping_support_progress_is_selected_by_checkpoint(self):
        sim = PointLegSimulation(STEPPING_TASK, scene=COURSE)
        replay = sim.stepping
        dt = sim.model.opt.timestep
        actor = make_actor(STEPPING_TASK, 105)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stepping.pt"
            saved = {"actor_state_dict": actor.state_dict()}
            torch.save(saved, path)
            self.assertFalse(load_policy(path, STEPPING_TASK, 105).use_support_progress)
            saved["infos"] = {SUPPORT_PROGRESS_INFO_KEY: True}
            torch.save(saved, path)
            policy = load_policy(path, STEPPING_TASK, 105)
            sim.configure_stepping_policy(policy)
            self.assertEqual(sim.obs_dim, 105)
            self.assertFalse(replay.fixed_enabled)
            effective_support = torch.tensor([[True, True, False, False]])
            for _ in range(4):
                replay.left_front_handoff.update(
                    torch.zeros(1, 4, dtype=torch.long), effective_support, replay.handoff_active, dt,
                    traversing=replay.handoff_active,
                )
            obs = sim.observe()
            torch.testing.assert_close(obs[0, 96:100], torch.tensor([.25, .25, 0., 0.]))
            self.assertEqual(tuple(policy(obs).shape), (1, 12))
            sim.reset()
            torch.testing.assert_close(sim.observe()[0, 96:100], torch.zeros(4))

    def test_switch_to_stepping_stones_preserves_state_and_changes_observations(self):
        sim = PointLegSimulation(TASK, scene=COURSE)
        sim.mj_data.qpos[:2] = [3.00, 1.55]
        sim.mj_data.qvel[0] = .25
        mujoco.mj_forward(sim.model, sim.mj_data)
        pose = sim.mj_data.qpos.copy()
        velocity = sim.mj_data.qvel.copy()
        controls = sim.mj_data.ctrl.copy()
        self.assertEqual(tuple(sim.observe().shape), (1, 45))

        sim.switch_task(STEPPING_TASK)
        np.testing.assert_array_equal(sim.mj_data.qpos, pose)
        np.testing.assert_array_equal(sim.mj_data.qvel, velocity)
        np.testing.assert_array_equal(sim.mj_data.ctrl, controls)
        self.assertTrue(sim.in_transition)
        obs = sim.observe()[0].numpy()
        self.assertEqual(obs.size, 123)
        np.testing.assert_allclose(obs[45:48], ([3.29, 1.75, 0.20] - pose[:3]) * 0.25, atol=1e-6)
        np.testing.assert_array_equal(obs[69:77], [1, 1, 1, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(obs[77:85], 0)
        np.testing.assert_array_equal(obs[91:95], [0, 0, 1, 0])
        np.testing.assert_array_equal(obs[101:105], [1, 0, 1, 1])
        self.assertAlmostEqual(sim._stepping_episode_s, 60)

        sim.switch_task(TASK)
        np.testing.assert_array_equal(sim.mj_data.qpos, pose)
        self.assertEqual(tuple(sim.observe().shape), (1, 45))

    def test_stepping_entry_spawn_and_policy_selection_have_the_same_start_in_either_order(self):
        spawn_first = PointLegSimulation(TASK, scene=COURSE)
        spawn_first.spawn_slot = STEPPING_ENTRY_SLOT
        spawn_first.reset()
        spawn_first.mj_data.qpos[0] += .03
        spawn_first.mj_data.qpos[7] += .1
        spawn_first.mj_data.qvel[0] = .25
        spawn_first.mj_data.ctrl[:] += .05
        spawn_first.mj_data.time = .5
        spawn_first.env.action_manager.process_action(torch.full((1, 12), .2))
        spawn_first.switch_task(STEPPING_TASK)

        select_first = PointLegSimulation(TASK, scene=COURSE)
        select_first.switch_task(STEPPING_TASK)
        self.assertTrue(select_first.in_transition)
        select_first.spawn_slot = STEPPING_ENTRY_SLOT
        select_first.reset()

        for sim in (spawn_first, select_first):
            self.assertFalse(sim.in_transition)
            self.assertEqual(sim.mj_data.time, 0.)
            self.assertFalse(sim.stepping.visited.any())
            sim.command[0, 0] = STEPPING_DEFAULT_SPEED
        for name in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(spawn_first.mj_data, name), getattr(select_first.mj_data, name))
        torch.testing.assert_close(spawn_first.observe(), select_first.observe(), atol=0., rtol=0.)

        spawn_first.switch_task(TASK)
        spawn_first.mj_data.qpos[0] += .25
        spawn_first.mj_data.time = .5
        spawn_first.switch_task(STEPPING_TASK)
        self.assertTrue(spawn_first.in_transition)
        self.assertEqual(spawn_first.mj_data.time, .5)

    def test_stepping_switch_requires_course_but_preserves_flat_task(self):
        sim = PointLegSimulation(TASK)
        with self.assertRaisesRegex(ValueError, "requires a --scene"):
            sim.switch_task(STEPPING_TASK)
        self.assertEqual(sim.task, TASK)
        self.assertEqual(tuple(sim.observe().shape), (1, 45))

    def test_stepping_contact_tracks_real_stone_top_and_resets(self):
        trace = []
        sim = PointLegSimulation(STEPPING_TASK, scene=COURSE, stepping_trace=trace.append)
        replay = sim.stepping
        foot = replay.foot_geoms[RIGHT_REAR]
        stone = next(geom for geom, index in replay.stone_indices.items() if index == replay.right_stones[0])
        center = sim.mj_data.geom_xpos[foot].copy()
        target = replay.stones[replay.stone_indices[stone]] + [0, 0, sim.model.geom_size[foot, 0] - 0.002]
        sim.mj_data.qpos[:3] += target - center
        mujoco.mj_forward(sim.model, sim.mj_data)
        contact = replay._foot_contacts()[-1]
        self.assertIsNotNone(contact)
        self.assertEqual(contact[1], stone)
        self.assertLess(contact[2], -0.7)
        visited_index = 77 + replay.stone_indices[stone]
        for _ in range(3):
            replay.record_substep()
        self.assertEqual(sim.observe()[0, visited_index], 0)
        replay.record_substep()
        self.assertEqual(sim.observe()[0, visited_index], 1)
        row = dict(zip(STEPPING_TRACE_FIELDS, trace[0], strict=True))
        self.assertEqual(row["rr_target"], sim.model.geom(stone).name)
        self.assertAlmostEqual(row["rr_foot_minus_target_x_m"], 0.)
        self.assertGreaterEqual(row["rr_foot_normal_n"], 8.)
        self.assertEqual(row["rr_foot_supported"], 1)
        self.assertEqual(row["ground_feet"], "")
        sim.reset()
        self.assertEqual(sim.observe()[0, visited_index], 0)

    def test_right_front_trace_records_rearward_offset_once_at_confirmed_landing(self):
        trace = []
        sim = PointLegSimulation(STEPPING_TASK, scene=COURSE, stepping_trace=trace.append)
        replay = sim.stepping
        foot = replay.foot_geoms[RIGHT_FRONT]
        first_right = replay.right_stones[0]
        center = sim.mj_data.geom_xpos[foot].copy()
        target = replay.stones[first_right] + [-.04, 0, sim.model.geom_size[foot, 0] - .002]
        sim.mj_data.qpos[:3] += target - center
        mujoco.mj_forward(sim.model, sim.mj_data)
        for _ in range(5):
            replay.record_substep()
        rows = [dict(zip(STEPPING_TRACE_FIELDS, row, strict=True)) for row in trace]
        self.assertEqual([row["fr_stone_landing"] for row in rows], ["", "", "", "stone_1_0", ""])
        self.assertAlmostEqual(rows[3]["fr_landing_offset_x_m"], -.04)
        self.assertGreaterEqual(rows[3]["fr_foot_normal_n"], 8.)

    def test_replay_blocks_left_front_stone_coverage_and_reports_the_violation(self):
        trace = []
        sim = PointLegSimulation(STEPPING_TASK, scene=COURSE, stepping_trace=trace.append)
        replay = sim.stepping
        entry = replay.platforms[0]
        stone_id = next(geom for geom, index in replay.stone_indices.items() if index == 0)
        entry_contacts = [(0., replay.entry_id, -1., entry, foot) for foot in range(4)]
        contacts = entry_contacts.copy()
        loads = np.full(4, 10.)

        def contact_force(model, data, index, force):
            force[:] = 0.
            force[0] = loads[index]

        with (
            patch.object(replay, "_foot_contacts", side_effect=lambda: contacts),
            patch.object(replay, "_shin_load", return_value=(np.zeros(4), None)),
            patch("mujoco.mj_contactForce", side_effect=contact_force),
        ):
            for _ in range(4):
                replay.record_substep()
            contacts[LEFT_FRONT] = None
            loads[RIGHT_REAR] = 6.
            for _ in range(4):
                replay.record_substep()
            opening = dict(zip(STEPPING_TRACE_FIELDS, trace[-1], strict=True))
            self.assertEqual(opening["rr_first_step_complete"], 0)
            self.assertEqual(opening["lf_overlap_event"], 0)
            loads[RIGHT_REAR] = 10.
            for _ in range(16):
                replay.record_substep()
            on_entry = dict(zip(STEPPING_TRACE_FIELDS, trace[-1], strict=True))
            self.assertEqual(on_entry["rr_first_step_complete"], 0)
            right_stone = replay.right_stones[0]
            right_stone_id = next(geom for geom, index in replay.stone_indices.items() if index == right_stone)
            contacts[RIGHT_REAR] = (0., right_stone_id, -1., replay.stones[right_stone], RIGHT_REAR)
            for _ in range(16):
                replay.record_substep()

            loads[RIGHT_REAR] = 6.
            replay.record_substep()
            event = dict(zip(STEPPING_TRACE_FIELDS, trace[-1], strict=True))
            self.assertEqual(event["rr_first_step_complete"], 1)
            self.assertEqual(event["rr_foot_normal_n"], 6.)
            self.assertEqual(event["rr_transfer_active"], 1)
            self.assertEqual(event["lf_overlap_event"], 1)

            loads[RIGHT_REAR] = 10.
            for _ in range(17):
                replay.record_substep()
            touchdown = (0., stone_id, -1., replay.stones[0], LEFT_FRONT)
            contacts[LEFT_FRONT] = touchdown
            for _ in range(4):
                replay.record_substep()
            blocked = dict(zip(STEPPING_TRACE_FIELDS, trace[-1], strict=True))
            self.assertEqual(blocked["lf_foot_normal_n"], 10.)
            self.assertEqual(blocked["rr_transfer_active"], 0)
            self.assertEqual(blocked["lf_stone_landing"], replay.stone_names[0])
            self.assertEqual(blocked["lf_stone_landing_eligible"], 0)
            self.assertFalse(replay.visited[0])
            for _ in range(16):
                replay.record_substep()
            self.assertFalse(replay.visited[0])

            contacts[LEFT_FRONT] = None
            replay.record_substep()
            contacts[LEFT_FRONT] = touchdown
            for _ in range(4):
                replay.record_substep()
            clean = dict(zip(STEPPING_TRACE_FIELDS, trace[-1], strict=True))
            self.assertEqual(clean["lf_stone_landing_eligible"], 1)
            self.assertEqual(clean["lf_new_stone_covered"], 1)
            self.assertTrue(replay.visited[0])

    def test_stepping_replay_keeps_policy_active_until_stable_exit_or_timeout(self):
        sim = PointLegSimulation(STEPPING_TASK, scene=COURSE)
        replay = sim.stepping
        sim.mj_data.qpos[:2] = replay.platforms[1, :2]
        mujoco.mj_forward(sim.model, sim.mj_data)
        replay.visited[:replay.count] = True
        replay.exit_visited[:] = True
        on_exit = np.ones(4, dtype=bool)
        support = on_exit.copy()
        required = math.ceil(EXIT_SETTLE_TIME_S / sim.model.opt.timestep)
        for _ in range(required - 1):
            replay._record_exit_settle(on_exit, support)
        self.assertTrue(replay.settling)
        self.assertFalse(replay.finished or replay.failed)
        self.assertEqual(replay.command(.35), 0.)
        np.testing.assert_array_equal(sim.observe()[0, 91:95], [0, 0, 0, 1])

        support[0] = False
        replay._record_exit_settle(on_exit, support)
        self.assertEqual(replay.stable_substeps, 0)
        support[0] = True
        for _ in range(required):
            replay._record_exit_settle(on_exit, support)
        self.assertTrue(replay.finished)

        replay.reset()
        replay.visited[:replay.count] = True
        replay.exit_visited[:] = True
        replay.settle_substeps = math.ceil(EXIT_SETTLE_TIMEOUT_S / sim.model.opt.timestep) - 1
        replay._record_exit_settle(on_exit, np.zeros(4, dtype=bool))
        self.assertTrue(replay.failed and replay.settle_timed_out)

    def test_stepping_replay_skipped_left_front_target_does_not_count_coverage(self):
        trace = []
        sim = PointLegSimulation(STEPPING_TASK, scene=COURSE, stepping_trace=trace.append)
        replay = sim.stepping
        contacts = [(0., replay.entry_id, -1., replay.platforms[0], foot) for foot in range(4)]
        first = next(geom for geom, index in replay.stone_indices.items() if index == 0)
        skipped = next(geom for geom, index in replay.stone_indices.items() if index == 2)

        def contact_force(model, data, index, force):
            force[:] = 0.
            force[0] = 10.

        with (
            patch.object(replay, "_foot_contacts", side_effect=lambda: contacts),
            patch.object(replay, "_shin_load", return_value=(np.zeros(4), None)),
            patch("mujoco.mj_contactForce", side_effect=contact_force),
        ):
            for _ in range(4):
                replay.record_substep()
            contacts[LEFT_FRONT] = (0., skipped, -1., replay.stones[2], LEFT_FRONT)
            for _ in range(16):
                replay.record_substep()
            self.assertFalse(replay.visited[2])
            self.assertEqual(replay.left_front_handoff.last_surface[0, LEFT_FRONT].item(), 3)
            self.assertEqual(replay.left_front_handoff.left_front_target_surface(replay.left_stones).item(), 1)

            contacts[LEFT_FRONT] = (0., replay.entry_id, -1., replay.platforms[0], LEFT_FRONT)
            for _ in range(16):
                replay.record_substep()
            contacts[LEFT_FRONT] = (0., first, -1., replay.stones[0], LEFT_FRONT)
            for _ in range(16):
                replay.record_substep()
            latest = dict(zip(STEPPING_TRACE_FIELDS, trace[-1], strict=True))
            self.assertEqual(latest["rr_follow_pending"], 1)

            contacts[LEFT_FRONT] = (0., skipped, -1., replay.stones[2], LEFT_FRONT)
            for _ in range(16):
                replay.record_substep()
            self.assertFalse(replay.visited[2])

    def test_all_tasks_compile_and_use_task_default_pose(self):
        for name in ("Flat", "Rough", "Crawl"):
            with self.subTest(task=name):
                sim = PointLegSimulation(f"PointLeg-{name}-v0")
                self.assertEqual((sim.model.nq, sim.model.nv, sim.model.nu), (19, 18, 12))
                self.assertEqual(tuple(sim.observe().shape), (1, 45))
                self.assertAlmostEqual(sim.control_dt, 0.02)
                expected_limits = np.tile([-robot_cfg.EFFORT_LEG, robot_cfg.EFFORT_LEG], (12, 1))
                np.testing.assert_allclose(sim.model.actuator_forcerange, expected_limits)
                np.testing.assert_allclose(sim.model.jnt_actfrcrange[sim.robot.joint_ids], expected_limits)
                self.assertAlmostEqual(sim.mj_data.qpos[2], sim.cfg.scene.entities["robot"].init_state.pos[2])
                for _ in range(50):
                    sim.step(torch.zeros(1, 12))
                self.assertTrue(np.isfinite(sim.mj_data.qpos).all())

    def test_observation_order_scales_gravity_and_previous_raw_action(self):
        sim = PointLegSimulation(TASK)
        sim.command[:] = torch.tensor([[0.2, -0.3, 0.4]])
        sim.mj_data.qpos[3:7] = [np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0]
        sim.mj_data.qpos[sim.robot.qpos_ids] += 0.01
        sim.mj_data.qvel[sim.robot.qvel_ids] = np.arange(12)
        sim.mj_data.qvel[3:6] = [0.4, -0.8, 1.2]
        raw = torch.arange(12, dtype=torch.float32)[None] / 10
        sim.env.action_manager.process_action(raw)
        mujoco.mj_forward(sim.model, sim.mj_data)
        obs = sim.observe()[0].numpy()
        np.testing.assert_allclose(obs[:3], [0.1, -0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(obs[3:6], [0.0, -np.sqrt(0.5), -np.sqrt(0.5)], atol=1e-6)
        np.testing.assert_allclose(obs[6:9], [0.2, -0.3, 0.4], atol=1e-6)
        np.testing.assert_allclose(obs[9:21], 0.01, atol=1e-6)
        np.testing.assert_allclose(obs[21:33], np.arange(12) * 0.05, atol=1e-6)
        np.testing.assert_allclose(obs[33:45], raw[0].numpy())

    def test_filter_delay_soft_limits_and_reset(self):
        sim = PointLegSimulation(TASK, action_delay=2)
        action = torch.ones(1, 12)
        for _ in range(2):
            sim.env.action_manager.process_action(action)
            sim.env.action_manager.apply_action()
            np.testing.assert_allclose(sim.mj_data.ctrl[sim.robot.ctrl_ids], sim.robot.data.default_joint_pos[0])
        sim.env.action_manager.process_action(action)
        sim.env.action_manager.apply_action()
        alpha = 1 - np.exp(-2 * np.pi * 5 / 50)
        scale = np.tile([0.125, 0.25, 0.25], 4)
        expected = sim.robot.data.default_joint_pos[0].numpy() + alpha * scale
        np.testing.assert_allclose(sim.mj_data.ctrl[sim.robot.ctrl_ids], expected, atol=1e-6)
        for _ in range(5):
            sim.env.action_manager.process_action(torch.full((1, 12), 1e6))
        sim.env.action_manager.apply_action()
        targets = sim.mj_data.ctrl[sim.robot.ctrl_ids]
        limits = sim.robot.data.soft_joint_pos_limits[0].numpy()
        self.assertTrue((targets <= limits[:, 1] + 1e-6).all())
        self.assertTrue((targets >= limits[:, 0] - 1e-6).all())
        sim.reset()
        self.assertEqual(torch.count_nonzero(sim.env.action_manager.action).item(), 0)
        self.assertEqual(torch.count_nonzero(sim.observe()[:, 33:]).item(), 0)

    def test_checkpoint_matches_native_actor_and_runs_physics(self):
        torch.manual_seed(7)
        actor = make_actor()
        obs = torch.randn(3, 45)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
            policy = load_policy(checkpoint, TASK)
            with torch.inference_mode():
                expected = actor(TensorDict({"actor": obs}, batch_size=[3]))
                torch.testing.assert_close(policy(obs), expected)
                sim = PointLegSimulation(TASK)
                for _ in range(10):
                    sim.step(policy(sim.observe()))

    def test_old_wheel_leg_checkpoint_and_missing_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            for state in (make_actor(obs_dim=53, action_dim=16).state_dict(), {}):
                torch.save({"actor_state_dict": state}, checkpoint)
                with self.assertRaisesRegex(ValueError, "45 observations, 12 actions"):
                    load_policy(checkpoint, TASK)

    def test_nonfinite_and_wrong_size_actions_are_rejected(self):
        sim = PointLegSimulation(TASK)
        for actions in (torch.zeros(1, 16), torch.full((1, 12), float("nan"))):
            with self.assertRaises(ValueError):
                sim.step(actions)

    def test_terrain_only_scene_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "terrain.xml"
            spec = mujoco.MjSpec()
            spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
            spec.compile()
            spec.to_file(str(scene))
            sim = PointLegSimulation(TASK, scene=scene)
            self.assertEqual(sim.model.nu, 12)
            self.assertGreaterEqual(sim.model.geom("ground").id, 0)
            sim.step(torch.zeros(1, 12))

    def test_robot_scene_is_rejected(self):
        scene = Path(__file__).resolve().parents[2] / "mjcf" / "scene.xml"
        with self.assertRaisesRegex(ValueError, "terrain only"):
            PointLegSimulation(TASK, scene=scene)

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "ONNX Runtime is optional")
    def test_onnx_matches_native_actor_and_rejects_wheel_leg_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.onnx"
            actor = make_actor().as_jit().eval()
            obs = torch.randn(1, 45)
            torch.onnx.export(
                actor, obs, str(checkpoint), input_names=["obs"], output_names=["actions"],
                opset_version=14, dynamo=False,
            )
            policy = load_policy(checkpoint, TASK)
            with torch.inference_mode():
                torch.testing.assert_close(policy(obs), actor(obs), atol=1e-6, rtol=1e-5)
            old_policy = make_actor(obs_dim=53, action_dim=16).as_jit().eval()
            torch.onnx.export(
                old_policy, torch.zeros(1, 53), str(checkpoint),
                input_names=["obs"], output_names=["actions"], opset_version=14, dynamo=False,
            )
            with self.assertRaisesRegex(ValueError, "45 observations and 12 actions"):
                load_policy(checkpoint, TASK)

    def test_checkpoint_command_line_headless(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            actor = make_actor()
            torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
            root = Path(__file__).resolve().parents[2]
            result = subprocess.run(
                [sys.executable, str(root / "sim2sim" / "sim2sim.py"),
                 "--checkpoint", str(checkpoint), "--headless", "--duration", "0.1"],
                cwd=root, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("observations: 45; actions: 12", result.stdout)
            self.assertIn("N6014B-12.6 x12; supply: 48 V", result.stdout)
            self.assertIn("continuous rating unknown", result.stdout)
            self.assertIn("reference only, not an enforced speed limit", result.stdout)
            self.assertIn("no MuJoCo warnings", result.stdout)


if __name__ == "__main__":
    unittest.main()
