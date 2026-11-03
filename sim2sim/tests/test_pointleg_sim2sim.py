"""Check the point-foot CPU interface against training data and inference."""

from dataclasses import asdict
from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_rl_cfg
from pointleg import robot_cfg
from rsl_rl.models import MLPModel
from sim2sim.pointleg_sim2sim import PointLegSimulation, load_policy


TASK = "PointLeg-Flat-v0"


def make_actor(task=TASK, obs_dim=45, action_dim=12):
    cfg = asdict(load_rl_cfg(task).actor)
    for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        cfg.pop(key, None)
    return MLPModel(
        TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1]),
        {"actor": ["actor"]}, "actor", action_dim, **cfg,
    ).eval()


class PointLegSim2SimTests(unittest.TestCase):
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
