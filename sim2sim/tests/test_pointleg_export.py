"""Verify strict point-foot export, normalization, and failure handling."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.models import MLPModel
from sim2sim.export_onnx import export_policy, main
from sim2sim.pointleg_sim2sim import load_policy
from pointleg.mdp.fixed_stone_gait import FIXED_GAIT_INFO_KEY
from pointleg.mdp.stepping_stones import SUPPORT_PROGRESS_INFO_KEY
from sim2sim.pointleg_sim2sim import STEPPING_TASK, stepping_policy_observation_dim


TASK = "PointLeg-Flat-v0"


def make_actor(cfg, obs_dim=45, action_dim=12):
    actor_cfg = asdict(cfg.actor)
    for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        actor_cfg.pop(key, None)
    return MLPModel(
        TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1]),
        {"actor": ["actor"]}, "actor", action_dim, **actor_cfg,
    ).eval()


class PointLegExportTests(unittest.TestCase):
    def test_stepping_export_preserves_legacy_and_fixed_gait_interfaces(self):
        import onnxruntime as ort

        for width in (105, 123):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "stepping.pt"
                actor = make_actor(load_rl_cfg(STEPPING_TASK), obs_dim=width)
                torch.save({"actor_state_dict": actor.state_dict(),
                            "infos": {SUPPORT_PROGRESS_INFO_KEY: True}}, checkpoint)
                output, max_diff = export_policy(checkpoint, STEPPING_TASK)
                self.assertLess(max_diff, 1e-5)
                self.assertEqual(stepping_policy_observation_dim(output), width)
                policy = load_policy(output, STEPPING_TASK, width)
                self.assertEqual(policy.fixed_gait, width == 123)
                self.assertTrue(policy.use_support_progress)
                session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
                self.assertEqual(session.get_modelmeta().custom_metadata_map[FIXED_GAIT_INFO_KEY],
                                 "true" if width == 123 else "false")

    def test_roundtrip_native_actor_with_and_without_normalization(self):
        for normalized in (False, True):
            with self.subTest(normalized=normalized), tempfile.TemporaryDirectory() as directory:
                cfg = load_rl_cfg(TASK)
                cfg.actor.obs_normalization = normalized
                actor = make_actor(cfg)
                if normalized:
                    actor.obs_normalizer._mean.fill_(0.3)
                    actor.obs_normalizer._std.fill_(1.7)
                    actor.obs_normalizer._var.fill_(1.7 ** 2)
                checkpoint = Path(directory) / "model.pt"
                torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
                with patch("sim2sim.pointleg_sim2sim.load_rl_cfg", return_value=cfg):
                    output, max_diff = export_policy(checkpoint, TASK)
                self.assertEqual(output, checkpoint.with_suffix(".onnx"))
                self.assertLess(max_diff, 1e-5)
                policy = load_policy(output, TASK)
                obs = torch.randn(7, 45)
                with torch.inference_mode():
                    expected = actor(TensorDict({"actor": obs}, batch_size=[7]))
                    torch.testing.assert_close(policy(obs), expected, atol=1e-6, rtol=1e-5)

    def test_old_checkpoint_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old.pt"
            actor = make_actor(load_rl_cfg(TASK), obs_dim=53, action_dim=16)
            torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
            output = checkpoint.with_suffix(".onnx")
            output.write_bytes(b"existing output")
            with self.assertRaisesRegex(ValueError, "45 observations, 12 actions"):
                export_policy(checkpoint, TASK)
            self.assertEqual(output.read_bytes(), b"existing output")

    def test_failed_verification_does_not_publish_export(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"actor_state_dict": make_actor(load_rl_cfg(TASK)).state_dict()}, checkpoint)
            output = Path(directory) / "policy.onnx"
            output.write_bytes(b"existing output")
            with patch("sim2sim.export_onnx.np.testing.assert_allclose", side_effect=AssertionError("mismatch")):
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    export_policy(checkpoint, TASK, output)
            self.assertEqual(output.read_bytes(), b"existing output")
            self.assertFalse(any(path.is_dir() for path in Path(directory).iterdir()))

    def test_task_and_checkpoint_are_required(self):
        for arguments in (["export_onnx.py"], ["export_onnx.py", "--task", TASK]):
            with self.subTest(arguments=arguments), patch("sys.argv", arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_cli_pt_path_alias_and_custom_output(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"actor_state_dict": make_actor(load_rl_cfg(TASK)).state_dict()}, checkpoint)
            output = Path(directory) / "exported" / "policy.onnx"
            arguments = ["export_onnx.py", "--task", TASK, "--pt-path", str(checkpoint), "--output", str(output)]
            with patch("sys.argv", arguments), redirect_stdout(io.StringIO()) as messages:
                main()
            self.assertTrue(output.is_file())
            self.assertIn("Exported and verified", messages.getvalue())

    def test_nonfinite_actor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            actor = make_actor(load_rl_cfg(TASK))
            with torch.no_grad():
                actor.mlp[0].weight.fill_(float("nan"))
            torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "finite"):
                export_policy(checkpoint, TASK)
            self.assertFalse(checkpoint.with_suffix(".onnx").exists())


if __name__ == "__main__":
    unittest.main()
