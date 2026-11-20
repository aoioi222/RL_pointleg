"""Check observation expansion and checkpoint restoration without a physical scene."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from pointleg.mdp.stepping_stones import SUPPORT_PROGRESS_INFO_KEY, SteppingStoneCurriculum
from pointleg.mdp.fixed_stone_gait import FIXED_GAIT_INFO_KEY
from pointleg.transfer import _TransferEnv, prepare_transfer


TASK = "PointLeg-SteppingStones-v0"


def make_runner(task_id, *, use_registered_runner=False):
    cfg = load_rl_cfg(task_id)
    cfg.upload_model = False
    cfg.num_steps_per_env = 1
    cfg.algorithm.num_mini_batches = 1
    env = _TransferEnv(task_id)
    runner_cls = MjlabOnPolicyRunner
    if use_registered_runner:
        env.observation_manager = SimpleNamespace(
            get_term_cfg=lambda group, term: env.cfg.observations[group].terms[term],
            reset=Mock(),
        )
        runner_cls = load_runner_cls(task_id)
    return runner_cls(env, asdict(cfg), device="cpu")


class SteppingStonesTransferTests(unittest.TestCase):
    def test_full_resume_restores_curriculum_cap_and_success_window(self):
        source = make_runner(TASK, use_registered_runner=True)
        course = SteppingStoneCurriculum(source.env.cfg.curriculum["stepping_stone_levels"], source.env)
        course.level_cap = 2
        course.window.extend([True, False, True])
        source.current_learning_iteration = 53
        source.env.common_step_counter = 1200
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "course.pt"
            source.save(str(path))
            restored = make_runner(TASK, use_registered_runner=True)
            target = SteppingStoneCurriculum(restored.env.cfg.curriculum["stepping_stone_levels"], restored.env)
            restored.load(str(path), map_location="cpu")
            self.assertEqual(target.level_cap, 2)
            self.assertEqual(list(target.window), [True, False, True])
            self.assertEqual(restored.current_learning_iteration, 53)
            self.assertEqual(restored.env.common_step_counter, 1200)

    def test_actor_only_load_preserves_current_curriculum_and_counters(self):
        runner = make_runner(TASK, use_registered_runner=True)
        course = SteppingStoneCurriculum(runner.env.cfg.curriculum["stepping_stone_levels"], runner.env)
        course.level_cap = 2
        course.window.extend([True, True])
        runner.current_learning_iteration = 53
        runner.env.common_step_counter = 1200
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "course.pt"
            runner.save(str(path))
            course.level_cap = 1
            course.window.clear()
            course.window.append(False)
            runner.current_learning_iteration = 7
            runner.env.common_step_counter = 42
            runner.load(str(path), load_cfg={"actor": True}, map_location="cpu")
            self.assertEqual(course.level_cap, 1)
            self.assertEqual(list(course.window), [False])
            self.assertEqual(runner.current_learning_iteration, 7)
            self.assertEqual(runner.env.common_step_counter, 42)

    def test_training_uses_progress_and_actor_only_load_selects_checkpoint_observations(self):
        runner = make_runner(TASK, use_registered_runner=True)
        term = runner.env.cfg.observations["actor"].terms["foot_contact"]
        self.assertTrue(term.params["support_progress"])
        self.assertFalse(load_env_cfg(TASK, play=True).observations["actor"].terms["foot_contact"].params["support_progress"])
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.pt"
            legacy = Path(directory) / "legacy.pt"
            runner.save(str(current))
            make_runner(TASK).save(str(legacy))
            runner.load(str(legacy), map_location="cpu")
            self.assertTrue(term.params["support_progress"])
            runner.load(str(legacy), load_cfg={"actor": True}, map_location="cpu")
            self.assertFalse(term.params["support_progress"])
            runner.load(str(current), load_cfg={"actor": True}, map_location="cpu")
            self.assertTrue(term.params["support_progress"])
            self.assertEqual(runner.env.observation_manager.reset.call_count, 2)
            output = prepare_transfer(TASK, current, "progress", output_root=directory)
            runner.load(str(output), load_cfg={"actor": True}, map_location="cpu")
            self.assertTrue(term.params["support_progress"])
            self.assertTrue(torch.load(output, weights_only=True)["infos"][SUPPORT_PROGRESS_INFO_KEY])

    def test_source_observation_columns_remain_a_prefix_of_the_target(self):
        source = load_env_cfg("PointLeg-Flat-v0")
        target = load_env_cfg(TASK)
        for group in ("actor", "critic"):
            original = source.observations[group].terms
            adapted = target.observations[group].terms
            self.assertEqual(list(adapted)[:len(original)], list(original))
            for name, term in original.items():
                self.assertEqual(adapted[name].scale, term.scale)

    def test_expansion_preserves_weights_and_zeros_new_columns_with_fresh_training_state(self):
        source_runner = make_runner("PointLeg-Flat-v0")
        for parameter in source_runner.alg.actor.parameters():
            parameter.grad = torch.ones_like(parameter)
        source_runner.alg.optimizer.step()
        source_runner.current_learning_iteration = 8700
        source_runner.env.common_step_counter = 123456
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            source_runner.save(str(source))
            original = torch.load(source, map_location="cpu", weights_only=False)
            self.assertTrue(original["optimizer_state_dict"]["state"])
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            output = prepare_transfer(TASK, source, "flat_pretrained", output_root=root)
            saved = torch.load(output, map_location="cpu", weights_only=False)
            for key, width, target_width in (("actor_state_dict", 45, 123), ("critic_state_dict", 325, 399)):
                self.assertEqual(saved[key]["mlp.0.weight"].shape, (512, target_width))
                for name, value in original[key].items():
                    if name == "mlp.0.weight":
                        torch.testing.assert_close(saved[key][name][:, :width], value, rtol=0, atol=0)
                        self.assertEqual(saved[key][name][:, width:].count_nonzero().item(), 0)
                    else:
                        torch.testing.assert_close(saved[key][name], value, rtol=0, atol=0)
            self.assertEqual(saved["iter"], 0)
            self.assertEqual(saved["optimizer_state_dict"]["state"], {})
            self.assertEqual(saved["infos"]["env_state"]["common_step_counter"], 0)
            metadata = json.loads((output.parent / "transfer.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_sha256"], source_hash)
            self.assertEqual((metadata["actor_added_columns"], metadata["critic_added_columns"]), (78, 74))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            resumed = make_runner(TASK)
            resumed.load(str(output), strict=True, map_location="cpu")
            self.assertEqual(resumed.current_learning_iteration, 0)
            for old_model, new_model, width, extra in (
                (source_runner.alg.actor, resumed.alg.actor, 45, 78),
                (source_runner.alg.critic, resumed.alg.critic, 325, 74),
            ):
                observation = torch.randn(8, width)
                extended = torch.cat((observation, torch.randn(8, extra)), dim=1)
                with torch.no_grad():
                    torch.testing.assert_close(old_model.mlp(observation), new_model.mlp(extended))

    def test_same_dimension_transfer_keeps_learned_task_columns(self):
        runner = make_runner(TASK)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            runner.save(str(source))
            output = prepare_transfer(TASK, source, "same_dimensions", output_root=directory)
            saved = torch.load(output, map_location="cpu", weights_only=False)
            for key, model in (("actor_state_dict", runner.alg.actor), ("critic_state_dict", runner.alg.critic)):
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(saved[key][name], value, rtol=0, atol=0)

    def test_legacy_105_column_checkpoint_expands_to_fixed_gait(self):
        runner = make_runner(TASK)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.pt"
            runner.save(str(source))
            original = torch.load(source, weights_only=False)
            for key, width in (("actor_state_dict", 105), ("critic_state_dict", 381)):
                original[key]["mlp.0.weight"] = original[key]["mlp.0.weight"][:, :width].clone()
            torch.save(original, source)
            output = prepare_transfer(TASK, source, "fixed_gait", output_root=directory)
            saved = torch.load(output, weights_only=False)
            for key, width in (("actor_state_dict", 105), ("critic_state_dict", 381)):
                weight = saved[key]["mlp.0.weight"]
                torch.testing.assert_close(weight[:, :width], original[key]["mlp.0.weight"], atol=0, rtol=0)
                self.assertEqual(weight[:, width:].count_nonzero(), 0)
                self.assertEqual(weight.shape[1] - width, 18)
            self.assertEqual(saved["infos"][FIXED_GAIT_INFO_KEY], 1)
            make_runner(TASK).load(str(output), strict=True, map_location="cpu")

    def test_unknown_input_size_is_rejected_before_creating_an_output(self):
        runner = make_runner("PointLeg-Flat-v0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            runner.save(str(source))
            saved = torch.load(source, weights_only=False)
            saved["actor_state_dict"]["mlp.0.weight"] = torch.zeros(512, 46)
            torch.save(saved, source)
            with self.assertRaisesRegex(ValueError, "45/325 or 105/381"):
                prepare_transfer(TASK, source, "invalid", output_root=root)
            self.assertFalse((root / "pointleg_stepping_stones" / "invalid").exists())

    def test_output_directory_is_never_overwritten(self):
        runner = make_runner("PointLeg-Flat-v0")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            runner.save(str(source))
            output = prepare_transfer(TASK, source, "initial", output_root=directory)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):
                prepare_transfer(TASK, source, "initial", output_root=directory)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)

    def test_requested_training_command_parses_without_launching(self):
        import mjlab
        import tyro
        from mjlab.scripts.train import TrainConfig
        args = ["--env.scene.num-envs", "512", "--agent.resume", "True",
                "--agent.load-run", "flat_8700_pretrained", "--agent.load-checkpoint", "model_0.pt",
                "--agent.max-iterations", "10000", "--agent.run-name", "flat8700_stepping",
                "--agent.logger", "tensorboard", "--agent.upload-model", "False"]
        cfg = tyro.cli(TrainConfig, args=args, default=TrainConfig.from_task(TASK), config=mjlab.TYRO_FLAGS)
        self.assertEqual(cfg.env.scene.num_envs, 512)
        self.assertEqual(cfg.agent.max_iterations, 10000)
        self.assertTrue(cfg.agent.resume)
        self.assertEqual(cfg.agent.load_run, "flat_8700_pretrained")
        self.assertEqual(cfg.agent.load_checkpoint, "model_0.pt")
        self.assertFalse(cfg.agent.upload_model)


if __name__ == "__main__":
    unittest.main()
