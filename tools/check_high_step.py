"""Run a bounded HighStep physics and optional pretrained-policy compatibility check."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
import warp

import pointleg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-file")
    parser.add_argument("--output", default="artifacts/high_step_check.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    root = Path(__file__).resolve().parents[1]
    warp.config.kernel_cache_dir = str(root / ".venv" / "warp-diagnostics-cache")
    cfg = load_env_cfg("PointLeg-HighStep-v0", play=True)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device=args.device)
    source_hash = None
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (4, 45)
        assert obs["critic"].shape == (4, 325)
        policy = None
        wrapped = RslRlVecEnvWrapper(env)
        if args.checkpoint_file:
            source = Path(args.checkpoint_file).resolve(strict=True)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            agent = load_rl_cfg("PointLeg-HighStep-v0")
            runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
            runner.load(str(source), load_cfg={"actor": True, "critic": True}, map_location=args.device)
            policy = runner.get_inference_policy()
        supported = False
        reward_sum = 0.0
        sample_counts = {key: 0 for key in ("peak_height_m", "cycle_s", "step_length_m")}
        with torch.inference_mode():
            for _ in range(args.steps):
                command = env.command_manager.get_term("twist")
                command.is_standing_env[:] = False
                command.vel_command_b[:] = torch.tensor([0.2, 0.0, 0.0], device=env.device)
                action = (policy(wrapped.get_observations()) if policy is not None
                          else torch.zeros(4, 12, device=env.device))
                obs, reward, _, _, extras = env.step(action)
                assert torch.isfinite(action).all() and torch.isfinite(reward).all()
                assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
                assert torch.isfinite(env.reward_manager._step_reward).all()
                supported |= bool((env.scene["feet_ground_contact"].data.found.sum(dim=1) >= 2).any())
                reward_sum += reward.mean().item()
                for key in sample_counts:
                    values = extras["log"].get(f"Metrics/high_step_{key}")
                    if values is not None:
                        assert torch.isfinite(values).all()
                        sample_counts[key] += values.numel()
        if args.steps >= 64:
            assert supported, "No two-foot physical support detected"
        if args.checkpoint_file:
            assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        report = {
            "task": "PointLeg-HighStep-v0", "device": args.device, "steps": args.steps,
            "actor_dim": 45, "critic_dim": 325, "action_dim": 12,
            "physical_support_detected": supported, "reward_sum_mean": reward_sum,
            "completed_cycle_samples": sample_counts, "source_sha256": source_hash,
            "policy_validation": False,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report))
    finally:
        env.close()


if __name__ == "__main__":
    main()
