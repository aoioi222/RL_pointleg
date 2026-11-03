"""Check all stepping-stone levels in the real simulation without training a policy."""

import argparse
import json
from pathlib import Path

import torch
import warp

import pointleg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="artifacts/stepping_stones_reward_check.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    warp.config.kernel_cache_dir = str(root / ".venv" / "warp-diagnostics-cache")
    cfg = load_env_cfg("PointLeg-SteppingStones-v0")
    cfg.scene.num_envs = 7
    cfg.curriculum = {}
    cfg.scene.terrain.max_init_terrain_level = 6
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        terrain = env.scene.terrain
        levels = torch.arange(7, device=env.device)
        terrain.terrain_levels[:] = levels
        terrain.env_origins[:] = terrain.get_terrain_origins(levels, terrain.terrain_types)
        obs, _ = env.reset()
        assert obs["actor"].shape == (7, 105)
        assert obs["critic"].shape == (7, 381)
        assert env.action_manager.total_action_dim == 12
        assert env._stepping_stones["active_count"].tolist() == list(range(2, 9))
        saw_support = False
        reward_min, reward_max = float("inf"), float("-inf")
        for _ in range(args.steps):
            obs, reward, _, _, _ = env.step(torch.zeros(7, 12, device=env.device))
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(env._stepping_stones["goal_reward_value"]).all()
            saw_support |= bool((env._stepping_stones["last_support_count"] >= 2).any())
            reward_min = min(reward_min, reward.min().item())
            reward_max = max(reward_max, reward.max().item())
        assert saw_support, "No physical foot support was detected."
        report = {
            "task": "PointLeg-SteppingStones-v0",
            "device": args.device,
            "steps": args.steps,
            "actor_dim": 105,
            "critic_dim": 381,
            "action_dim": 12,
            "stone_counts": env._stepping_stones["active_count"].tolist(),
            "physical_support_detected": saw_support,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "reward_terms": env.reward_manager.active_terms,
            "metrics": env.metrics_manager.active_terms,
            "policy_validation": False,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
