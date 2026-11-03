"""Evaluate every physical height/gap combination for both jump terrains."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from .config.evaluation import configure_evaluation
from .evaluate import EVALUATION_SEEDS, _split_episode_count
from .terrains.jump import LEVELS_PER_AXIS, TERRAIN_NAMES, jump_profile


@dataclass(frozen=True)
class JumpEvaluateConfig:
    checkpoint_file: str
    profile: str = "nominal"
    episodes_per_case: int = 100
    output: str = "jump_evaluation.json"
    device: str | None = None
    final_only: bool = False


REASONS = {0: "incomplete", 1: "non_platform_top_contact", 2: "body_contact", 3: "wall_contact",
           4: "launch_deadline", 5: "runup", 6: "jump_distance", 7: "multiple_jump",
           8: "outside_tilt_or_joint", 9: "timeout", 10: "invalid_simulation"}


def evaluation_cases(final_only=False):
    levels = (4,) if final_only else range(LEVELS_PER_AXIS)
    return [{"terrain_type": name, "height_level": h, "gap_level": g,
             "terrain_level": h * LEVELS_PER_AXIS + g,
             "platform_height_m": jump_profile(name, h, g).platform_height,
             "gap_m": jump_profile(name, h, g).gap, "wall_rise_m": jump_profile(name, h, g).wall_rise}
            for name in TERRAIN_NAMES for h in levels for g in levels]


def run_batch(checkpoint, options, cases, seed, repetitions, device):
    cfg = configure_evaluation(load_env_cfg("PointLeg-Jump-v0"), options.profile)
    cfg.episode_length_s = cfg.events["reset_jump"].params["settings"].episode_duration
    cfg.auto_reset = False
    cfg.seed = seed
    cfg.scene.num_envs = len(cases) * repetitions
    cfg.scene.terrain.terrain_generator.seed = seed
    raw = ManagerBasedRlEnv(cfg, device=device)
    try:
        raw._jump_fixed_cases = True
        raw.scene.terrain.terrain_types[:] = torch.tensor(
            [TERRAIN_NAMES.index(c["terrain_type"]) for c in cases], device=device).repeat_interleave(repetitions)
        raw.scene.terrain.terrain_levels[:] = torch.tensor(
            [c["terrain_level"] for c in cases], device=device).repeat_interleave(repetitions)
        raw.reset()
        agent = load_rl_cfg("PointLeg-Jump-v0")
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        runner = load_runner_cls("PointLeg-Jump-v0")(wrapped, asdict(agent), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        finished = torch.zeros(raw.num_envs, dtype=torch.bool, device=device)
        results = [None] * raw.num_envs
        for _ in range(raw.max_episode_length + 2):
            if bool(finished.all()):
                break
            with torch.no_grad():
                _, _, done, _ = wrapped.step(policy(wrapped.get_observations()))
            state = raw._jump
            for index in ((done > 0) & ~finished).nonzero(as_tuple=False).flatten().cpu().tolist():
                success = bool(state["complete"][index]) and not bool(state["failed"][index])
                results[index] = {"success": success, "reason": "complete" if success else REASONS[int(state["reason"][index])],
                                  "seed": seed, **{key: float(state[key][index]) for key in (
                                      "elapsed", "distance_min", "distance_max", "distance_mean", "takeoff_latency",
                                      "stable_time", "peak_clearance", "landing_speed", "height", "gap", "wall_rise")}}
                finished[index] = True
            ids = (done > 0).nonzero(as_tuple=False).flatten()
            if len(ids):
                raw.reset(env_ids=ids)
        if any(item is None for item in results):
            raise RuntimeError("Jump evaluation did not terminate every case")
        return [results[i * repetitions:(i + 1) * repetitions] for i in range(len(cases))]
    finally:
        raw.close()


def run_evaluation(options):
    if options.profile not in ("nominal", "robust") or options.episodes_per_case < 1:
        raise ValueError("Choose nominal/robust and a positive episode count")
    checkpoint = Path(options.checkpoint_file).resolve(strict=True)
    cases = evaluation_cases(options.final_only)
    results = [[] for _ in cases]
    device = options.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    for seed, count in zip(EVALUATION_SEEDS, _split_episode_count(options.episodes_per_case, len(EVALUATION_SEEDS))):
        if count:
            batch = run_batch(checkpoint, options, cases, seed, count, device)
            for dest, values in zip(results, batch):
                dest.extend(values)
    threshold = .9 if options.profile == "nominal" else .8
    reports = []
    for case, episodes in zip(cases, results):
        rate = sum(item["success"] for item in episodes) / len(episodes)
        reports.append({**case, "success_rate": rate, "accepted": rate >= threshold, "episodes": episodes})
    report = {"task_id": "PointLeg-Jump-v0", "revision": "triggered_platform_jump_v1",
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "profile": options.profile, "actor": 45, "critic": 325, "actions": 12,
              "command": "trigger, elapsed_seconds/3, zero", "required_success_rate": threshold,
              "distance_definition": "last launch front contact to first landing rear contact; all spans strictly (0.400, 0.500) m",
              "accepted": all(case["accepted"] for case in reports), "cases": reports}
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(task_id, arguments):
    if task_id != "PointLeg-Jump-v0":
        raise ValueError("Expected PointLeg-Jump-v0")
    options = tyro.cli(JumpEvaluateConfig, args=arguments, prog=f"pointleg evaluate {task_id}")
    run_evaluation(options)
    print(Path(options.output).resolve())
