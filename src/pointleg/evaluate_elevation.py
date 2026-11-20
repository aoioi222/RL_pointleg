"""Evaluate fixed-height stair variants and graded slopes for elevation walking."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from .config.evaluation import configure_evaluation
from .evaluate import EVALUATION_SEEDS, _split_episode_count
from .terrains.elevation import ELEVATION_TASK_TERRAINS, FLAT_NAME, WALL_NAME, walking_terrain_names
from .robot_cfg import LEG_NAMES


@dataclass(frozen=True)
class ElevationEvaluateConfig:
    checkpoint_file: str
    profile: str = "nominal"
    episodes_per_case: int = 100
    output: str = "elevation_evaluation.json"
    device: str | None = None
    forward_speed: float = 0.20
    wall_index: int = 0
    include_target_wall: bool = True
    episode_length_s: float | None = None


REASONS = {0: "incomplete", 1: "outside_lane", 2: "missed_tread", 3: "invalid_wall_crossing",
           4: "fall", 5: "joint_limit", 6: "wrong_route", 7: "deadline", 8: "invalid_simulation", 9: "stalled"}


def functional_episode_metrics(env, index):
    """Read event-weighted diagnostics before the finished environment is reset."""
    gait = getattr(env, "_elevation_gait_term", None)
    body = getattr(env, "_elevation_body_clearance_term", None)
    touchdown = getattr(env, "_elevation_touchdown_term", None)
    result = {"feet": {}}
    for i, leg in enumerate(LEG_NAMES):
        foot = {}
        if gait is not None:
            count = float(gait.leg_swing_count[index, i])
            foot.update(swing_count=count, true_lift_sum_m=float(gait.leg_lift_sum[index, i]),
                        mean_true_lift_m=float(gait.leg_lift_sum[index, i]) / count if count else None)
        if touchdown is not None:
            count = float(touchdown.count[index, i])
            foot.update(touchdown_count=count, touchdown_speed_sum_m_s=float(touchdown.speed_sum[index, i]),
                        mean_touchdown_speed_m_s=float(touchdown.speed_sum[index, i]) / count if count else None,
                        max_touchdown_speed_m_s=float(touchdown.maximum[index, i]) if count else None)
        result["feet"][leg] = foot
    if body is not None:
        result["minimum_body_clearance_m"] = float(body.episode_minimum[index]) if body.samples[index] > 0 else None
    if gait is not None:
        count = float(gait.landing_count[index])
        result.update(tread_landing_count=count, tread_margin_sum_m=float(gait.landing_margin_sum[index]),
                      edge_landing_count=float(gait.edge_landings[index]),
                      mean_tread_edge_margin_m=float(gait.landing_margin_sum[index]) / count if count else None,
                      edge_landing_fraction=float(gait.edge_landings[index]) / count if count else None)
    return result


def summarize_functional_metrics(episodes):
    """Combine events rather than averaging episodes with different step counts."""
    records = [episode.get("functional_posture", {}) for episode in episodes]
    result = {"feet": {}}
    for leg in LEG_NAMES:
        feet = [record.get("feet", {}).get(leg, {}) for record in records]
        swings = sum(foot.get("swing_count", 0.) for foot in feet)
        touches = sum(foot.get("touchdown_count", 0.) for foot in feet)
        speeds = [foot["max_touchdown_speed_m_s"] for foot in feet if foot.get("max_touchdown_speed_m_s") is not None]
        result["feet"][leg] = {
            "swing_count": swings,
            "mean_true_lift_m": sum(foot.get("true_lift_sum_m", 0.) for foot in feet) / swings if swings else None,
            "touchdown_count": touches,
            "mean_touchdown_speed_m_s": sum(foot.get("touchdown_speed_sum_m_s", 0.) for foot in feet) / touches if touches else None,
            "max_touchdown_speed_m_s": max(speeds) if speeds else None,
        }
    minimums = [record["minimum_body_clearance_m"] for record in records if record.get("minimum_body_clearance_m") is not None]
    landings = sum(record.get("tread_landing_count", 0.) for record in records)
    result.update(minimum_body_clearance_m=min(minimums) if minimums else None,
                  tread_landing_count=landings,
                  mean_tread_edge_margin_m=sum(record.get("tread_margin_sum_m", 0.) for record in records) / landings if landings else None,
                  edge_landing_fraction=sum(record.get("edge_landing_count", 0.) for record in records) / landings if landings else None)
    return result


def evaluation_cases(task_id, include_target_wall=True):
    if task_id not in ELEVATION_TASK_TERRAINS:
        raise ValueError("Expected Elevation or a walking specialization; platform Jump uses pointleg.evaluate_jump")
    terrain_kind = ELEVATION_TASK_TERRAINS[task_id]
    names = walking_terrain_names(terrain_kind)
    cases = [{"terrain_type": name, "terrain_level": level, "wall_height_override_m": None,
              "fixed_step_height_m": (.06 + .015 * level if terrain_kind == "stairs" else .12)
              if name.startswith("stairs") else None}
             for name in names for level in (range(1) if name == FLAT_NAME else range(5))]
    return cases


def _run_batch(task_id, checkpoint, options, seed, cases, repetitions, device):
    cfg = load_env_cfg(task_id)
    episode_length_s = cfg.episode_length_s
    cfg = configure_evaluation(cfg, options.profile)
    cfg.episode_length_s = episode_length_s
    if options.episode_length_s is not None:
        cfg.episode_length_s = options.episode_length_s
    cfg.auto_reset = False
    cfg.seed = seed
    cfg.scene.num_envs = repetitions * len(cases)
    cfg.scene.terrain.num_envs = cfg.scene.num_envs
    cfg.scene.terrain.terrain_generator.seed = seed
    cfg.scene.terrain.terrain_generator.sub_terrains[WALL_NAME].wall_height_override = cases[0]["wall_height_override_m"]
    raw = ManagerBasedRlEnv(cfg, device=device)
    raw._elevation_fixed_cases = True
    raw._elevation_wall_index = options.wall_index
    raw.scene.terrain.terrain_types[:] = torch.tensor(
        [list(cfg.scene.terrain.terrain_generator.sub_terrains).index(case["terrain_type"]) for case in cases], device=device).repeat_interleave(repetitions)
    raw.scene.terrain.terrain_levels[:] = torch.tensor(
        [case["terrain_level"] for case in cases], device=device).repeat_interleave(repetitions)
    agent = load_rl_cfg(task_id)
    wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        raw._elevation["speed"][:] = options.forward_speed
        raw.command_manager.get_term("twist")._apply_forward()
        raw.obs_buf = raw.observation_manager.compute(update_history=False)
        runner = load_runner_cls(task_id)(wrapped, asdict(agent), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        count = raw.num_envs
        finished = torch.zeros(count, dtype=torch.bool, device=device)
        results = [None] * count
        start = raw.scene["robot"].data.root_link_pos_w[:, 0].clone()
        approach = raw._elevation["entry_x"] - raw._elevation["start_x"]
        slip = torch.zeros(count, device=device)
        energy = torch.zeros_like(slip)
        samples = torch.zeros_like(slip)
        for _ in range(raw.max_episode_length + 1):
            active = ~finished
            if not bool(active.any()):
                break
            with torch.no_grad():
                obs = wrapped.get_observations()
                _, _, done, _ = wrapped.step(policy(obs))
            state = raw._elevation
            robot = raw.scene["robot"].data
            slip[active] += state["slip"][active]
            power = (robot.qfrc_actuator * robot.joint_vel).abs().sum(dim=1)
            energy[active] += power[active] * raw.step_dt
            samples[active] += 1
            for index in ((done > 0) & active).nonzero(as_tuple=False).flatten().cpu().tolist():
                complete = bool(state["complete"][index]) and not bool(state["failed"][index])
                results[index] = {
                    "success": complete, "reason": "complete" if complete else REASONS[int(state["reason"][index])],
                    "seed": seed, "elapsed_s": float(state["elapsed"][index]),
                    "geometry_level": int(state["geometry_level"][index]),
                    "progress_m": float(robot.root_link_pos_w[index, 0] - start[index]),
                    "mean_normalized_slip": float(slip[index] / samples[index].clamp_min(1)),
                    "mechanical_energy_j": float(energy[index]),
                    "covered_treads": int(state["covered"][index, :3].sum()) if task_id == "PointLeg-Stairs-v0" else int(state["covered"][index].sum()),
                    "initial_approach_distance_m": float(approach[index]),
                    "stair_platform_length_m": float(state["stair_platform_length"][index]),
                    "airborne": bool(state["airborne"][index]), "wall_crossed": bool(state["crossed"][index]),
                    "landed": bool(state["landed"][index]),
                    "functional_posture": functional_episode_metrics(raw, index),
                }
                finished[index] = True
            reset_ids = (done > 0).nonzero(as_tuple=False).flatten()
            if len(reset_ids):
                raw.reset(env_ids=reset_ids)
                raw._elevation["speed"][reset_ids] = options.forward_speed
        if not all(result is not None for result in results):
            raise RuntimeError("Evaluation did not produce a terminal outcome for every case")
        return [results[i * repetitions:(i + 1) * repetitions] for i in range(len(cases))]
    finally:
        wrapped.close()


def run_evaluation(task_id, options):
    if task_id not in ELEVATION_TASK_TERRAINS:
        raise ValueError("Expected Elevation or a walking specialization; platform Jump uses pointleg.evaluate_jump")
    if options.profile not in ("nominal", "robust") or options.episodes_per_case < 1:
        raise ValueError("Choose nominal/robust and a positive episode count")
    if not math.isfinite(options.forward_speed) or options.forward_speed <= 0 or options.wall_index not in (0, 1, 2):
        raise ValueError("Forward speed must be positive and wall index must lie in [0, 2]")
    if options.episode_length_s is not None and (not math.isfinite(options.episode_length_s) or options.episode_length_s <= 0):
        raise ValueError("episode_length_s must be positive")
    checkpoint = Path(options.checkpoint_file).resolve(strict=True)
    device = options.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = evaluation_cases(task_id, options.include_target_wall)
    outcomes = [[] for _ in cases]
    for seed, count in zip(EVALUATION_SEEDS, _split_episode_count(options.episodes_per_case, len(EVALUATION_SEEDS))):
        if not count:
            continue
        for override in (None, 0.330):
            indices = [i for i, case in enumerate(cases) if case["wall_height_override_m"] == override]
            if not indices:
                continue
            batch = _run_batch(task_id, checkpoint, options, seed, [cases[i] for i in indices], count, device)
            for index, values in zip(indices, batch):
                outcomes[index].extend(values)
    threshold = 0.9 if options.profile == "nominal" else 0.8
    reports = []
    for case, episodes in zip(cases, outcomes):
        rate = sum(item["success"] for item in episodes) / len(episodes)
        reports.append({**case, "attempts": len(episodes), "success_rate": rate,
                        "accepted": rate >= threshold, "functional_posture_summary": summarize_functional_metrics(episodes),
                        "episodes": episodes})
    report = {
        "task_id": task_id, "profile": options.profile, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "actor_observations": 45, "critic_observations": 325, "actions": 12,
        "command": (options.forward_speed, 0.0, 0.0), "wall_index": options.wall_index,
        "required_success_rate": threshold, "accepted": all(case["accepted"] for case in reports),
        "episode_length_s_override": options.episode_length_s,
        "scope": "single-skill; multi-policy handoff and three-wall route need separate integration validation",
        "cases": reports,
    }
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(task_id, arguments):
    cfg = tyro.cli(ElevationEvaluateConfig, args=arguments, prog=f"pointleg evaluate {task_id}")
    run_evaluation(task_id, cfg)
    print(Path(cfg.output).resolve())
