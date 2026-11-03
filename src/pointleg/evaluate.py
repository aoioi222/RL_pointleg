"""Reproducible checkpoint evaluation for point-foot locomotion tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from pointleg.config.evaluation import evaluation_env_cfg
from pointleg.config.mixed_obstacles import MIXED_EPISODE_LENGTH_S
from pointleg.mdp.rewards import contact_forces, joint_power


EvaluationProfile = Literal["nominal", "robust"]
EVALUATION_SEEDS = (104729, 130363, 155921, 181081, 205759)
DEFAULT_COMMAND = (0.4, 0.0, 0.0)
MIXED_EVALUATION_COMMAND = (0.25, 0.0, 0.0)
HIGH_STEP_EVALUATION_COMMAND = (0.2, 0.0, 0.0)


def evaluation_command(task_id):
    if task_id == "PointLeg-HighStep-v0":
        return HIGH_STEP_EVALUATION_COMMAND
    return MIXED_EVALUATION_COMMAND if task_id == "PointLeg-RoughMixed-v0" else DEFAULT_COMMAND


@dataclass(frozen=True)
class EvaluateConfig:
    checkpoint_file: str
    profile: EvaluationProfile = "nominal"
    episodes_per_case: int = 100
    output: str = "rough_evaluation.json"
    device: str | None = None


@dataclass
class EpisodeResult:
    success: bool
    termination_reason: str
    velocity_error: float
    progress: float
    foot_slip: float
    energy: float
    contact_overload: float
    seed: int
    ring_peak_displacement_m: float = 0.0
    obstacle_completed: bool | None = None
    gait: dict | None = None


def checkpoint_sha256(path: Path) -> str:
    """Return the SHA256 digest without loading checkpoint tensors."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_episode_count(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (index < remainder) for index in range(parts)]


def _configure_case_env(task_id: str, profile: EvaluationProfile, seed: int, num_envs: int):
    cfg = evaluation_env_cfg(task_id, mode=profile)
    cfg.seed = seed
    cfg.episode_length_s = MIXED_EPISODE_LENGTH_S if task_id == "PointLeg-RoughMixed-v0" else 20.0
    if "reset_mixed_obstacles" in cfg.events:
        cfg.events["reset_mixed_obstacles"].params["forward_only"] = True
    cfg.auto_reset = False
    cfg.scene.num_envs = num_envs
    terrain_cfg = cfg.scene.terrain
    if terrain_cfg is None or terrain_cfg.terrain_generator is None:
        raise ValueError("Evaluation requires a generated terrain configuration")
    terrain_cfg.num_envs = num_envs
    terrain_cfg.terrain_generator.seed = seed
    terrain_cfg.terrain_generator.num_rows = 10
    terrain_cfg.terrain_generator.num_cols = len(terrain_cfg.terrain_generator.sub_terrains)
    return cfg


def _set_cases(raw_env: ManagerBasedRlEnv, terrain_level: int, count_per_terrain: int) -> None:
    terrain = raw_env.scene.terrain
    if terrain is None or terrain.terrain_origins is None or terrain.env_origins is None:
        raise RuntimeError("Generated terrain origins are unavailable")
    terrain_types = torch.arange(
        terrain.terrain_origins.shape[1], device=raw_env.device, dtype=torch.long
    ).repeat_interleave(count_per_terrain)
    if len(terrain_types) != raw_env.num_envs:
        raise ValueError("Evaluation environment count must be balanced across terrain types")
    terrain.terrain_types[:] = terrain_types
    terrain.terrain_levels.fill_(terrain_level)
    resolver = getattr(terrain, "get_terrain_origins", None)
    if callable(resolver):
        terrain.env_origins[:] = resolver(terrain.terrain_levels, terrain.terrain_types)
    else:
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
    raw_env.reset()


def _set_fixed_command(raw_env: ManagerBasedRlEnv, command: tuple[float, float, float]) -> None:
    if hasattr(raw_env, "_mixed_obstacles"):
        raw_env._mixed_obstacles["speed"][:] = command[0]
        raw_env._mixed_obstacles["direction"][:] = 1.0
    term = raw_env.command_manager.get_term("twist")
    value = torch.tensor(command, device=raw_env.device, dtype=torch.float32)
    term.vel_command_b[:] = value
    term.vel_command_w[:] = value
    term.is_standing_env[:] = False
    term.is_forward_env[:] = command[0] > 0.0 and command[1] == 0.0 and command[2] == 0.0
    if hasattr(term, "is_lateral_env"):
        term.is_lateral_env[:] = False
    if hasattr(term, "is_yaw_env"):
        term.is_yaw_env[:] = False
    if hasattr(term, "is_backward_env"):
        term.is_backward_env[:] = command[0] < 0.0
    if term.cfg.heading_command:
        term.heading_target[:] = term.robot.data.heading_w
        term.is_heading_env[:] = True
    heading = term.robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    direction_w = torch.empty((raw_env.num_envs, 2), device=raw_env.device)
    direction_w[:, 0] = cos_h * command[0] - sin_h * command[1]
    direction_w[:, 1] = sin_h * command[0] + cos_h * command[1]
    term.progress_direction_w[:] = direction_w
    raw_env.obs_buf = raw_env.observation_manager.compute(update_history=False)


def _resolve_reason(raw_env: ManagerBasedRlEnv, env_index: int) -> str:
    for name in raw_env.termination_manager.active_terms:
        if name == "time_out":
            continue
        if bool(raw_env.termination_manager.get_term(name)[env_index].item()):
            return name
    return "time_out"


def _evaluate_seed_batch(
    task_id: str,
    checkpoint: Path,
    profile: EvaluationProfile,
    seed: int,
    terrain_names: list[str],
    terrain_level: int,
    count_per_terrain: int,
    device: str,
) -> dict[str, list[EpisodeResult]]:
    count = count_per_terrain * len(terrain_names)
    env_cfg = _configure_case_env(task_id, profile, seed, count)
    agent_cfg = load_rl_cfg(task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        _set_cases(raw_env, terrain_level, count_per_terrain)
        fixed_command = evaluation_command(task_id)
        _set_fixed_command(raw_env, fixed_command)
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)

        start_pos = raw_env.scene["robot"].data.root_link_pos_w[:, :2].clone()
        direction_w = raw_env.command_manager.get_term("twist").progress_direction_w.clone()
        speed_error_sum = torch.zeros(count, device=device)
        slip_sum = torch.zeros(count, device=device)
        energy_sum = torch.zeros(count, device=device)
        overload_sum = torch.zeros(count, device=device)
        sample_count = torch.zeros(count, device=device)
        stable_steps = torch.zeros(count, dtype=torch.long, device=device)
        finished = torch.zeros(count, dtype=torch.bool, device=device)
        results: list[EpisodeResult | None] = [None] * count

        slip_cfg = raw_env.reward_manager.get_term_cfg("feet_slip")
        required_stable_steps = math.ceil(0.5 / raw_env.step_dt)

        for _ in range(raw_env.max_episode_length + 1):
            active = ~finished
            if not active.any():
                break
            with torch.no_grad():
                observations = wrapped_env.get_observations()
                actions = policy(observations)
                actions[finished] = 0.0
                _, _, terminated, truncated, _ = raw_env.step(actions)

            robot = raw_env.scene["robot"]
            command = raw_env.command_manager.get_command("twist")
            speed_error = torch.linalg.vector_norm(
                robot.data.root_link_lin_vel_b[:, :2] - command[:, :2], dim=1
            )
            slip = slip_cfg.func(raw_env, **slip_cfg.params)
            energy = joint_power(raw_env)
            overload = contact_forces(raw_env, "feet_ground_contact", threshold=100.0)
            speed_error_sum[active] += speed_error[active]
            slip_sum[active] += slip[active]
            energy_sum[active] += energy[active]
            overload_sum[active] += overload[active]
            sample_count[active] += 1.0

            stable = (-robot.data.projected_gravity_b[:, 2] >= math.cos(0.5))
            stable_steps = torch.where(active & stable, stable_steps + 1, torch.zeros_like(stable_steps))
            done = (terminated | truncated) & active
            if done.any():
                terminal_pos = robot.data.root_link_pos_w[:, :2].clone()
                for env_index in done.nonzero(as_tuple=False).flatten().tolist():
                    progress = torch.dot(
                        terminal_pos[env_index] - start_pos[env_index], direction_w[env_index]
                    ).item()
                    direction_norm = torch.linalg.vector_norm(direction_w[env_index]).item()
                    if direction_norm > 0.0:
                        progress /= direction_norm
                    samples = max(sample_count[env_index].item(), 1.0)
                    mean_error = speed_error_sum[env_index].item() / samples
                    reason = _resolve_reason(raw_env, env_index)
                    target = min(
                        direction_norm * env_cfg.episode_length_s,
                        env_cfg.scene.terrain.terrain_generator.size[0] * 0.45,
                    ) * 0.60
                    is_stable = stable_steps[env_index].item() >= required_stable_steps
                    success = (
                        reason in ("time_out", "obstacle_completed")
                        and progress >= target
                        and mean_error <= (0.05 if task_id == "PointLeg-HighStep-v0" else 0.25)
                        and is_stable
                    )
                    obstacle_complete = None
                    gait = None
                    if task_id == "PointLeg-HighStep-v0":
                        from pointleg.high_step_evaluation import snapshot_high_step

                        gait = snapshot_high_step(raw_env, env_index)
                        success = success and gait["accepted"]
                        if reason == "time_out" and not gait["accepted"]:
                            reason = "gait_quality"
                    peak_push = 0.0
                    if hasattr(raw_env, "_mixed_obstacles"):
                        from pointleg.mdp.terrain_utils import terrain_type_mask
                        from pointleg.terrains.mixed_obstacles import FENCE_NAMES, RING_TERRAIN
                        special = terrain_type_mask(raw_env, (*FENCE_NAMES, RING_TERRAIN))
                        if bool(special[env_index]):
                            obstacle_complete = bool(raw_env._mixed_obstacles["complete"][env_index])
                            peak_push = float(raw_env._mixed_obstacles["peak_push"][env_index])
                            success = success and obstacle_complete
                            if reason == "time_out" and not obstacle_complete:
                                reason = "obstacle_incomplete"
                    if reason == "time_out" and not success:
                        if progress < target:
                            reason = "insufficient_progress"
                        elif mean_error > (0.05 if task_id == "PointLeg-HighStep-v0" else 0.25):
                            reason = "velocity_error"
                        else:
                            reason = "post_completion_instability"
                    results[env_index] = EpisodeResult(
                        success=success,
                        termination_reason=reason,
                        velocity_error=mean_error,
                        progress=progress,
                        foot_slip=slip_sum[env_index].item() / samples,
                        energy=energy_sum[env_index].item() / samples,
                        contact_overload=overload_sum[env_index].item() / samples,
                        seed=seed,
                        ring_peak_displacement_m=peak_push,
                        obstacle_completed=obstacle_complete,
                        gait=gait,
                    )
                finished |= done
            ended = terminated | truncated
            if ended.any():
                raw_env.reset(env_ids=ended.nonzero(as_tuple=False).flatten())
        if not finished.all():
            raise RuntimeError("Evaluation episode did not reach a termination condition")
        completed_results = [result for result in results if result is not None]
        return {
            terrain_name: completed_results[
                terrain_index * count_per_terrain:(terrain_index + 1) * count_per_terrain
            ]
            for terrain_index, terrain_name in enumerate(terrain_names)
        }
    finally:
        wrapped_env.close()


def _mean(results: list[EpisodeResult], field: str) -> float:
    return sum(float(getattr(result, field)) for result in results) / max(len(results), 1)


def build_case_report(
    terrain_name: str,
    terrain_level: int,
    results: list[EpisodeResult],
    required_success_rate: float,
    command: tuple[float, float, float] = DEFAULT_COMMAND,
) -> dict:
    """Aggregate episode outcomes into the stable JSON report schema."""
    successes = sum(result.success for result in results)
    attempts = len(results)
    success_rate = successes / max(attempts, 1)
    return {
        "terrain_type": terrain_name,
        "terrain_level": terrain_level,
        "command": {"lin_vel_x": command[0], "lin_vel_y": command[1], "ang_vel_z": command[2]},
        "attempts": attempts,
        "successes": successes,
        "success_rate": success_rate,
        "accepted": attempts > 0 and success_rate >= required_success_rate,
        "termination_reasons": dict(sorted(Counter(result.termination_reason for result in results).items())),
        "mean_velocity_error_mps": _mean(results, "velocity_error"),
        "mean_progress_m": _mean(results, "progress"),
        "mean_foot_slip": _mean(results, "foot_slip"),
        "mean_energy_w": _mean(results, "energy"),
        "mean_contact_overload_n": _mean(results, "contact_overload"),
        "maximum_ring_displacement_m": max((r.ring_peak_displacement_m for r in results), default=0.0),
        "obstacle_completions": sum(r.obstacle_completed is True for r in results),
        "seeds": sorted({result.seed for result in results}),
        "gait_episodes": [result.gait for result in results if result.gait is not None],
    }


def run_evaluation(task_id: str, cfg: EvaluateConfig) -> dict:
    """Run every terrain case and write a deterministic JSON report."""
    if cfg.episodes_per_case < 1:
        raise ValueError("episodes_per_case must be positive")
    checkpoint = Path(cfg.checkpoint_file).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    preview_cfg = _configure_case_env(task_id, cfg.profile, EVALUATION_SEEDS[0], 1)
    terrain_names = list(preview_cfg.scene.terrain.terrain_generator.sub_terrains)
    terrain_level = preview_cfg.scene.terrain.terrain_generator.num_rows - 1
    required_rate = 0.90 if cfg.profile == "nominal" else 0.80
    episode_counts = _split_episode_count(cfg.episodes_per_case, len(EVALUATION_SEEDS))

    results_by_terrain: dict[str, list[EpisodeResult]] = {name: [] for name in terrain_names}
    for seed, count in zip(EVALUATION_SEEDS, episode_counts, strict=True):
        if count:
            seed_results = _evaluate_seed_batch(
                task_id, checkpoint, cfg.profile, seed, terrain_names,
                terrain_level, count, device,
            )
            for terrain_name, results in seed_results.items():
                results_by_terrain[terrain_name].extend(results)
    cases = [
        build_case_report(
            name, terrain_level, results_by_terrain[name], required_rate,
            command=evaluation_command(task_id),
        )
        for name in terrain_names
    ]

    report = {
        "schema_version": 1,
        "task": task_id,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256(checkpoint)},
        "profile": cfg.profile,
        "episodes_per_case": cfg.episodes_per_case,
        "seeds": list(EVALUATION_SEEDS),
        "criteria": {
            "minimum_success_rate": required_rate,
            "minimum_progress_ratio": 0.60,
            "maximum_mean_velocity_error_mps": 0.05 if task_id == "PointLeg-HighStep-v0" else 0.25,
            "high_step_gait_required": task_id == "PointLeg-HighStep-v0",
            "post_completion_stability_s": 0.5,
        },
        "cases": cases,
        "accepted": bool(cases) and all(case["accepted"] for case in cases),
    }
    output = Path(cfg.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(task_id: str, arguments: list[str] | None = None) -> None:
    """Parse the evaluate subcommand and print its report location."""
    cfg = tyro.cli(
        EvaluateConfig,
        args=arguments,
        prog=f"pointleg evaluate {task_id}",
    )
    report = run_evaluation(task_id, cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))

