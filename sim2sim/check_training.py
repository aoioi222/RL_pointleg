"""Exercise real PPO updates, checkpoint resume, ONNX export and CPU playback.

Run from the project root with ``python -m sim2sim.check_training``. This is a
small integration check, not training to convergence or obstacle acceptance.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import pointleg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from pointleg.robot_cfg import BASE_BODY_NAME
from pointleg.config.debug import configure_debug_training
from sim2sim.export_onnx import export_policy
from sim2sim.pointleg_sim2sim import PointLegSimulation, load_policy


def pose_diagnostic(task: str, duration: float = 5.0) -> dict:
    """Measure the untrained zero-action pose on a CPU flat floor."""
    sim = PointLegSimulation(task)
    with torch.inference_mode():
        for _ in range(round(duration / sim.control_dt)):
            sim.step(torch.zeros(1, 12))
    gravity_z = sim.observe()[0, 5].item()
    return {
        "duration_s": duration,
        "initial_root_height_m": sim.cfg.scene.entities["robot"].init_state.pos[2],
        "final_root_height_m": float(sim.mj_data.qpos[2]),
        "final_tilt_deg": float(np.degrees(np.arccos(np.clip(-gravity_z, -1.0, 1.0)))),
        "note": "zero actions on flat ground; not a trained policy or hardware calibration",
    }


def check_training(task: str, output: Path, device: str = "cuda:0") -> dict:
    """Preserve artifacts and raise on training, resume or inference mismatch."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this GPU integration check cannot run")
    cfg, agent = load_env_cfg(task), load_rl_cfg(task)
    configure_debug_training(cfg, agent)
    cfg.seed = agent.seed
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", asdict(agent))
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        robot = env.scene["robot"]
        local_ids, _ = robot.find_bodies(BASE_BODY_NAME)
        body_id = int(robot.indexing.body_ids[local_ids[0]].item())
        mass = env.sim.model.body_mass[:, body_id]
        inertia = env.sim.model.body_inertia[:, body_id]
        mass_ratio = mass / float(env.sim.mj_model.body_mass[body_id])
        default_inertia = torch.as_tensor(
            env.sim.mj_model.body_inertia[body_id].copy(), device=device, dtype=inertia.dtype,
        )
        inertia_ratio = inertia.sort(dim=-1).values / default_inertia.sort().values
        torch.testing.assert_close(inertia_ratio, mass_ratio[:, None].expand_as(inertia_ratio),
                                   rtol=1e-4, atol=1e-4)
        if not ((mass_ratio >= 0.85 - 1e-5) & (mass_ratio <= 1.45 + 1e-5)).all():
            raise RuntimeError("Base mass-density randomization is outside its configured range")
        inertial_range = [float(mass_ratio.min()), float(mass_ratio.max())]
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), str(output / "train"), device)
        before = {name: value.clone() for name, value in runner.alg.get_policy().state_dict().items()}
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
        after = runner.alg.get_policy().state_dict()
        if not all(torch.isfinite(value).all().item() for value in after.values()):
            raise RuntimeError("PPO produced non-finite policy parameters")
        if not any(not torch.equal(before[name], value) for name, value in after.items()):
            raise RuntimeError("PPO did not update any policy parameter")
        checkpoint = output / "trained.pt"
        runner.save(str(checkpoint))
        counter = env.common_step_counter
        resumed = MjlabOnPolicyRunner(wrapped, asdict(agent), str(output / "resume"), device)
        env.common_step_counter = 0
        resumed.load(str(checkpoint), map_location=device)
        if env.common_step_counter != counter:
            raise RuntimeError("Checkpoint did not restore the environment step counter")
        for name, value in resumed.alg.get_policy().state_dict().items():
            torch.testing.assert_close(value, after[name], rtol=0.0, atol=0.0)
        old_optimizer = runner.alg.optimizer.state_dict()["state"]
        new_optimizer = resumed.alg.optimizer.state_dict()["state"]
        if old_optimizer.keys() != new_optimizer.keys():
            raise RuntimeError("Checkpoint did not restore optimizer state")
        for key, state in old_optimizer.items():
            for name, value in state.items():
                torch.testing.assert_close(new_optimizer[key][name].cpu(), value.cpu(), rtol=0.0, atol=0.0)
        resumed.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        if env.common_step_counter <= counter:
            raise RuntimeError("Resumed training did not advance the environment")
        checkpoint = output / "resumed.pt"
        resumed.save(str(checkpoint))
        final_counter = env.common_step_counter
    finally:
        env.close()

    onnx_path, export_error = export_policy(checkpoint, task, output / "policy.onnx")
    pt_policy, onnx_policy = load_policy(checkpoint, task), load_policy(onnx_path, task)
    playback = PointLegSimulation(task)
    with torch.inference_mode():
        for _ in range(50):
            obs = playback.observe()
            pt_action, onnx_action = pt_policy(obs), onnx_policy(obs)
            torch.testing.assert_close(pt_action, onnx_action, rtol=1e-5, atol=1e-6)
            playback.step(onnx_action)
    report = {
        "task": task, "device": device, "num_envs": cfg.scene.num_envs,
        "ppo_updates": 3, "rollout_steps_per_env": agent.num_steps_per_env,
        "final_global_step": final_counter, "checkpoint": str(checkpoint),
        "density_scale_range_observed": inertial_range,
        "optimizer_and_policy_restored": True,
        "onnx": str(onnx_path), "onnx_max_abs_error": export_error,
        "cpu_playback_s": 50 * playback.control_dt,
        "zero_action_pose": pose_diagnostic(task),
        "scope": "training smoke check only; no convergence or obstacle success claim",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[name for name, spec in pointleg.TASK_SPECS.items() if spec.enabled],
                        default="PointLeg-Flat-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pose-only", action="store_true", help="Only measure the CPU zero-action pose; no GPU training")
    args = parser.parse_args()
    if args.pose_only:
        print(json.dumps(pose_diagnostic(args.task), ensure_ascii=False, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("artifacts/pointleg_checks") / f"{stamp}_{args.task}"
    check_training(args.task, output, args.device)


if __name__ == "__main__":
    main()
