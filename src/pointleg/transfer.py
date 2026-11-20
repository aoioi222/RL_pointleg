"""Create fresh task checkpoints, expanding stepping-stone inputs without changing the initial policy."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
import tyro

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.rl import MjlabOnPolicyRunner

from .mdp.elevation import ElevationTerrainCurriculum
from .rsl_rl.runners.elevation_runner import ElevationOnPolicyRunner
from .mdp.jump import JumpTerrainCurriculum
from .mdp.stepping_stones import SUPPORT_PROGRESS_INFO_KEY
from .mdp.fixed_stone_gait import FIXED_GAIT_INFO_KEY, FIXED_GAIT_OBS_DIM
from .rsl_rl.runners.jump_runner import JumpOnPolicyRunner
from .terrains.elevation import ELEVATION_TASK_TERRAINS


@dataclass
class TransferConfig:
    checkpoint_file: str
    output_run: str = "rough_pretrained"
    warmup: bool = False


class _TransferEnv:
    def __init__(self, task_id, warmup=False):
        self.cfg = load_env_cfg(task_id)
        if warmup:
            raise ValueError("Jump now uses the platform curriculum starting at 15%; --warmup is retired")
        self.num_envs, self.num_actions, self.device = 1, 12, torch.device("cpu")
        self.common_step_counter = 0
        self.step_dt = 0.02
        self.max_episode_length = int(self.cfg.episode_length_s / 0.02)
        self.unwrapped = self
        self.actor_observations = 105 + FIXED_GAIT_OBS_DIM if task_id == "PointLeg-SteppingStones-v0" else 45
        self.critic_observations = 381 + FIXED_GAIT_OBS_DIM if task_id == "PointLeg-SteppingStones-v0" else 325
        if task_id == "PointLeg-Jump-v0":
            self._jump_settings = self.cfg.events["reset_jump"].params["settings"]
            JumpTerrainCurriculum(self.cfg.curriculum["jump_levels"], self)
        elif task_id in ELEVATION_TASK_TERRAINS:
            self._elevation_settings = self.cfg.events["reset_elevation"].params["settings"]
            ElevationTerrainCurriculum(self.cfg.curriculum["elevation_levels"], self)

    def get_observations(self):
        return TensorDict({
            "actor": torch.zeros(1, self.actor_observations),
            "critic": torch.zeros(1, self.critic_observations),
        }, batch_size=[1])


def _load_stepping_stones_weights(runner, source):
    """Zero-pad only new input columns while loading every other parameter strictly."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_dims = tuple(checkpoint[key]["mlp.0.weight"].shape[1]
                        for key in ("actor_state_dict", "critic_state_dict"))
    if source_dims not in ((45, 325), (105, 381), (123, 399)):
        raise ValueError(f"Stepping-stone transfer requires 45/325 or 105/381 or 123/399 inputs, got {source_dims}")
    adapted = {}
    for key, model in (("actor_state_dict", runner.alg.actor), ("critic_state_dict", runner.alg.critic)):
        incoming = dict(checkpoint[key])
        expected = model.state_dict()["mlp.0.weight"]
        original = incoming["mlp.0.weight"]
        if original.shape[0] != expected.shape[0]:
            raise ValueError(f"Hidden layer size changed for {key}: {tuple(original.shape)} vs {tuple(expected.shape)}")
        expanded = torch.zeros_like(expected)
        expanded[:, :original.shape[1]] = original
        incoming["mlp.0.weight"] = expanded
        adapted[key] = incoming
    runner.alg.load(adapted, load_cfg={"actor": True, "critic": True}, strict=True)
    return {
        "source_actor_observations": source_dims[0],
        "source_critic_observations": source_dims[1],
        "input_adaptation": "copy existing input columns; zero-initialize additional task columns",
        "actor_added_columns": runner.env.actor_observations - source_dims[0],
        "critic_added_columns": runner.env.critic_observations - source_dims[1],
        SUPPORT_PROGRESS_INFO_KEY: True,
        FIXED_GAIT_INFO_KEY: 1,
    }


def prepare_transfer(task_id, checkpoint_file, output_run="rough_pretrained", output_root=None, warmup=False):
    if task_id not in ELEVATION_TASK_TERRAINS and task_id not in ("PointLeg-Jump-v0", "PointLeg-SteppingStones-v0"):
        raise ValueError("Weights-only transfer is defined for Elevation, its specializations, Jump and SteppingStones")
    if not output_run or Path(output_run).name != output_run or output_run in (".", "..") or "/" in output_run or "\\" in output_run:
        raise ValueError("output_run must be a single directory name")
    source = Path(checkpoint_file).expanduser().resolve(strict=True)
    cfg = load_rl_cfg(task_id)
    if warmup:
        if task_id != "PointLeg-Jump-v0":
            raise ValueError("Flat-jump warmup applies only to PointLeg-Jump-v0")
        cfg.experiment_name = "pointleg_jump_warmup"
    cfg.upload_model = False
    cfg.num_steps_per_env = 1
    cfg.algorithm.num_mini_batches = 1
    root = Path(output_root) if output_root is not None else Path(__file__).resolve().parents[2] / "logs" / "rsl_rl"
    output = root / cfg.experiment_name / output_run
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite transfer run: {output}")
    env = _TransferEnv(task_id, warmup=warmup)
    if task_id == "PointLeg-SteppingStones-v0":
        runner_cls = MjlabOnPolicyRunner
    else:
        runner_cls = JumpOnPolicyRunner if task_id == "PointLeg-Jump-v0" else ElevationOnPolicyRunner
    runner = runner_cls(env, asdict(cfg), device="cpu")
    adaptation = {}
    if task_id == "PointLeg-SteppingStones-v0":
        adaptation = _load_stepping_stones_weights(runner, source)
    else:
        runner.load(str(source), load_cfg={"actor": True, "critic": True}, strict=True, map_location="cpu")
    metadata = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "task_id": task_id, "actor_observations": env.actor_observations, "critic_observations": env.critic_observations,
        "actions": 12, "optimizer": "fresh", "iteration": 0, "warmup": warmup,
        "command_semantics": "trigger, elapsed_seconds/3, zero" if task_id == "PointLeg-Jump-v0" else "velocity",
        "terrain_kind": ELEVATION_TASK_TERRAINS.get(task_id),
        **adaptation,
    }
    output.mkdir(parents=True, exist_ok=False)
    infos = {"transfer": metadata}
    if SUPPORT_PROGRESS_INFO_KEY in adaptation:
        infos[SUPPORT_PROGRESS_INFO_KEY] = adaptation[SUPPORT_PROGRESS_INFO_KEY]
        infos[FIXED_GAIT_INFO_KEY] = 1
    runner.save(str(output / "model_0.pt"), infos=infos)
    (output / "transfer.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output / "model_0.pt"


def main(task_id, arguments):
    cfg = tyro.cli(TransferConfig, args=arguments, prog=f"pointleg transfer {task_id}")
    print(prepare_transfer(task_id, cfg.checkpoint_file, cfg.output_run, warmup=cfg.warmup))
