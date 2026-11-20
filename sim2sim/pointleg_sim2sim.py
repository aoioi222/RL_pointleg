"""Point-foot inference using the task's actual observations and action pipeline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

import pointleg
from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionManager
from mjlab.managers.observation_manager import ObservationManager
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pointleg.jump_trigger import JumpTrigger
from pointleg.mdp.stepping_stones import STEPPING_FORWARD_SPEED_RANGE, SUPPORT_PROGRESS_INFO_KEY
from pointleg.mdp.fixed_stone_gait import FIXED_GAIT_OBS_DIM
from pointleg.robot_cfg import (
    BASE_BODY_NAME,
    EFFORT_LEG,
    ESTIMATED_NO_LOAD_SPEED_RAD_S,
    LEG_JOINT_NAMES,
    LEG_MOTOR_SPEC,
    MOTOR_SUPPLY_VOLTAGE_V,
)
from rsl_rl.models import MLPModel

if __package__:
    from .stepping_stones_replay import STEPPING_TASK, STEPPING_TRACE_FIELDS, SteppingStonesReplay, stepping_observation
else:
    from stepping_stones_replay import STEPPING_TASK, STEPPING_TRACE_FIELDS, SteppingStonesReplay, stepping_observation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIRECTORY = PROJECT_ROOT / "sim2sim" / "policy" / "optical_policy"
TERRAIN_SPAWN = (-4.1, 0.0, 0.0)
STEPPING_ENTRY_SLOT = "f5"
SPAWN_SITES = {
    "1": ("start_1", "Start area 1"),
    "2": ("start_2", "Start area 2 / platform jump"),
    "3": ("spawn_bridge", "Balance beam"),
    "4": ("spawn_wall", "High wall"),
    "5": ("spawn_stones", "Stepping stones"),
    STEPPING_ENTRY_SLOT: ("spawn_stones_top", "Stepping stones entry platform top"),
    "6": ("spawn_clearance", "Clearance frame"),
    "7": ("slalom_entry", "Slalom"),
    "8": ("spawn_tires", "Tire array"),
    "9": ("spawn_pit", "Brick and bark pit"),
    "0": ("spawn_ramp", "Ramp approach"),
    "-": ("spawn_stairs", "Stairs approach"),
}
JUMP_TASK = "PointLeg-Jump-v0"
MULTI_POLICY_FILES = {
    "1": "model_1999.pt",   #平动flat
    "2": "model_14100.pt",  #高抬腿high_step
    "3": "model_4997.pt",   #坡slope
    "4": "model_3692.pt",   #限高框架crawl
    "5": "model_12000.pt",  #爬梯策略stairs
    "6": "model_42700.pt",  #梅花桩steppingstone
    "7": "PointLeg_Bridge10cm_model_7485_20261111.pt",#单边桥
    "jump": "model_jump.pt",
}
MULTI_POLICY_TASKS = {"4": "PointLeg-Crawl-v0", "6": STEPPING_TASK, "jump": JUMP_TASK}
POLICY_SWITCH_DURATION_S = 1.0
STEPPING_OBS_DIM = 105 + FIXED_GAIT_OBS_DIM
STEPPING_DEFAULT_SPEED = sum(STEPPING_FORWARD_SPEED_RANGE) / 2

class CpuRobot:
    """Expose native MuJoCo state to the training observation and action terms."""

    def __init__(self, entity: Entity, model: mujoco.MjModel, data: mujoco.MjData,
                 default_pos: np.ndarray | None = None):
        self.entity = entity
        self.model = model
        self.mj_data = data
        self.joint_ids = np.array([model.joint(name).id for name in LEG_JOINT_NAMES])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = model.jnt_dofadr[self.joint_ids]
        self.ctrl_ids = []
        for joint_id in self.joint_ids:
            matches = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == joint_id)
            )
            if len(matches) != 1:
                raise ValueError("Each point-foot joint must have exactly one position actuator")
            self.ctrl_ids.append(int(matches[0]))
        if default_pos is None:
            default_pos = model.key_qpos[0, self.qpos_ids].copy()
        default_vel = model.key_qvel[0, self.qvel_ids].copy()
        limits = model.jnt_range[self.joint_ids].copy()
        factor = entity.cfg.articulation.soft_joint_pos_limit_factor
        centers = limits.mean(axis=1)
        half_ranges = (limits[:, 1] - limits[:, 0]) * factor / 2.0
        soft_limits = np.column_stack((centers - half_ranges, centers + half_ranges))
        self.data = SimpleNamespace(
            default_joint_pos=torch.tensor(default_pos, dtype=torch.float32)[None],
            default_joint_vel=torch.tensor(default_vel, dtype=torch.float32)[None],
            soft_joint_pos_limits=torch.tensor(soft_limits, dtype=torch.float32)[None],
            encoder_bias=torch.zeros(1, 12),
        )
        self.update()

    def __getattr__(self, name):
        return getattr(self.entity, name)

    def update(self):
        body_id = self.model.body(BASE_BODY_NAME).id
        rotation = self.mj_data.xmat[body_id].reshape(3, 3)
        angular_velocity = rotation.T @ self.mj_data.cvel[body_id, :3]
        self.data.root_link_ang_vel_b = torch.tensor(angular_velocity, dtype=torch.float32)[None]
        self.data.projected_gravity_b = torch.tensor(
            rotation.T @ np.array([0.0, 0.0, -1.0]), dtype=torch.float32
        )[None]
        self.data.joint_pos = torch.tensor(self.mj_data.qpos[self.qpos_ids], dtype=torch.float32)[None]
        self.data.joint_vel = torch.tensor(self.mj_data.qvel[self.qvel_ids], dtype=torch.float32)[None]

    def set_joint_position_target(self, target, joint_ids):
        indices = joint_ids.cpu().numpy()
        self.mj_data.ctrl[np.asarray(self.ctrl_ids)[indices]] = target[0].cpu().numpy()


class PointLegSimulation:
    """Run one deterministic, non-randomized CPU instance of a point-foot task."""

    def __init__(self, task: str, scene: Path | None = None, action_delay: int = 0, *, stepping_trace=None):
        cfg = load_env_cfg(task, play=True)
        if set(cfg.actions) != {"leg_joint_pos"}:
            raise ValueError("Sim2Sim currently supports the point-foot position action only")
        entity = Entity(cfg.scene.entities["robot"])
        if tuple(entity.joint_names) != LEG_JOINT_NAMES:
            raise ValueError("The model does not match the 12-joint point-foot policy order")
        if scene is None:
            spec = entity.spec
            spec.worldbody.add_geom(
                name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0.0, 0.0, 0.05], contype=1, conaffinity=1,
                friction=[0.8, 0.05, 0.01], rgba=[0.55, 0.6, 0.65, 1.0],
            )
            spec.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
        else:
            spec = mujoco.MjSpec.from_file(str(scene.resolve()))
            if list(spec.joints) or list(spec.actuators) or list(spec.keys):
                raise ValueError("--scene must contain terrain only, without robot joints, actuators or keyframes")
            spec.attach(entity.spec, prefix="", frame=spec.worldbody.add_frame())
        self.model = spec.compile()
        cfg.sim.mujoco.apply(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, 12):
            raise ValueError("Scene must contain exactly one free-base, 12-joint point-foot robot")
        self.mj_data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.mj_data, 0)
        mujoco.mj_forward(self.model, self.mj_data)
        self._initial_task = task
        self._action_delay = action_delay
        self._stepping_trace = stepping_trace
        self._spawn = TERRAIN_SPAWN if scene is not None else None
        self.spawn_slot = None
        self._stepping_entry_pending = False
        self.spawn_sites = {
            slot: (self.mj_data.site(name).xpos.copy(), self.mj_data.site(name).xmat.copy())
            for slot, (name, _) in SPAWN_SITES.items()
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
        }
        self._transition_start_ctrl = None
        self._transition_step = 0
        self.stepping = None
        self._stepping_episode_s = None
        self._configure_task(task, entity=entity, cfg=cfg)
        self.reset()

    def _configure_task(self, task: str, entity: Entity | None = None, cfg=None):
        """Configure observations and action references for one policy task."""
        cfg = cfg if cfg is not None else load_env_cfg(task, play=True)
        if set(cfg.actions) != {"leg_joint_pos"}:
            raise ValueError("Sim2Sim currently supports the point-foot position action only")
        entity = entity if entity is not None else Entity(cfg.scene.entities["robot"])
        if tuple(entity.joint_names) != LEG_JOINT_NAMES:
            raise ValueError("The task has a different joint ordering")
        training_cfg = load_env_cfg(task)
        reference_model = self.model if task == self._initial_task else entity.spec.compile()
        key_qpos = reference_model.key_qpos[0].copy()
        default_pos = np.array([
            reference_model.key_qpos[0, reference_model.jnt_qposadr[reference_model.joint(name).id]]
            for name in LEG_JOINT_NAMES
        ])
        stepping = None
        if task == STEPPING_TASK:
            try:
                stepping = SteppingStonesReplay(
                    self.model, self.mj_data, self._stepping_trace,
                    standing_height=key_qpos[2], standing_joint_pos=default_pos,
                )
            except KeyError as exc:
                raise ValueError(f"Stepping-stone scene is missing a required geom: {exc}") from exc
        self._task_key_qpos = key_qpos
        previous_control_dt = getattr(self, "control_dt", None)
        cfg.sim.mujoco.apply(self.model)
        self.task = task
        self.cfg = cfg
        self.stepping = stepping
        self._stepping_episode_s = training_cfg.episode_length_s if stepping is not None else None
        self.robot = CpuRobot(entity, self.model, self.mj_data, default_pos=default_pos)
        self.command = torch.zeros(1, 3)
        self.env = SimpleNamespace(
            num_envs=1, device="cpu", scene={"robot": self.robot},
            command_manager=SimpleNamespace(get_command=lambda name: self.command),
            stepping=stepping,
        )
        action_cfg = cfg.actions["leg_joint_pos"]
        training_action = training_cfg.actions["leg_joint_pos"]
        if not training_action.min_delay <= self._action_delay <= training_action.max_delay:
            raise ValueError(f"Action delay must be in [{training_action.min_delay}, {training_action.max_delay}]")
        action_cfg.min_delay = action_cfg.max_delay = self._action_delay
        self.control_dt = self.model.opt.timestep * cfg.decimation
        if previous_control_dt is not None and not np.isclose(self.control_dt, previous_control_dt):
            raise ValueError("Cannot switch between tasks with different control periods")
        if not np.isclose(action_cfg.control_frequency, 1.0 / self.control_dt):
            raise ValueError("Action filter frequency does not match the simulation control rate")
        self.env.action_manager = ActionManager(cfg.actions, self.env)
        if stepping is not None:
            stepping.refresh_observations(self._stepping_episode_s)
            for name in stepping.values:
                term = cfg.observations["actor"].terms[name]
                term.func = stepping_observation
                term.params = {"name": name}
        self.observations = ObservationManager({"actor": cfg.observations["actor"]}, self.env)
        self.obs_dim = self.observations.group_obs_dim["actor"][0]
        expected_dim = STEPPING_OBS_DIM if stepping is not None else 45
        if self.obs_dim != expected_dim:
            raise ValueError(f"{task} requires {expected_dim} actor observations, got {self.obs_dim}")
        self.env.action_manager.reset()
        self.observations.reset()

    def configure_stepping_policy(self, policy):
        if self.stepping is None:
            return
        if policy.fixed_gait and not 3 <= self.stepping.count <= 6:
            raise ValueError("Fixed-gait playback requires three through six stones")
        self.stepping.use_support_progress = policy.use_support_progress
        self.stepping.fixed_enabled = policy.fixed_gait
        terms = self.cfg.observations["actor"].terms
        if policy.fixed_gait:
            from mjlab.managers.observation_manager import ObservationTermCfg
            terms["fixed_gait"] = ObservationTermCfg(func=stepping_observation, params={"name": "fixed_gait"})
        else:
            terms.pop("fixed_gait", None)
        self.observations = ObservationManager({"actor": self.cfg.observations["actor"]}, self.env)
        self.obs_dim = self.observations.group_obs_dim["actor"][0]
        self.observations.reset()

    def switch_task(self, task: str, *, transition: bool = True):
        """Switch policies in place, or start SteppingStones at a queued entry spawn."""
        if task == self.task:
            return
        start_ctrl = self.mj_data.ctrl.copy()
        pose = self.mj_data.qpos.copy()
        velocity = self.mj_data.qvel.copy()
        self._configure_task(task)
        if not np.array_equal(self.mj_data.qpos, pose) or not np.array_equal(self.mj_data.qvel, velocity):
            raise RuntimeError("Task switch unexpectedly changed the robot state")
        if task == STEPPING_TASK and self._stepping_entry_pending:
            self.reset()
            return
        self._transition_start_ctrl = start_ctrl if transition else None
        self._transition_step = 0

    @property
    def in_transition(self) -> bool:
        return self._transition_start_ctrl is not None

    def reset(self):
        self._stepping_entry_pending = self.stepping is None and self.spawn_slot == STEPPING_ENTRY_SLOT
        mujoco.mj_resetDataKeyframe(self.model, self.mj_data, 0)
        self.mj_data.qpos[:] = self._task_key_qpos
        spawn_slot = self.spawn_slot
        if spawn_slot is None and self.stepping is not None:
            spawn_slot = STEPPING_ENTRY_SLOT if STEPPING_ENTRY_SLOT in self.spawn_sites else None
        if spawn_slot is not None:
            position, orientation = self.spawn_sites[spawn_slot]
            self.mj_data.qpos[:2] = position[:2]
            self.mj_data.qpos[2] += position[2]
            mujoco.mju_mat2Quat(self.mj_data.qpos[3:7], orientation)
        elif self.stepping is not None:
            self.mj_data.qpos[:3] += np.r_[
                self.stepping.platforms[0, :2] + self.stepping.spawn_offset * self.stepping.direction,
                self.stepping.platforms[0, 2],
            ]
            dx, dy = self.stepping.direction
            yaw = np.arctan2(dy, dx)
            self.mj_data.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        elif self._spawn is not None:
            x, y, yaw = self._spawn
            self.mj_data.qpos[:2] = (x, y)
            self.mj_data.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        self.mj_data.ctrl[self.robot.ctrl_ids] = self.robot.data.default_joint_pos[0].numpy()
        mujoco.mj_forward(self.model, self.mj_data)
        if self.stepping is not None:
            self.stepping.reset()
        self.robot.update()
        self.env.action_manager.reset()
        self.observations.reset()
        self._transition_start_ctrl = None
        self._transition_step = 0

    def observe(self):
        self.robot.update()
        if self.stepping is not None:
            self.stepping.refresh_observations(self._stepping_episode_s)
        return self.observations.compute(update_history=True)["actor"]

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.shape != (1, 12) or not torch.isfinite(actions).all():
            raise ValueError("Policy must return 12 finite actions with shape (1, 12)")
        self.env.action_manager.process_action(actions)
        alpha = 1.0
        if self._transition_start_ctrl is not None:
            self._transition_step += 1
            alpha = min(1.0, self._transition_step * self.control_dt / POLICY_SWITCH_DURATION_S)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        for _ in range(self.cfg.decimation):
            self.env.action_manager.apply_action()
            if self._transition_start_ctrl is not None:
                ids = self.robot.ctrl_ids
                self.mj_data.ctrl[ids] = (
                    (1.0 - alpha) * self._transition_start_ctrl[ids]
                    + alpha * self.mj_data.ctrl[ids]
                )
            mujoco.mj_step(self.model, self.mj_data)
            if self.stepping is not None:
                self.stepping.record_substep()
        if self._transition_start_ctrl is not None and self._transition_step * self.control_dt >= POLICY_SWITCH_DURATION_S:
            self._transition_start_ctrl = None
        mujoco.mj_forward(self.model, self.mj_data)
        if not np.isfinite(self.mj_data.qpos).all() or not np.isfinite(self.mj_data.qvel).all():
            raise RuntimeError("Simulation produced non-finite states")
        if any(warning.number for warning in self.mj_data.warning):
            raise RuntimeError("MuJoCo reported a simulation warning")


def check_jump_contract(task, contract):
    if task == JUMP_TASK:
        if not isinstance(contract, dict) or contract.get("command") != "trigger, elapsed_seconds/3, zero":
            raise ValueError("Jump requires a triggered-platform checkpoint with the trigger/time command contract")
    elif contract:
        raise ValueError("A triggered-platform checkpoint requires PointLeg-Jump-v0")


def spawn_key(key):
    if getattr(key, "name", None) == "f5":
        return "f5"
    vk = getattr(key, "vk", None)
    if vk is not None:
        if 0x30 <= vk <= 0x39:
            return str(vk - 0x30)
        if 0x60 <= vk <= 0x69:
            return str(vk - 0x60)
        if vk in (0xBD, 0x6D):
            return "-"
    char = getattr(key, "char", None)
    return char if char in SPAWN_SITES else None


def stepping_policy_observation_dim(checkpoint):
    if checkpoint.suffix.lower() == ".onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
        width = session.get_inputs()[0].shape[1]
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        width = saved["actor_state_dict"]["mlp.0.weight"].shape[1]
    if width not in (105, STEPPING_OBS_DIM):
        raise ValueError(f"Stepping-stone policy requires 105 legacy or 123 fixed-gait inputs, got {width}")
    return width


def load_policy(checkpoint: Path, task: str, obs_dim: int = 45):
    """Load a strict current-task actor or a dimension-checked ONNX policy."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint.suffix.lower() == ".onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX playback requires onnxruntime; use sim2sim/requirements.txt") from exc
        session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
        metadata_map = session.get_modelmeta().custom_metadata_map
        jump_metadata = metadata_map.get("pointleg_jump_contract")
        use_support_progress = metadata_map.get(SUPPORT_PROGRESS_INFO_KEY) == "true"
        check_jump_contract(task, json.loads(jump_metadata) if jump_metadata else None)
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("ONNX must accept one float32 observation tensor and return one action tensor")
        for metadata, width in ((inputs[0], obs_dim), (outputs[0], 12)):
            if len(metadata.shape) != 2 or (isinstance(metadata.shape[1], int) and metadata.shape[1] != width):
                raise ValueError(f"Point-foot ONNX requires {obs_dim} observations and 12 actions for {task}")

        def policy(obs):
            return torch.from_numpy(session.run(None, {inputs[0].name: obs.numpy()})[0])
    else:
        cfg = load_rl_cfg(task)
        if cfg.actor.class_name != "MLPModel" or tuple(cfg.obs_groups["actor"]) != ("actor",):
            raise ValueError("Only the current MLP actor with the actor observation group is supported")
        actor_cfg = asdict(cfg.actor)
        for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            actor_cfg.pop(key, None)
        actor = MLPModel(
            TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1]),
            {"actor": ["actor"]}, "actor", 12, **actor_cfg,
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or "actor_state_dict" not in saved:
            raise ValueError("Expected an MJLab checkpoint containing actor_state_dict")
        infos = saved.get("infos") or {}
        check_jump_contract(task, infos.get("jump_contract"))
        use_support_progress = bool(infos.get(SUPPORT_PROGRESS_INFO_KEY))
        try:
            actor.load_state_dict(saved["actor_state_dict"], strict=True)
        except RuntimeError as exc:
            first = saved["actor_state_dict"].get("mlp.0.weight")
            actual = f"; checkpoint input dimension: {first.shape[1]}" if first is not None and first.ndim == 2 else ""
            raise ValueError(f"Checkpoint does not match this task's current actor configuration ({obs_dim} observations, 12 actions{actual})") from exc
        policy = actor.eval().as_jit().eval()
    with torch.inference_mode():
        probe = policy(torch.zeros(1, obs_dim))
    if probe.shape != (1, 12) or not torch.isfinite(probe).all():
        raise ValueError("Policy did not produce 12 finite point-foot actions")
    if task == STEPPING_TASK:
        policy.use_support_progress = use_support_progress
        policy.fixed_gait = obs_dim == STEPPING_OBS_DIM
    return policy


def load_available_policies(directory: Path, default_task: str) -> dict:
    """Load compatible slots while preserving the required initial policy."""
    policies = {}
    for slot, filename in MULTI_POLICY_FILES.items():
        checkpoint = directory / filename
        if not checkpoint.is_file():
            print(f"Policy {slot} unavailable: {checkpoint}")
            continue
        policy_task = MULTI_POLICY_TASKS.get(slot, default_task)
        try:
            obs_dim = stepping_policy_observation_dim(checkpoint) if policy_task == STEPPING_TASK else 45
            policies[slot] = load_policy(checkpoint, policy_task, obs_dim)
        except (ValueError, RuntimeError) as exc:
            if slot == "1":
                raise ValueError(f"Policy {slot} ({checkpoint.name}) unavailable: {exc}") from exc
            print(f"Policy {slot} ({checkpoint.name}) unavailable: {exc}")
            continue
        print(f"Loaded policy {slot}: {checkpoint.name} ({policy_task})")
    if "1" not in policies:
        raise FileNotFoundError(f"Slot 1 policy not found: {directory / MULTI_POLICY_FILES['1']}")
    return policies


def main():
    parser = argparse.ArgumentParser(description="Point-foot policy validation in native CPU MuJoCo")
    tasks = [name for name, spec in pointleg.TASK_SPECS.items() if spec.enabled]
    parser.add_argument("--task", choices=tasks, default="PointLeg-Flat-v0")
    parser.add_argument("--checkpoint", type=Path, help="Point-foot MJLab .pt checkpoint or exported .onnx policy")
    parser.add_argument("--scene", type=Path, help="Terrain-only MJCF; required for SteppingStones")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--action-delay", type=int, default=0, help="Fixed control-step delay, 0 to 2")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation seconds, 0 for unlimited viewer mode")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer or keyboard")
    parser.add_argument("--check-model", action="store_true", help="Run zero actions without loading a policy")
    parser.add_argument("--multi-policy", action="store_true", help="Load policies configured in MULTI_POLICY_FILES")
    parser.add_argument("--stepping-trace", type=Path, help="Write per-substep contacts, landing errors and support handoffs to a new CSV")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0 or (args.headless and args.duration == 0):
        parser.error("--duration must be finite and positive in headless mode")
    if args.command is None:
        args.command = [STEPPING_DEFAULT_SPEED if args.task == STEPPING_TASK else 0.0, 0.0, 0.0]
    if not np.isfinite(args.command).all():
        parser.error("--command values must be finite")
    if args.check_model and (args.checkpoint is not None or args.multi_policy):
        parser.error("Use --check-model alone, without --checkpoint or --multi-policy")
    if args.checkpoint is not None and args.multi_policy:
        parser.error("Use either --checkpoint or --multi-policy")
    if args.checkpoint is None and not args.check_model and not args.multi_policy:
        parser.error("Pass --checkpoint, use --multi-policy, or pass --check-model")
    if args.multi_policy and args.task != "PointLeg-Flat-v0":
        parser.error("Multi-policy starts with Flat slot 1; use --task PointLeg-Flat-v0")
    if args.stepping_trace is None:
        run(args, parser)
        return
    if args.task != STEPPING_TASK and not args.multi_policy:
        parser.error("--stepping-trace requires SteppingStones or multi-policy mode")
    try:
        with args.stepping_trace.open("x", encoding="utf-8", newline="") as trace:
            writer = csv.writer(trace)
            writer.writerow(STEPPING_TRACE_FIELDS)
            run(args, parser, writer.writerow)
    except OSError as exc:
        parser.exit(2, f"Sim2Sim trace error: {exc}\n")


def run(args, parser, stepping_trace=None):
    try:
        sim = PointLegSimulation(args.task, args.scene, args.action_delay, stepping_trace=stepping_trace)
        policies = {}
        if args.multi_policy:
            policies = load_available_policies(POLICY_DIRECTORY, args.task)
        elif not args.check_model:
            width = stepping_policy_observation_dim(args.checkpoint) if args.task == STEPPING_TASK else sim.obs_dim
            policies["single"] = load_policy(args.checkpoint.resolve(), args.task, width)
            if sim.stepping is not None:
                sim.configure_stepping_policy(policies["single"])
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"Sim2Sim error: {exc}\n")
    command = np.array(args.command, dtype=np.float32)
    sim.command[:] = torch.from_numpy(command)
    controls = {
        "reset": False,
        "paused": False,
        "active_policy": "1" if args.multi_policy else "single",
        "requested_policy": None,
        "requested_spawn": None,
        "jump_requested": False,
    }
    jump_trigger = JumpTrigger()
    return_policy = None
    print(f"Task: {args.task}; observations: {sim.obs_dim}; actions: 12; control: {1 / sim.control_dt:.1f} Hz")
    print(f"Scene: {args.scene or 'flat floor'}; delay: {args.action_delay} steps")
    print(
        f"Motor: {LEG_MOTOR_SPEC.model} x12; supply: {MOTOR_SUPPLY_VOLTAGE_V:g} V; "
        f"simulation effort cap: {EFFORT_LEG:g} Nm (continuous rating unknown)"
    )
    print(
        f"Estimated no-load output speed: {ESTIMATED_NO_LOAD_SPEED_RAD_S:.2f} rad/s; "
        "reference only, not an enforced speed limit or a voltage-aware motor model"
    )
    if args.check_model:
        print("Model check: zero actions, no policy")
    elif args.multi_policy:
        print("Multi-policy mode: slot 4 selects Crawl; slot 6 selects the SteppingStones actor")
    else:
        print(f"Policy: {args.checkpoint}")

    def activate_policy(target_policy):
        if target_policy not in policies:
            filename = MULTI_POLICY_FILES[target_policy]
            print(f"Policy {target_policy} is unavailable: {POLICY_DIRECTORY / filename}")
            return
        if target_policy == controls["active_policy"]:
            return
        policy_task = MULTI_POLICY_TASKS.get(target_policy, args.task)
        was_stepping = sim.stepping is not None
        try:
            sim.switch_task(policy_task, transition=policy_task != JUMP_TASK)
        except ValueError as exc:
            print(f"Cannot switch to policy {target_policy}: {exc}")
            return
        controls["active_policy"] = target_policy
        if policy_task == STEPPING_TASK:
            sim.configure_stepping_policy(policies[target_policy])
            command[:] = (STEPPING_DEFAULT_SPEED, 0, 0)
        elif was_stepping:
            command.fill(0)
        print(f"Switched to policy {target_policy} ({policy_task})")

    def advance(velocity_command, requested_policy=None, jump_requested=False):
        nonlocal jump_trigger, return_policy
        if requested_policy is not None:
            activate_policy(requested_policy)
            velocity_command = command.copy()
            return_policy = None
        if jump_requested and sim.task != JUMP_TASK:
            return_policy = controls["active_policy"]
            activate_policy("jump")
            jump_trigger = JumpTrigger()
        jump_command = (
            jump_trigger.update(jump_requested, sim.mj_data.time)
            if sim.task == JUMP_TASK else (0., 0., 0.)
        )
        if sim.task == JUMP_TASK and not jump_command[0] and return_policy is not None:
            activate_policy(return_policy)
            return_policy = None
        if sim.task == JUMP_TASK:
            sim.command[0] = torch.tensor(jump_command)
        elif sim.in_transition:
            sim.command.zero_()
        elif sim.stepping is not None:
            sim.command[0] = torch.tensor((sim.stepping.command(velocity_command[0]), 0., 0.))
        else:
            sim.command[0] = torch.from_numpy(velocity_command)
        with torch.inference_mode():
            obs = sim.observe()
            stepping_done = sim.stepping is not None and (sim.stepping.finished or sim.stepping.failed)
            if args.check_model or sim.in_transition or stepping_done or (sim.task == JUMP_TASK and not jump_command[0]):
                action = torch.zeros(1, 12)
            else:
                action = policies[controls["active_policy"]](obs)
        sim.step(action)
        if sim.stepping is not None and not stepping_done:
            if sim.stepping.settle_timed_out:
                print("Stepping stones: exit stance did not settle. Press R to reset or select another policy.")
            elif sim.stepping.failed:
                print("Stepping stones: foot touched ground. Press R to reset or select another policy.")
            elif sim.stepping.finished:
                print("Stepping stones: all stones covered and the robot stood steadily on the exit platform.")

    if args.headless:
        while sim.mj_data.time < args.duration:
            advance(command)
    else:
        import threading
        import mujoco.viewer
        from pynput import keyboard

        lock = threading.Lock()
        ctrl_keys = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
        held_ctrl = set()
        held_spawn_keys = set()
        space_held = False
        jump_policy = "jump" if args.multi_policy else "single" if args.task == JUMP_TASK else None

        def switch_policy(target_policy):
            if target_policy not in policies:
                filename = MULTI_POLICY_FILES[target_policy]
                print(f"Policy {target_policy} is unavailable: {POLICY_DIRECTORY / filename}")
            else:
                controls["requested_policy"] = target_policy
                print(f"Queued policy {target_policy}")

        def on_press(key):
            nonlocal space_held
            with lock:
                if key in ctrl_keys:
                    held_ctrl.add(key)
                    return
                slot = spawn_key(key)
                if slot in held_spawn_keys:
                    return
                if held_ctrl:
                    if slot is not None:
                        held_spawn_keys.add(slot)
                        if slot in sim.spawn_sites:
                            controls["requested_spawn"] = slot
                        else:
                            print(f"Spawn {slot} unavailable in the current scene")
                    return
                if key == keyboard.Key.space or getattr(key, "char", None) == " ":
                    if not space_held:
                        if jump_policy in policies:
                            command.fill(0)
                            controls["jump_requested"] = True
                        else:
                            print(f"Jump unavailable: load a compatible {JUMP_TASK} checkpoint as "
                                  f"{POLICY_DIRECTORY / MULTI_POLICY_FILES['jump']} in multi-policy mode")
                    space_held = True
                    return
                motion_keys = {
                    keyboard.Key.up: (0, 0.1),
                    keyboard.Key.down: (0, -0.1),
                    keyboard.Key.left: (1, 0.1),
                    keyboard.Key.right: (1, -0.1),
                }
                if key in motion_keys:
                    axis, delta = motion_keys[key]
                    command[axis] += delta
                    print(f"Command vx/vy/wz: {command.round(2).tolist()}")
                    return
                if not hasattr(key, "char") or key.char is None:
                    return
                char = key.char.lower()
                if args.multi_policy and slot in {"1", "2", "3", "4", "5", "6"}:
                    switch_policy(slot)
                elif char == "q":
                    command[2] += 0.1
                elif char == "e":
                    command[2] -= 0.1
                elif char == "z":
                    command[:] = 0.0
                elif char == "r":
                    controls["reset"] = True
                elif char == "p":
                    controls["paused"] = not controls["paused"]
                    state = "paused" if controls["paused"] else "running"
                    print(f"Simulation {state}")
                print(f"Command vx/vy/wz: {command.round(2).tolist()}")

        def on_release(key):
            nonlocal space_held
            with lock:
                held_ctrl.discard(key)
                held_spawn_keys.discard(spawn_key(key))
                if key == keyboard.Key.space or getattr(key, "char", None) == " ":
                    space_held = False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print("Global keyboard: Up/Down forward, Left/Right lateral, Q/E yaw, Z stop, R reset, P pause/resume, Space jump; speed increment 0.1")
        if args.multi_policy:
            print("Global multi-policy keys: 1-6 select policy slots")
        if sim.spawn_sites:
            print("Spawn presets: " + ", ".join(
                f"Ctrl+{slot.upper()} {SPAWN_SITES[slot][1]}" for slot in sim.spawn_sites
            ))
        elapsed_sim = 0.0
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.mj_data) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -20.0
                while viewer.is_running() and (args.duration == 0 or elapsed_sim < args.duration):
                    started = time.perf_counter()
                    with lock:
                        reset = controls["reset"]
                        spawn = controls["requested_spawn"]
                        paused = controls["paused"]
                        if reset or spawn is not None:
                            controls["reset"] = False
                            controls["requested_spawn"] = None
                            queued_task = MULTI_POLICY_TASKS.get(controls["requested_policy"])
                            if reset or spawn != STEPPING_ENTRY_SLOT or queued_task != STEPPING_TASK:
                                controls["requested_policy"] = None
                            controls["jump_requested"] = False
                            if spawn is not None:
                                command.fill(0)
                                if spawn == STEPPING_ENTRY_SLOT and sim.stepping is not None:
                                    command[0] = STEPPING_DEFAULT_SPEED
                        requested_policy = controls["requested_policy"]
                        jump_requested = controls["jump_requested"]
                        velocity_command = command.copy()
                        if not paused:
                            controls["requested_policy"] = None
                            controls["jump_requested"] = False
                    if reset or spawn is not None:
                        if reset and args.multi_policy:
                            activate_policy("1")
                        elif return_policy is not None:
                            activate_policy(return_policy)
                        if spawn is not None:
                            sim.spawn_slot = spawn
                            print(f"Spawn Ctrl+{spawn}: {SPAWN_SITES[spawn][1]}")
                        sim.reset()
                        jump_trigger = JumpTrigger()
                        return_policy = None
                    if not paused:
                        advance(velocity_command, requested_policy, jump_requested)
                        elapsed_sim += sim.control_dt
                    viewer.cam.lookat[:] = sim.mj_data.xpos[sim.model.body(BASE_BODY_NAME).id]
                    viewer.opt.geomgroup[:] = 1
                    viewer.sync()
                    time.sleep(max(0.0, sim.control_dt - (time.perf_counter() - started)))
        finally:
            listener.stop()
    body_id = sim.model.body(BASE_BODY_NAME).id
    print(f"Completed {sim.mj_data.time:.2f}s; base xyz: {sim.mj_data.xpos[body_id].round(3).tolist()}; finite states, no MuJoCo warnings")


if __name__ == "__main__":
    main()
