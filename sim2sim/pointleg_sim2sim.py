"""Point-foot inference using the task's actual observations and action pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from pointleg.robot_cfg import (
    BASE_BODY_NAME,
    EFFORT_LEG,
    ESTIMATED_NO_LOAD_SPEED_RAD_S,
    LEG_JOINT_NAMES,
    LEG_MOTOR_SPEC,
    MOTOR_SUPPLY_VOLTAGE_V,
)
from rsl_rl.models import MLPModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIRECTORY = PROJECT_ROOT / "sim2sim" / "policy" / "optical_policy"
MULTI_POLICY_FILES = {
    "1": "model_1999.pt",   #平动flat   
    "2": "model_14100.pt",  #高抬腿high_step
    "3": "model_4997.pt",   #坡slope
    "4": "model_2694.pt",   #限高框架crawl
    "5": "model_5.pt",
    "6": "model_6.pt",

    "jump": "model_jump.pt",
}
MULTI_POLICY_TASKS = {"4": "PointLeg-Crawl-v0"}
POLICY_SWITCH_DURATION_S = 1.0

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

    def __init__(self, task: str, scene: Path | None = None, action_delay: int = 0):
        if task == "PointLeg-Jump-v0":
            raise ValueError("Platform Jump needs trigger/time commands and platform contact rules; use pointleg play/evaluate PointLeg-Jump-v0")
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
        self._transition_start_ctrl = None
        self._transition_step = 0
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
        reference_model = self.model if task == self._initial_task else entity.spec.compile()
        self._task_key_qpos = reference_model.key_qpos[0].copy()
        default_pos = np.array([
            reference_model.key_qpos[0, reference_model.jnt_qposadr[reference_model.joint(name).id]]
            for name in LEG_JOINT_NAMES
        ])
        previous_control_dt = getattr(self, "control_dt", None)
        cfg.sim.mujoco.apply(self.model)
        self.task = task
        self.cfg = cfg
        self.robot = CpuRobot(entity, self.model, self.mj_data, default_pos=default_pos)
        self.command = torch.zeros(1, 3)
        self.env = SimpleNamespace(
            num_envs=1, device="cpu", scene={"robot": self.robot},
            command_manager=SimpleNamespace(get_command=lambda name: self.command),
        )
        action_cfg = cfg.actions["leg_joint_pos"]
        training_action = load_env_cfg(task).actions["leg_joint_pos"]
        if not training_action.min_delay <= self._action_delay <= training_action.max_delay:
            raise ValueError(f"Action delay must be in [{training_action.min_delay}, {training_action.max_delay}]")
        action_cfg.min_delay = action_cfg.max_delay = self._action_delay
        self.control_dt = self.model.opt.timestep * cfg.decimation
        if previous_control_dt is not None and not np.isclose(self.control_dt, previous_control_dt):
            raise ValueError("Cannot switch between tasks with different control periods")
        if not np.isclose(action_cfg.control_frequency, 1.0 / self.control_dt):
            raise ValueError("Action filter frequency does not match the simulation control rate")
        self.env.action_manager = ActionManager(cfg.actions, self.env)
        self.observations = ObservationManager({"actor": cfg.observations["actor"]}, self.env)
        self.obs_dim = self.observations.group_obs_dim["actor"][0]
        if self.obs_dim != 45:
            raise ValueError(f"Expected current-frame point-foot observations (45), got {self.obs_dim}")
        self.env.action_manager.reset()
        self.observations.reset()

    def switch_task(self, task: str):
        """Switch policy semantics while preserving the physical state."""
        if task == self.task:
            return
        start_ctrl = self.mj_data.ctrl.copy()
        pose = self.mj_data.qpos.copy()
        velocity = self.mj_data.qvel.copy()
        self._configure_task(task)
        if not np.array_equal(self.mj_data.qpos, pose) or not np.array_equal(self.mj_data.qvel, velocity):
            raise RuntimeError("Task switch unexpectedly changed the robot state")
        self._transition_start_ctrl = start_ctrl
        self._transition_step = 0

    @property
    def in_transition(self) -> bool:
        return self._transition_start_ctrl is not None

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.mj_data, 0)
        self.mj_data.qpos[:] = self._task_key_qpos
        self.mj_data.ctrl[self.robot.ctrl_ids] = self.robot.data.default_joint_pos[0].numpy()
        mujoco.mj_forward(self.model, self.mj_data)
        self.robot.update()
        self.env.action_manager.reset()
        self.observations.reset()
        self._transition_start_ctrl = None
        self._transition_step = 0

    def observe(self):
        self.robot.update()
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
        if self._transition_start_ctrl is not None and self._transition_step * self.control_dt >= POLICY_SWITCH_DURATION_S:
            self._transition_start_ctrl = None
        mujoco.mj_forward(self.model, self.mj_data)
        if not np.isfinite(self.mj_data.qpos).all() or not np.isfinite(self.mj_data.qvel).all():
            raise RuntimeError("Simulation produced non-finite states")
        if any(warning.number for warning in self.mj_data.warning):
            raise RuntimeError("MuJoCo reported a simulation warning")


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
        jump_contract = session.get_modelmeta().custom_metadata_map.get("pointleg_jump_contract")
        if (task == "PointLeg-Jump-v0") != bool(jump_contract):
            raise ValueError("Jump uses trigger/time commands; task and ONNX observation contract do not match")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("ONNX must accept one float32 observation tensor and return one action tensor")
        for metadata, width in ((inputs[0], obs_dim), (outputs[0], 12)):
            if len(metadata.shape) != 2 or (isinstance(metadata.shape[1], int) and metadata.shape[1] != width):
                raise ValueError("Point-foot ONNX requires 45 observations and 12 actions; old wheel-leg models are incompatible")

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
        jump_contract = (saved.get("infos") or {}).get("jump_contract")
        if (task == "PointLeg-Jump-v0") != bool(jump_contract):
            raise ValueError("Jump uses trigger/time commands; task and checkpoint observation contract do not match")
        try:
            actor.load_state_dict(saved["actor_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError("Checkpoint does not match this task's current actor configuration (45 observations, 12 actions)") from exc
        policy = actor.eval().as_jit().eval()
    with torch.inference_mode():
        probe = policy(torch.zeros(1, obs_dim))
    if probe.shape != (1, 12) or not torch.isfinite(probe).all():
        raise ValueError("Policy did not produce 12 finite point-foot actions")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Point-foot policy validation in native CPU MuJoCo")
    tasks = [name for name, spec in pointleg.TASK_SPECS.items() if spec.enabled]
    parser.add_argument("--task", choices=tasks, default="PointLeg-Flat-v0")
    parser.add_argument("--checkpoint", type=Path, help="Point-foot MJLab .pt checkpoint or exported .onnx policy")
    parser.add_argument("--scene", type=Path, help="Terrain-only MJCF; default is a flat floor for all tasks")
    parser.add_argument("--command", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--action-delay", type=int, default=0, help="Fixed control-step delay, 0 to 2")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation seconds, 0 for unlimited viewer mode")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer or keyboard")
    parser.add_argument("--check-model", action="store_true", help="Run zero actions without loading a policy")
    parser.add_argument("--multi-policy", action="store_true", help="Load policies configured in MULTI_POLICY_FILES")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0 or (args.headless and args.duration == 0):
        parser.error("--duration must be finite and positive in headless mode")
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
    try:
        sim = PointLegSimulation(args.task, args.scene, args.action_delay)
        policies = {}
        if args.multi_policy:
            for slot, filename in MULTI_POLICY_FILES.items():
                checkpoint = POLICY_DIRECTORY / filename
                if checkpoint.is_file():
                    policy_task = MULTI_POLICY_TASKS.get(slot, args.task)
                    policies[slot] = load_policy(checkpoint, policy_task, sim.obs_dim)
                    print(f"Loaded policy {slot}: {checkpoint.name} ({policy_task})")
                else:
                    print(f"Policy {slot} unavailable: {checkpoint}")
            if "1" not in policies:
                raise FileNotFoundError(f"Slot 1 policy not found: {POLICY_DIRECTORY / MULTI_POLICY_FILES['1']}")
        elif not args.check_model:
            policies["single"] = load_policy(args.checkpoint.resolve(), args.task, sim.obs_dim)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"Sim2Sim error: {exc}\n")
    command = np.array(args.command, dtype=np.float32)
    sim.command[:] = torch.from_numpy(command)
    controls = {
        "reset": False,
        "paused": False,
        "active_policy": "1" if args.multi_policy else "single",
        "requested_policy": None,
    }
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
        print("Multi-policy mode: slot 4 switches to Crawl observations and action references")
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
        sim.switch_task(policy_task)
        controls["active_policy"] = target_policy
        print(f"Switched to policy {target_policy} ({policy_task}); crouching transition: {POLICY_SWITCH_DURATION_S:.2f}s")

    def advance():
        requested_policy = controls["requested_policy"]
        controls["requested_policy"] = None
        if requested_policy is not None:
            activate_policy(requested_policy)
        sim.command[:] = torch.from_numpy(np.zeros(3, dtype=np.float32) if sim.in_transition else command)
        with torch.inference_mode():
            obs = sim.observe()
            if args.check_model:
                action = torch.zeros(1, 12)
            else:
                action = torch.zeros(1, 12) if sim.in_transition else policies[controls["active_policy"]](obs)
        sim.step(action)

    if args.headless:
        while sim.mj_data.time < args.duration:
            advance()
    else:
        import threading
        import mujoco.viewer
        from pynput import keyboard

        lock = threading.Lock()

        def switch_policy(target_policy):
            if target_policy not in policies:
                filename = MULTI_POLICY_FILES[target_policy]
                print(f"Policy {target_policy} is unavailable: {POLICY_DIRECTORY / filename}")
            else:
                controls["requested_policy"] = target_policy
                print(f"Queued policy {target_policy}")

        def on_press(key):
            with lock:
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
                if args.multi_policy and char in {"1", "2", "3", "4", "5", "6"}:
                    switch_policy(char)
                elif args.multi_policy and char == "j":
                    switch_policy("jump")
                elif char == "q":
                    command[2] += 0.1
                elif char == "e":
                    command[2] -= 0.1
                elif char in {"z", " "}:
                    command[:] = 0.0
                elif char == "r":
                    controls["reset"] = True
                elif char == "p":
                    controls["paused"] = not controls["paused"]
                    state = "paused" if controls["paused"] else "running"
                    print(f"Simulation {state}")
                print(f"Command vx/vy/wz: {command.round(2).tolist()}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("Global keyboard: Up/Down forward, Left/Right lateral, Q/E yaw, Z/Space stop, R reset, P pause/resume; each press changes 0.1")
        if args.multi_policy:
            print("Global multi-policy keys: 1-6 select slots 1-6, J selects the jump policy")
        elapsed_sim = 0.0
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.mj_data) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -20.0
                while viewer.is_running() and (args.duration == 0 or elapsed_sim < args.duration):
                    started = time.perf_counter()
                    with lock:
                        if controls["reset"]:
                            if args.multi_policy:
                                sim.switch_task(args.task)
                                controls["active_policy"] = "1"
                                controls["requested_policy"] = None
                            sim.reset()
                            controls["reset"] = False
                        paused = controls["paused"]
                    if not paused:
                        advance()
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
