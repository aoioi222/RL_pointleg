"""Checkpoint contracts for the fixed triggered platform jump."""

from dataclasses import asdict
import math

import torch

from mjlab.rl import MjlabOnPolicyRunner

from ...terrains.jump import TILE_SIZE, TERRAIN_NAMES, LEVEL_COUNT, LAUNCH_EDGE, SPAWN_X


class JumpOnPolicyRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        raw = env.unwrapped
        expected = raw._jump_settings.gamma ** (raw.step_dt / .02)
        if not math.isclose(train_cfg["algorithm"]["gamma"], expected, rel_tol=1e-7):
            raise ValueError("Jump PPO gamma must match the potential discount")
        super().__init__(env, train_cfg, log_dir, device)

    def contract(self):
        cfg = self.env.unwrapped.cfg
        return {"revision": 1, "task": "triggered_platform_jump", "actor": 45, "critic": 325, "actions": 12,
                "command": "trigger, elapsed_seconds/3, zero", "settings": asdict(self.env.unwrapped._jump_settings),
                "tile_size": TILE_SIZE, "terrain_names": TERRAIN_NAMES, "level_count": LEVEL_COUNT,
                "launch_edge": LAUNCH_EDGE, "spawn_x": SPAWN_X,
                "height_range": (.030, .200), "gap_range": (.054, .360), "wall_rise_range": (.045, .300),
                "wall_thickness": .080, "wall_gap_min": .080,
                "sole_offset": cfg.events["reset_jump"].params["sole_offset"],
                "episode_duration": cfg.episode_length_s,
                "rewards": {name: term.weight for name, term in cfg.rewards.items()},
                "action": {name: asdict(term) for name, term in cfg.actions.items()}}

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """A trigger always starts at time zero, including the first rollout."""
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def save(self, path, infos=None):
        raw = self.env.unwrapped
        extra = dict(infos or {})
        extra["jump_contract"] = self.contract()
        if hasattr(raw, "_jump_curriculum"):
            extra["jump_curriculum"] = raw._jump_curriculum.state_dict()
        super().save(path, infos=extra)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        raw = self.env.unwrapped
        restore = load_cfg is None or bool(load_cfg.get("iteration", False))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        infos = saved.get("infos") or {}
        if restore:
            if infos.get("jump_contract") != self.contract() or "jump_curriculum" not in infos:
                raise ValueError("Jump geometry, rewards or policy contract changed; use pointleg transfer")
        elif load_cfg and load_cfg.get("actor") and not load_cfg.get("critic"):
            previous = infos.get("jump_contract")
            if previous is None or previous.get("command") != self.contract()["command"]:
                raise ValueError("Inference needs a triggered-platform Jump actor; use transfer for old policies")
        counter, iteration = raw.common_step_counter, self.current_learning_iteration
        result = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        if restore and hasattr(raw, "_jump_curriculum"):
            raw._jump_curriculum.load_state_dict(infos["jump_curriculum"])
        if not restore:
            raw.common_step_counter, self.current_learning_iteration = counter, iteration
        return result
