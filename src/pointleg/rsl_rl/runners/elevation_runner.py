"""Persist elevation curricula without restoring them for inference or transfer."""

from dataclasses import asdict
import math
import warnings

import torch

from mjlab.rl import MjlabOnPolicyRunner


class ElevationOnPolicyRunner(MjlabOnPolicyRunner):
    def _walking_contract(self):
        settings = asdict(self.env.unwrapped._elevation_settings)
        rewards = self.env.unwrapped.cfg.rewards
        if "swing_clearance" not in rewards:
            revision = 9 if "forward_stall" in rewards else 8
            gait_style = "basic_slope_anti_stall" if revision == 9 else "basic_forward_walking"
            if "step_timing" in rewards:
                revision, gait_style = 11, "slope_longer_steps_and_alignment"
                if settings["terrain_kind"] == "stairs":
                    revision, gait_style = 34, "stairs_right_front_joint_limits"
            completion = "four_loaded_feet_on_exit_platform"
            if settings["terrain_kind"] == "stairs":
                completion = "first_three_treads_covered_and_four_loaded_feet_on_middle_platform"
            return {"revision": revision, "terrain_kind": settings["terrain_kind"],
                    "flat_rehearsal": False, "gait_style": gait_style,
                    "speed_range": settings["speed_range"],
                    "walking_approach_distance": settings["walking_approach_distance"],
                    "rewards": {name: {"weight": term.weight, "params": dict(term.params)}
                                for name, term in rewards.items()},
                    **({"episode_length_s": self.env.unwrapped.cfg.episode_length_s} if revision >= 10 else {}),
                    **({"completion": completion, "completion_force_n": 8.,
                        "completion_hold_s": 0.} if revision >= 11 else {}),
                    **({"stair_clearance_termination": False} if revision >= 13 else {}),
                    **{key: value for key, value in settings.items() if key.startswith("stall_")}}
        return {"revision": 7, "flat_rehearsal": settings["terrain_kind"] is None, "gait_style": "functional_clearance_and_soft_landing",
                "speed_range": settings["speed_range"],
                "walking_approach_distance": settings["walking_approach_distance"],
                "target_clearance_m": self.env.unwrapped.cfg.rewards["swing_clearance"].params["target_clearance"],
                "height_reference": "takeoff_support_normal", "landing_edge_margin_m": rewards["swing_clearance"].params["landing_edge_margin"],
                "body_clearance": dict(rewards["body_clearance"].params),
                "touchdown_speed": dict(rewards["touchdown_speed"].params),
                "body_clearance_weight": rewards["body_clearance"].weight,
                "touchdown_speed_weight": rewards["touchdown_speed"].weight,
                "contact_cost_aggregation": "half_group_means_half_worst",
                **{key: value for key, value in settings.items() if key.startswith("stall_")}}

    def _layout_contract(self):
        generator = self.env.unwrapped.cfg.scene.terrain.terrain_generator
        return {name: generator.sub_terrains[name].layout_contract()
                for name in ("stairs_up", "stairs_down")}

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        raw = env.unwrapped
        expected = raw._elevation_settings.gamma ** (raw.step_dt / 0.02)
        if not math.isclose(train_cfg["algorithm"]["gamma"], expected, rel_tol=1e-7):
            raise ValueError(f"PPO gamma must match the task shaping discount {expected}; update both when changing the control period")
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path, infos=None):
        raw = self.env.unwrapped
        extra = dict(infos or {})
        if hasattr(raw, "_elevation_curriculum"):
            extra["elevation_curriculum"] = raw._elevation_curriculum.state_dict()
        extra["elevation_settings"] = asdict(raw._elevation_settings)
        extra["policy_contract"] = {"actor": 45, "critic": 325, "actions": 12}
        extra["elevation_layout"] = self._layout_contract()
        if not raw._elevation_settings.jump:
            extra["elevation_walking"] = self._walking_contract()
        if hasattr(raw, "scene"):
            extra["wall_height_override"] = raw.scene.terrain.cfg.terrain_generator.sub_terrains["wall_sequence"].wall_height_override
        else:
            extra["wall_height_override"] = raw.cfg.scene.terrain.terrain_generator.sub_terrains["wall_sequence"].wall_height_override
        super().save(path, infos=extra)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        raw = self.env.unwrapped
        restore = load_cfg is None or bool(load_cfg.get("iteration", False))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        infos = saved.get("infos") or {}
        if restore:
            if "elevation_curriculum" not in infos:
                raise ValueError("Use pointleg transfer for initial Rough migration; resume needs an elevation checkpoint")
            if infos.get("elevation_settings", {}).get("jump") != raw._elevation_settings.jump:
                raise ValueError("Resume requires the same elevation or jump task")
            if infos.get("elevation_settings", {}).get("terrain_kind") != raw._elevation_settings.terrain_kind:
                raise ValueError("Terrain specialization changed; use weights-only transfer instead of resume")
            expected = raw.cfg.scene.terrain.terrain_generator.sub_terrains["wall_sequence"].wall_height_override
            if infos.get("wall_height_override") != expected:
                raise ValueError("Wall geometry changed; use weights-only transfer instead of resume")
            if not raw._elevation_settings.jump and infos.get("elevation_layout") != self._layout_contract():
                raise ValueError("Stair geometry changed; use weights-only transfer instead of resume")
            if not raw._elevation_settings.jump:
                course = infos["elevation_curriculum"]
                if hasattr(raw, "_elevation_curriculum"):
                    expected_count = raw._elevation_curriculum.level_caps.numel()
                    if (len(course.get("level_caps", ())) != expected_count
                            or len(course.get("windows", ())) != expected_count
                            or course.get("window_size", raw._elevation_curriculum.window_size) != raw._elevation_curriculum.window_size):
                        raise ValueError("Curriculum structure changed; use weights-only transfer instead of resume")
                if infos.get("elevation_walking") != self._walking_contract():
                    warnings.warn(
                        "Elevation reward settings changed. Resuming network, optimizer, iteration and curriculum "
                        "with the current reward configuration; reward-term episode state starts fresh.",
                        UserWarning, stacklevel=2,
                    )
        counter, iteration = raw.common_step_counter, self.current_learning_iteration
        result = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        if restore and hasattr(raw, "_elevation_curriculum"):
            raw._elevation_curriculum.load_state_dict(infos["elevation_curriculum"])
            if (raw._elevation_settings.terrain_kind == "stairs"
                    and (infos.get("elevation_walking") or {}).get("completion")
                    != self._walking_contract().get("completion")):
                for name in ("stairs_up", "stairs_down"):
                    raw._elevation_curriculum.windows[raw._elevation_curriculum.terrain_names.index(name)].clear()
                warnings.warn("Stair completion changed; cleared stair curriculum success windows while retaining unlocked levels.",
                              UserWarning, stacklevel=2)
        if not restore:
            raw.common_step_counter, self.current_learning_iteration = counter, iteration
        return result
