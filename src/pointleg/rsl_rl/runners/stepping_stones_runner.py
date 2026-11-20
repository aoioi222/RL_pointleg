"""Keep stepping-stone contact observations paired with their checkpoints."""

from mjlab.rl import MjlabOnPolicyRunner

from ...mdp.stepping_stones import SUPPORT_PROGRESS_INFO_KEY
from ...mdp.fixed_stone_gait import FIXED_GAIT_INFO_KEY


class SteppingStonesOnPolicyRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        raw = self.env.unwrapped
        manager = raw.observation_manager
        term = manager.get_term_cfg("actor", "foot_contact")
        extra = {**(infos or {}), SUPPORT_PROGRESS_INFO_KEY: term.params["support_progress"],
                 FIXED_GAIT_INFO_KEY: 1}
        if hasattr(raw, "_stepping_stone_curriculum"):
            extra["stepping_stone_curriculum"] = raw._stepping_stone_curriculum.state_dict()
        super().save(path, infos=extra)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        raw = self.env.unwrapped
        restore = load_cfg is None or bool(load_cfg.get("iteration", False))
        counter, iteration = raw.common_step_counter, self.current_learning_iteration
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        course = (infos or {}).get("stepping_stone_curriculum")
        if restore and course is not None and hasattr(raw, "_stepping_stone_curriculum"):
            raw._stepping_stone_curriculum.load_state_dict(course)
        if not restore:
            raw.common_step_counter, self.current_learning_iteration = counter, iteration
        if load_cfg and load_cfg.get("actor") and not load_cfg.get("critic"):
            manager = self.env.unwrapped.observation_manager
            support_progress = bool((infos or {}).get(SUPPORT_PROGRESS_INFO_KEY))
            actor = manager.get_term_cfg("actor", "foot_contact")
            critic = manager.get_term_cfg("critic", "foot_contact")
            if actor.params["support_progress"] != support_progress:
                actor.params["support_progress"] = support_progress
                critic.params["support_progress"] = support_progress
                manager.reset()
        return infos
