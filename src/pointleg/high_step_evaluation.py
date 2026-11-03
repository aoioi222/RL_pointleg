"""Per-foot acceptance criteria for the high-step foundation."""

import math

from .robot_cfg import LEG_NAMES


def gait_accepted(summary):
    """Require every leg to meet clearance, rhythm and cycle coverage criteria."""
    if summary and summary.get("profile") == "basic":
        # 基础阶段只验收持续有效换步，不沿用旧15厘米和0.6秒目标。
        return all(
            foot["valid_cycles"] >= 5
            and all(math.isfinite(float(v)) for v in foot.values())
            for foot in summary["feet"].values()
        ) and len(summary["feet"]) == 4
    if not summary or not math.isfinite(summary["diagonal_error"]) or summary["diagonal_error"] > .15:
        return False
    feet = summary["feet"]
    for leg in LEG_NAMES:
        foot = feet[leg]
        if not all(math.isfinite(float(v)) for v in foot.values()):
            return False
        if foot["cycles"] < 5 or foot["qualified_fraction"] < .8:
            return False
        if not .8 * summary["target_height"] <= foot["height_m"] <= 1.2 * summary["target_height"]:
            return False
        if abs(foot["cycle_s"] - summary["target_cycle"]) > summary["cycle_tolerance"]:
            return False
    for left, right in (("fl", "fr"), ("rl", "rr")):
        for key, floor in (("height_m", .01), ("cycle_s", .02), ("stride_m", .01)):
            a, b = feet[left][key], feet[right][key]
            if abs(a - b) / max((abs(a) + abs(b)) / 2., floor) > .2:
                return False
    return True


def snapshot_high_step(env, index):
    """Read episode accumulators before the evaluator resets a finished env."""
    basic = getattr(env, "_high_step_basic_term", None)
    if basic is not None:
        summary = {"profile": "basic", "height_reward_enabled": True, "feet": {}}
        for i, leg in enumerate(LEG_NAMES):
            count = basic.stats["cycles"][index, i]
            summary["feet"][leg] = {
                "cycles": int(count),
                "valid_cycles": int(basic.stats["valid_cycles"][index, i]),
                **{f"{key}_{unit}": float(basic.stats[f"{key}_sum"][index, i] / count.clamp_min(1.))
                   for key, unit in (("height", "m"), ("cycle", "s"), ("stride", "m"))},
            }
        summary["accepted"] = gait_accepted(summary)
        return summary
    term = env._high_step_cycle_term
    diagonal = env._diagonal_gait_term
    params = env.reward_manager.get_term_cfg("high_step_cycle").params
    summary = {
        "target_height": params["target_height"], "target_cycle": params["target_cycle"],
        "cycle_tolerance": params["cycle_std"],
        "diagonal_error": float(diagonal.cost_sum[index] / diagonal.samples[index].clamp_min(1.)),
        "feet": {},
    }
    if diagonal.samples[index] == 0:
        summary["diagonal_error"] = 1.
    for i, leg in enumerate(LEG_NAMES):
        count = term.stats["cycles"][index, i]
        summary["feet"][leg] = {
            "cycles": int(count),
            "qualified_fraction": float(term.stats["qualified"][index, i] / count.clamp_min(1.)),
            **{f"{key}_{unit}": float(term.stats[f"{key}_sum"][index, i] / count.clamp_min(1.))
               for key, unit in (("height", "m"), ("cycle", "s"), ("stride", "m"))},
        }
    summary["accepted"] = gait_accepted(summary)
    return summary
