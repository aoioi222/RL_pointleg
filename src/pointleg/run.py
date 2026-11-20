"""Launch installed point-foot tasks with the MJLab runtime.

Usage:
    python -m pointleg.run list
    python -m pointleg.run train PointLeg-Rough-v0
    python -m pointleg.run play PointLeg-Rough-v0 --checkpoint MODEL
    python -m pointleg.run evaluate PointLeg-Rough-v0 --checkpoint-file MODEL

Set POINTLEG_XML before launch when the completed XML lives outside mjcf.
"""

from __future__ import annotations

import runpy
import sys


COMPETITION_TRAINING_FLAGS = ("--competition-six", "--competition-six-curriculum")


def _run_training_with_presets(
    task_id: str, arguments: list[str], *, debug: bool = False, competition_flag: str | None = None,
) -> None:
    """Apply training presets before parsing the native overrides."""
    import mjlab
    import tyro

    from mjlab.scripts.train import TrainConfig, launch_training

    defaults = TrainConfig.from_task(task_id)
    program = f"pointleg train {task_id}"
    if debug:
        from pointleg.config.debug import configure_debug_training

        configure_debug_training(defaults.env, defaults.agent)
        program += " --debug"
    if competition_flag is not None:
        from pointleg.config.stepping_stones import configure_competition_six

        configure_competition_six(defaults.env, gap_curriculum=competition_flag == "--competition-six-curriculum")
        program += f" {competition_flag}"
    args = tyro.cli(
        TrainConfig, args=arguments, default=defaults,
        prog=program, config=mjlab.TYRO_FLAGS,
    )
    launch_training(task_id, args)


def run_debug_training(task_id: str, arguments: list[str]) -> None:
    """Apply debug defaults before parsing the native training overrides."""
    _run_training_with_presets(task_id, arguments, debug=True)


def run_jump_warmup(arguments: list[str]) -> None:
    """Reject the retired wall-free rehearsal instead of launching the wrong task."""
    raise SystemExit("Jump 已使用从15%开始的平台课程，请移除 --warmup；旧模型需通过 transfer 迁移。")


def main() -> None:
    """Dispatch task listing, training, or playback to the installed MJLab scripts."""
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "train", "play", "evaluate", "transfer"):
        raise SystemExit("Usage: pointleg {list|train|play|evaluate|transfer} [task_id] [options]")
    command = sys.argv.pop(1)
    import pointleg
    task_specs = pointleg.TASK_SPECS

    if command == "list":
        from mjlab.tasks.registry import list_tasks

        for task_id in list_tasks():
            if task_id.startswith("PointLeg-"):
                print(task_id)
        for task_id, task_spec in task_specs.items():
            if not task_spec.enabled:
                print(f"{task_id} [占位，未启用] {task_spec.description}")
        return
    if len(sys.argv) > 1:
        task_spec = task_specs.get(sys.argv[1])
        if task_spec is not None and not task_spec.enabled:
            raise SystemExit(
                f"{sys.argv[1]} 是占位任务，尚不能训练或回放。"
                f"请实现 config/env_cfgs.py 中的 {task_spec.env_cfg_factory.__name__}，"
                "再将 __init__.py 中该任务的 enabled 改为 True。"
            )
    if command == "transfer":
        if len(sys.argv) < 2:
            raise SystemExit("Usage: pointleg transfer TASK_ID --checkpoint-file PATH")
        from pointleg.transfer import main as run_transfer

        run_transfer(sys.argv[1], sys.argv[2:])
        return
    if command == "evaluate":
        if len(sys.argv) < 2:
            raise SystemExit("Usage: pointleg evaluate TASK_ID --checkpoint-file PATH")
        from pointleg.terrains.elevation import ELEVATION_TASK_TERRAINS

        if sys.argv[1] == "PointLeg-Jump-v0":
            from pointleg.evaluate_jump import main as run_evaluate
        elif sys.argv[1] in ELEVATION_TASK_TERRAINS:
            from pointleg.evaluate_elevation import main as run_evaluate
        else:
            from pointleg.evaluate import main as run_evaluate

        run_evaluate(sys.argv[1], sys.argv[2:])
        return
    if "--warmup" in sys.argv[1:]:
        if command != "train" or len(sys.argv) < 2 or sys.argv[1] != "PointLeg-Jump-v0":
            raise SystemExit("--warmup requires: pointleg train PointLeg-Jump-v0 --warmup")
        run_jump_warmup([argument for argument in sys.argv[2:] if argument != "--warmup"])
        return
    competition_flags = [flag for flag in COMPETITION_TRAINING_FLAGS if flag in sys.argv[1:]]
    if competition_flags:
        if len(competition_flags) > 1:
            raise SystemExit("Choose one competition training preset.")
        competition_flag = competition_flags[0]
        if command != "train" or len(sys.argv) < 2 or sys.argv[1] != "PointLeg-SteppingStones-v0":
            raise SystemExit(f"{competition_flag} requires: pointleg train PointLeg-SteppingStones-v0 {competition_flag}")
        flags = {competition_flag, "--debug"}
        arguments = [argument for argument in sys.argv[2:] if argument not in flags]
        _run_training_with_presets(
            sys.argv[1], arguments, debug="--debug" in sys.argv[2:], competition_flag=competition_flag,
        )
        return
    if "--debug" in sys.argv[1:]:
        if command != "train" or len(sys.argv) < 2 or sys.argv[1] not in task_specs:
            raise SystemExit("--debug requires: pointleg train PointLeg-<task>-v0 --debug")
        arguments = [argument for argument in sys.argv[2:] if argument != "--debug"]
        run_debug_training(sys.argv[1], arguments)
        return
    runpy.run_module(f"mjlab.scripts.{command}", run_name="__main__")


if __name__ == "__main__":
    main()
