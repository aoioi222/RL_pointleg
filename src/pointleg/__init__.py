"""点足障碍赛任务入口，通过 mjlab.tasks 插件入口加载。

TASK_SPECS 集中维护核心技能、高程专项和跳跃任务。
环境和 PPO 工厂位于 config/，高程专项各自提供独立配置模块。
占位任务完成地形、奖励、规则和终止配置后，将 enabled 改为 True 即可注册。
恢复、返回启动区和快速保守模式由后续任务调度处理，不单独注册模型。
"""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from .config.env_cfgs import (
    bridge_env_cfg,
    crawl_env_cfg,
    elevation_env_cfg,
    flat_env_cfg,
    jump_env_cfg,
    rough_env_cfg,
    stepping_stones_env_cfg,
)
from .config.rl_cfg import (
    bridge_ppo_runner_cfg,
    crawl_ppo_runner_cfg,
    elevation_ppo_runner_cfg,
    flat_ppo_runner_cfg,
    high_step_ppo_runner_cfg,
    jump_ppo_runner_cfg,
    rough_ppo_runner_cfg,
    stepping_stones_ppo_runner_cfg,
)
from .config.mixed_obstacles import rough_mixed_env_cfg
from .config.high_step import high_step_env_cfg
from .config.stairs import stairs_env_cfg, stairs_ppo_runner_cfg
from .config.slope import slope_env_cfg, slope_ppo_runner_cfg
from .rsl_rl.runners.elevation_runner import ElevationOnPolicyRunner
from .rsl_rl.runners.jump_runner import JumpOnPolicyRunner
from .rsl_rl.runners.stepping_stones_runner import SteppingStonesOnPolicyRunner


@dataclass(frozen=True)
class TaskSpec:
    """保存任务说明、配置工厂及注册开关，不在定义时构建环境配置。"""

    description: str
    env_cfg_factory: Callable[[bool], ManagerBasedRlEnvCfg]
    rl_cfg_factory: Callable[[], RslRlOnPolicyRunnerCfg]
    enabled: bool = True
    runner_cls: type | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    "PointLeg-Flat-v0": TaskSpec(
        description="平地行走与机动，后续扩展绕桩和精定位",
        env_cfg_factory=flat_env_cfg,
        rl_cfg_factory=flat_ppo_runner_cfg,
    ),
    "PointLeg-HighStep-v0": TaskSpec(
        description="平地高抬腿、小步幅、快周期步态，作为爬梯迁移的基础",
        env_cfg_factory=high_step_env_cfg,
        rl_cfg_factory=high_step_ppo_runner_cfg,
    ),
    "PointLeg-Elevation-v0": TaskSpec(
        description="800 mm 通道内向前上下三级楼梯与持续坡面",
        env_cfg_factory=elevation_env_cfg,
        rl_cfg_factory=elevation_ppo_runner_cfg,
        runner_cls=ElevationOnPolicyRunner,
    ),
    "PointLeg-Stairs-v0": TaskSpec(
        description="上下台阶合并训练，固定120毫米，基础奖励",
        env_cfg_factory=stairs_env_cfg,
        rl_cfg_factory=stairs_ppo_runner_cfg,
        runner_cls=ElevationOnPolicyRunner,
    ),
    "PointLeg-Slope-v0": TaskSpec(
        description="上下坡合并训练，约6.67至20度课程，基础奖励",
        env_cfg_factory=slope_env_cfg,
        rl_cfg_factory=slope_ppo_runner_cfg,
        runner_cls=ElevationOnPolicyRunner,
    ),
    "PointLeg-Rough-v0": TaskSpec(
        description="崎岖地形，后续扩展轮胎阵和砖木坑",
        env_cfg_factory=rough_env_cfg,
        rl_cfg_factory=rough_ppo_runner_cfg,
    ),
    "PointLeg-RoughMixed-v0": TaskSpec(
        description="崎岖全向、正反跨栏与可移动圆环阵混合训练",
        env_cfg_factory=rough_mixed_env_cfg,
        rl_cfg_factory=rough_ppo_runner_cfg,
    ),
    "PointLeg-Bridge-v0": TaskSpec(
        description="单木桥平衡与完整通行",
        env_cfg_factory=bridge_env_cfg,
        rl_cfg_factory=bridge_ppo_runner_cfg,
        enabled=False,
    ),
    "PointLeg-SteppingStones-v0": TaskSpec(
        description="梅花桩精准踏点与全部桩顶接触覆盖",
        env_cfg_factory=stepping_stones_env_cfg,
        rl_cfg_factory=stepping_stones_ppo_runner_cfg,
        runner_cls=SteppingStonesOnPolicyRunner,
    ),
    "PointLeg-Crawl-v0": TaskSpec(
        description="限高匍匐，后续扩展全身净空和出框判定",
        env_cfg_factory=crawl_env_cfg,
        rl_cfg_factory=crawl_ppo_runner_cfg,
    ),
    "PointLeg-Jump-v0": TaskSpec(
        description="触发后固定前跳跨平台间隙及中墙，前脚到后脚净跨距400至500毫米，四足稳定落地",
        env_cfg_factory=jump_env_cfg,
        rl_cfg_factory=jump_ppo_runner_cfg,
        runner_cls=JumpOnPolicyRunner,
    ),
}


# MJLab 注册时需要完整配置，因此占位工厂在启用前不调用。
for task_id, task_spec in TASK_SPECS.items():
    if task_spec.enabled:
        register_mjlab_task(
            task_id=task_id,
            env_cfg=task_spec.env_cfg_factory(False),
            play_env_cfg=task_spec.env_cfg_factory(True),
            rl_cfg=task_spec.rl_cfg_factory(),
            runner_cls=task_spec.runner_cls,
        )
