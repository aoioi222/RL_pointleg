# =============================================================================
# curriculums.py — 自适应课程（Curriculum Manager 的自定义课程函数）
#
# 【模块定位】
#   本文件属于 mdp 组件库中的「课程（Curriculum）」侧，向上挂载到 CurriculumManager，
#   在 env_cfgs.py 的 cfg.curriculum 段被引用，例如：
#       cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
#           func=terrain_levels_obstacle_release, ...)
#       cfg.curriculum["lin_vel_x"] = CurriculumTermCfg(
#           func=command_levels_adaptive, ...)
#
# 【课程分两类】
#   1) 指令范围课程（command curriculum）：根据奖励表现好坏，动态扩/缩指令速度范围
#      - adaptive_command_vel       ：EMA 判断跟踪奖励，超阈值扩范围、低阈值缩范围
#      - command_axis_levels_vel    ：按训练步数线性展开单轴指令范围（时间课程）
#      - command_levels_adaptive    ：按 episode 级表现逐步放宽单轴范围（表现课程）
#   2) 地形等级课程（terrain curriculum）：根据机器人跑动表现，升降地形难度等级
#      - terrain_levels_vel_strict      ：标准 legged_gym 逻辑（走得远升、走得差降）
#      - terrain_levels_ramp_strict     ：起步即生效 + 难度上限按步数逐步解锁
#      - terrain_levels_flat_warmup     ：前 N 步锁在平地，再逐步释放各地形
#      - terrain_levels_obstacle_release：按时间表逐批释放障碍地形（当前主用）
#
# 【课程的本质：改配置】
#   课程函数不产生奖励、不控制机器人，它只做一件事——动态修改其他 manager 的配置。
#   两类"改写通道"：
#     1) 改指令范围：env.command_manager.get_term("twist").cfg.ranges.xxx = (lo, hi)
#        注意拿到的是共享的 cfg 对象，改了就全局生效，影响下一轮重采样。
#     2) 改地形归属：terrain.terrain_levels / terrain.terrain_types / terrain.env_origins
#        三个数组描述"每个环境站在哪块地形上、那块地形有多难"，改完调用
#        update_env_origins 把机器人瞬移到新地形块的原点。
#
# 【表现数据从哪来】
#   env.reward_manager._step_reward[term_idx]     —— 本步该奖励项的加权值
#   env.reward_manager._episode_sums[name][ids]   —— 该 episode 的累计值
#   除以 weight 还原成"原始奖励"（指数奖励在 [0,1] 之间），再除以 episode 时长
#   得到平均每步得分，这个数可以直接和 0.8 之类的阈值比较。
#
# 【关键交叉依赖】
#   - 指令课程会直接改写 commands.py 的 cfg.ranges（通过 get_term("twist").cfg）
#   - 表现判断读取 rewards.py 各奖励 term 的 buffer（_step_reward / _episode_sums）
#   - 地形课程调用 terrain.update_env_origins / env_origins 重定位
# =============================================================================

"""Adaptive command curriculum based on tracking reward performance."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast
import math
import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ─────────────────────────────────────────────────────────────────────────────
# 指令范围课程（Command Curriculum）
# ─────────────────────────────────────────────────────────────────────────────


class adaptive_command_vel:
    """Expand velocity command ranges dynamically when tracking performance exceeds threshold.

    Uses the reward manager's per-step reward buffer directly to compute the mean raw
    tracking reward (in [0, 1] for exponential-based rewards).
    """

    # 基于跟踪奖励表现的指令范围自适应：
    #   - 每步读取目标奖励项的原始值（step_reward / weight 还原到 [0,1]）
    #   - EMA 平滑（0.5 系数）防止振荡
    #   - EMA > upper_threshold  → _expand()  扩大指令范围（机器人跟得好了，给它加难度）
    #   - EMA < lower_threshold  → _shrink()  缩小指令范围（跟得差，降低难度）
    #   - 扩/缩都有边界（_max_lin_vel_x 等配置上限；缩小最低不低于 ±0.3）

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

        p = cfg.params
        self._command_name: str = p["command_name"]
        self._reward_name: str = p["reward_name"]
        self._upper_threshold: float = p.get("upper_threshold", 0.8)
        self._lower_threshold: float = p.get("lower_threshold", 0.4)
        self._delta: float = p.get("delta", 0.2)
        self._max_lin_vel_x: tuple = tuple(p.get("max_lin_vel_x", (-1.0, 1.0)))
        self._max_lin_vel_y: tuple = tuple(p.get("max_lin_vel_y", (-0.5, 0.5)))
        self._max_ang_vel_z: tuple = tuple(p.get("max_ang_vel_z", (-1.0, 1.0)))
        self._min_range: float = 0.3

        # 拿到命令 term 的配置对象（就是 commands.py 的 UniformThresholdVelocityCommandCfg）
        # 后续 _expand/_shrink 直接改这个对象的 ranges 字段——这就是"课程改指令"的通道
        command_term = env.command_manager.get_term(self._command_name)
        self._cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

        # Locate the index of the target reward term
        self._reward_idx = list(env.reward_manager._term_names).index(self._reward_name)
        self._reward_weight = env.reward_manager.get_term_cfg(self._reward_name).weight

        # Exponential moving average (EMA) smoothing to prevent oscillations
        self._ema = 0.5
        self._running_mean = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Retrieve raw step rewards from step_reward buffer
        # step_reward[:, idx] = raw * weight (already scaled by dt)
        step_rewards = env.reward_manager._step_reward[:, self._reward_idx]
        # raw = step_reward / weight -> [0, 1] for exponential rewards
        # 还原原始奖励（除以 weight）：指数奖励在误差=0 时取 1，所以原始值在 [0,1]
        mean_raw = (torch.mean(step_rewards) / self._reward_weight).item()

        # Apply EMA smoothing to prevent jitter
        self._running_mean = self._ema * mean_raw + (1 - self._ema) * self._running_mean

        if self._running_mean > self._upper_threshold:
            self._expand()
        elif self._running_mean < self._lower_threshold:
            self._shrink()

        return self._log()

    def _expand(self):
        # 扩大指令范围：各轴向两边各放宽 delta（y 轴只放宽一半），并 clamp 到配置上限
        # y 轴减半的原因：点足四足横移通常弱于前后运动，同等放宽容易让
        # 大量环境卡在"横移跟不上"的状态，把课程拖住。该函数当前无环境使用，属备用。
        lo, hi = self._cfg.ranges.lin_vel_x
        self._cfg.ranges.lin_vel_x = (
            max(lo - self._delta, self._max_lin_vel_x[0]),
            min(hi + self._delta, self._max_lin_vel_x[1]),
        )
        lo, hi = self._cfg.ranges.lin_vel_y
        self._cfg.ranges.lin_vel_y = (
            max(lo - self._delta * 0.5, self._max_lin_vel_y[0]),
            min(hi + self._delta * 0.5, self._max_lin_vel_y[1]),
        )
        lo, hi = self._cfg.ranges.ang_vel_z
        self._cfg.ranges.ang_vel_z = (
            max(lo - self._delta, self._max_ang_vel_z[0]),
            min(hi + self._delta, self._max_ang_vel_z[1]),
        )

    def _shrink(self):
        # 缩小指令范围：只缩 x 轴（其他轴保持），且不低于 ±_min_range
        lo, hi = self._cfg.ranges.lin_vel_x
        new_lo = min(lo + self._delta, -self._min_range)
        new_hi = max(hi - self._delta, self._min_range)
        if new_hi - new_lo >= 2 * self._min_range:
            self._cfg.ranges.lin_vel_x = (new_lo, new_hi)

    def _log(self) -> dict[str, torch.Tensor]:
        # 输出课程统计（供 TensorBoard 记录范围变化）
        return {
            "lin_vel_x_min": torch.tensor(self._cfg.ranges.lin_vel_x[0]),
            "lin_vel_x_max": torch.tensor(self._cfg.ranges.lin_vel_x[1]),
            "ang_vel_z_max": torch.tensor(self._cfg.ranges.ang_vel_z[1]),
        }


class command_axis_levels_vel:
    """Linearly expand one command axis over training steps."""

    # 时间型单轴课程：指令范围随训练步数从初始小范围线性展开到全范围
    #   - warmup_steps: 预热期内保持最小范围（策略先学基础）
    #   - ramp_steps  : 预热后线性展开所需步数（默认 800*24 ≈ 800 轮 × 24 步）
    #   - 可选 initial_range 指定起点；否则用 full_range × range_multiplier[0] 起步
    # 与 adaptive_command_vel 的区别：这个只依赖训练步数（时间表），不依赖表现

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

        p = cfg.params
        self._command_name: str = p.get("command_name", "twist")
        self._reward_name: str = p["reward_term_name"]
        self._axis: str = p["axis"]
        self._range_multiplier: tuple[float, float] = tuple(p.get("range_multiplier", (0.4, 1.0)))
        self._initial_range_param: tuple[float, float] | None = (
            tuple(p["initial_range"]) if "initial_range" in p else None
        )
        self._ema_alpha: float = p.get("ema_alpha", 0.5)
        self._warmup_steps: int = p.get("warmup_steps", 0)
        self._ramp_steps: int = p.get("ramp_steps", 800 * 24)

        command_term = env.command_manager.get_term(self._command_name)
        self._cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
        self._reward_idx = list(env.reward_manager._term_names).index(self._reward_name)
        self._reward_weight = env.reward_manager.get_term_cfg(self._reward_name).weight
        self._running_mean = 0.0

        # full_range = 配置里的最终完整范围；min_range = 起步范围
        self._full_range = self._get_range()
        self._min_range = self._initial_range_param or self._scaled_range(self._range_multiplier[0])
        self._set_range(self._min_range)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # 用 episode 平均奖励的 EMA 做表现统计（仅记录，不作为展开依据）
        if len(env_ids) > 0:
            episode_sums = env.reward_manager._episode_sums[self._reward_name][env_ids]
            mean_raw = torch.mean(episode_sums / env.cfg.episode_length_s / self._reward_weight).item()
            self._running_mean = self._ema_alpha * mean_raw + (1.0 - self._ema_alpha) * self._running_mean
        # 按当前训练步数计算本步应达到的范围，并写回 cfg
        self._set_range(self._current_max_range(env.common_step_counter))

        lo, hi = self._get_range()
        return {
            f"{self._axis}_range_min": torch.tensor(lo),
            f"{self._axis}_range_max": torch.tensor(hi),
            f"{self._axis}_tracking_ema": torch.tensor(self._running_mean),
            f"{self._axis}_warmup_active": torch.tensor(float(env.common_step_counter < self._warmup_steps)),
            f"{self._axis}_range_cap": torch.tensor(self._current_multiplier(env.common_step_counter)),
        }

    def _get_range(self) -> tuple[float, float]:
        # 按轴名取当前指令范围
        if self._axis == "x":
            return tuple(self._cfg.ranges.lin_vel_x)
        if self._axis == "y":
            return tuple(self._cfg.ranges.lin_vel_y)
        if self._axis == "yaw":
            return tuple(self._cfg.ranges.ang_vel_z)
        raise ValueError(f"Unknown command curriculum axis: {self._axis}")

    def _set_range(self, value: tuple[float, float]) -> None:
        # 按轴名写回指令范围（改的是 commands.py 共享的 cfg 对象）
        if self._axis == "x":
            self._cfg.ranges.lin_vel_x = value
        elif self._axis == "y":
            self._cfg.ranges.lin_vel_y = value
        elif self._axis == "yaw":
            self._cfg.ranges.ang_vel_z = value
        else:
            raise ValueError(f"Unknown command curriculum axis: {self._axis}")

    def _scaled_range(self, multiplier: float) -> tuple[float, float]:
        # 全范围 × 倍率 → 起步范围
        lo, hi = self._full_range
        return (lo * multiplier, hi * multiplier)

    def _current_multiplier(self, step: int) -> float:
        # 当前展开比例：warmup 前 = start，ramp 期间线性插值到 end
        start, end = self._range_multiplier
        if self._ramp_steps <= 0:
            return end
        progress = max(0.0, min(1.0, (step - self._warmup_steps) / self._ramp_steps))
        return start + (end - start) * progress

    def _current_max_range(self, step: int) -> tuple[float, float]:
        # 无 initial_range：直接用 全范围 × 当前倍率
        if self._initial_range_param is None:
            return self._scaled_range(self._current_multiplier(step))
        # 有 initial_range：从起点按 progress 线性过渡到全范围
        progress = self._current_progress(step)
        lo = self._min_range[0] + (self._full_range[0] - self._min_range[0]) * progress
        hi = self._min_range[1] + (self._full_range[1] - self._min_range[1]) * progress
        return (lo, hi)

    def _current_progress(self, step: int) -> float:
        # 0→1 的展开进度（warmup 后开始计）
        if self._ramp_steps <= 0:
            return 1.0
        return max(0.0, min(1.0, (step - self._warmup_steps) / self._ramp_steps))


class command_levels_adaptive:
    """Adaptive command range curriculum based on average tracking performance.

    为什么只放宽不缩回：缩回会造成"能过→放宽→过不了→缩回"的极限环，
    策略在边界反复横跳。当前设计把"难度只会上升"作为硬约束，
    表现差就停在当前范围继续练，符合课程学习的单调性要求。
    在独立时间窗口内累计已结束回合，窗口到期后的首次重置触发评估。
    不要求重置恰好发生在全局步数的整数倍，提前终止仍按完整回合时长计分。
    """

    # 表现型单轴课程（当前 env_cfgs 主用，与 command_axis_levels_vel 二选一）：
    #   - 起步范围 = initial_range（如 [-0.5, 0.5]）
    #   - 时间窗口到期后看回合平均跟踪表现（EMA），超过 target_ratio 则放宽 delta_command
    #   - 只放宽（表现差不会缩回），clamp 到 full_range（配置全范围）之内
    # 特点：难度跟随能力走，机器人学得多快，指令就给得多大

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        p = cfg.params
        self._command_name: str = p.get("command_name", "twist")
        self._reward_name: str = p["reward_term_name"]
        self._axis: str = p["axis"]
        self._delta_command: float = p.get("delta_command", 0.05)
        self._target_ratio: float = p.get("target_ratio", 0.8)
        self._ema_alpha: float = p.get("ema_alpha", 0.5)

        command_term = env.command_manager.get_term(self._command_name)
        self._cfg = command_term.cfg
        self._reward_weight = env.reward_manager.get_term_cfg(self._reward_name).weight
        self._running_mean = 0.0
        interval_s = p.get("update_interval_s", env.cfg.episode_length_s)
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("Command curriculum update interval must be positive")
        if self._reward_weight <= 0.0 or not 0.0 < self._ema_alpha <= 1.0:
            raise ValueError("Command curriculum requires positive reward weight and EMA alpha in (0, 1]")
        if self._delta_command <= 0.0 or not 0.0 <= self._target_ratio <= 1.0:
            raise ValueError("Command curriculum requires positive delta and target ratio in [0, 1]")
        self._update_interval_steps = max(1, math.ceil(interval_s / env.step_dt))
        self._last_update_step = env.common_step_counter
        self._window_score_sum = 0.0
        self._window_episode_count = 0

        # Read the full range configured in the environment configuration
        self._full_range = self._get_range()

        # Set the command range to initial range at the start
        self._initial_range = list(p["initial_range"])  # e.g. [-0.5, 0.5]
        if not (
            self._full_range[0] <= self._initial_range[0]
            <= self._initial_range[1] <= self._full_range[1]
        ):
            raise ValueError("Initial command range must be contained in the full range")
        self._current_range = list(self._initial_range)
        self._set_range(self._current_range)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **kwargs) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        episode_sums = env.reward_manager._episode_sums[self._reward_name][ids]
        completed = env.episode_length_buf[ids] > 0
        finite = torch.isfinite(episode_sums)
        scores = torch.where(finite, episode_sums, torch.zeros_like(episode_sums))
        scores = scores / env.cfg.episode_length_s / self._reward_weight
        self._window_score_sum += scores[completed].sum().item()
        self._window_episode_count += int(completed.sum().item())

        step = env.common_step_counter
        due = step - self._last_update_step >= self._update_interval_steps
        if due and self._window_episode_count:
            mean_raw = self._window_score_sum / self._window_episode_count
            self._running_mean = self._ema_alpha * mean_raw + (1.0 - self._ema_alpha) * self._running_mean
            if self._running_mean > self._target_ratio:
                lo, hi = self._current_range
                new_lo = max(self._full_range[0], lo - self._delta_command)
                new_hi = min(self._full_range[1], hi + self._delta_command)
                self._current_range = [new_lo, new_hi]
                self._set_range(self._current_range)
            self._last_update_step = step
            self._window_score_sum = 0.0
            self._window_episode_count = 0

        lo, hi = self._current_range
        return {
            f"{self._axis}_range_min": torch.tensor(lo, device=env.device),
            f"{self._axis}_range_max": torch.tensor(hi, device=env.device),
            f"{self._axis}_tracking_ema": torch.tensor(self._running_mean, device=env.device),
            f"{self._axis}_target_ratio": torch.tensor(self._target_ratio, device=env.device),
        }

    def _get_range(self) -> tuple[float, float]:
        if self._axis == "x":
            return tuple(self._cfg.ranges.lin_vel_x)
        if self._axis == "y":
            return tuple(self._cfg.ranges.lin_vel_y)
        if self._axis == "yaw":
            return tuple(self._cfg.ranges.ang_vel_z)
        raise ValueError(f"Unknown command curriculum axis: {self._axis}")

    def _set_range(self, value: list[float]) -> None:
        if self._axis == "x":
            self._cfg.ranges.lin_vel_x = tuple(value)
        elif self._axis == "y":
            self._cfg.ranges.lin_vel_y = tuple(value)
        elif self._axis == "yaw":
            self._cfg.ranges.ang_vel_z = tuple(value)
        else:
            raise ValueError(f"Unknown command curriculum axis: {self._axis}")


def _get_terrain_origins(
    terrain: object, terrain_levels: torch.Tensor, terrain_types: torch.Tensor
) -> torch.Tensor:
    """Resolve semantic terrain types to their physical terrain tiles."""
    resolver = getattr(terrain, "get_terrain_origins", None)
    if callable(resolver):
        return resolver(terrain_levels, terrain_types)
    return terrain.terrain_origins[terrain_levels, terrain_types]


class rough_terrain_levels:
    """Gate terrain difficulty using command-aligned episode progress."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        params = cfg.params
        self._command_name: str = params.get("command_name", "twist")
        self._window_size: int = params.get("window_size", 100)
        self._success_threshold: float = params.get("success_threshold", 0.8)
        self._success_progress_ratio: float = params.get("success_progress_ratio", 0.6)
        self._demote_progress_ratio: float = params.get("demote_progress_ratio", 0.4)
        self._minimum_command_speed: float = params.get("minimum_command_speed", 0.15)
        self._target_tile_fraction: float = params.get("target_tile_fraction", 0.45)
        self._require_obstacle_completion: bool = params.get("require_obstacle_completion", False)
        if self._window_size <= 0:
            raise ValueError("Terrain curriculum window size must be positive")
        if not 0.0 <= self._success_threshold <= 1.0:
            raise ValueError("Terrain curriculum success threshold must be in [0, 1]")
        if not 0.0 <= self._demote_progress_ratio < self._success_progress_ratio <= 1.0:
            raise ValueError("Terrain progress thresholds must be ordered within [0, 1]")
        terrain = env.scene.terrain
        if terrain is None or terrain.cfg.terrain_generator is None:
            raise ValueError("Rough terrain curriculum requires generated terrain")
        self._maximum_level = terrain.cfg.terrain_generator.num_rows - 1
        self._level_cap = int(params.get("initial_level_cap", 0))
        if not 0 <= self._level_cap <= self._maximum_level:
            raise ValueError("Initial terrain level cap is outside the terrain rows")
        self._success_window: deque[bool] = deque(maxlen=self._window_size)
        self._last_success_rate = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)

        terrain = env.scene.terrain
        assert terrain is not None
        generator = terrain.cfg.terrain_generator
        assert generator is not None
        assert terrain.terrain_origins is not None
        assert terrain.env_origins is not None

        command_term = env.command_manager.get_term(self._command_name)
        command = env.command_manager.get_command(self._command_name)
        direction_w = getattr(command_term, "progress_direction_w", None)
        if direction_w is None:
            direction_w = command[:, :2]
        direction = direction_w[env_ids]
        command_speed = torch.linalg.vector_norm(direction, dim=1)
        unit_direction = direction / torch.clamp(command_speed[:, None], min=1.0e-6)
        displacement = (
            env.scene["robot"].data.root_link_pos_w[env_ids, :2]
            - env.scene.env_origins[env_ids, :2]
        )
        effective_progress = torch.sum(displacement * unit_direction, dim=1)
        elapsed_s = env.episode_length_buf[env_ids].float() * env.step_dt
        target_distance = command_speed * elapsed_s

        metrics_manager = getattr(env, "metrics_manager", None)
        episode_metrics = getattr(metrics_manager, "_episode_sums", {})
        if "effective_progress" in episode_metrics and "cmd_lin_vel" in episode_metrics:
            effective_progress = episode_metrics["effective_progress"][env_ids] * env.step_dt
            target_distance = episode_metrics["cmd_lin_vel"][env_ids] * env.step_dt
            command_speed = target_distance / torch.clamp(elapsed_s, min=env.step_dt)

        target_distance = torch.minimum(
            target_distance,
            torch.full_like(target_distance, generator.size[0] * self._target_tile_fraction),
        )
        progress_ratio = effective_progress / torch.clamp(target_distance, min=1.0e-6)

        completed = env.episode_length_buf[env_ids] > 0
        eligible = completed & (command_speed >= self._minimum_command_speed)
        reset_terminated = getattr(
            env, "reset_terminated", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        reset_time_outs = getattr(
            env, "reset_time_outs", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        failed = reset_terminated[env_ids] & ~reset_time_outs[env_ids]
        success = eligible & ~failed & (progress_ratio >= self._success_progress_ratio)
        if self._require_obstacle_completion and hasattr(env, "_mixed_obstacles"):
            from .terrain_utils import terrain_type_mask
            from ..terrains.mixed_obstacles import FENCE_NAMES, RING_TERRAIN
            obstacle = terrain_type_mask(env, (*FENCE_NAMES, RING_TERRAIN))[env_ids]
            complete = env._mixed_obstacles["complete"][env_ids]
            failed |= obstacle & ~complete
            # Obstacle completion already checks crossing distance and stable exit.
            success = torch.where(obstacle, eligible & ~failed & complete, success)

        for value in success[eligible].detach().cpu().tolist():
            self._success_window.append(bool(value))
        if self._success_window:
            self._last_success_rate = sum(self._success_window) / len(self._success_window)
        if (
            len(self._success_window) == self._window_size
            and self._last_success_rate >= self._success_threshold
            and self._level_cap < self._maximum_level
        ):
            self._level_cap += 1
            self._success_window.clear()

        current_levels = terrain.terrain_levels[env_ids]
        move_up = success
        move_down = eligible & ~success & (failed | (progress_ratio < self._demote_progress_ratio))
        next_levels = current_levels + move_up.long() - move_down.long()
        terrain.terrain_levels[env_ids] = torch.clamp(next_levels, min=0, max=self._level_cap)
        terrain.env_origins[env_ids] = _get_terrain_origins(
            terrain, terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
        )

        levels = terrain.terrain_levels.float()
        result: dict[str, torch.Tensor] = {
            "mean": torch.mean(levels),
            "max": torch.max(levels),
            "level_cap": torch.tensor(float(self._level_cap), device=env.device),
            "effective_progress": effective_progress[eligible].mean()
            if eligible.any() else torch.zeros((), device=env.device),
            "success_rate": torch.tensor(self._last_success_rate, device=env.device),
            "window_episodes": torch.tensor(float(len(self._success_window)), device=env.device),
            "failure_rate": failed[eligible].float().mean()
            if eligible.any() else torch.zeros((), device=env.device),
        }
        for terrain_type, name in enumerate(generator.sub_terrains):
            mask = terrain.terrain_types == terrain_type
            if mask.any():
                result[name] = levels[mask].mean()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 地形等级课程（Terrain Curriculum）
# ─────────────────────────────────────────────────────────────────────────────


def terrain_levels_vel_strict(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Velocity-based terrain curriculum aligned with standard legged_gym logic.

    Upgrade:   robot travels beyond half the terrain tile width (> 4m).
    Downgrade: actual distance < 50% of commanded target distance.

    This matches the standard DreamWaQ / HIMLoco locomotion curriculum behaviour:
    - Promotion is easy (any traversal past 4m qualifies).
    - Demotion requires consistently failing to cover half the expected distance.
    """
    # 标准速度型地形课程（legged_gym 风格）：
    #   - 升级（move_up）  ：本 episode 已走过的水平距离 > 地形块宽的一半（4m），
    #                       说明当前难度已经轻松通过 → 地形等级 +1
    #   - 降级（move_down）：实际行进距离 < 指令速度 × 最长时长 × 33%，
    #                       说明连目标距离的一半都走不到 → 地形等级 -1
    #   与 DreamWaQ / HIMLoco 的升易降严策略对齐，
    #   30s episode 比标准的 20s 长，故比例取 0.33 以保持约 10m 的绝对阈值。
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    # Horizontal displacement from episode start
    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    cmd_speed = torch.norm(command[env_ids, :2], dim=1)

    # Upgrade: crossed half the tile width
    move_up = distance > terrain_generator.size[0] / 2

    # Downgrade: traveled less than 33% of commanded target distance.
    # Absolute threshold ≈ cmd_speed × 10m, matching DreamWaQ / HIMLoco
    # which use 50% × 20s episode = 10m. Adjusted for the longer 30s episode here.
    move_down = (distance < cmd_speed * env.max_episode_length_s * 0.33) & ~move_up



    # 按升降标志更新环境起点（升级/降级后机器人被挪到新的难度地形上）
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    # 统计各子地形上的平均等级（便于观察每种地形的推进情况）
    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    num_cols = terrain_origins.shape[1]
    if num_cols == len(sub_terrain_names):
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])

    return result


def terrain_levels_ramp_strict(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    ramp_steps: int = 50 * 24,
    move_up_expected_distance_ratio: float = 0.60,
    move_down_distance_ratio: float = 0.50,
    min_command_speed: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Terrain curriculum active from start, with strict promotion and time-ramped max level.

    严格版地形课程（升级门槛更高 + 难度上限随时间解锁）：
      - 升级：实际距离 > 预期距离(指令速度×时长) × 60%（比 vel_strict 的 50% 更严）
      - 降级：实际距离 < 预期距离 × 50%
      - 只统计"有运动指令"的环境（cmd_speed >= min_command_speed，静止不算失败）
      - level_cap 随时间线性解锁：训练越久允许到达的地形等级越高（ramp_steps 后全解锁）

    核心思想——难度上限由训练进度解锁，具体等级由表现决定：
      1) 达标/不达标只决定"该升还是该降"（move_up / move_down）
      2) 但最终等级会被 clamp 到 level_cap，而 level_cap 随 common_step_counter 线性增长
      3) clamp 完之后必须重写 terrain.env_origins，否则机器人还在旧地形块上，
         等级数字变了但实际难度没变，等于课程失效。
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None
    assert terrain.terrain_origins is not None
    assert terrain.env_origins is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    cmd_speed = torch.norm(command[env_ids, :2], dim=1)
    active_command = cmd_speed >= min_command_speed

    expected_distance = cmd_speed * env.max_episode_length_s
    move_up = (distance > expected_distance * move_up_expected_distance_ratio) & active_command
    move_down = (
        (distance < expected_distance * move_down_distance_ratio)
        & active_command
        & ~move_up
    )

    # 直接原地增减等级（不调 update_env_origins，最后统一 clamp + 重定位）
    terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

    # 难度上限随训练步数解锁：progress=0 时 cap=0（全平地），线性涨到最大等级
    max_level = max(int(terrain.max_terrain_level) - 1, 0)
    if ramp_steps <= 0:
        level_cap = max_level
    else:
        progress = max(0.0, min(1.0, env.common_step_counter / ramp_steps))
        level_cap = int(round(progress * max_level))
    terrain.terrain_levels[env_ids] = torch.clamp(
        terrain.terrain_levels[env_ids],
        min=0,
        max=min(level_cap, max_level),
    )

    # 按新等级重设每个环境的世界原点（搬到对应地形块）
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "level_cap": torch.tensor(float(level_cap), device=env.device),
    }

    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    num_cols = terrain.terrain_origins.shape[1]
    if num_cols == len(sub_terrain_names):
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])

    return result


def terrain_levels_flat_warmup(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    warmup_steps: int = 4800,
    flat_terrain_name: str = "flat",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Keep all reset envs on flat level 0 before enabling terrain curriculum."""
    # 平地热身课程：训练前 warmup_steps 步，所有 reset 的环境强制待在平地 level 0
    #   - 热身期内：terrain_levels=0、terrain_types=flat、env_origins 指向平地
    #   - 热身期后：每个环境首次 reset 时按子地形比例 multinomial 抽一种地形
    #     （新地形从 level 0 起步），之后再交给 terrain_levels_vel_strict 正常升降
    # 目的：先让策略学会走路，再上复杂地形，避免一开始就被台阶难住
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None
    assert terrain.terrain_origins is not None
    assert terrain.env_origins is not None

    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    flat_type = sub_terrain_names.index(flat_terrain_name) if flat_terrain_name in sub_terrain_names else 0

    if env.common_step_counter < warmup_steps:
        # ── 热身期：强制平地 ──
        terrain.terrain_levels[env_ids] = 0
        terrain.terrain_types[env_ids] = flat_type
        terrain.env_origins[env_ids] = terrain.terrain_origins[0, flat_type]
        if not hasattr(terrain, "_flat_warmup_env_released"):
            terrain._flat_warmup_env_released = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
        terrain._flat_warmup_env_released[env_ids] = False

        levels = terrain.terrain_levels.float()
        result: dict[str, torch.Tensor] = {
            "mean": torch.mean(levels),
            "max": torch.max(levels),
            "warmup_active": torch.ones((), device=env.device),
        }
        for i, name in enumerate(sub_terrain_names):
            mask = terrain.terrain_types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])
        return result

    # ── 热身期结束：逐环境释放 ──
    if not hasattr(terrain, "_flat_warmup_released"):
        terrain._flat_warmup_released = True
        terrain._flat_warmup_release_counts = 0
    if not hasattr(terrain, "_flat_warmup_env_released"):
        terrain._flat_warmup_env_released = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    # 找出"还没释放"的环境，按子地形配置的比例随机分配一种地形（level 归 0）
    newly_released = env_ids[~terrain._flat_warmup_env_released[env_ids]]
    if len(newly_released) > 0:
        proportions = torch.tensor(
            [sub.proportion for sub in terrain_generator.sub_terrains.values()],
            device=env.device,
            dtype=torch.float,
        )
        proportions = proportions / torch.clamp(proportions.sum(), min=1.0e-6)
        terrain.terrain_types[newly_released] = torch.multinomial(
            proportions, len(newly_released), replacement=True
        )
        terrain.terrain_levels[newly_released] = 0
        terrain.env_origins[newly_released] = terrain.terrain_origins[
            terrain.terrain_levels[newly_released], terrain.terrain_types[newly_released]
        ]
        terrain._flat_warmup_env_released[newly_released] = True
        terrain._flat_warmup_release_counts += len(newly_released)

    # 已释放的环境走标准升降课程
    result = terrain_levels_vel_strict(env, env_ids, command_name, asset_cfg=asset_cfg)
    result["warmup_active"] = torch.zeros((), device=env.device)
    result["released_envs"] = torch.tensor(float(getattr(terrain, "_flat_warmup_release_counts", 0)), device=env.device)
    return result


def terrain_levels_obstacle_release(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    release_schedule: tuple[tuple[int, tuple[str, ...]], ...],
    initial_terrain_names: tuple[str, ...] = ("flat", "random_rough", "perlin_noise", "sloped_terrain"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Release obstacle terrain types gradually while keeping standard level progression."""
    # 障碍逐批释放课程（当前 env_cfgs 主用的地形课程）：
    #   - 训练一开始只允许 4 种基础地形（平底/粗糙/噪声/斜坡）
    #   - 按 release_schedule 时间表，在指定训练步数解锁新地形类型
    #     （如 (20000, ("pyramid_stairs",)) 表示 20000 步后允许金字塔台阶）
    #   - 新解锁地形时，所有未匹配的环境强制重新按比例抽地形（level 归 0）
    #   - 已允许的地形内部，仍按 terrain_levels_vel_strict 正常升降等级
    # 作用：由易到难逐步引入障碍，避免课程初期被楼梯/短墙打崩
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None
    assert terrain.terrain_origins is not None
    assert terrain.env_origins is not None

    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    # 累计允许的地形：初始集合 + 已到时间点的解锁集合
    allowed_names = list(initial_terrain_names)
    for step, names in release_schedule:
        if env.common_step_counter >= step:
            allowed_names.extend(names)
    allowed_type_ids = [
        sub_terrain_names.index(name) for name in allowed_names if name in sub_terrain_names
    ]
    if not allowed_type_ids:
        allowed_type_ids = [0]

    if not hasattr(terrain, "_obstacle_release_env_allowed"):
        terrain._obstacle_release_env_allowed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        terrain._last_allowed_count = len(allowed_type_ids)

    # If new terrains were released, force all envs to eventually resample upon their next reset
    # 若本次解锁了新地形，把所有环境标记为"需要重抽"（下次 reset 时重抽地形）
    if len(allowed_type_ids) > terrain._last_allowed_count:
        terrain._obstacle_release_env_allowed.fill_(False)
        terrain._last_allowed_count = len(allowed_type_ids)

    allowed_tensor = torch.tensor(allowed_type_ids, dtype=torch.long, device=env.device)
    current_allowed = torch.isin(terrain.terrain_types[env_ids], allowed_tensor)
    need_resample = env_ids[~terrain._obstacle_release_env_allowed[env_ids] | ~current_allowed]

    if len(need_resample) > 0:
        # 按允许地形的比例权重多项式抽样，分配新地形
        proportions = torch.tensor(
            [terrain_generator.sub_terrains[sub_terrain_names[i]].proportion for i in allowed_type_ids],
            device=env.device,
            dtype=torch.float,
        )
        proportions = proportions / torch.clamp(proportions.sum(), min=1.0e-6)
        sampled = allowed_tensor[torch.multinomial(proportions, len(need_resample), replacement=True)]

        # Check which envs actually changed terrain type
        changed_mask = terrain.terrain_types[need_resample] != sampled
        changed_envs = need_resample[changed_mask]

        terrain.terrain_types[need_resample] = sampled
        # Only reset the level to 0 if the terrain type was actually changed
        # 只有地形类型真的变了才把等级归零（同类地形重抽不算降级）
        if len(changed_envs) > 0:
            terrain.terrain_levels[changed_envs] = 0

        terrain.env_origins[need_resample] = terrain.terrain_origins[
            terrain.terrain_levels[need_resample], terrain.terrain_types[need_resample]
        ]
        terrain._obstacle_release_env_allowed[need_resample] = True

    # 已释放的地形走标准升降课程；附加"当前允许地形数"与"各障碍是否已解锁"的统计
    result = terrain_levels_vel_strict(env, env_ids, command_name, asset_cfg=asset_cfg)
    result["allowed_types"] = torch.tensor(float(len(allowed_type_ids)), device=env.device)
    for name in ("pyramid_stairs", "pyramid_stairs_inv", "random_grid", "rc_wall"):
        result[f"{name}_released"] = torch.tensor(float(name in allowed_names), device=env.device)
    return result
