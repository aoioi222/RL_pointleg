# =============================================================================
# lowpass_actions.py — 动作低通滤波 + 随机延迟（自定义 Action Term）
#
# 【模块定位】
#   本文件属于 mdp 组件库中的「动作（Action）」侧，向上挂载到 ActionManager，
#   在 env_cfgs.py 中以 action term 形式被使用：
#       cfg.actions["leg_joint_pos"] = JointPositionDelayedLowPassActionCfg(...)
#
# 【解决的问题（sim2real）】
#   真实电机/舵机存在响应滞后与执行延迟，若策略动作直接下发，仿真与实物行为差异大。
#   本模块在策略原始输出上做两件事：
#     1) 一阶低通滤波（IIR）：平滑控制信号，抑制高频抖动；
#        - 腿关节使用 5Hz 截止频率，抑制高频抖动
#     2) 随机延迟（max_delay 步）：模拟通信/执行延迟，增强策略对延迟的鲁棒性
#        （每次 reset 随机采样延迟步数，属于域随机化的一种）
#
# 【处理链路】process_actions：
#   raw action →（可选）延迟缓冲取旧动作 → 一阶低通 → 父类 scale/offset → 关节目标
# =============================================================================

"""Low-pass filtered action terms for mjlab (native implementation).

Provides first-order IIR low-pass filtering on the policy output before applying
scale/offset. This smooths control signals and reduces sim-to-real gap.

- Leg joints: 5 Hz cutoff
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import (
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _lowpass_weights(control_freq: float, cutoff_freq: float) -> list[float]:
    """Compute 1st-order IIR low-pass filter weights.

    alpha = 1 - exp(-2π * f_c / f_s)
    filtered = alpha * current + (1 - alpha) * previous_filtered
    """
    # 一阶 IIR 低通系数：alpha 由截止频率与控制频率决定
    #   f_c 越小 → alpha 越小 → 滤波越"重"（越平滑）
    #   filtered = alpha*current + (1-alpha)*previous（指数滑动平均形式）
    if not 0.0 < cutoff_freq < 0.5 * control_freq:
        raise ValueError("cutoff_freq must be positive and below half control_freq")
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_freq / control_freq)
    return [alpha, 1.0 - alpha]


# -- Position low-pass --


@dataclass(kw_only=True)
class JointPositionLowPassActionCfg(JointPositionActionCfg):
    """Joint position action with 1st-order low-pass filter on policy output."""

    control_frequency: float = 50.0
    """Control frequency in Hz (= 1 / (dt * decimation))."""
    cut_off_frequency: float = 5.0
    """Cut-off frequency in Hz for the low-pass filter."""

    clip_to_joint_limits: bool = True
    """Clamp processed position targets to the XML soft joint limits."""

    expected_joint_names: tuple[str, ...] | None = None
    """Required entity and action joint order, checked when this action is built."""

    def build(self, env: ManagerBasedRlEnv) -> JointPositionLowPassAction:
        return JointPositionLowPassAction(self, env)


class JointPositionLowPassAction(JointPositionAction):
    """Applies a 1st-order low-pass filter to raw actions before processing."""

    def __init__(self, cfg: JointPositionLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.expected_joint_names is not None:
            expected = tuple(cfg.expected_joint_names)
            if tuple(self._entity.joint_names) != expected:
                raise ValueError(
                    "Point-foot XML must contain exactly the expected 12 leg joints "
                    f"in policy order; expected {expected}, got {tuple(self._entity.joint_names)}."
                )
            if tuple(self.target_names) != expected:
                raise ValueError(
                    "Position action targets must match the expected 12 leg joints; "
                    f"got {tuple(self.target_names)}."
                )
        # 滤波状态保存上一帧滤波输出，用于一阶 IIR 递推。
        self._weights = _lowpass_weights(cfg.control_frequency, cfg.cut_off_frequency)
        self._prev_filtered = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        # 当前输入与上一帧滤波输出共同决定本帧滤波结果。
        filtered = self._weights[0] * actions + self._weights[1] * self._prev_filtered
        self._prev_filtered[:] = filtered
        # 交给父类做 scale/offset 与关节目标转换
        super().process_actions(filtered)
        self._clip_to_joint_limits()

    def _clip_to_joint_limits(self) -> None:
        """Clamp position targets using the limits resolved from the loaded XML."""
        if not self.cfg.clip_to_joint_limits:
            return
        limits = self._entity.data.soft_joint_pos_limits
        assert limits is not None
        target_limits = limits[:, self._target_ids]
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=target_limits[..., 0],
            max=target_limits[..., 1],
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # reset 时清空历史动作缓存，避免新 episode 带上旧帧动作
        self._prev_filtered[env_ids] = 0.0
        super().reset(env_ids)


# -- Velocity low-pass --


@dataclass(kw_only=True)
class JointVelocityLowPassActionCfg(JointVelocityActionCfg):
    """Joint velocity action with 1st-order low-pass filter on policy output."""

    control_frequency: float = 50.0
    """Control frequency in Hz (= 1 / (dt * decimation))."""
    cut_off_frequency: float = 15.0
    """Cut-off frequency in Hz for the low-pass filter."""

    def build(self, env: ManagerBasedRlEnv) -> JointVelocityLowPassAction:
        return JointVelocityLowPassAction(self, env)


class JointVelocityLowPassAction(JointVelocityAction):
    """Applies a 1st-order low-pass filter to raw actions before processing."""

    def __init__(self, cfg: JointVelocityLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 速度模式同样保存上一帧滤波输出。
        self._weights = _lowpass_weights(cfg.control_frequency, cfg.cut_off_frequency)
        self._prev_filtered = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        filtered = self._weights[0] * actions + self._weights[1] * self._prev_filtered
        self._prev_filtered[:] = filtered
        super().process_actions(filtered)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._prev_filtered[env_ids] = 0.0
        super().reset(env_ids)


# -- Position low-pass + random delay --


@dataclass(kw_only=True)
class JointPositionDelayedLowPassActionCfg(JointPositionLowPassActionCfg):
    """Joint position action with random delay and 1st-order low-pass filter."""

    # 随机延迟步数范围：每个环境在 reset 时从 [min_delay, max_delay] 采样一个固定延迟
    # 模拟真实系统的执行延迟，让策略学会"动作下去后晚一点才生效"的鲁棒性
    min_delay: int = 0
    max_delay: int = 2

    def build(self, env: ManagerBasedRlEnv) -> JointPositionDelayedLowPassAction:
        return JointPositionDelayedLowPassAction(self, env)


class JointPositionDelayedLowPassAction(JointPositionLowPassAction):
    """Applies a random delay and a 1st-order low-pass filter to raw actions."""

    def __init__(self, cfg: JointPositionDelayedLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.min_delay = cfg.min_delay
        self.max_delay = cfg.max_delay
        if not 0 <= self.min_delay <= self.max_delay:
            raise ValueError("Delays must satisfy 0 <= min_delay <= max_delay")

        # Buffer shape: (max_delay + 1, num_envs, action_dim)
        # 环形延迟缓冲：按时间顺序保存最近 max_delay+1 帧动作，按需取旧帧
        self._action_buffer = torch.zeros(
            self.max_delay + 1, self.num_envs, self.action_dim, device=self.device
        )

        # Active delays for each env, shape: (num_envs,)
        # 每个环境当前生效的延迟步数（reset 时随机采样，回合内固定）
        self._active_delays = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_delays(torch.arange(self.num_envs, device=self.device))

    def reset_delays(self, env_ids: torch.Tensor):
        # 重新采样延迟：区间相同则固定；否则在 [min, max] 均匀随机
        if self.min_delay == self.max_delay:
            self._active_delays[env_ids] = self.min_delay
        else:
            sampled = torch.randint(
                self.min_delay,
                self.max_delay + 1,
                (len(env_ids),),
                dtype=torch.long,
                device=self.device,
            )
            self._active_delays[env_ids] = sampled

    def process_actions(self, actions: torch.Tensor):
        # 延迟缓冲丢弃最旧动作，并在尾部保存当前动作。
        if self.max_delay > 0:
            self._action_buffer = torch.cat(
                [self._action_buffer[1:], actions.unsqueeze(0)], dim=0
            )
        else:
            self._action_buffer[0] = actions

        # 每个环境按独立的延迟步数选择历史动作。
        indices = self.max_delay - self._active_delays
        actions_delayed = self._action_buffer[
            indices, torch.arange(self.num_envs, device=self.device)
        ]

        # 父类依次完成 IIR 滤波、缩放和关节软限位裁剪。
        super().process_actions(actions_delayed)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # reset 时：重采样延迟 + 清空缓冲与历史缓存
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]

        self.reset_delays(env_ids)
        self._action_buffer[:, env_ids] = 0.0
        super().reset(env_ids)


# -- Velocity low-pass + random delay --


@dataclass(kw_only=True)
class JointVelocityDelayedLowPassActionCfg(JointVelocityLowPassActionCfg):
    """Joint velocity action with random delay and 1st-order low-pass filter."""

    min_delay: int = 0
    max_delay: int = 2

    def build(self, env: ManagerBasedRlEnv) -> JointVelocityDelayedLowPassAction:
        return JointVelocityDelayedLowPassAction(self, env)


class JointVelocityDelayedLowPassAction(JointVelocityLowPassAction):
    """Applies a random delay and a 1st-order low-pass filter to raw actions."""

    # 与 JointPositionDelayedLowPassAction 逻辑完全对称，仅父类（速度版）不同
    def __init__(self, cfg: JointVelocityDelayedLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.min_delay = cfg.min_delay
        self.max_delay = cfg.max_delay
        if not 0 <= self.min_delay <= self.max_delay:
            raise ValueError("Delays must satisfy 0 <= min_delay <= max_delay")

        # Buffer shape: (max_delay + 1, num_envs, action_dim)
        self._action_buffer = torch.zeros(
            self.max_delay + 1, self.num_envs, self.action_dim, device=self.device
        )

        # Active delays for each env, shape: (num_envs,)
        self._active_delays = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_delays(torch.arange(self.num_envs, device=self.device))

    def reset_delays(self, env_ids: torch.Tensor):
        if self.min_delay == self.max_delay:
            self._active_delays[env_ids] = self.min_delay
        else:
            sampled = torch.randint(
                self.min_delay,
                self.max_delay + 1,
                (len(env_ids),),
                dtype=torch.long,
                device=self.device,
            )
            self._active_delays[env_ids] = sampled

    def process_actions(self, actions: torch.Tensor):
        if self.max_delay > 0:
            self._action_buffer = torch.cat(
                [self._action_buffer[1:], actions.unsqueeze(0)], dim=0
            )
        else:
            self._action_buffer[0] = actions

        indices = self.max_delay - self._active_delays
        actions_delayed = self._action_buffer[
            indices, torch.arange(self.num_envs, device=self.device)
        ]

        super().process_actions(actions_delayed)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]

        self.reset_delays(env_ids)
        self._action_buffer[:, env_ids] = 0.0
        super().reset(env_ids)
