# =============================================================================
# disturbances.py — 连续低频外力扰动（自定义 Event/Step 事件）
#
# 【模块定位】
#   本文件属于 mdp 组件库中的「扰动（Disturbance）」侧，用于在训练中给机器人
#   施加平滑的随机外力/力矩，增强策略抗扰鲁棒性（sim2real 常用手段）。
#   以 EventTerm 方式使用，调用模式为 mode="step"（每个仿真步执行一次）：
#       cfg.events["external_disturbance"] = EventTermCfg(
#           func=apply_continuous_disturbance, mode="step",
#           params={... force_range / torque_range / resample_time_range ...})
#
# 【当前状态】备用：已实现并在 env_cfgs.py 中导入，但尚未挂到 events 段。
#
# 【设计要点】
#   - 不是每步独立随机（那样是白噪声，机器人学不出抗扰），而是：
#     周期性采样一个目标力/力矩 → 用一阶低通（指数平滑）逐步逼近目标
#     → 形成"平滑且持续变化"的低频扰动，更接近真实风载/推搡。
#   - 定时器到点才重新采样目标值与下一次采样间隔。
#   - reset() 时清零所有状态并清除仿真中的外力，避免跨 episode 残留。
# =============================================================================

"""Continuous low-frequency external wrench disturbance for mjlab."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class apply_continuous_disturbance:
    """Apply smooth, continuous low-frequency force and torque disturbances to bodies.

    Achieved by sampling random target forces and torques periodically, and interpolating
    towards them via a 1st-order low-pass filter (exponential smoothing).
    """

    # 核心机制（每步 __call__）：
    #   1) 定时器倒数；到点 → 重新采样目标力/力矩（force_range/torque_range 内均匀随机）
    #                     + 重新采样下一次采样间隔（resample_time_range 内均匀随机）
    #   2) 当前力/力矩向目标做一阶指数平滑：
    #        alpha = 1 - exp(-dt / time_constant)
    #        current = (1-alpha)*current + alpha*target
    #      time_constant 越大 → alpha 越小 → 变化越缓慢平滑
    #   3) 通过 write_external_wrench_to_sim 写入仿真，作用于指定 body

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._body_ids = cfg.params["asset_cfg"].body_ids
        self._num_envs = env.num_envs
        self._device = env.device
        self._step_dt = env.step_dt

        self._num_bodies = (
            len(self._body_ids)
            if isinstance(self._body_ids, list)
            else self._asset.num_bodies
        )

        # Disturbance states: shape (num_envs, num_bodies, 3)
        # 当前力/力矩（逐步逼近目标）+ 目标力/力矩（定时重采样的随机值）
        self._current_force = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        self._current_torque = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        self._target_force = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        self._target_torque = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)

        # Periodic resampling timers: shape (num_envs,)
        # 每个环境的重新采样倒计时（秒）
        self._timer = torch.zeros(self._num_envs, device=self._device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        resample_time_range: tuple[float, float],
        time_constant: float,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Tick disturbance state: interpolate towards targets, periodically resample targets.

        Use with mode="step".
        """
        del env, env_ids, asset_cfg  # Unused, step events act on all envs.
        dt = self._step_dt

        # Decrement timers
        self._timer -= dt

        # Identify envs that need resampling
        resample = self._timer <= 0
        if resample.any():
            resample_ids = resample.nonzero(as_tuple=False).squeeze(-1)
            n = len(resample_ids)

            # Sample new targets
            size = (n, self._num_bodies, 3)
            self._target_force[resample_ids] = sample_uniform(*force_range, size, self._device)
            self._target_torque[resample_ids] = sample_uniform(*torque_range, size, self._device)

            # Sample new timer durations
            t_low, t_high = resample_time_range
            self._timer[resample_ids] = (
                torch.rand(n, device=self._device) * (t_high - t_low) + t_low
            )

        # Exponential smoothing step: x_new = (1 - alpha) * x + alpha * x_target
        alpha = 1.0 - math.exp(-dt / time_constant)

        self._current_force = (1.0 - alpha) * self._current_force + alpha * self._target_force
        self._current_torque = (1.0 - alpha) * self._current_torque + alpha * self._target_torque

        # Apply smooth wrenches to simulation
        # 对全部环境写入当前平滑后的外力/力矩（body_ids 指定作用在哪些刚体上）
        all_env_ids = torch.arange(self._num_envs, device=self._device)
        self._asset.write_external_wrench_to_sim(
            self._current_force, self._current_torque, env_ids=all_env_ids, body_ids=self._body_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset disturbance states and clear external wrenches for the given envs."""
        if env_ids is None:
            env_ids = slice(None)

        # Reset states
        # 清零目标与当前力/力矩及定时器，防止上一 episode 的扰动残留
        self._current_force[env_ids] = 0.0
        self._current_torque[env_ids] = 0.0
        self._target_force[env_ids] = 0.0
        self._target_torque[env_ids] = 0.0
        self._timer[env_ids] = 0.0

        if isinstance(env_ids, slice):
            reset_ids = torch.arange(self._num_envs, device=self._device)[env_ids]
        else:
            reset_ids = env_ids

        if len(reset_ids) > 0:
            # 同时在仿真中把该批环境的外力清为 0
            zeros = torch.zeros((len(reset_ids), self._num_bodies, 3), device=self._device)
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=reset_ids, body_ids=self._body_ids
            )
