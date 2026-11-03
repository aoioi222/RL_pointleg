# =============================================================================
# mode_command.py — 离散运动模式指令（多条件策略训练用 Command Term）
#
# 【模块定位】
#   「指令（Command）」侧的离散模式生成器，为「多模式条件化策略」提供训练条件：
#   每个 episode 随机（或按地形映射）给机器人指定一个运动模式：
#       MODE_WALK=0  行走  → 观测值 -1.0
#       MODE_CLIMB=1 爬越  → 观测值  0.0
#       MODE_CROUCH=2 匍匐 → 观测值 +1.0
#   模式作为一维观测喂给策略网络，同时可被奖励函数读取（见 rewards.get_mode_id）。
#
# 【当前状态】
#   注意：本文件暂未被 env_cfgs.py 导入（处于「备用」状态）。
#   若启用，在 cfg.commands 中添加：
#       "mode": ModeCommandCfg(mode_probs=(0.5, 0.3, 0.2), ...)
#   部署（真实机器人）时设置 fixed_mode=0/1/2 固定模式。
#
# 【架构参考】
#   结构与 mjlab 的 UniformVelocityCommand 一致，继承 CommandTerm 基类：
#   - _resample_command(): episode 边界时采样新模式
#   - _update_command(): 每步更新（离散模式回合内不变，故为空实现）
#   - command 属性: 暴露给观测管线的归一化模式值 [num_envs, 1]
# =============================================================================

"""Discrete mode command term for multi-conditioned policy training.

Architecture is modeled after UniformVelocityCommand / UniformVelocityCommandCfg in
mjlab/tasks/velocity/mdp/velocity_command.py.

During training: mode_id in {0, 1, 2} is sampled randomly per episode based on mode_probs.
During deployment: ModeCommandCfg.fixed_mode is set to 0, 1, or 2, which can be modified externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# Locomotion Mode ID Constants
MODE_WALK = 0
MODE_CLIMB = 1
MODE_CROUCH = 2
NUM_MODES = 3


class ModeCommand(CommandTerm):
    """Discrete locomotion mode command term.

    The command tensor shape is [num_envs, 1], holding normalized mode values in [-1.0, 1.0].
    Mapping details: mode_id -> observation value
      0 (walk)   -> -1.0
      1 (climb)  ->  0.0
      2 (crouch) -> +1.0
    Using normalized scalar values keeps observation dimensions minimal (1D).
    """

    cfg: ModeCommandCfg

    def __init__(self, cfg: ModeCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Stores current integer mode_id for each environment
        self._mode_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Normalized command observations: shape [num_envs, 1]
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

        # Pre-calculate cumulative sum of mode probabilities (normalized for multinomial sampling)
        probs = torch.tensor(cfg.mode_probs, dtype=torch.float32, device=self.device)
        self._mode_probs = probs / probs.sum()

    @property
    def command(self) -> torch.Tensor:
        """Normalized mode command observation tensor of shape [num_envs, 1]."""
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Episode 边界采样新模式：fixed_mode > 地形映射 > 随机采样 三选一。"""
        if self.cfg.fixed_mode is not None:
            # Deployment mode: fixed mode
            self._mode_id[env_ids] = self.cfg.fixed_mode
        else:
            # Training mode: map modes based on active terrain type if configured
            terrain = getattr(self._env.scene, "terrain", None)
            terrain_types = getattr(terrain, "terrain_types", None)
            if self.cfg.terrain_mode_mapping is not None and terrain_types is not None:
                # 地形映射：按子地形序号查表得到模式（如平底→walk、楼梯→climb、低矮地形→crouch）
                mapping = torch.tensor(
                    self.cfg.terrain_mode_mapping, dtype=torch.long, device=self.device
                )
                types = terrain_types[env_ids]
                # Defensive clamp to prevent out of bounds indexing
                types_safe = torch.clamp(types, 0, len(mapping) - 1)
                self._mode_id[env_ids] = mapping[types_safe]
            else:
                # Fallback to random multinomial sampling
                # 按 mode_probs 多项式采样（概率自动归一化）
                sampled = torch.multinomial(
                    self._mode_probs.expand(len(env_ids), -1),
                    num_samples=1,
                    replacement=True,
                ).squeeze(-1)
                self._mode_id[env_ids] = sampled

        # Update normalized command observations
        self._update_obs_from_mode_id(env_ids)

    def _update_command(self) -> None:
        # Modes are persistent across episodes, no step-level command updates required
        pass

    def _update_obs_from_mode_id(self, env_ids: torch.Tensor) -> None:
        """把整数 mode_id 线性投影到 [-1, +1]：walk=0 -> -1.0, climb=1 -> 0.0, crouch=2 -> +1.0。"""
        mode = self._mode_id[env_ids].float()
        # Linear projection: [0, NUM_MODES - 1] -> [-1.0, +1.0]
        normalized = 2.0 * mode / (NUM_MODES - 1) - 1.0
        self._command[env_ids, 0] = normalized

    def create_gui(
        self,
        name: str,
        server,  # ViserServer instance (late-bound to avoid top-level dependency)
        get_env_idx,
        on_change=None,
        request_action=None,
    ) -> None:
        """Construct interactive mode selection controls in the Viser visualizer GUI.

        When enabled is active, slider values override target environment mode IDs in real-time.
        Discrete modes: 0=walk, 1=climb, 2=crouch (step size = 1).
        """
        # 在 Viser 可视化界面中创建模式选择控件（checkbox 开关 + 0~2 的整数滑杆）
        with server.gui.add_folder(f"{name.capitalize()} (0=walk 1=climb 2=crouch)"):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            mode_slider = server.gui.add_slider(
                "mode_id",
                min=0,
                max=NUM_MODES - 1,
                step=1,
                initial_value=0,
            )

        # Store GUI handles for compute() execution
        self._gui_enabled = enabled
        self._gui_slider = mode_slider
        self._gui_get_env_idx = get_env_idx

    def compute(self, dt: float) -> None:
        """Perform step updates: override mode_id with interactive sliders if visualizer GUI is active."""
        super().compute(dt)
        # 若可视化 GUI 开启，用滑杆实时覆盖目标环境的模式（便于人为引导训练）
        if (
            getattr(self, "_gui_enabled", None) is not None
            and self._gui_enabled.value
            and self._gui_get_env_idx is not None
        ):
            idx = self._gui_get_env_idx()
            new_mode = int(round(self._gui_slider.value))
            env_ids = torch.tensor([idx], dtype=torch.long, device=self.device)
            self._mode_id[env_ids] = new_mode
            self._update_obs_from_mode_id(env_ids)

    def get_mode_id(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Retrieve active environment mode IDs for reward/termination conditioning."""
        # 供奖励/终止函数读取当前模式（rewards.get_mode_id 内部调用）
        if env_ids is None:
            return self._mode_id
        return self._mode_id[env_ids]


@dataclass(kw_only=True)
class ModeCommandCfg(CommandTermCfg):
    """Discrete ModeCommand configuration class."""

    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    """Discrete mode resampling duration (resampled at episode boundary, equal to episode length)."""

    mode_probs: tuple[float, ...] = (0.5, 0.3, 0.2)
    """Probabilities for discrete modes (walk, climb, crouch). Normalized automatically."""

    fixed_mode: int | None = None
    """None for active random sampling during training; 0/1/2 for deployment."""

    terrain_mode_mapping: tuple[int, ...] | None = None
    """Force maps discrete modes based on environment sub-terrain grid index.
    
    E.g., (0, 0, 0, 0, 1, 1, 2) corresponds to walk mode for first four sub-terrains, 
    climb mode for the next two sub-terrains, and crouch mode for the last sub-terrain.
    Overrides mode_probs if active.
    """

    def build(self, env: ManagerBasedRlEnv) -> ModeCommand:
        return ModeCommand(self, env)
