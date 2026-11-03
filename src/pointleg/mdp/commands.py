"""Velocity commands with exclusive sampling groups and terrain constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from .terrain_utils import terrain_type_mask

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """Sample stand, forward, backward, lateral, yaw and mixed body-frame commands."""

    cfg: UniformThresholdVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.is_lateral_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_yaw_env = torch.zeros_like(self.is_lateral_env)
        self.is_backward_env = torch.zeros_like(self.is_lateral_env)
        self.was_climbing = torch.zeros_like(self.is_lateral_env)
        self.progress_direction_w = torch.zeros(self.num_envs, 2, device=self.device)

    def _climbing_mask(self):
        return terrain_type_mask(self._env, self.cfg.climbing_terrain_names) & ~self.is_standing_env

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.is_lateral_env[env_ids] = False
        self.is_yaw_env[env_ids] = False
        self.is_backward_env[env_ids] = False
        active = env_ids[~self.is_standing_env[env_ids]]
        forward = active[self.is_forward_env[active]]
        self.is_heading_env[forward] = False
        candidates = active[~self.is_forward_env[active]]
        if self.cfg.rel_backward_envs > 0.0 and len(candidates):
            # 后退从非前进组中抽取，换算条件概率以保持活动环境中的目标占比。
            probability = self.cfg.rel_backward_envs / (1.0 - self.cfg.rel_forward_envs)
            selected = torch.rand(len(candidates), device=self.device) < probability
            backward = candidates[selected & ~self._climbing_mask()[candidates]]
            self.is_backward_env[backward] = True
            self.vel_command_b[backward, 0] = torch.empty(len(backward), device=self.device).uniform_(
                *self.cfg.backward_velocity_range
            )
            self.vel_command_b[backward, 1:] = 0.0
            if self.cfg.heading_command:
                self.heading_target[backward] = self.robot.data.heading_w[backward]
                self.is_heading_env[backward] = True
            candidates = candidates[~self.is_backward_env[candidates]]
        if len(candidates):
            flat = terrain_type_mask(self._env, ("flat",))[candidates]
            lateral_probability = torch.where(
                flat, self.cfg.flat_lateral_probability, self.cfg.rel_lateral_envs
            )
            yaw_probability = torch.where(
                flat, self.cfg.flat_yaw_probability, self.cfg.rel_yaw_envs
            )
            draw = torch.rand(len(candidates), device=self.device)
            lateral = candidates[draw < lateral_probability]
            yaw = candidates[(draw >= lateral_probability) & (draw < lateral_probability + yaw_probability)]
            self.is_lateral_env[lateral] = True
            self.is_yaw_env[yaw] = True
            self.vel_command_b[lateral, 0] = 0.0
            self.vel_command_b[lateral, 2] = 0.0
            self.vel_command_b[yaw, :2] = 0.0
            if self.cfg.heading_command:
                self.heading_target[lateral] = self.robot.data.heading_w[lateral]
                self.is_heading_env[lateral] = True
                delta = torch.empty(len(yaw), device=self.device).uniform_(
                    -self.cfg.yaw_only_heading_range, self.cfg.yaw_only_heading_range
                )
                self.heading_target[yaw] = self.robot.data.heading_w[yaw] + delta
                self.is_heading_env[yaw] = True
        climbing = env_ids[self._climbing_mask()[env_ids]]
        if len(climbing):
            self.is_lateral_env[climbing] = False
            self.is_yaw_env[climbing] = False
            maximum = max(0.0, self.cfg.ranges.lin_vel_x[1])
            self.vel_command_b[climbing, 0] = self.vel_command_b[climbing, 0].abs().clamp(
                min=min(self.cfg.minimum_climbing_speed, maximum), max=maximum
            )
            self.vel_command_b[climbing, 1:] = 0.0
            self.is_heading_env[climbing] = self.cfg.heading_command
        # 在记录世界坐标进度方向前应用高速横移限制，与后续指令更新保持一致。
        high_speed = env_ids[self.vel_command_b[env_ids, 0].abs() >= self.cfg.high_speed_threshold]
        self.vel_command_b[high_speed, 1] = 0.0
        self._apply_deadbands()
        self.vel_command_b[self.is_standing_env] = 0.0
        heading = self.robot.data.heading_w[env_ids]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        command_x = self.vel_command_b[env_ids, 0]
        command_y = self.vel_command_b[env_ids, 1]
        self.progress_direction_w[env_ids, 0] = cos_h * command_x - sin_h * command_y
        self.progress_direction_w[env_ids, 1] = sin_h * command_x + cos_h * command_y
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    def _apply_deadbands(self):
        small_linear = torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=1) < self.cfg.linear_deadband
        self.vel_command_b[small_linear, :2] = 0.0
        small_yaw = self.vel_command_b[:, 2].abs() < self.cfg.angular_deadband
        self.vel_command_b[small_yaw, 2] = 0.0

    def _update_command(self):
        super()._update_command()
        climbing = self._climbing_mask()
        departed = self.was_climbing & ~climbing
        if departed.any():
            self._resample_command(departed.nonzero(as_tuple=False).flatten())
            super()._update_command()
        self.vel_command_b[climbing, 1] = 0.0
        self.is_backward_env[climbing] = False
        if self.cfg.heading_command:
            yaw_limit = self.cfg.climbing_yaw_limit
            self.vel_command_b[climbing, 2] = torch.clamp(
                self.cfg.heading_control_stiffness * self.heading_error[climbing],
                min=max(-yaw_limit, self.cfg.ranges.ang_vel_z[0]),
                max=min(yaw_limit, self.cfg.ranges.ang_vel_z[1]),
            )
        else:
            self.vel_command_b[climbing, 2] = 0.0
        self.was_climbing[:] = climbing
        self.vel_command_b[self.is_forward_env & ~climbing, 1:] = 0.0
        self.vel_command_b[self.is_backward_env & ~climbing, 1] = 0.0
        high_speed = self.vel_command_b[:, 0].abs() >= self.cfg.high_speed_threshold
        self.vel_command_b[high_speed, 1] = 0.0
        self._apply_deadbands()
        self.vel_command_b[self.is_standing_env] = 0.0


@dataclass(kw_only=True)
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    """Forward/backward probabilities apply to non-standing samples.

    Lateral/yaw probabilities apply to the remaining samples. Flat terrain uses
    its own configurable mutually exclusive lateral/yaw probabilities.
    Mixed world-frame commands are not supported by these body-frame constraints.
    """

    yaw_only_heading_range: float = 1.0
    rel_lateral_envs: float = 0.0
    rel_yaw_envs: float = 0.0
    rel_backward_envs: float = 0.0
    backward_velocity_range: tuple[float, float] = (-0.4, -0.2)
    flat_lateral_probability: float = 0.35
    flat_yaw_probability: float = 0.35
    linear_deadband: float = 0.2
    angular_deadband: float = 0.1
    minimum_climbing_speed: float = 0.3
    climbing_yaw_limit: float = 0.3
    high_speed_threshold: float = 0.8
    climbing_terrain_names: tuple[str, ...] = ("pyramid_stairs", "pyramid_stairs_inv", "rc_wall")

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.rel_forward_envs <= 1.0 or not 0.0 <= self.rel_backward_envs <= 1.0:
            raise ValueError("Forward and backward probabilities must lie between zero and one")
        if self.rel_forward_envs + self.rel_backward_envs > 1.0:
            raise ValueError("Forward and backward probabilities must sum to at most one")
        if self.rel_backward_envs > 0.0:
            lower, upper = self.backward_velocity_range
            if not self.ranges.lin_vel_x[0] <= lower <= upper < 0.0:
                raise ValueError("Backward velocity range must be negative and within the x velocity range")
            if upper > self.ranges.lin_vel_x[1] or -upper < self.linear_deadband:
                raise ValueError("Backward velocities must respect the x range and linear deadband")
        for lateral, yaw in ((self.rel_lateral_envs, self.rel_yaw_envs),
                             (self.flat_lateral_probability, self.flat_yaw_probability)):
            if lateral < 0 or yaw < 0 or lateral + yaw > 1:
                raise ValueError("Lateral and yaw probabilities must be nonnegative and sum to at most one")
        if self.rel_world_envs != 0:
            raise ValueError("Terrain-conditioned velocity commands require rel_world_envs=0")

    def build(self, env: ManagerBasedRlEnv) -> UniformThresholdVelocityCommand:
        return UniformThresholdVelocityCommand(self, env)
