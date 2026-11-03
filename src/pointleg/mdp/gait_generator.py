"""Reference gait generation for the 12-DoF point-foot quadruped."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PointFootGaitCfg:
    """Parameters for a diagonal walking gait in policy leg-major order."""

    frequency: float = 2.0
    duty_factor: float = 0.60
    hip_abduction: float = 0.0
    hip_pitch: float = 0.550
    knee: float = -1.125
    hip_pitch_amplitude: float = 0.22
    knee_lift_amplitude: float = 0.35
    hip_abduction_amplitude: float = 0.08
    command_deadband: float = 0.05


class GaitGenerator:
    """Generate diagonal walking references for four legs and twelve joints.

    The output order is ``fl, fr, rl, rr``. Each leg contributes
    ``hip_abduction, hip_pitch, knee``. The reference is intended for warm-start
    analysis and future imitation rewards; the current RL action path remains a
    direct 12-dimensional joint-position policy.
    """

    def __init__(self, cfg: PointFootGaitCfg | None = None):
        self.cfg = cfg or PointFootGaitCfg()
        if not 0.0 < self.cfg.duty_factor < 1.0:
            raise ValueError("duty_factor must be between 0 and 1")
        if self.cfg.frequency <= 0.0:
            raise ValueError("frequency must be positive")

    def advance_phase(self, phase: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance a batch of gait phases by one control interval."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        return torch.remainder(phase + self.cfg.frequency * dt, 1.0)

    def generate(
        self,
        phase: torch.Tensor,
        command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a batch of 12-dimensional point-foot joint references.

        Args:
            phase: Global gait phase in cycles with shape ``(num_envs,)``.
            command: Optional ``(num_envs, 3)`` planar velocity command.

        Returns:
            Tensor with shape ``(num_envs, 12)`` in leg-major joint order.
        """
        if phase.ndim == 0:
            phase = phase.unsqueeze(0)
        if phase.ndim != 1:
            raise ValueError("phase must have shape (num_envs,)")
        if not torch.is_floating_point(phase):
            raise TypeError("phase must use a floating-point dtype")

        num_envs = phase.shape[0]
        device = phase.device
        dtype = phase.dtype
        if command is None:
            command = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        if command.shape != (num_envs, 3):
            raise ValueError("command must have shape (num_envs, 3)")

        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        active = (command_speed > self.cfg.command_deadband).to(dtype)
        stride_scale = torch.clamp(command_speed, max=1.0) * active
        lateral_scale = torch.clamp(torch.abs(command[:, 1]), max=1.0) * active

        diagonal_offsets = torch.tensor(
            [0.0, 0.5, 0.5, 0.0], device=device, dtype=dtype
        )
        side_signs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device, dtype=dtype)
        leg_phase = torch.remainder(phase[:, None] + diagonal_offsets, 1.0)
        swing_phase = torch.clamp(
            (leg_phase - self.cfg.duty_factor) / (1.0 - self.cfg.duty_factor),
            0.0,
            1.0,
        )
        in_swing = leg_phase >= self.cfg.duty_factor
        swing_lift = torch.where(
            in_swing,
            torch.sin(torch.pi * swing_phase),
            torch.zeros_like(leg_phase),
        )
        stride = torch.sin(2.0 * torch.pi * leg_phase)

        hip_abduction = self.cfg.hip_abduction + (
            side_signs[None, :]
            * self.cfg.hip_abduction_amplitude
            * lateral_scale[:, None]
            * stride
        )
        hip_pitch = self.cfg.hip_pitch + (
            self.cfg.hip_pitch_amplitude * stride_scale[:, None] * stride
        )
        knee = self.cfg.knee - (
            self.cfg.knee_lift_amplitude * stride_scale[:, None] * swing_lift
        )

        # XML待核对：三个关节的正方向与站立关节角必须与最终 XML 一致。
        return torch.stack((hip_abduction, hip_pitch, knee), dim=-1).reshape(num_envs, 12)
