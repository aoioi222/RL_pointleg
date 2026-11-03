"""N6014B-12.6 output-shaft specifications supplied with the motor datasheet.

The torque figures are peaks, not continuous ratings or stall-torque samples.
The supplied FOC curve was measured at 60 V and 25 degrees Celsius.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MotorSpec:
    """Keep manufacturer limits separate from simulation operating limits."""

    model: str
    mass_kg: float
    diameter_m: float
    length_m: float
    reduction_ratio: float
    input_voltage_range_v: tuple[float, float]
    max_line_current_a: float
    driving_peak_torque_nm: float
    braking_peak_torque_nm: float
    no_load_speed_rad_s: float
    speed_reference_voltage_v: float
    continuous_torque_nm: float | None = None

    def estimate_no_load_speed(self, supply_voltage_v: float) -> float:
        """Estimate unloaded output speed by voltage ratio, not a loaded limit."""
        low, high = self.input_voltage_range_v
        if not math.isfinite(supply_voltage_v) or not low <= supply_voltage_v <= high:
            raise ValueError(f"Motor supply voltage must be within [{low}, {high}] V")
        return self.no_load_speed_rad_s * supply_voltage_v / self.speed_reference_voltage_v

    def validate_simulation_effort_limit(self, effort_limit_nm: float) -> None:
        """Reject limits above either peak without implying continuous capability."""
        peak = min(self.driving_peak_torque_nm, self.braking_peak_torque_nm)
        if not math.isfinite(effort_limit_nm) or not 0.0 < effort_limit_nm <= peak:
            raise ValueError(f"Simulation effort limit must be positive and at most {peak} Nm")


LEG_MOTOR_SPEC = MotorSpec(
    model="N6014B-12.6",
    mass_kg=0.535,
    diameter_m=0.065,
    length_m=0.060,
    reduction_ratio=38.0 / 3.0,
    input_voltage_range_v=(24.0, 75.0),
    max_line_current_a=40.0,
    driving_peak_torque_nm=31.7,
    braking_peak_torque_nm=34.4,
    no_load_speed_rad_s=54.2,
    speed_reference_voltage_v=60.0,
)
