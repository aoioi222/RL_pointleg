"""Keep supplied motor data distinct from the provisional simulation model."""

from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg
from pointleg import robot_cfg
from pointleg.motor_spec import LEG_MOTOR_SPEC


MODEL_DIR = Path(__file__).resolve().parents[3] / "mjcf"


class MotorSpecificationTests(unittest.TestCase):
    def assert_effort_limits(self, model):
        joint_ids = [model.joint(name).id for name in robot_cfg.LEG_JOINT_NAMES]
        self.assertTrue(model.jnt_actfrclimited[joint_ids].all())
        np.testing.assert_allclose(
            model.jnt_actfrcrange[joint_ids],
            np.tile([-robot_cfg.EFFORT_LEG, robot_cfg.EFFORT_LEG], (12, 1)),
        )
        if model.nu:
            self.assertEqual(model.nu, 12)
            self.assertTrue(model.actuator_forcelimited.all())
            np.testing.assert_allclose(
                model.actuator_forcerange,
                np.tile([-robot_cfg.EFFORT_LEG, robot_cfg.EFFORT_LEG], (12, 1)),
            )
            np.testing.assert_allclose(model.actuator_gear[:, 0], 1.0)

    def test_datasheet_units_and_unknown_continuous_rating(self):
        motor = LEG_MOTOR_SPEC
        self.assertEqual(motor.model, "N6014B-12.6")
        self.assertEqual(motor.mass_kg, 0.535)
        self.assertEqual((motor.diameter_m, motor.length_m), (0.065, 0.060))
        self.assertAlmostEqual(motor.reduction_ratio, 38 / 3)
        self.assertEqual(motor.input_voltage_range_v, (24.0, 75.0))
        self.assertEqual(motor.max_line_current_a, 40.0)
        self.assertEqual(motor.driving_peak_torque_nm, 31.7)
        self.assertEqual(motor.braking_peak_torque_nm, 34.4)
        self.assertIsNone(motor.continuous_torque_nm)

    def test_voltage_estimate_is_unloaded_and_reference_based(self):
        self.assertEqual(robot_cfg.MOTOR_SUPPLY_VOLTAGE_V, 48.0)
        self.assertAlmostEqual(LEG_MOTOR_SPEC.estimate_no_load_speed(60.0), 54.2)
        self.assertAlmostEqual(robot_cfg.ESTIMATED_NO_LOAD_SPEED_RAD_S, 43.36)
        for voltage in (0.0, 23.9, 75.1, float("nan"), float("inf")):
            with self.subTest(voltage=voltage):
                with self.assertRaisesRegex(ValueError, "supply voltage"):
                    LEG_MOTOR_SPEC.estimate_no_load_speed(voltage)

    def test_effort_cap_cannot_exceed_driving_peak(self):
        for effort in (robot_cfg.EFFORT_LEG, 31.7):
            LEG_MOTOR_SPEC.validate_simulation_effort_limit(effort)
        for effort in (-1.0, 0.0, 31.8, 34.4, float("nan"), float("inf")):
            with self.subTest(effort=effort):
                with self.assertRaisesRegex(ValueError, "effort limit"):
                    LEG_MOTOR_SPEC.validate_simulation_effort_limit(effort)

    def test_standalone_xml_and_scene_share_operating_cap(self):
        for filename in ("pointleg.xml", "scene.xml"):
            with self.subTest(filename=filename):
                model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / filename))
                self.assert_effort_limits(model)

    def test_all_tasks_share_limits_without_double_reduction(self):
        for task in ("Flat", "Rough", "Crawl"):
            with self.subTest(task=task):
                cfg = load_env_cfg(f"PointLeg-{task}-v0")
                model = Entity(cfg.scene.entities["robot"]).spec.compile()
                self.assert_effort_limits(model)
                np.testing.assert_allclose(model.actuator_gainprm[:, 0], robot_cfg.STIFFNESS_LEG)
                np.testing.assert_allclose(model.actuator_biasprm[:, 2], -robot_cfg.DAMPING_LEG)

    def test_large_pd_targets_saturate_at_operating_cap(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "pointleg.xml"))
        data = mujoco.MjData(model)
        joint_ids = [model.joint(name).id for name in robot_cfg.LEG_JOINT_NAMES]
        dof_ids = model.jnt_dofadr[joint_ids]
        for direction in (-1.0, 1.0):
            with self.subTest(direction=direction):
                mujoco.mj_resetDataKeyframe(model, data, 0)
                data.ctrl[:] = direction * 1.0e6
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.actuator_force, direction * 25.0)
                np.testing.assert_allclose(data.qfrc_actuator[dof_ids], direction * 25.0)

    def test_rebuilding_spec_replaces_stale_joint_force_limits(self):
        spec = mujoco.MjSpec.from_file(str(MODEL_DIR / "pointleg.xml"))
        for name in robot_cfg.LEG_JOINT_NAMES:
            joint = spec.joint(name)
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
            joint.actfrcrange = [-2.0, 2.0]
        with patch("pointleg.robot_cfg.mujoco.MjSpec") as loader:
            loader.from_file.return_value = spec
            rebuilt = robot_cfg.get_spec()
        self.assert_effort_limits(rebuilt.compile())


if __name__ == "__main__":
    unittest.main()
