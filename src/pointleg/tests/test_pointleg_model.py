"""Compile and exercise the ball-foot XML and its native force sensors."""

from pathlib import Path
import unittest

import mujoco
import numpy as np

from test_pointleg import PACKAGE_DIR
from mjlab.entity import Entity
from pointleg import robot_cfg


MODEL_DIR = PACKAGE_DIR.parents[1] / "mjcf"


class PointLegModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "pointleg.xml"))

    def test_both_model_entries_and_scene_compile(self):
        for filename in ("pointleg.xml", "scene.xml"):
            with self.subTest(filename=filename):
                model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / filename))
                self.assertEqual((model.nq, model.nv, model.nu), (19, 18, 12))
                self.assertEqual((model.nsensor, model.nsensordata), (8, 16))
                self.assertEqual(tuple(model.joint(i).name for i in range(1, model.njnt)), robot_cfg.LEG_JOINT_NAMES)

    def test_ball_geometry_mass_and_bottom_site(self):
        for leg in robot_cfg.LEG_NAMES:
            with self.subTest(leg=leg):
                body_id = self.model.body(f"{leg}_foot_Link").id
                geom_id = self.model.geom(f"{leg}_foot_Link_collision").id
                site_id = self.model.site(f"{leg}_foot_site").id
                self.assertEqual(self.model.body_jntnum[body_id], 0)
                self.assertAlmostEqual(self.model.body_mass[body_id], 0.08)
                np.testing.assert_allclose(self.model.body_inertia[body_id], [0.00002] * 3)
                self.assertEqual(self.model.geom_type[geom_id], mujoco.mjtGeom.mjGEOM_SPHERE)
                self.assertAlmostEqual(self.model.geom_size[geom_id, 0], 0.025)
                self.assertEqual(self.model.geom_group[geom_id], 3)
                np.testing.assert_allclose(self.model.site_pos[site_id], [0.0, 0.0, -0.05])

    def test_straight_shin_reaches_ball_without_wheel_housing(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        mesh_names = {self.model.mesh(i).name for i in range(self.model.nmesh)}
        geom_names = {self.model.geom(i).name for i in range(self.model.ngeom)}
        for leg in robot_cfg.LEG_NAMES:
            with self.subTest(leg=leg):
                self.assertNotIn(f"{leg}_knee_Link", mesh_names)
                self.assertNotIn(f"{leg}_wheel_Link", mesh_names)
                self.assertNotIn(f"{leg}_knee_Link_0_collision", geom_names)
                self.assertNotIn(f"{leg}_knee_extension_collision", geom_names)
                self.assertNotIn(f"{leg}_knee_extension_visual", geom_names)
                shin_id = self.model.geom(f"{leg}_shin_collision").id
                self.assertEqual(self.model.geom_type[shin_id], mujoco.mjtGeom.mjGEOM_BOX)
                shin_axis = data.geom_xmat[shin_id].reshape(3, 3)[:, 2]
                shin_end = data.geom_xpos[shin_id] + shin_axis * self.model.geom_size[shin_id, 2]
                ball_center = data.geom(f"{leg}_foot_Link_collision").xpos
                np.testing.assert_allclose(shin_end, ball_center, atol=1e-6)
                self.assertAlmostEqual(self.model.body_mass[self.model.body(f"{leg}_knee_Link").id], 0.6965)

    def test_mjlab_rebuilds_actuators_and_task_keyframes(self):
        for factory, expected_z, expected_knee in (
            (robot_cfg.get_robot_cfg, 0.42, -1.125),
            (robot_cfg.get_robot_crawl_cfg, 0.20, robot_cfg.CRAWL_KNEE_ANGLE),
        ):
            with self.subTest(task=factory.__name__):
                entity = Entity(factory())
                self.assertEqual(len(entity.spec.keys), 1)
                self.assertEqual(entity.spec.keys[0].name, "init_state")
                model = entity.spec.compile()
                self.assertAlmostEqual(model.key_qpos[0, 2], expected_z)
                self.assertAlmostEqual(model.key_qpos[0, 9], expected_knee)
                self.assertEqual(model.nu, 12)
                self.assertEqual(model.nsensor, 8)
                visual = np.array([not model.geom(i).name.endswith("_collision") for i in range(model.ngeom)])
                self.assertFalse(model.geom_contype[visual].any())
                self.assertFalse(model.geom_conaffinity[visual].any())

    def test_force_sensors_respond_to_transmitted_load(self):
        spec = mujoco.MjSpec.from_file(str(MODEL_DIR / "pointleg.xml"))
        for collection in (list(spec.keys), list(spec.actuators), list(spec.joints)):
            for element in collection:
                spec.delete(element)
        model = spec.compile()
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for leg in robot_cfg.LEG_NAMES:
            with self.subTest(leg=leg):
                sensor_id = model.sensor(f"{leg}_foot_force").id
                self.assertEqual(model.sensor_type[sensor_id], mujoco.mjtSensor.mjSENS_FORCE)
                self.assertEqual(model.sensor_dim[sensor_id], 3)
                data.xfrc_applied[:] = 0.0
                applied_force = np.array([5.0, -3.0, 7.0])
                data.xfrc_applied[model.body(f"{leg}_foot_Link").id, :3] = applied_force
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(np.abs(data.sensor(f"{leg}_foot_force").data), np.abs(applied_force), atol=1e-6)

    def test_ground_touch_positive_and_airborne_touch_zero(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        for _ in range(1500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.sensordata).all())
        self.assertEqual(sum(warning.number for warning in data.warning), 0)
        for leg in robot_cfg.LEG_NAMES:
            self.assertGreater(data.sensor(f"{leg}_foot_touch").data[0], 0.0)
        data.qpos[2] += 2.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for leg in robot_cfg.LEG_NAMES:
            self.assertEqual(data.sensor(f"{leg}_foot_touch").data[0], 0.0)


if __name__ == "__main__":
    unittest.main()
