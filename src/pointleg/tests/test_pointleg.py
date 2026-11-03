"""Regression checks for point-foot business code without loading an XML model."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
import pointleg

from mjlab.managers.action_manager import ActionManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from pointleg import robot_cfg
from pointleg.mdp import lowpass_actions, rewards, terminations


class FakeScene(dict):
    """Supply per-environment origins for real reset and termination functions."""

    def __init__(self, robot):
        super().__init__(robot=robot)
        self.env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [4.0, 0.0, -1.0]])


class FakeRobot:
    """Provide joint data and record control targets without a physics model."""

    def __init__(self, names=None, target_names=None):
        self.joint_names = list(names if names is not None else robot_cfg.LEG_JOINT_NAMES)
        self.target_names = list(target_names if target_names is not None else self.joint_names)
        self.num_joints = len(self.joint_names)
        self.is_fixed_base = False
        root_state = torch.zeros(3, 13)
        root_state[:, 2] = robot_cfg.STANDING_BASE_HEIGHT
        root_state[:, 3] = 1.0
        self.data = SimpleNamespace(
            default_root_state=root_state,
            root_link_pos_w=root_state[:, :3].clone(),
            default_joint_pos=torch.zeros(3, self.num_joints),
            default_joint_vel=torch.zeros(3, self.num_joints),
            encoder_bias=torch.zeros(3, self.num_joints),
            joint_pos=torch.zeros(3, self.num_joints),
            joint_pos_limits=torch.tensor([-1.0, 1.0]).expand(3, self.num_joints, 2).clone(),
            soft_joint_pos_limits=torch.tensor([-0.5, 0.5]).expand(3, self.num_joints, 2).clone(),
        )

    def find_joints_by_actuator_names(self, patterns):
        return [self.joint_names.index(name) for name in self.target_names], self.target_names

    def set_joint_position_target(self, target, joint_ids):
        self.position_target = target.clone()

    def set_joint_velocity_target(self, target, joint_ids):
        self.velocity_target = target.clone()

    def write_root_link_pose_to_sim(self, pose, env_ids):
        self.data.root_link_pos_w[env_ids] = pose[:, :3]

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        self.velocity = velocity


def make_env(robot=None):
    """Construct tensor-only parallel environment inputs for manager checks."""
    return SimpleNamespace(scene=FakeScene(robot or FakeRobot()), num_envs=3, device="cpu")


class PointLegRegressionTests(unittest.TestCase):
    def action_cfg(self):
        cfg = load_env_cfg("PointLeg-Rough-v0").actions["leg_joint_pos"]
        cfg.min_delay = 0
        cfg.max_delay = 0
        return cfg

    def test_task_registration_and_default_runner(self):
        import rsl_rl

        self.assertNotIn(PACKAGE_DIR / "rsl_rl", Path(rsl_rl.__file__).resolve().parents)
        for mode in ("Flat", "Rough", "Crawl"):
            task_id = f"PointLeg-{mode}-v0"
            self.assertIn(task_id, list_tasks())
            self.assertIsNone(load_runner_cls(task_id))
            self.assertEqual(load_rl_cfg(task_id).experiment_name, f"pointleg_{mode.lower()}")

    def test_config_parameters_and_sensor_rays(self):
        for mode in ("Flat", "Rough", "Crawl"):
            for play in (False, True):
                with self.subTest(mode=mode, play=play):
                    cfg = load_env_cfg(f"PointLeg-{mode}-v0", play=play)
                    for kind in ("rewards", "terminations", "metrics", "curriculum", "events"):
                        for term in getattr(cfg, kind).values():
                            if term is None or inspect.isclass(term.func):
                                continue
                            args = (None, None) if kind in ("events", "curriculum") else (None,)
                            inspect.signature(term.func).bind(*args, **term.params)
                    for group in cfg.observations.values():
                        for term in group.terms.values():
                            inspect.signature(term.func).bind(None, **term.params)
                    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
                    ground_offsets, _ = sensors["base_height_sensor"].pattern.generate_rays(None, "cpu")
                    scan_offsets, _ = sensors["height_scanner"].pattern.generate_rays(None, "cpu")
                    self.assertEqual(tuple(ground_offsets.shape), (1, 3))
                    self.assertEqual(scan_offsets.shape[0], 273)
                    self.assertEqual(set(cfg.actions), {"leg_joint_pos"})
                    self.assertEqual(cfg.actions["leg_joint_pos"].expected_joint_names, robot_cfg.LEG_JOINT_NAMES)
                    self.assertEqual(cfg.terminations["base_height_out_of_range"].params["sensor_name"], "base_height_sensor")

    def test_flat_config_penalizes_asymmetric_static_postures(self):
        cfg = load_env_cfg("PointLeg-Flat-v0")
        self.assertEqual(
            set(("feet_contact_without_cmd", "static_joint_posture", "static_leg_symmetry", "long_foot_air_time"))
            - set(cfg.rewards),
            set(),
        )
        symmetry = cfg.rewards["static_leg_symmetry"]
        self.assertEqual(symmetry.params["command_name"], "twist")
        self.assertEqual(symmetry.params["command_threshold"], 0.1)
        self.assertEqual(symmetry.weight, -0.35)
        self.assertEqual(cfg.rewards["feet_air_time"].weight, 0.4)
        self.assertEqual(cfg.rewards["feet_air_time"].params["threshold_max"], 0.7)
        self.assertEqual(cfg.rewards["long_foot_air_time"].weight, -3.0)
        self.assertEqual(cfg.rewards["long_foot_air_time"].params["max_air_time"], 0.8)

    def test_hip_abduction_cost_allows_small_adjustments_and_selects_only_hips(self):
        env = make_env()
        env.extras = {"log": {}}
        data = env.scene["robot"].data
        data.default_joint_pos[:] = 0.03
        data.joint_pos[:] = data.default_joint_pos
        data.joint_pos[:, [1, 2, 4, 5, 7, 8, 10, 11]] += 3.0
        data.joint_pos[0, [0, 3, 6, 9]] += 0.1
        data.joint_pos[1, [0, 3]] += 0.37
        data.joint_pos[2, [6, 9]] -= 0.37
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["hip_abduction_excess"]
        self.assertEqual(cfg.params["asset_cfg"].joint_names, (".*_hip_abduction_joint",))
        selected = SceneEntityCfg("robot", joint_ids=[0, 3, 6, 9])
        params = {**cfg.params, "asset_cfg": selected}
        torch.testing.assert_close(cfg.func(env, **params), torch.tensor([0.0, 2.0, 2.0]))
        data.joint_pos[1, [0, 3]] += 0.25
        torch.testing.assert_close(cfg.func(env, **params), torch.tensor([0.0, 8.0, 2.0]))
        self.assertNotIn("hip_abduction_excess", load_env_cfg("PointLeg-Crawl-v0").rewards)

    def test_long_foot_air_time_penalty_ignores_normal_swings_and_static_commands(self):
        env = make_env()
        env.command_manager = SimpleNamespace(get_command=lambda name: torch.tensor([
            [0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]))
        env.scene["feet_ground_contact"] = SimpleNamespace(
            data=SimpleNamespace(current_air_time=torch.tensor([
                [0.20, 0.55, 0.95, 0.40],
                [1.20, 1.20, 1.20, 1.20],
                [0.30, 0.40, 0.50, 0.55],
            ]))
        )
        penalty = rewards.long_foot_air_time_penalty(
            env, "feet_ground_contact", "twist", max_air_time=0.55
        )
        torch.testing.assert_close(penalty, torch.tensor([0.10, 0.0, 0.0]))

    def test_swing_height_rewards_completed_swings_and_resets_state(self):
        env = self.ground_env([0.42, 0.42, 0.42])
        env.step_dt = 0.02
        env.extras = {"log": {}}
        robot = env.scene["robot"]
        robot.find_sites = lambda name: ([robot_cfg.FOOT_SITE_NAMES.index(name)], [name])
        robot.data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)
        robot.data.site_pos_w = torch.zeros(3, 4, 3)
        ground_z = env.scene.env_origins[:, 2, None]
        commands = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
        env.command_manager = SimpleNamespace(get_command=lambda name: commands)
        found = torch.zeros(3, 4)
        found[:, 2:] = 1.0
        sensor = SimpleNamespace(
            primary_names=list(reversed(robot_cfg.FOOT_BODY_NAMES)),
            data=SimpleNamespace(
                found=found, current_contact_time=torch.zeros(3, 4),
                last_air_time=torch.full((3, 4), 0.34),
            ),
            compute_first_contact=lambda dt: found.bool(),
        )
        env.scene["feet_ground_contact"] = sensor
        cfg = load_env_cfg("PointLeg-Flat-v0").rewards["foot_swing_height"]
        term = cfg.func(cfg, env)
        self.assertEqual(term.site_ids, [3, 2, 1, 0])
        robot.data.site_pos_w[:, :, 2] = ground_z + torch.tensor([[0.1], [0.01], [0.1]])
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        found[:] = 1.0
        robot.data.site_pos_w[:, :, 2] = ground_z
        sensor.data.current_contact_time[:] = 0.02
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        sensor.data.current_contact_time[:] = 0.16
        result = term(env, **cfg.params)
        self.assertAlmostEqual(result[0].item(), 2.0 / env.step_dt, places=3)
        self.assertLess(result[1].item(), result[0].item() * 0.1)
        self.assertEqual(result[2].item(), 0.0)
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        found[:, :2] = 0.0
        robot.data.site_pos_w[:, :, 2] = ground_z + 0.1
        term(env, **cfg.params)
        found[:] = 0.0
        term(env, **cfg.params)
        found[:, 2:] = 1.0
        term(env, **cfg.params)
        found[:] = 1.0
        self.assertEqual(term(env, **cfg.params).tolist(), [0.0, 0.0, 0.0])
        found[:] = 0.0
        term(env, **cfg.params)
        term.reset(torch.tensor([0]))
        self.assertFalse(term.has_swing[0].any())
        self.assertTrue(term.has_swing[1].all())
        term.reset()
        self.assertFalse(term.has_swing.any())
        self.assertTrue(term.supported_swing.all())
        self.assertFalse(term.pending_valid.any())

    def test_support_rewards_penalize_early_takeoff_and_suppress_flight_rewards(self):
        env = make_env()
        env.extras = {"log": {}}
        found = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        env.scene["feet_ground_contact"] = SimpleNamespace(data=SimpleNamespace(
            found=found, current_air_time=(found == 0).float() * 0.2,
        ))
        env.command_manager = SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.4, 0.0, 0.0]]).repeat(3, 1)
        )
        torch.testing.assert_close(
            rewards.support_deficit_penalty(env, "feet_ground_contact"),
            torch.tensor([0.0, 1.0, 2.0]),
        )
        torch.testing.assert_close(
            rewards.supported_feet_air_time(env, "feet_ground_contact"),
            torch.tensor([2.0, 0.0, 0.0]),
        )
        self.assertIn("support_deficit", load_env_cfg("PointLeg-Rough-v0").rewards)
        self.assertNotIn("support_deficit", load_env_cfg("PointLeg-Crawl-v0").rewards)

    def test_observation_functions_produce_45_and_325_dimensions(self):
        cfg = load_env_cfg("PointLeg-Rough-v0")
        env = make_env()
        data = env.scene["robot"].data
        data.root_link_ang_vel_b = torch.zeros(3, 3)
        data.root_link_lin_vel_b = torch.zeros(3, 3)
        data.projected_gravity_b = torch.tensor([0.0, 0.0, -1.0]).expand(3, 3)
        data.joint_vel = torch.zeros(3, 12)
        env.action_manager = ActionManager({"leg_joint_pos": self.action_cfg()}, env)
        env.command_manager = SimpleNamespace(get_command=lambda name: torch.zeros(3, 3))
        env.scene["feet_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=torch.ones(3, 4)))
        env.scene["height_scanner"] = SimpleNamespace(
            num_frames=1, num_rays_per_frame=273, cfg=SimpleNamespace(max_distance=5.0),
            data=SimpleNamespace(
                distances=torch.full((3, 273), 0.42),
                frame_pos_w=torch.tensor([[[0.0, 0.0, 0.42]]]).expand(3, 1, 3),
                hit_pos_w=torch.zeros(3, 273, 3),
            ),
        )
        for name, expected_dim in (("actor", 45), ("critic", 325)):
            terms = cfg.observations[name].terms.values()
            observed = torch.cat([term.func(env, **term.params).reshape(3, -1) for term in terms], dim=1)
            self.assertEqual(tuple(observed.shape), (3, expected_dim))

    def test_real_reset_uses_default_height_once(self):
        for mode in ("Flat", "Rough", "Crawl"):
            with self.subTest(mode=mode):
                cfg = load_env_cfg(f"PointLeg-{mode}-v0")
                env = make_env()
                default_z = cfg.scene.entities["robot"].init_state.pos[2]
                env.scene["robot"].data.default_root_state[:, 2] = default_z
                event = cfg.events["reset_base"]
                event.func(env, torch.arange(3), **event.params)
                local_z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                lower, upper = event.params["pose_range"].get("z", (0.0, 0.0))
                self.assertTrue(torch.all(local_z >= default_z + lower - 1e-6))
                self.assertTrue(torch.all(local_z <= default_z + upper + 1e-6))
                self.assertTrue(torch.all(local_z < cfg.terminations["base_height_out_of_range"].params["maximum_height"]))

    def ground_env(self, clearances):
        env = make_env()
        ground_z = env.scene.env_origins[:, 2]
        env.scene["robot"].data.root_link_pos_w[:, 2] = ground_z + torch.tensor(clearances)
        hit_pos = torch.zeros(3, 1, 3)
        hit_pos[:, 0, 2] = ground_z
        normals = torch.zeros(3, 1, 3)
        normals[:, 0, 2] = 1.0
        env.scene["base_height_sensor"] = SimpleNamespace(data=SimpleNamespace(
            hit_pos_w=hit_pos, distances=torch.tensor(clearances)[:, None], normals_w=normals,
        ))
        return env

    def test_height_is_invariant_to_world_elevation(self):
        env = self.ground_env([0.42, 0.42, 0.42])
        done = terminations.root_height_out_of_range(env, 0.18, 0.65, sensor_name="base_height_sensor")
        self.assertEqual(done.tolist(), [False, False, False])
        flat_done = terminations.root_height_out_of_range(env, 0.18, 0.65)
        self.assertEqual(flat_done.tolist(), [False, False, False])

    def test_height_detects_low_high_and_missing_ground(self):
        env = self.ground_env([0.05, 0.9, 0.42])
        env.scene["base_height_sensor"].data.distances[2] = -1.0
        done = terminations.root_height_out_of_range(env, 0.18, 0.65, sensor_name="base_height_sensor")
        self.assertEqual(done.tolist(), [True, True, True])

    def test_height_rejects_nan_and_terrain_backfaces(self):
        env = self.ground_env([0.42, 0.42, 0.42])
        env.scene["robot"].data.root_link_pos_w[0, 2] = float("nan")
        env.scene["base_height_sensor"].data.normals_w[1, 0, 2] = -1.0
        done = terminations.root_height_out_of_range(env, 0.18, 0.65, sensor_name="base_height_sensor")
        self.assertEqual(done.tolist(), [True, True, False])

    def test_default_action_manager_checks_joint_order_and_count(self):
        expected = list(robot_cfg.LEG_JOINT_NAMES)
        wrong_orders = [expected[:-1], expected + ["extra_joint"], list(reversed(expected))]
        for names in wrong_orders:
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "12 leg joints"):
                ActionManager({"leg_joint_pos": self.action_cfg()}, make_env(FakeRobot(names)))
        with self.assertRaisesRegex(ValueError, "12 leg joints"):
            ActionManager({"leg_joint_pos": self.action_cfg()}, make_env(FakeRobot(target_names=expected[::-1])))
        manager = ActionManager({"leg_joint_pos": self.action_cfg()}, make_env())
        self.assertEqual(manager.total_action_dim, 12)
        with self.assertRaises(ValueError):
            manager.process_action(torch.zeros(3, 16))

    def test_soft_limits_are_applied_to_real_control_targets(self):
        env = make_env()
        limits = env.scene["robot"].data.soft_joint_pos_limits
        limits[1, :, 0] = -0.2
        limits[1, :, 1] = 0.3
        manager = ActionManager({"leg_joint_pos": self.action_cfg()}, env)
        for value in (100.0, -100.0):
            manager.process_action(torch.full((3, 12), value))
            manager.apply_action()
            target = env.scene["robot"].position_target
            self.assertTrue(torch.all(target >= limits[..., 0]))
            self.assertTrue(torch.all(target <= limits[..., 1]))

    def test_iir_response_for_position_and_velocity_actions(self):
        configs = [
            lowpass_actions.JointPositionLowPassActionCfg,
            lowpass_actions.JointPositionDelayedLowPassActionCfg,
            lowpass_actions.JointVelocityLowPassActionCfg,
            lowpass_actions.JointVelocityDelayedLowPassActionCfg,
        ]
        for cfg_type in configs:
            with self.subTest(action=cfg_type.__name__):
                kwargs = dict(entity_name="robot", actuator_names=robot_cfg.LEG_JOINT_PATTERNS,
                              scale=0.25, control_frequency=50.0, cut_off_frequency=5.0)
                if "Delayed" in cfg_type.__name__:
                    kwargs.update(min_delay=0, max_delay=0)
                term = cfg_type(**kwargs).build(make_env())
                expected = 0.0
                alpha = term._weights[0]
                for value in (1.0, 1.0, 0.0, -1.0):
                    expected = alpha * value + (1.0 - alpha) * expected
                    term.process_actions(torch.full((3, 12), value))
                    torch.testing.assert_close(term._processed_actions, torch.full((3, 12), 0.25 * expected))

    def test_per_environment_delay_and_partial_reset(self):
        cfg = self.action_cfg()
        cfg.max_delay = 2
        term = cfg.build(make_env())
        term._active_delays[:] = torch.tensor([0, 1, 2])
        expected = torch.zeros(3, 12)
        for value in (1.0, 2.0, 3.0):
            delayed = torch.tensor([value, max(value - 1.0, 0.0), max(value - 2.0, 0.0)])[:, None]
            expected = term._weights[0] * delayed + term._weights[1] * expected
            term.process_actions(torch.full((3, 12), value))
            torch.testing.assert_close(term._prev_filtered, expected)
        before_reset = term._prev_filtered.clone()
        term.reset(torch.tensor([1]))
        torch.testing.assert_close(term._prev_filtered[[0, 2]], before_reset[[0, 2]])
        self.assertEqual(torch.count_nonzero(term._prev_filtered[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(term._action_buffer[:, 1]).item(), 0)
        term.reset()
        self.assertEqual(torch.count_nonzero(term._prev_filtered).item(), 0)
        self.assertEqual(torch.count_nonzero(term._action_buffer).item(), 0)

    def test_joint_limit_termination_selects_only_requested_joints(self):
        env = make_env()
        env.scene["robot"].data.joint_pos[1, 0] = 0.99
        selected = SceneEntityCfg("robot", joint_ids=[0])
        self.assertEqual(terminations.joint_pos_out_of_limits(env, asset_cfg=selected).tolist(), [False, True, False])
        other = SceneEntityCfg("robot", joint_ids=[1])
        self.assertFalse(terminations.joint_pos_out_of_limits(env, asset_cfg=other).any())

    def test_model_path_resolves_to_project_mjcf(self):
        import os

        if "POINTLEG_XML" not in os.environ:
            self.assertEqual(robot_cfg.ROBOT_XML, PACKAGE_DIR.parents[1] / "mjcf" / "pointleg.xml")
        if not robot_cfg.ROBOT_XML.exists():
            with self.assertRaisesRegex(FileNotFoundError, "POINTLEG_XML"):
                robot_cfg.get_spec()


if __name__ == "__main__":
    unittest.main()
