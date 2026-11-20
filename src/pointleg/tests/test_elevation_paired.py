"""Paired sampling, basic feedback and checkpoint checks without physics stepping."""

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from pointleg.evaluate_elevation import evaluation_cases
from pointleg.mdp.elevation import (
    ElevationSettings, ElevationTerrainCurriculum, collect_elevation_contacts,
    elevation_cost, elevation_event, reset_elevation, select_elevation_terrain,
)
from pointleg.mdp.elevation_basic import (
    ElevationBasicProlongedSwingPenalty, ElevationBasicStallPenalty,
    elevation_basic_support_deficit, elevation_basic_velocity,
)
from pointleg.rsl_rl.runners.elevation_runner import ElevationOnPolicyRunner
from pointleg.transfer import _TransferEnv, prepare_transfer
from test_elevation import bounds, contacts, fake_env


PAIRS = {"PointLeg-Stairs-v0": (0, 1), "PointLeg-Slope-v0": (2, 3)}


def paired_env(task, count=12):
    env = fake_env(count=count, stair_height_override=None if task == "PointLeg-Stairs-v0" else .12,
                   stair_height_range=(.06, .12) if task == "PointLeg-Stairs-v0" else None)
    env.cfg = load_env_cfg(task)
    env.scene.terrain.cfg = env.cfg.scene.terrain
    env.extras = {"log": {}}
    settings = env.cfg.events["reset_elevation"].params["settings"]
    select_elevation_terrain(env, None, settings)
    reset_elevation(env, None, settings, randomize=False)
    return env


def runner_for(task):
    return ElevationOnPolicyRunner(_TransferEnv(task), asdict(load_rl_cfg(task)), device="cpu")


class PairedTerrainTests(unittest.TestCase):
    def test_curriculum_initializes_before_the_first_reset_event(self):
        env = fake_env(count=2, stair_height_override=.12)
        env.cfg = load_env_cfg("PointLeg-Slope-v0")
        env.scene.terrain.cfg = env.cfg.scene.terrain
        del env._elevation_settings
        course = ElevationTerrainCurriculum(env.cfg.curriculum["elevation_levels"], env)

        self.assertFalse(hasattr(env, "_elevation_settings"))
        logged = course(env, None)

        self.assertEqual(set(logged), {"slope_up_cap", "slope_down_cap"})

    def test_both_directions_and_only_those_directions_survive_partial_reset(self):
        for task, kinds in PAIRS.items():
            env = paired_env(task)
            terrain = env.scene.terrain
            self.assertEqual(set(terrain.terrain_types.tolist()), set(kinds))
            ids = torch.tensor([0, 3, 7])
            untouched = torch.tensor([1, 2, 4, 5, 6, 8, 9, 10, 11])
            terrain.terrain_types[ids] = 5
            before = terrain.terrain_types[untouched].clone()
            select_elevation_terrain(env, ids, env._elevation_settings)
            self.assertTrue(set(terrain.terrain_types[ids].tolist()) <= set(kinds))
            torch.testing.assert_close(terrain.terrain_types[untouched], before)
            reset_elevation(env, ids, env._elevation_settings, randomize=False)
            down = terrain.terrain_types.remainder(2) == 1
            root = env.scene["robot"].data
            expected = torch.where(down, env._elevation["top_height"], 0.)
            torch.testing.assert_close(root.root_link_pos_w[:, 2] - root.default_root_state[:, 2], expected)

    def test_fixed_evaluation_allows_both_directions_and_rejects_other_families(self):
        for task, kinds in PAIRS.items():
            cases = evaluation_cases(task)
            self.assertEqual(len(cases), 10)
            self.assertEqual(len({case["terrain_type"] for case in cases}), 2)
            env = paired_env(task)
            env._elevation_fixed_cases = True
            select_elevation_terrain(env, None, env._elevation_settings)
            env.scene.terrain.terrain_types[0] = 2 if kinds == (0, 1) else 0
            with self.assertRaisesRegex(ValueError, "specialized terrain scope"):
                select_elevation_terrain(env, None, env._elevation_settings)
        with self.assertRaises(ValueError):
            ElevationSettings(jump=True, terrain_kind="slope")

    def test_uphill_and_downhill_keep_independent_curriculum_caps(self):
        env = paired_env("PointLeg-Slope-v0", count=2)
        env.cfg.curriculum["elevation_levels"].params["window_size"] = 2
        course = ElevationTerrainCurriculum(env.cfg.curriculum["elevation_levels"], env)
        env.episode_length_buf[:] = 1
        env._elevation["complete"][:] = torch.tensor([True, False])
        for _ in range(2):
            values = course(env, None)
        self.assertEqual(course.level_caps.tolist(), [0, 0, 1, 0, 0, 0])
        self.assertEqual(set(values), {"slope_up_cap", "slope_down_cap"})
        select_elevation_terrain(env, None, env._elevation_settings)
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [1, 0])
        stairs = paired_env("PointLeg-Stairs-v0", count=2)
        stair_course = ElevationTerrainCurriculum(stairs.cfg.curriculum["elevation_levels"], stairs)
        logged = stair_course(stairs, None)
        self.assertEqual(set(logged), {"stairs_up_cap", "stairs_down_cap"})
        for value in logged.values():
            self.assertEqual(value.item(), 0.)


class BasicElevationRewardTests(unittest.TestCase):
    def test_whitelist_and_interfaces_exclude_old_compound_terms(self):
        expected = {"velocity", "lateral", "yaw_rate", "edge", "slip", "attitude", "angular",
                    "joint", "torque", "action", "contact_cost", "support_deficit", "task_completed", "task_failed"}
        legacy = load_env_cfg("PointLeg-Elevation-v0")
        self.assertIn("step_quality", legacy.rewards)
        speed_ranges = {
            "PointLeg-Stairs-v0": (.10, .20),
            "PointLeg-Slope-v0": (.15, .30),
        }
        for task in PAIRS:
            self.assertIn(task, list_tasks())
            for play in (False, True):
                cfg = load_env_cfg(task, play=play)
                expected_rewards = expected | ({"forward_stall", "prolonged_stalled_swing", "hip_abduction_zero",
                                                 "step_timing", "mean_leg_posture"}
                                               if task in ("PointLeg-Slope-v0", "PointLeg-Stairs-v0") else set())
                if task == "PointLeg-Stairs-v0":
                    expected_rewards.update({"stairs_progress", "tread_foot_placement", "four_foot_standstill", "swing_lift_deficit", "thigh_body_alignment", "left_front_rear_tracking", "swing_knee_flexion", "stuck_foot_recovery", "swing_joint_speed", "down_shin_contact", "front_support_loss", "front_premature_takeoff", "front_overreach", "rear_thigh_above_body", "fr_hip_forward_limit", "fr_knee_min_flexion"})
                self.assertEqual(set(cfg.rewards), expected_rewards)
                self.assertTrue(all(term.weight != 0. for term in cfg.rewards.values()))
                self.assertEqual(cfg.observations, load_env_cfg("PointLeg-Elevation-v0", play=play).observations)
                self.assertEqual(cfg.actions, load_env_cfg("PointLeg-Elevation-v0", play=play).actions)
                specialized = task in ("PointLeg-Slope-v0", "PointLeg-Stairs-v0")
                self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, speed_ranges[task])
                self.assertEqual(cfg.episode_length_s, 120. if specialized else 60.)
                settings = cfg.events["reset_elevation"].params["settings"]
                self.assertEqual(settings.speed_range, cfg.commands["twist"].ranges.lin_vel_x)
                self.assertIs(settings, cfg.events["select_elevation_terrain"].params["settings"])
                self.assertNotIn("elevation_true_lift_height", cfg.metrics)
            self.assertEqual(load_rl_cfg(task).algorithm, load_rl_cfg("PointLeg-Elevation-v0").algorithm)

    def test_slope_stall_and_prolonged_swing_only_penalize_stalled_motion(self):
        env = paired_env("PointLeg-Slope-v0", count=2)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(2, 1)})()
        contacts(env, [2., 2., 1.8, 1.8])
        env.scene["elevation_contacts"].data.found[0, 0] = 0.
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0., .12])
        state = env._elevation
        stall = ElevationBasicStallPenalty(SimpleNamespace(), env)
        swing = ElevationBasicProlongedSwingPenalty(SimpleNamespace(), env)

        with patch("pointleg.mdp.elevation_basic.update_elevation_state", return_value=state):
            for step in range(40):
                env.common_step_counter = step
                stall_cost = stall(env, grace_period=.20, rise_time=.20)
                swing_cost = swing(env, max_air_time=.20, rise_time=.20)

        torch.testing.assert_close(stall_cost, torch.tensor([1., 0.]))
        torch.testing.assert_close(swing_cost, torch.tensor([1., 0.]))
        env.common_step_counter += 1
        env.scene["robot"].data.root_link_lin_vel_b[0, 0] = .12
        with patch("pointleg.mdp.elevation_basic.update_elevation_state", return_value=state):
            torch.testing.assert_close(stall(env, grace_period=.20, rise_time=.20), torch.zeros(2))
            torch.testing.assert_close(swing(env, max_air_time=.20, rise_time=.20), torch.zeros(2))

    def test_slope_hip_abduction_target_is_zero_degrees(self):
        cfg = load_env_cfg("PointLeg-Slope-v0").rewards["hip_abduction_zero"]
        self.assertEqual(cfg.weight, -7.2)
        self.assertEqual(cfg.params["tolerance"], 0.)
        self.assertEqual(cfg.params["deviation_scale"], .25)
        self.assertEqual(cfg.params["asset_cfg"].joint_names, (".*_hip_abduction_joint",))

    def test_forward_tracking_is_independent_and_suppresses_invalid_or_failed_positives(self):
        env = paired_env("PointLeg-Slope-v0", count=6)
        env.command_manager = type("Commands", (), {"get_command": lambda self, _: torch.tensor([[.2, 0., 0.]]).repeat(6, 1)})()
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([.2, 0., -.2, .2, .2, float("nan")])
        state = env._elevation
        state["failed"][3] = True
        state["complete"][4] = True
        with patch("pointleg.mdp.elevation_basic.update_elevation_state", return_value=state):
            score = elevation_basic_velocity(env)
        torch.testing.assert_close(score, torch.tensor([1., 0., 0., 0., 0., 0.]))
        self.assertFalse(hasattr(env, "_elevation_gait_term"))

    def test_lateral_cost_is_symmetric_and_pitch_follows_uphill_and_downhill(self):
        env = paired_env("PointLeg-Slope-v0", count=2)
        robot = env.scene["robot"].data
        robot.root_link_lin_vel_b[:, 1] = torch.tensor([.08, -.08])
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            cost = elevation_cost(env, "lateral", tolerance=.01, scale=.1)
        torch.testing.assert_close(cost, torch.tensor([.49, .49]))
        env._elevation["angle"][:] = math.radians(20.)
        robot.root_link_pos_w[:, 0] = env._elevation["corner"][:, 0] + 4.
        pitch = torch.tensor([-math.radians(20.), math.radians(20.)])
        robot.root_link_quat_w[:] = 0.
        robot.root_link_quat_w[:, 0] = (pitch / 2.).cos()
        robot.root_link_quat_w[:, 2] = (pitch / 2.).sin()
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            value = elevation_cost(env, "attitude")
        torch.testing.assert_close(value, torch.zeros(2), atol=1.e-6, rtol=0.)

    def test_falling_outside_a_lane_keeps_exact_failure_penalty(self):
        for task in PAIRS:
            env = paired_env(task, count=2)
            contacts(env, [2., 2., 1.8, 1.8], outside=True)
            collect_elevation_contacts(env)
            with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
                first = elevation_event(env, "failed")
                second = elevation_event(env, "failed")
            weight = env.cfg.rewards["task_failed"].weight
            torch.testing.assert_close(first * weight * env.step_dt, torch.full((2,), -50.))
            torch.testing.assert_close(first, second)
            self.assertTrue(env._elevation["failed"].all())
            self.assertTrue(env._elevation["outside"].all())

    def test_support_penalty_requires_actual_load_and_rejects_bad_contact_data(self):
        env = paired_env("PointLeg-Slope-v0", count=2)
        contacts(env, [2., 2., 1.8, 1.8])
        sensor = env.scene["elevation_contacts"].data
        sensor.force[0, 8:] = 0.
        sensor.force[1, 4:] = 0.
        sensor.force[1, 0] = float("nan")
        torch.testing.assert_close(elevation_basic_support_deficit(env), torch.tensor([0., 2.]))


class PairedCheckpointTests(unittest.TestCase):
    def test_transfer_and_strict_resume_reset_and_preserve_each_task_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            original = runner_for("PointLeg-Elevation-v0")
            original.current_learning_iteration = 14100
            original.save(str(source))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            outputs = {}
            for task in PAIRS:
                output = prepare_transfer(task, source, "basic_initial", output_root=root)
                outputs[task] = output
                data = torch.load(output, map_location="cpu", weights_only=False)
                self.assertEqual(data["iter"], 0)
                self.assertEqual(data["optimizer_state_dict"]["state"], {})
                self.assertEqual(data["infos"]["elevation_curriculum"]["level_caps"], [0] * 6)
                expected_revision = 11 if task == "PointLeg-Slope-v0" else 34
                self.assertEqual(data["infos"]["elevation_walking"]["revision"], expected_revision)
                self.assertEqual(data["infos"]["policy_contract"], {"actor": 45, "critic": 325, "actions": 12})
                resumed = runner_for(task)
                resumed.load(str(output))
                for incoming, expected in ((resumed.alg.actor, original.alg.actor), (resumed.alg.critic, original.alg.critic)):
                    for name, tensor in incoming.state_dict().items():
                        torch.testing.assert_close(tensor, expected.state_dict()[name], rtol=0., atol=0.)
                resumed.current_learning_iteration = 12
                resumed.env.common_step_counter = 100
                saved = root / (task + ".pt")
                resumed.save(str(saved))
                restored = runner_for(task)
                restored.load(str(saved))
                self.assertEqual(restored.current_learning_iteration, 12)
                self.assertEqual(restored.env.common_step_counter, 100)
            with self.assertRaisesRegex(ValueError, "specialization changed"):
                runner_for("PointLeg-Stairs-v0").load(str(outputs["PointLeg-Slope-v0"]))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
