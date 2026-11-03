"""Check high-step cycles, anti-tapping gates and isolation from normal walking."""

import inspect
import math
from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import list_tasks, load_rl_cfg
from legacy_high_step_config import load_env_cfg
from pointleg import robot_cfg
from pointleg.mdp.high_step import (
    HighStepCycleReward, HighStepMeanLegSymmetryPenalty,
    high_step_phase_penalty, high_step_pitch_penalty,
)
from test_flat_gait_rewards import gait_env
from test_audit_fixes import command_env


def cycle_env(step_dt=0.02, reverse_contacts=False):
    env, sensor = gait_env()
    env.step_dt = step_dt
    env.episode_length_buf = torch.full((3,), 100)
    env.commands[:] = torch.tensor([0.2, 0.0, 0.0])
    env.scene["robot"].data.root_link_quat_w = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    sensor.data.force = torch.zeros(3, 4, 3)
    sensor.data.force[:, :, 2] = 30.0
    sensor.data.current_air_time = torch.zeros(3, 4)
    sensor.data.current_contact_time[:] = 0.36
    sensor.data.last_contact_time[:] = 0.36
    sensor.data.last_air_time[:] = 0.24
    if reverse_contacts:
        sensor.primary_names = list(reversed(robot_cfg.FOOT_BODY_NAMES))
    sensor.compute_first_air = lambda dt: (
        (sensor.data.current_air_time > 0) & (sensor.data.current_air_time < dt + 1.e-6)
    )
    sensor.compute_first_contact = lambda dt: (
        (sensor.data.current_contact_time > 0) & (sensor.data.current_contact_time < dt + 1.e-6)
    )
    robot = env.scene["robot"]
    names = [f"{leg}_foot_contact_site" for leg in robot_cfg.LEG_NAMES]
    robot.find_sites = lambda name: ([names.index(name.replace("_foot_site", "_foot_contact_site"))], [name])
    robot.data.site_pos_w[:, :, 2] = env.scene.env_origins[:, 2, None] + 0.025
    robot.data.site_lin_vel_w = torch.zeros_like(robot.data.site_pos_w)
    robot.data.site_ang_vel_w = torch.zeros_like(robot.data.site_pos_w)
    sensor.data.pos = robot.data.site_pos_w.clone()
    sensor.data.pos[:, :, 2] -= .025
    sensor.data.normal = torch.zeros_like(sensor.data.pos)
    sensor.data.normal[:, :, 2] = -1.
    cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["high_step_cycle"]
    sensor.cfg = next(s for s in load_env_cfg("PointLeg-HighStep-v0").scene.sensors
                      if s.name == "feet_ground_contact")
    term = HighStepCycleReward(cfg, env)
    term(env, **cfg.params)
    return env, sensor, cfg, term


def run_cycle(env, sensor, cfg, term, *, height=0.15, stride=0.12,
              body_progress=0.048, air_time=0.24, stance_time=0.36,
              no_support=False, body_bounce=False):
    feet = [0, 3] if not no_support else [0, 1, 2, 3]
    contacts = [term.contact_ids[i] for i in feet]
    data = sensor.data
    data.found[:, contacts] = 0.0
    data.force[:, contacts] = 0.0
    data.current_contact_time[:, contacts] = 0.0
    data.current_air_time[:, contacts] = env.step_dt
    data.last_contact_time[:, contacts] = stance_time
    term(env, **cfg.params)
    data.current_air_time[:, contacts] = max(air_time, 2 * env.step_dt)
    env.scene["robot"].data.site_pos_w[:, feet, 2] += height
    if body_bounce:
        env.scene["robot"].data.root_link_pos_w[:, 2] += height
    term(env, **cfg.params)
    env.scene["robot"].data.site_pos_w[:, feet, 2] -= height
    if body_bounce:
        env.scene["robot"].data.root_link_pos_w[:, 2] -= height
    env.scene["robot"].data.site_pos_w[:, feet, 0] += stride
    env.scene["robot"].data.root_link_pos_w[:, 0] += body_progress
    data.found[:, contacts] = 1.0
    data.force[:, contacts, 2] = 30.0
    data.current_air_time[:, contacts] = 0.0
    data.last_air_time[:, contacts] = air_time
    data.current_contact_time[:, contacts] = env.step_dt
    immediate = term(env, **cfg.params)
    data.current_contact_time[:, contacts] = cfg.params["min_stance_time"]
    settled = term(env, **cfg.params)
    return immediate, settled


class HighStepTests(unittest.TestCase):
    def test_commands_really_sample_slow_forward_motion_without_parent_speed_floor(self):
        torch.manual_seed(27)
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        command = cfg.commands["twist"].build(command_env(cfg, count=4096))
        command._resample_command(torch.arange(4096))
        command._update_command()
        active = ~command.is_standing_env
        values = command.command[active, 0]
        self.assertGreater(values.numel(), 3000)
        self.assertGreaterEqual(values.min().item(), 0.15 - 1.e-6)
        self.assertLessEqual(values.max().item(), 0.25 + 1.e-6)
        self.assertGreater(values.std().item(), 0.02)
        self.assertEqual(command.command[:, 1:].abs().sum().item(), 0)
        self.assertEqual(command.command[~active].abs().sum().item(), 0)

    def test_pitch_tolerance_and_forward_backward_tilt_are_symmetric(self):
        env, _, _, _ = cycle_env()
        angles = torch.deg2rad(torch.tensor([2.0, 10.0, -10.0]))
        env.scene["robot"].data.projected_gravity_b[:] = torch.stack(
            (angles.sin(), torch.zeros(3), -angles.cos()), dim=1,
        )
        cost = high_step_pitch_penalty(env)
        torch.testing.assert_close(cost, torch.tensor([0.0, (7.0 / 8.0) ** 2, (7.0 / 8.0) ** 2]))

    def test_mean_symmetry_penalizes_slow_left_right_and_front_rear_bias(self):
        env, _, _, _ = cycle_env()
        cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["high_step_mean_leg_symmetry"]
        term = HighStepMeanLegSymmetryPenalty(cfg, env)
        q = env.scene["robot"].data.joint_pos
        q[0, 1] = math.radians(20)
        q[0, 5] = math.radians(-20)
        q[1, [1, 4, 7, 10]] = math.radians(20)
        q[1, [2, 5, 8, 11]] = math.radians(-30)
        q[2, [1, 4]] = math.radians(20)
        for _ in range(200):
            cost = term(env, **cfg.params)
        self.assertGreater(cost[0].item(), 0.5)
        self.assertEqual(cost[1].item(), 0.)
        self.assertGreater(cost[2].item(), 0.)
        bias = env.extras["log"]["Metrics/high_step_front_rear_hip_bias_deg"]
        self.assertGreater(bias[2].item(), 19)

    def test_mean_symmetry_allows_alternation_and_resets_individual_environments(self):
        env, _, _, _ = cycle_env()
        cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["high_step_mean_leg_symmetry"]
        term = HighStepMeanLegSymmetryPenalty(cfg, env)
        q = env.scene["robot"].data.joint_pos
        for step in range(250):
            angle = math.radians(30) * math.sin(2 * math.pi * 2.5 * step * env.step_dt)
            q[:, [1, 10]] = angle
            q[:, [4, 7]] = -angle
            q[:, [2, 11]] = -angle
            q[:, [5, 8]] = angle
            cost = term(env, **cfg.params)
            if step > 100:
                self.assertLess(cost.max().item(), 0.08)
        term.reset(torch.tensor([0]))
        self.assertEqual(term.age[0].item(), 0)
        self.assertGreater(term.age[1].item(), 1)
        env.commands[:] = 0
        self.assertEqual(term(env, **cfg.params).tolist(), [0., 0., 0.])
        self.assertEqual(term.mean.abs().sum().item(), 0)

    def test_posture_terms_only_apply_to_high_step_and_keep_flat_rewards_unchanged(self):
        high_step = load_env_cfg("PointLeg-HighStep-v0")
        flat = load_env_cfg("PointLeg-Flat-v0")
        self.assertEqual(high_step.rewards["high_step_pitch"].weight, -1.5)
        self.assertEqual(high_step.rewards["high_step_mean_leg_symmetry"].weight, -0.2)
        self.assertNotIn("high_step_pitch", flat.rewards)
        self.assertNotIn("high_step_mean_leg_symmetry", flat.rewards)

    def test_registered_task_keeps_walking_interface_and_flat_unchanged(self):
        self.assertIn("PointLeg-HighStep-v0", list_tasks())
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        flat = load_env_cfg("PointLeg-Flat-v0")
        self.assertEqual(load_rl_cfg("PointLeg-HighStep-v0").experiment_name, "pointleg_high_step")
        self.assertEqual(cfg.actions, flat.actions)
        self.assertEqual(list(cfg.observations["actor"].terms), list(flat.observations["actor"].terms))
        self.assertEqual(list(cfg.observations["critic"].terms), list(flat.observations["critic"].terms))
        self.assertEqual(cfg.scene.entities, flat.scene.entities)
        self.assertEqual(flat.rewards["foot_swing_height"].params["target_height"], 0.10)
        self.assertEqual(flat.rewards["short_stance"].params["min_stance_time"], 0.16)
        self.assertIn("feet_air_time", flat.rewards)
        self.assertNotIn("feet_air_time", cfg.rewards)
        self.assertNotIn("foot_swing_height", cfg.rewards)
        self.assertEqual(cfg.rewards["high_step_cycle"].params["target_height"], 0.15)
        self.assertEqual(cfg.rewards["high_step_cycle"].params["max_step_length"], 0.18)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (0.15, 0.25))
        self.assertLess(cfg.commands["twist"].linear_deadband, 0.15)
        self.assertEqual(cfg.commands["twist"].rel_backward_envs, 0)
        self.assertEqual(cfg.commands["twist"].rel_forward_envs, 0)
        for term in cfg.rewards.values():
            if not inspect.isclass(term.func):
                inspect.signature(term.func).bind(None, **term.params)

    def test_valid_cycle_waits_for_stance_then_pays_once_and_is_dt_invariant(self):
        for dt in (0.01, 0.02):
            env, sensor, cfg, term = cycle_env(dt)
            immediate, settled = run_cycle(env, sensor, cfg, term)
            self.assertEqual(immediate.tolist(), [0., 0., 0.])
            torch.testing.assert_close(settled * dt, torch.full((3,), 2.0))
            self.assertEqual(term(env, **cfg.params).tolist(), [0., 0., 0.])
            self.assertAlmostEqual(env.extras["log"]["Metrics/high_step_cycle_s"].mean().item(), 0.6)

    def test_low_and_excessively_high_lifts_score_less_than_target(self):
        scores = []
        for height in (0.02, 0.12, 0.15, 0.30):
            env, sensor, cfg, term = cycle_env()
            scores.append(run_cycle(env, sensor, cfg, term, height=height)[1].mean().item())
        self.assertGreater(scores[0], 0.)
        self.assertGreater(scores[2], scores[0])
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[2], 50 * scores[3])

    def test_overlong_and_too_short_cycles_score_less_than_target(self):
        scores = []
        for stance, air in ((0.18, 0.14), (0.36, 0.24), (0.60, 0.24)):
            env, sensor, cfg, term = cycle_env()
            settled = run_cycle(env, sensor, cfg, term, stance_time=stance, air_time=air)[1]
            scores.append(settled.mean().item() * env.step_dt / (stance + air))
        self.assertGreater(scores[0], 0.)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_taps_overlong_swing_long_stride_stationary_and_backward_cycles_do_not_pay(self):
        for params in ({"air_time": 0.02}, {"air_time": 0.40}, {"stride": 0.20},
                       {"stride": 0.0}, {"body_progress": 0.0}, {"stride": -0.08}):
            with self.subTest(params=params):
                env, sensor, cfg, term = cycle_env()
                self.assertEqual(run_cycle(env, sensor, cfg, term, **params)[1].tolist(), [0., 0., 0.])

    def test_all_feet_airborne_and_body_bounce_do_not_pay(self):
        for params in ({"no_support": True}, {"body_bounce": True}):
            env, sensor, cfg, term = cycle_env()
            self.assertEqual(run_cycle(env, sensor, cfg, term, **params)[1].tolist(), [0., 0., 0.])

    def test_contact_order_does_not_change_reward(self):
        env, sensor, cfg, term = cycle_env(reverse_contacts=True)
        torch.testing.assert_close(run_cycle(env, sensor, cfg, term)[1] * env.step_dt, torch.full((3,), 2.0))

    def test_static_command_and_partial_reset_clear_pending_cycles(self):
        env, sensor, cfg, term = cycle_env()
        run_cycle(env, sensor, cfg, term)
        term.pending[:] = True
        term.pending_score[:] = 1.0
        term.reset(torch.tensor([0]))
        self.assertFalse(term.pending[0].any())
        self.assertTrue(term.pending[1:].all())
        env.commands[:] = 0.0
        self.assertEqual(term(env, **cfg.params).tolist(), [0., 0., 0.])
        self.assertFalse(term.pending.any())
        term.reset()
        self.assertEqual(term.lift_position.abs().sum().item(), 0)

    def test_phase_penalty_detects_one_stuck_foot_but_not_static_or_startup(self):
        env, sensor, _, _ = cycle_env()
        params = {"sensor_name": "feet_ground_contact", "command_name": "twist"}
        self.assertEqual(high_step_phase_penalty(env, **params).tolist(), [0., 0., 0.])
        sensor.data.current_contact_time[0, 0] = 0.8
        sensor.data.current_air_time[1, 3] = 0.6
        result = high_step_phase_penalty(env, **params)
        self.assertGreater(result[0].item(), 0)
        self.assertGreater(result[1].item(), 0)
        self.assertEqual(result[2].item(), 0)
        env.commands[:] = 0.0
        self.assertEqual(high_step_phase_penalty(env, **params).tolist(), [0., 0., 0.])
        env.commands[:, 0] = 0.2
        env.episode_length_buf[:] = 0
        self.assertEqual(high_step_phase_penalty(env, **params).tolist(), [0., 0., 0.])

    def test_invalid_parameters_are_rejected_and_invalid_ground_cancels_reward(self):
        env, sensor, cfg, term = cycle_env()
        with self.assertRaises(ValueError):
            term(env, **{**cfg.params, "target_cycle": float("nan")})
        env.scene["base_height_sensor"].data.distances[:] = -1
        self.assertEqual(run_cycle(env, sensor, cfg, term)[1].tolist(), [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
