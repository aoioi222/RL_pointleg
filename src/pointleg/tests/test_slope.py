"""Check slope timing, posture and faster command resets without physics stepping."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from pointleg.mdp.elevation import (
    collect_elevation_contacts, reset_elevation, slope_exit_supported_feet, update_elevation_state,
)
from pointleg.mdp.slope import slope_task_metric
from pointleg.robot_cfg import LEG_NAMES
from test_elevation import bounds, contacts
from test_elevation_paired import paired_env, runner_for


def timing_env(count=1, dt=.02):
    env = paired_env("PointLeg-Slope-v0", count=count)
    env.step_dt = dt
    command = torch.tensor([[.375, 0., 0.]]).repeat(count, 1)
    env.command_manager = type("Commands", (), {"get_command": lambda self, _: command})()
    contacts(env, [2., 2., 1.8, 1.8])
    cfg = env.cfg.rewards["step_timing"]
    return env, cfg.func(cfg, env), cfg


def tick(env, term, cfg, loaded):
    data = env.scene["elevation_contacts"].data
    data.found[:, [0, 4, 8, 12]] = torch.as_tensor(loaded).float()
    env.common_step_counter += 1
    env.episode_length_buf += 1
    with patch("pointleg.mdp.slope.update_elevation_state", return_value=env._elevation):
        return term(env, **cfg.params).clone()


def gait_cost(period, dt=.02):
    env, term, cfg = timing_env(dt=dt)
    cfg = deepcopy(cfg)
    cfg.params["startup_grace"] = .01
    total = 0.
    for step in range(round(4.2 / dt)):
        phase = (step % round(period / dt)) * dt
        loaded = [phase < period * .60, True, True, True]
        cost = tick(env, term, cfg, loaded)
        total += float(cost[0]) * dt
    return total


def destination_contacts(env):
    state = env._elevation
    contacts(env, [5.4, 5.4, 5.02, 5.02])
    sensor = env.scene["elevation_contacts"].data
    height = torch.where(env.scene.terrain.terrain_types == 2, state["top_height"], 0.)
    sensor.pos[:, :, 2] = height[:, None]
    sensor.pos += state["corner"][:, None, :]
    sensor.force[:] = 0.
    sensor.force[:, [0, 4, 8, 12], 2] = 10.
    env.scene["robot"].data.root_link_pos_w[:, 0] = state["corner"][:, 0] + 5.15
    env.scene["robot"].data.root_link_pos_w[:, 1] = state["corner"][:, 1] + 4.
    env.scene["robot"].data.root_link_pos_w[:, 2] = state["corner"][:, 2] + height + .42
    env.episode_length_buf[:] = 1
    collect_elevation_contacts(env)


class SlopeTests(unittest.TestCase):
    def test_both_directions_complete_on_four_loaded_exit_feet_and_pay_before_reset(self):
        env = paired_env("PointLeg-Slope-v0", count=2)
        destination_contacts(env)
        self.assertEqual(env.scene.terrain.terrain_types.tolist(), [2, 3])
        manager = TerminationManager(env.cfg.terminations, env)
        reward = RewardManager({name: env.cfg.rewards[name] for name in ("task_completed", "task_failed")}, env)
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            done = manager.compute()
            value = reward.compute(env.step_dt)
            state = update_elevation_state(env)
        self.assertTrue(done.all())
        self.assertTrue(state["complete"].all())
        self.assertFalse(state["failed"].any())
        torch.testing.assert_close(value, torch.full((2,), 20.))
        self.assertTrue(env.cfg.auto_reset)
        self.assertLess(float(state["stable"].max()), .5)
        reset_elevation(env, torch.tensor([0]), env._elevation_settings, randomize=False)
        self.assertFalse(bool(state["complete"][0]))
        self.assertTrue(bool(state["complete"][1]))

    def test_three_feet_ramp_contact_wrong_height_weak_load_side_contact_and_nan_do_not_complete(self):
        env = paired_env("PointLeg-Slope-v0", count=6)
        destination_contacts(env)
        data = env.scene["elevation_contacts"].data
        data.found[0, 12] = 0.
        data.pos[1, 12, 0] = env._elevation["corner"][1, 0] + 4.99
        data.pos[2, 12, 2] += .10
        data.force[3, 12, 2] = 7.9
        data.normal[4, 12] = torch.tensor([-1., 0., 0.])
        data.force[5, 12, 2] = float("nan")
        self.assertFalse(slope_exit_supported_feet(env).all(dim=1).any())
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            self.assertFalse(update_elevation_state(env)["complete"].any())

    def test_downhill_initial_upper_platform_is_not_the_destination(self):
        env = paired_env("PointLeg-Slope-v0", count=2)
        contacts(env, [2.3, 2.3, 1.8, 1.8])
        data = env.scene["elevation_contacts"].data
        data.pos[1, :, 2] = env._elevation["top_height"][1]
        data.force[:] = 0.
        data.force[:, [0, 4, 8, 12], 2] = 10.
        collect_elevation_contacts(env)
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            self.assertFalse(update_elevation_state(env)["complete"].any())

    def test_failure_overrides_platform_arrival_and_destination_uses_local_coordinates(self):
        env = paired_env("PointLeg-Slope-v0", count=2)
        env._elevation["corner"][:] = torch.tensor([12., 8., 1.])
        destination_contacts(env)
        self.assertTrue(slope_exit_supported_feet(env).all())
        env._elevation["outside"][0] = True
        with patch("pointleg.mdp.elevation.geometry_bounds", bounds):
            state = update_elevation_state(env)
        self.assertEqual(state["complete"].tolist(), [False, True])
        self.assertEqual(state["failed"].tolist(), [True, False])

    def test_assignment_metrics_report_only_uphill_and_downhill(self):
        env = paired_env("PointLeg-Slope-v0", count=256)
        self.assertEqual(float(slope_task_metric(env, "up_fraction").mean()), .5)
        self.assertEqual(float(slope_task_metric(env, "down_fraction").mean()), .5)
        self.assertEqual(set(env.scene.terrain.terrain_types.tolist()), {2, 3})

    def test_longer_period_costs_less_than_quick_shuffling(self):
        self.assertGreater(gait_cost(.30), gait_cost(.70) + .05)

    def test_timing_integrates_consistently_at_two_control_rates(self):
        self.assertAlmostEqual(gait_cost(.30, .02), gait_cost(.30, .01), delta=.03)

    def test_persistent_support_and_air_do_not_escape_the_timing_cost(self):
        env, term, cfg = timing_env()
        for _ in range(100):
            cost = tick(env, term, cfg, [False, True, True, True])
        self.assertGreater(float(cost[0]), .5)
        self.assertGreater(float(term.air_time[0, 0]), 1.)

    def test_single_tick_contact_does_not_confirm_a_landing(self):
        env, term, cfg = timing_env()
        for _ in range(40):
            tick(env, term, cfg, [False, True, True, True])
        tick(env, term, cfg, [True, True, True, True])
        self.assertFalse(bool(term.loaded[0, 0]))
        self.assertFalse(bool(term.anchored[0, 0]))

    def test_same_tick_cache_partial_reset_failure_and_nan_are_isolated(self):
        env, term, cfg = timing_env(count=2)
        for _ in range(50):
            tick(env, term, cfg, [False, True, True, True])
        before = term.air_time.clone()
        with patch("pointleg.mdp.slope.update_elevation_state", return_value=env._elevation):
            term(env, **cfg.params)
        torch.testing.assert_close(term.air_time, before)
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.air_time[1], before[1])
        self.assertEqual(float(term.air_time[0].sum()), 0.)
        env._elevation["failed"][0] = True
        env.scene["elevation_contacts"].data.force[1, 0, 0] = float("nan")
        result = tick(env, term, cfg, [False, True, True, True])
        torch.testing.assert_close(result, torch.zeros(2))
        self.assertTrue(torch.isfinite(term.cached).all())

    def test_hip_penalty_detects_one_bad_leg_symmetrically_and_ignores_pitch(self):
        env, _, _ = timing_env(count=3)
        cfg = env.cfg.rewards["hip_abduction_zero"]
        cfg.params["asset_cfg"] = SceneEntityCfg("robot", joint_ids=[0, 3, 6, 9])
        q = env.scene["robot"].data.joint_pos
        q[0, 0], q[1, 0] = .2, -.2
        q[2, [1, 2]] = 1.
        cost = cfg.func(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([.64, .64, 0.]))
        for leg in LEG_NAMES:
            self.assertIn(f"Metrics/slope_hip_abduction_{leg}_deg", env.extras["log"])

    def test_slow_single_leg_posture_is_penalized_and_unequal_footholds_are_relaxed(self):
        env, _, _ = timing_env(count=3)
        contacts(env, [2., 2., 1.8, 1.8])
        cfg = env.cfg.rewards["mean_leg_posture"]
        term = cfg.func(cfg, env)
        q = env.scene["robot"].data.joint_pos
        q[:2, [1, 2]] = .8
        q[2, [1, 4, 7, 10]] = .8
        q[2, [2, 5, 8, 11]] = .8
        env.scene["robot"].data.site_pos_w[1, 0, 2] += .12
        for _ in range(120):
            env.common_step_counter += 1
            cost = term(env, **cfg.params)
        self.assertGreater(float(cost[0]), .1)
        self.assertLess(float(cost[1]), float(cost[0]))
        self.assertEqual(float(cost[2]), 0.)

    def test_reset_samples_requested_commands_and_play_uses_extended_deadline(self):
        env = paired_env("PointLeg-Slope-v0", count=32)
        settings = env.cfg.events["reset_elevation"].params["settings"]
        reset_elevation(env, None, settings, randomize=True)
        self.assertTrue((env._elevation["speed"] >= .15).all())
        self.assertTrue((env._elevation["speed"] <= .30).all())
        self.assertGreater(float(env._elevation["speed"].max()), .20)
        self.assertEqual(env.cfg.episode_length_s, 120.)

    def test_previous_slope_checkpoint_resumes_with_new_rewards_and_deadline(self):
        source = runner_for("PointLeg-Slope-v0")
        source.current_learning_iteration = 123
        source.env._elevation_curriculum.level_caps[2] = 2
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "previous_slope.pt"
            source.save(str(checkpoint))
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved["infos"]["elevation_settings"]["speed_range"] = (.15, .25)
            contract = saved["infos"]["elevation_walking"]
            contract.update(revision=9, gait_style="basic_slope_anti_stall", speed_range=(.15, .25))
            contract.pop("episode_length_s")
            for key in ("completion", "completion_force_n", "completion_hold_s"):
                contract.pop(key)
            contract["rewards"].pop("step_timing")
            contract["rewards"].pop("mean_leg_posture")
            contract["rewards"]["hip_abduction_zero"]["weight"] = -3.6
            torch.save(saved, checkpoint)
            restored = runner_for("PointLeg-Slope-v0")
            with self.assertWarnsRegex(UserWarning, "reward settings changed"):
                restored.load(str(checkpoint))
            self.assertEqual(restored.current_learning_iteration, 123)
            self.assertEqual(int(restored.env._elevation_curriculum.level_caps[2]), 2)
            self.assertEqual(restored.env._elevation_settings.speed_range, (.15, .30))
            self.assertEqual(restored.env.cfg.episode_length_s, 120.)
            for key, expected in source.alg.actor.state_dict().items():
                torch.testing.assert_close(restored.alg.actor.state_dict()[key], expected, rtol=0., atol=0.)


if __name__ == "__main__":
    unittest.main()
