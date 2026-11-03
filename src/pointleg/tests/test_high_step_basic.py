"""Validate the live basic objective with tensors and configuration only."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pointleg.high_step_evaluation import gait_accepted, snapshot_high_step
from pointleg.mdp.high_step_basic import (
    basic_height_event, basic_height_progress, basic_high_step_slip,
    basic_high_step_velocity, basic_support_deficit,
)
from test_high_step import cycle_env


def basic_env(reverse=False, step_dt=.02):
    env, sensor, _, _ = cycle_env(step_dt=step_dt, reverse_contacts=reverse)
    del env._high_step_cycle_term
    env.common_step_counter = 0
    env.scene["robot"].data.root_link_lin_vel_b = torch.zeros(3, 3)
    env.termination_manager = SimpleNamespace(terminated=torch.zeros(3, dtype=torch.bool))
    cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["support_timing"]
    term = cfg.func(cfg, env)
    return env, sensor, cfg, term


def tick(bundle):
    env, sensor, cfg, term = bundle
    env.common_step_counter += 1
    robot = env.scene["robot"].data
    for foot, contact in enumerate(term.ids):
        sensor.data.pos[:, contact] = robot.site_pos_w[:, foot] + torch.tensor([0., 0., -.025])
    env.extras["log"] = {}
    return term(env, **cfg.params).clone()


def contact(bundle, foot, loaded):
    _, sensor, _, term = bundle
    idx = term.ids[foot]
    sensor.data.found[:, idx] = float(loaded)
    sensor.data.force[:, idx] = 0.
    if loaded:
        sensor.data.force[:, idx, 2] = 30.


def observe_cycle(bundle, height=.10):
    env, _, _, _ = bundle
    robot = env.scene["robot"].data
    base_z = robot.site_pos_w[:, 0, 2].clone()
    for _ in range(18):
        tick(bundle)
    contact(bundle, 0, False)
    for i in range(10):
        robot.site_pos_w[:, 0, 2] = base_z + (height if i == 5 else 0.)
        tick(bundle)
    robot.site_pos_w[:, 0, 0] += .05
    contact(bundle, 0, True)
    tick(bundle)


def active_lift(bundle, height=.15, forward=.05, stable=True):
    env, _, _, term = bundle
    robot = env.scene["robot"].data
    base_z = robot.site_pos_w[:, 0, 2].clone()
    tick(bundle)
    for _ in range(20):
        robot.root_link_pos_w[:, 0] += .0035
        tick(bundle)
    contact(bundle, 0, False)
    tick(bundle)
    event = torch.zeros(env.num_envs)
    progress = torch.zeros(env.num_envs)
    for _ in range(5):
        robot.site_pos_w[:, 0, 2] = base_z + height
        tick(bundle)
        event += basic_height_event(env) * env.step_dt
        progress += basic_height_progress(env) * env.step_dt
    robot.site_pos_w[:, 0, 2] = base_z
    robot.site_pos_w[:, 0, 0] += forward
    contact(bundle, 0, True)
    tick(bundle)
    for _ in range(10 if stable else 2):
        tick(bundle)
        event += basic_height_event(env) * env.step_dt
        progress += basic_height_progress(env) * env.step_dt
    return event, progress


class BasicHighStepTests(unittest.TestCase):
    def test_live_configuration_has_the_approved_terms(self):
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        expected = {
            "upright", "body_ang_vel", "stand_still", "joint_torques", "action_rate",
            "joint_pos_limits", "body_collision", "is_terminated", "command_orthogonal_drift",
            "track_lin_vel", "high_step_yaw_rate", "high_step_mean_leg_symmetry",
            "support_deficit", "support_timing", "feet_slip", "active_lift_event",
            "active_lift_progress", "hip_abduction_excess", "high_step_pitch",
            "hip_swing_coordination", "front_foot_workspace", "diagonal_rear_tracking",
            "stance_backstroke", "lf_rr_pose_tracking", "active_lift_overshoot",
        }
        self.assertEqual(set(cfg.rewards), expected)
        self.assertTrue(all(term.weight != 0. for term in cfg.rewards.values()))
        self.assertNotIn("high_step_cycle", cfg.rewards)
        self.assertIn("high_step_contact_force", cfg.metrics)
        self.assertEqual(cfg.rewards["track_lin_vel"].weight, 2.)
        self.assertEqual(cfg.rewards["active_lift_event"].weight, .35)
        self.assertEqual(cfg.rewards["active_lift_progress"].weight, .08)

    def test_play_interface_ppo_and_other_tasks_are_preserved(self):
        flat = load_env_cfg("PointLeg-Flat-v0")
        for play in (False, True):
            cfg = load_env_cfg("PointLeg-HighStep-v0", play=play)
            self.assertEqual(cfg.actions, load_env_cfg("PointLeg-Flat-v0", play=play).actions)
            self.assertEqual(list(cfg.observations["actor"].terms), list(flat.observations["actor"].terms))
            self.assertEqual(list(cfg.observations["critic"].terms), list(flat.observations["critic"].terms))
            self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (.15, .25))
            self.assertEqual(cfg.commands["twist"].rel_standing_envs, .10)
            self.assertEqual(len(cfg.rewards), 25)
        self.assertEqual(load_rl_cfg("PointLeg-HighStep-v0").algorithm,
                         load_rl_cfg("PointLeg-Flat-v0").algorithm)
        self.assertIn("foot_swing_height", flat.rewards)
        for task in ("PointLeg-Flat-v0", "PointLeg-Elevation-v0", "PointLeg-Jump-v0"):
            self.assertNotIn("support_timing", load_env_cfg(task).rewards)

    def test_tracking_peaks_at_target_and_does_not_pay_for_stillness_or_reversal(self):
        env = basic_env()[0]
        velocity = env.scene["robot"].data.root_link_lin_vel_b
        values = []
        for speed in (0., -.2, .1, .2, .4):
            velocity[:, 0] = speed
            values.append(basic_high_step_velocity(env, "twist").clone())
        self.assertEqual(values[0].sum().item(), 0.)
        self.assertEqual(values[1].sum().item(), 0.)
        self.assertGreater(values[3].min().item(), .999)
        self.assertTrue((values[2] < values[3]).all())
        self.assertTrue((values[4] < values[3]).all())

    def test_tracking_is_independent_of_old_cycle_support_and_height(self):
        env = basic_env()[0]
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .2
        before = basic_high_step_velocity(env, "twist")
        env.scene["robot"].data.site_pos_w[:, :, 2] += .3
        env._high_step_support_term = object()
        env._high_step_cycle_term = object()
        torch.testing.assert_close(basic_high_step_velocity(env, "twist"), before)

    def test_tracking_masks_failure_standing_and_nonfinite_values(self):
        env = basic_env()[0]
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([.2, torch.nan, .2])
        env.termination_manager.terminated[0] = True
        env.commands[2] = 0.
        self.assertEqual(basic_high_step_velocity(env, "twist").sum().item(), 0.)
        env.commands[:, 0] = torch.inf
        self.assertTrue(torch.isfinite(basic_high_step_velocity(env, "twist")).all())

    def test_fixed_feet_have_overstay_cost_and_flying_foot_has_air_cost(self):
        for flying in (False, True):
            bundle = basic_env()
            for _ in range(32):
                tick(bundle)
            if flying:
                contact(bundle, 0, False)
            for _ in range(60):
                cost = tick(bundle)
            self.assertGreater(cost.min().item(), .6)

    def test_short_stance_event_occurs_once_and_mature_support_is_free(self):
        for frames, positive in ((4, True), (12, False)):
            bundle = basic_env()
            env, _, cfg, term = bundle
            cfg.params["startup_grace"] = 0.
            for _ in range(10):
                tick(bundle)
            contact(bundle, 0, False)
            tick(bundle)
            contact(bundle, 0, True)
            for _ in range(frames):
                tick(bundle)
            contact(bundle, 0, False)
            cost = tick(bundle)
            self.assertEqual(bool((cost > 0.).all()), positive)
            torch.testing.assert_close(term(env, **cfg.params), cost)
            self.assertEqual(tick(bundle).sum().item(), 0.)

    def test_short_stance_event_has_control_rate_independent_integral(self):
        values = []
        for dt in (.01, .02):
            bundle = basic_env(step_dt=dt)
            env, _, _, _ = bundle
            for _ in range(round(.64 / dt)):
                tick(bundle)
            contact(bundle, 0, False)
            tick(bundle)
            contact(bundle, 0, True)
            for _ in range(round(.08 / dt)):
                tick(bundle)
            contact(bundle, 0, False)
            values.append(tick(bundle) * dt)
        torch.testing.assert_close(*values, atol=1.e-6, rtol=0.)

    def test_real_rolling_is_not_slip_and_dragging_is(self):
        for reverse in (False, True):
            bundle = basic_env(reverse)
            env = bundle[0]
            robot = env.scene["robot"].data
            robot.site_lin_vel_w[:, 0, 0] = .1
            robot.site_ang_vel_w[:, 0, 1] = 4.
            for _ in range(6):
                tick(bundle)
            self.assertEqual(basic_high_step_slip(env).sum().item(), 0.)
            robot.site_ang_vel_w.zero_()
            tick(bundle)
            self.assertGreater(basic_high_step_slip(env).min().item(), .3)

    def test_weak_contacts_and_normal_motion_do_not_count_as_dragging(self):
        bundle = basic_env()
        env, sensor, _, _ = bundle
        robot = env.scene["robot"].data
        robot.site_lin_vel_w[:, 0, 2] = -.5
        for _ in range(5):
            tick(bundle)
        self.assertEqual(basic_high_step_slip(env).sum().item(), 0.)
        robot.site_lin_vel_w[:, 0, 0] = .5
        sensor.data.force[:, 0, 2] = 7.9
        tick(bundle)
        self.assertEqual(basic_high_step_slip(env).sum().item(), 0.)

    def test_support_deficit_uses_load_instead_of_contact_presence(self):
        env, sensor, _, _ = basic_env()
        sensor.data.force[:, :3, 2] = 7.9
        torch.testing.assert_close(basic_support_deficit(env, "feet_ground_contact"), torch.ones(3))

    def test_partial_reset_and_repeated_reads_do_not_advance_other_envs(self):
        bundle = basic_env()
        env, _, cfg, term = bundle
        for _ in range(10):
            tick(bundle)
        before = term.stance.clone()
        term(env, **cfg.params)
        basic_high_step_slip(env)
        torch.testing.assert_close(term.stance, before)
        term.reset(torch.tensor([0]))
        term(env, **cfg.params)
        torch.testing.assert_close(term.stance[1:], before[1:])
        self.assertAlmostEqual(term.stance[0, 0].item(), env.step_dt)

    def test_nonfinite_contact_and_failure_clear_state_without_nan(self):
        bundle = basic_env()
        env, sensor, cfg, term = bundle
        for _ in range(5):
            tick(bundle)
        env.scene["robot"].data.site_lin_vel_w[0, 0, 0] = torch.nan
        sensor.data.force[1, 0, 2] = torch.nan
        env.termination_manager.terminated[2] = True
        cost = tick(bundle)
        self.assertEqual(cost.sum().item(), 0.)
        self.assertEqual(term.stance.sum().item(), 0.)
        self.assertTrue(torch.isfinite(term.slip_cost).all())
        self.assertEqual(term.stats["cycles"].sum().item(), 0.)

    def test_height_is_observed_without_changing_support_penalty(self):
        costs = []
        for height in (.08, .15):
            bundle = basic_env()
            observe_cycle(bundle, height)
            env, _, _, term = bundle
            self.assertAlmostEqual(env.extras["log"]["Metrics/high_step_peak_fl_m"].mean().item(), height, places=5)
            self.assertEqual(term.stats["cycles"][:, 0].tolist(), [1., 1., 1.])
            costs.append(term.cost.clone())
        torch.testing.assert_close(*costs)

    def test_basic_evaluation_has_no_fixed_height_or_cycle_target(self):
        bundle = basic_env()
        observe_cycle(bundle, .05)
        env, _, _, term = bundle
        summary = snapshot_high_step(env, 0)
        self.assertEqual(summary["profile"], "basic")
        self.assertTrue(summary["height_reward_enabled"])
        self.assertFalse(summary["accepted"])
        term.stats["cycles"][:] = 5.
        term.stats["valid_cycles"][:] = 5.
        self.assertTrue(snapshot_high_step(env, 0)["accepted"])
        broken = deepcopy(snapshot_high_step(env, 0))
        broken["feet"]["rr"]["height_m"] = float("nan")
        self.assertFalse(gait_accepted(broken))

    def test_active_lift_rewards_are_ordered_and_sum_to_the_defined_scores(self):
        event, progress = zip(*(active_lift(basic_env(), height) for height in (.08, .12, .15)))
        for value, height in zip(event, (.08, .12, .15)):
            torch.testing.assert_close(value, torch.full((3,), (height / .15) ** 2), atol=1.e-5, rtol=0.)
        for value, height in zip(progress, (.08, .12, .15)):
            torch.testing.assert_close(value, torch.full((3,), height / .15), atol=1.e-5, rtol=0.)
        self.assertTrue((event[0] < event[1]).all())
        self.assertTrue((event[1] < event[2]).all())

    def test_active_lift_rejects_short_air_short_support_stationary_and_unstable_landings(self):
        for kwargs in (
            {"forward": 0.}, {"stable": False},
        ):
            event, _ = active_lift(basic_env(), **kwargs)
            self.assertEqual(event.sum().item(), 0.)
        bundle = basic_env()
        env, _, _, term = bundle
        for _ in range(2):
            tick(bundle)
        contact(bundle, 0, False)
        tick(bundle)
        self.assertFalse(term.eligible_swing[:, 0].any())

    def test_reward_manager_combines_independent_terms_and_resets_cached_state(self):
        from mjlab.managers.reward_manager import RewardManager

        bundle = basic_env()
        env = bundle[0]
        env.max_episode_length_s = 20.
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        names = ("track_lin_vel", "support_timing", "feet_slip", "support_deficit")
        manager = RewardManager({name: cfg.rewards[name] for name in names}, env)
        standing = manager.compute(env.step_dt).clone()
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .2
        env.common_step_counter += 1
        moving = manager.compute(env.step_dt).clone()
        self.assertTrue((moving > standing).all())
        torch.testing.assert_close(moving, torch.full((3,), .04))
        manager.reset(torch.tensor([0]))
        self.assertEqual(env._high_step_basic_term.age[0].item(), 0.)
        self.assertGreater(env._high_step_basic_term.age[1].item(), 0.)

    def test_contact_force_metric_records_overload_without_a_reward_term(self):
        from mjlab.managers.metrics_manager import MetricsManager

        env, sensor, _, _ = basic_env()
        sensor.data.force[:, 0, 2] = 200.
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        manager = MetricsManager({"force": cfg.metrics["high_step_contact_force"]}, env)
        manager.compute()
        self.assertEqual(env.extras["log"]["Metrics/high_step_contact_peak_fl_n"].tolist(), [200.] * 3)
        self.assertNotIn("high_step_contact_force", cfg.rewards)
        manager.reset(torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
