"""Verify short repeated steps, per-leg participation and slope-adapted references."""

import math
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.elevation import collect_elevation_contacts, elevation_cost, elevation_support_pitch
from pointleg.mdp.elevation_gait import elevation_gait_term, elevation_surface_frame
from pointleg.tests.test_elevation_clearance import clearance_env, set_foot, tick


def settle(env, term, cfg, seconds=.42):
    for _ in range(round(seconds / env.step_dt)):
        tick(env, term, cfg)


def step(env, term, cfg, foot, height=.09, stride=.10, body_progress=.08, landing_height=0.):
    corner = env._elevation["corner"]
    start = float(env.scene["robot"].data.site_pos_w[0, foot, 0] - corner[0, 0])
    for _ in range(5):
        set_foot(env, foot, start + stride * .6, height)
        tick(env, term, cfg)
    env.scene["robot"].data.root_link_pos_w[:, 0] += body_progress
    set_foot(env, foot, start + stride, landing_height, support=True)
    reward = 0.
    for _ in range(math.ceil(cfg.params["minimum_stance_time"] / env.step_dt) + 1):
        tick(env, term, cfg)
        reward += float(elevation_gait_term(env, "quality").sum()) * env.step_dt
    return reward


def workspace_env(kind=5, level=0):
    env, _, _ = clearance_env(kind=kind, level=level)
    robot = env.scene["robot"]
    robot.find_bodies = lambda name: ([['fl', 'fr', 'rl', 'rr'].index(name.split('_')[0])], [name])
    robot.data.body_link_pos_w = robot.data.site_pos_w.clone() - torch.tensor([
        [.035, .105, -.4], [.035, -.105, -.4], [.035, .105, -.4], [.035, -.105, -.4],
    ])
    cfg = env.cfg.rewards["translation_foot_workspace"]
    return env, cfg.func(cfg, env), cfg


class ElevationGaitTests(unittest.TestCase):
    def test_style_is_walking_only_and_reduces_the_push_for_large_forward_motion(self):
        cfg = load_env_cfg("PointLeg-Elevation-v0")
        self.assertEqual(cfg.rewards["command_aligned_progress"].weight, 0.)
        self.assertEqual(cfg.rewards["first_contact_events"].weight, .2)
        self.assertEqual(cfg.rewards["translation_foot_workspace"].params["forward_margin"], .14)
        self.assertEqual(cfg.rewards["feet_air_time"].weight, 0.)
        for name in ("step_quality", "foot_participation", "stride_length"):
            self.assertIn(name, cfg.rewards)
            for other in ("PointLeg-Jump-v0", "PointLeg-Rough-v0", "PointLeg-RoughMixed-v0"):
                self.assertNotIn(name, load_env_cfg(other).rewards)

    def test_every_leg_can_earn_a_bounded_short_high_step(self):
        for foot in range(4):
            env, term, cfg = clearance_env()
            settle(env, term, cfg)
            reward = step(env, term, cfg, foot)
            self.assertGreater(reward, 0.)
            self.assertLessEqual(reward, cfg.params["step_bonus"] + 1.e-6)
            self.assertLess(float(term.participation_age[0, foot]), .1)
            self.assertGreater(float(term.participation_age[0, (foot + 1) % 4]), .5)

    def test_two_advancing_steps_on_the_same_tread_both_receive_quality_feedback(self):
        env, term, cfg = clearance_env(kind=0)
        for foot in range(4):
            set_foot(env, foot, 3.04, .04, support=True)
        env._elevation["covered"][:, 0] = True
        settle(env, term, cfg)
        first = step(env, term, cfg, 0, height=.14, landing_height=.04)
        settle(env, term, cfg)
        second = step(env, term, cfg, 0, height=.14, landing_height=.04)
        self.assertGreater(first, 0.)
        self.assertGreater(second, 0.)
        self.assertLess(float(env.scene["robot"].data.site_pos_w[0, 0, 0]), 3.32)
        self.assertEqual(int(env._elevation["covered"].sum()), 1)

    def test_low_valid_swing_receives_feedback_but_is_not_counted_as_a_quality_step(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg, 1.)
        before = float(term.participation_age[0, 3])
        self.assertGreater(step(env, term, cfg, 3, height=.01), 0.)
        self.assertLess(float(term.participation_age[0, 3]), before)
        self.assertEqual(float(term.metrics["quality_steps_per_s"][0]), 0.)
        self.assertGreater(float(elevation_gait_term(env, "participation")[0]), 0.)

    def test_one_neglected_hind_leg_is_not_hidden_by_other_legs(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg, 2.)
        term.participation_age[:, :3] = 0.
        tick(env, term, cfg)
        self.assertGreaterEqual(float(elevation_gait_term(env, "participation")[0]), .62)
        self.assertGreater(float(env.extras["log"]["Metrics/elevation_cycle_age_rr_s"][0]), 1.8)

    def test_long_lunge_has_no_step_bonus_and_gets_a_stride_cost(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        reward = step(env, term, cfg, 0, height=.10, stride=.40)
        self.assertEqual(reward, 0.)
        self.assertGreater(float(term.participation_age[0, 0]), .5)
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        set_foot(env, 0, 1.7, .1)
        settle(env, term, cfg, .10)
        set_foot(env, 0, 1.8, 0., support=True)
        tick(env, term, cfg)
        self.assertGreater(float(elevation_gait_term(env, "stride")[0]), 0.)

    def test_body_bouncing_without_relative_leg_lift_does_not_qualify(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        env.scene["robot"].data.root_link_pos_w[:, 2] += .10
        reward = step(env, term, cfg, 0, height=.10)
        self.assertEqual(reward, 0.)
        self.assertGreater(float(term.participation_age[0, 0]), .5)

    def test_all_feet_airborne_cannot_earn_a_walking_cycle(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        for foot in range(4):
            set_foot(env, foot, 1.5, .1)
        settle(env, term, cfg, .1)
        env.scene["robot"].data.root_link_pos_w[:, 0] += .1
        for foot in range(4):
            set_foot(env, foot, 1.6, 0., support=True)
        settle(env, term, cfg, .1)
        self.assertEqual(float(elevation_gait_term(env, "quality")[0]), 0.)
        self.assertTrue((term.participation_age > .5).all())

    def test_stationary_repeated_taps_and_backtracking_do_not_create_new_progress(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        self.assertEqual(step(env, term, cfg, 0, body_progress=0.), 0.)
        settle(env, term, cfg)
        self.assertGreater(step(env, term, cfg, 0), 0.)
        settle(env, term, cfg)
        env.scene["robot"].data.root_link_pos_w[:, 0] -= .10
        self.assertEqual(step(env, term, cfg, 0, body_progress=.08), 0.)

    def test_same_tick_failure_masks_a_cached_positive_event(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        set_foot(env, 0, 1.46, .09)
        settle(env, term, cfg, .10)
        env.scene["robot"].data.root_link_pos_w[:, 0] += .08
        set_foot(env, 0, 1.5, 0., support=True)
        settle(env, term, cfg, cfg.params["minimum_stance_time"])
        self.assertGreater(float(elevation_gait_term(env, "quality")[0]), 0.)
        env._elevation["failed"][:] = True
        self.assertEqual(float(elevation_gait_term(env, "quality")[0]), 0.)

    def test_workspace_limits_both_front_and_rear_extension_without_limiting_flat_lift(self):
        env, term, cfg = workspace_env()
        self.assertAlmostEqual(float(term(env, **cfg.params)[0]), 0.)
        env.scene["robot"].data.site_pos_w[:, :, 2] += .20
        self.assertAlmostEqual(float(term(env, **cfg.params)[0]), 0.)
        env.scene["robot"].data.site_pos_w[:, 0, 0] += .4
        self.assertGreater(float(term(env, **cfg.params)[0]), .5)
        env, term, cfg = workspace_env()
        env.scene["robot"].data.site_pos_w[:, 3, 0] -= .4
        self.assertGreater(float(term(env, **cfg.params)[0]), .5)

    def test_ramp_workspace_allows_foot_movement_down_the_surface_normal(self):
        for kind in (2, 3):
            env, term, cfg = workspace_env(kind=kind, level=4)
            robot = env.scene["robot"].data
            robot.root_link_pos_w[:, 0] = 4.
            tangent, normal = elevation_surface_frame(env, torch.full((1, 4), 4.))
            hips = robot.body_link_pos_w
            for depth in (.35, .55):
                robot.site_pos_w[:] = hips + .035 * tangent - depth * normal
                robot.site_pos_w[:, :, 1] += torch.tensor([.105, -.105, .105, -.105])
                self.assertAlmostEqual(float(term(env, **cfg.params)[0]), 0., places=6)

    def test_loaded_contact_plane_sets_pitch_reference_instead_of_world_level(self):
        env, _, _ = clearance_env(kind=0)
        data = env.scene["elevation_contacts"].data
        for foot, (x, z) in enumerate(((3.16, .04), (3.16, .04), (3.48, .08), (3.48, .08))):
            data.pos[:, foot * 4] = torch.tensor([x, 4., z])
        expected = -math.atan(.04 / .32)
        self.assertAlmostEqual(float(elevation_support_pitch(env)[0]), expected, places=6)
        env._elevation["updated_tick"] = env.common_step_counter
        robot = env.scene["robot"].data
        robot.root_link_pos_w[:, 0] = 3.3
        robot.root_link_quat_w[:, 0] = math.cos(expected / 2)
        robot.root_link_quat_w[:, 2] = math.sin(expected / 2)
        self.assertEqual(float(elevation_cost(env, "attitude")[0]), 0.)

    def test_same_tick_and_partial_reset_do_not_double_pay_or_leak_cycles(self):
        env, term, cfg = clearance_env(count=2)
        settle(env, term, cfg)
        step(env, term, cfg, 0)
        before = term.participation_age.clone()
        term(env, **cfg.params)
        torch.testing.assert_close(term.participation_age, before)
        term.reset(torch.tensor([0]))
        self.assertTrue((term.participation_age[0] == 0.).all())
        torch.testing.assert_close(term.participation_age[1], before[1])

    def test_contact_point_velocity_does_not_mistake_foot_rotation_for_sliding(self):
        env, _, _ = clearance_env()
        robot = env.scene["robot"].data
        robot.site_pos_w[:] = torch.tensor([1.4, 4., .05])
        robot.site_lin_vel_w[..., 0] = .1
        robot.site_ang_vel_w[..., 1] = 2.
        sensor = env.scene["elevation_contacts"].data
        sensor.pos[:, ::4] = torch.tensor([1.4, 4., 0.])
        collect_elevation_contacts(env)
        self.assertAlmostEqual(float(env._elevation["max_slip"][0]), 0., places=6)
        robot.site_ang_vel_w[:] = 0.
        collect_elevation_contacts(env)
        self.assertGreater(float(env._elevation["max_slip"][0]), .4)

    def test_single_sliding_hind_leg_is_not_hidden_by_three_stable_feet(self):
        env, _, _ = clearance_env()
        env.scene["robot"].data.site_lin_vel_w[:, 3, 0] = .15
        collect_elevation_contacts(env)
        env._elevation["updated_tick"] = env.common_step_counter
        self.assertAlmostEqual(float(elevation_cost(env, "slip")[0]), .625, places=6)


if __name__ == "__main__":
    unittest.main()
