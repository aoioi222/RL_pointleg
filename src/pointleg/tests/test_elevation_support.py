"""Check loaded pushing, gait renewal and filtered posture without physics."""

import math
import unittest
from unittest.mock import patch

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.elevation import elevation_forward_velocity_reward, reset_elevation
from pointleg.mdp.elevation_gait import elevation_gait_term
from test_elevation import bounds, fake_env
from test_elevation_clearance import clearance_env, set_foot, tick
from test_elevation_gait import settle, step, workspace_env


def loaded_takeoff(seconds=.36, travel=.08, slip=0., rise=0., kind=5):
    env, term, cfg = clearance_env(kind=kind, level=4)
    robot = env.scene["robot"].data
    angle = float(env._elevation["angle"][0]) if kind == 2 else 0.
    if kind == 2:
        for foot in range(4):
            set_foot(env, foot, 4., math.tan(angle), support=True)
        robot.root_link_pos_w[:, 0] = env._elevation["corner"][:, 0] + 3.9
        robot.root_link_pos_w[:, 2] += .9 * math.tan(angle)
    tick(env, term, cfg)
    start_x = float(robot.site_pos_w[0, 0, 0] - env._elevation["corner"][0, 0])
    start_z = float(robot.site_pos_w[0, 0, 2] - env._elevation["corner"][0, 2])
    frames = round(seconds / env.step_dt)
    for i in range(frames):
        robot.root_link_pos_w[:, 0] += travel * math.cos(angle) / frames
        robot.root_link_pos_w[:, 2] += (travel * math.sin(angle) + rise) / frames
        set_foot(env, 0, start_x + slip * (i + 1) / frames, start_z, support=True)
        tick(env, term, cfg)
    set_foot(env, 0, start_x + slip + .01, start_z + .05)
    tick(env, term, cfg)
    return env, term, cfg


def posture_env(count=1):
    env, _, _ = clearance_env(count=count)
    cfg = env.cfg.rewards["mean_leg_posture"]
    return env, cfg.func(cfg, env), cfg


def posture_tick(env, term, cfg):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    env.extras["log"] = {}
    return term(env, **cfg.params).clone()


class ElevationSupportTests(unittest.TestCase):
    def test_full_loaded_backstroke_increases_the_actual_landing_bonus(self):
        rewards = []
        for duration in (.08, .36):
            env, term, cfg = loaded_takeoff(seconds=duration)
            set_foot(env, 0, 1.46, .15)
            for _ in range(5):
                tick(env, term, cfg)
            set_foot(env, 0, 1.5, 0., support=True)
            reward = 0.
            for _ in range(20):
                tick(env, term, cfg)
                reward += elevation_gait_term(env, "quality").item() * env.step_dt
            rewards.append(reward)
        self.assertGreater(rewards[0], 0.)
        self.assertGreater(rewards[1], rewards[0] * 3.9)
        self.assertAlmostEqual(rewards[1], .08, places=5)

    def test_world_fixed_foot_and_forward_body_motion_are_a_valid_backstroke(self):
        env, term, cfg = loaded_takeoff()
        self.assertGreater(term.swing_push_quality[0, 0].item(), .99)
        self.assertAlmostEqual(env.extras["log"]["Metrics/elevation_backstroke_fl_m"].item(), .08, places=5)
        self.assertGreaterEqual(env.extras["log"]["Metrics/elevation_stance_time_fl_s"].item(), .35)

    def test_short_touch_small_backstroke_sliding_and_vertical_bounce_are_not_good_pushes(self):
        for args in ({"seconds": .08}, {"travel": .005}, {"slip": .08}, {"travel": 0., "rise": .10}):
            with self.subTest(args=args):
                _, term, _ = loaded_takeoff(**args)
                self.assertLess(term.swing_push_quality[0, 0].item(), .15)
        env, term, _ = loaded_takeoff(rise=.06)
        self.assertGreater(term.swing_push_quality[0, 0].item(), .99)
        self.assertAlmostEqual(env.extras["log"]["Metrics/elevation_support_body_rise_fl_m"].item(), .06, places=5)

    def test_ramp_push_uses_the_support_tangent(self):
        env, term, _ = loaded_takeoff(kind=2)
        self.assertGreater(term.swing_push_quality[0, 0].item(), .99)
        self.assertAlmostEqual(env.extras["log"]["Metrics/elevation_backstroke_fl_m"].item(), .08, places=5)

    def test_backward_ground_sliding_cannot_increase_backstroke_credit(self):
        _, fixed, _ = loaded_takeoff(travel=.01)
        env, sliding, _ = loaded_takeoff(travel=.01, slip=-.005)
        self.assertLess(sliding.swing_push_quality[0, 0].item(), fixed.swing_push_quality[0, 0].item())
        self.assertLessEqual(env.extras["log"]["Metrics/elevation_backstroke_fl_m"].item(), .01 + 1.e-6)

    def test_sliding_out_and_back_does_not_restore_support_quality(self):
        env, term, cfg = clearance_env()
        tick(env, term, cfg)
        for x in (1.44, 1.4):
            set_foot(env, 0, x, 0., support=True)
            for _ in range(10):
                env.scene["robot"].data.root_link_pos_w[:, 0] += .004
                tick(env, term, cfg)
        set_foot(env, 0, 1.41, .05)
        tick(env, term, cfg)
        self.assertEqual(term.swing_push_quality[0, 0].item(), 0.)

    def test_fixed_feet_expire_velocity_credit_and_get_overstay_cost(self):
        env, term, cfg = clearance_env()
        env._elevation["speed"][:] = .2
        settle(env, term, cfg, 1.5)
        self.assertEqual(term.gait["renewal"].item(), 0.)
        self.assertGreater(term.gait["stance"].item(), .99)
        env.scene["robot"].data.root_link_pos_w[:, 0] += .004
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .2
        with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
            self.assertEqual(elevation_forward_velocity_reward(env).item(), 0.)
        self.assertGreater(step(env, term, cfg, 0, height=.04), 0.)
        self.assertGreater(term.gait["renewal"].item(), 0.)
        self.assertLess(term.gait["renewal"].item(), 1.)

    def test_same_tick_partial_reset_and_failure_do_not_leak_support_state(self):
        env, term, cfg = clearance_env(count=2)
        settle(env, term, cfg, .4)
        duration = term.stance_time.clone()
        term(env, **cfg.params)
        torch.testing.assert_close(term.stance_time, duration)
        term.reset(torch.tensor([0]))
        self.assertEqual(term.stance_time[0].sum().item(), 0.)
        torch.testing.assert_close(term.stance_time[1], duration[1])
        self.assertEqual(term.gait["renewal"][0].item(), 1.)
        env._elevation["failed"][:] = True
        self.assertEqual(elevation_gait_term(env, "quality").sum().item(), 0.)

    def test_front_workspace_is_shorter_and_rear_push_is_allowed(self):
        env, term, cfg = workspace_env()
        robot = env.scene["robot"].data
        hip_x = robot.body_link_pos_w[:, 0, 0].clone()
        robot.site_pos_w[:, 0, 0] = hip_x - .15
        self.assertEqual(term(env, **cfg.params).item(), 0.)
        robot.site_pos_w[:, 0, 0] = hip_x + .18
        self.assertGreater(term(env, **cfg.params).item(), 0.)

    def test_slow_front_rear_bias_is_penalized_but_common_flexion_is_allowed(self):
        env, term, cfg = posture_env(count=2)
        q = env.scene["robot"].data.joint_pos
        q[0, [1, 4]] = .55
        q[0, [2, 5]] = .6
        q[1, [1, 4, 7, 10]] = .55
        q[1, [2, 5, 8, 11]] = .6
        for _ in range(100):
            cost = posture_tick(env, term, cfg)
        self.assertGreater(cost[0].item(), .1)
        self.assertEqual(cost[1].item(), 0.)

    def test_normal_alternating_phases_are_not_forced_to_match(self):
        env, term, cfg = posture_env()
        for i in range(100):
            swing = .3 if i % 2 else -.3
            env.scene["robot"].data.joint_pos[:, [1, 4]] = swing
            env.scene["robot"].data.joint_pos[:, [7, 10]] = -swing
            cost = posture_tick(env, term, cfg)
        self.assertEqual(cost.item(), 0.)

    def test_unequal_footholds_relax_posture_and_invalid_joints_do_not_poison_it(self):
        env, term, cfg = posture_env(count=2)
        env.scene["robot"].data.joint_pos[:, [1, 4]] = .55
        env.scene["robot"].data.joint_pos[:, [2, 5]] = .6
        env.scene["robot"].data.site_pos_w[1, :2, 2] += .12
        for _ in range(100):
            cost = posture_tick(env, term, cfg)
        self.assertGreater(cost[0].item(), 0.)
        self.assertLess(cost[1].item(), cost[0].item() * .25)
        env.scene["robot"].data.joint_pos[0, 1] = torch.nan
        self.assertTrue(torch.isfinite(posture_tick(env, term, cfg)).all())
        self.assertEqual(term.cached[0].item(), 0.)
        mean = term.mean[1].clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.mean[1], mean)
        env._elevation["failed"][:] = True
        self.assertEqual(term(env, **cfg.params).sum().item(), 0.)

    def test_nearer_spawn_preserves_random_offsets_and_fixed_height_metadata(self):
        env = fake_env(count=100, stair_height_override=.12)
        settings = env._elevation_settings
        reset_elevation(env, None, settings, randomize=True)
        ids = torch.where(env.scene.terrain.terrain_types < 4)[0]
        distance = env._elevation["entry_x"][ids] - env._elevation["start_x"][ids]
        self.assertTrue((distance >= .7 - 1.e-6).all())
        self.assertTrue((distance <= 1.3 + 1.e-6).all())
        self.assertLess(distance.mean().item(), 1.1)
        reset_elevation(env, ids, settings, randomize=False)
        torch.testing.assert_close(env._elevation["entry_x"][ids] - env._elevation["start_x"][ids], torch.ones(len(ids)))
        down = torch.where(env.scene.terrain.terrain_types == 1)[0]
        torch.testing.assert_close(env._elevation["top_height"][down], torch.full((len(down),), .72))
        for task in ("PointLeg-HighStep-v0", "PointLeg-Rough-v0", "PointLeg-Jump-v0"):
            other = load_env_cfg(task)
            for key in ("mean_leg_posture", "stance_overstay"):
                self.assertNotIn(key, other.rewards)


if __name__ == "__main__":
    unittest.main()
