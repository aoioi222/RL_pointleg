"""Check forward credit and terrain-aware high steps with tensor trajectories."""

from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pointleg.evaluate_elevation import ElevationEvaluateConfig, _run_batch, evaluation_cases
from pointleg.mdp.elevation import (
    collect_elevation_contacts, elevation_forward_velocity_reward,
    reset_elevation, select_elevation_terrain,
)
from pointleg.mdp.elevation_gait import elevation_gait_term
from test_elevation import bounds, fake_env
from test_elevation_clearance import clearance_env, set_foot, tick
from test_elevation_gait import settle, step


def movement_env(count=1):
    env = fake_env(count=count, stair_height_override=.12)
    settings = env.cfg.events["reset_elevation"].params["settings"]
    env._elevation_fixed_cases = True
    env.scene.terrain.terrain_types[:] = 5
    select_elevation_terrain(env, None, settings)
    reset_elevation(env, None, settings, randomize=False)
    env._elevation["support_count"][:] = 4.
    return env


def move(env, distance, speed=.2):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    robot = env.scene["robot"].data
    robot.root_link_pos_w[:, 0] += torch.as_tensor(distance)
    robot.root_link_lin_vel_b[:, 0] = torch.as_tensor(speed)
    robot.root_link_lin_vel_w[:, 0] = torch.as_tensor(speed)
    with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
        return elevation_forward_velocity_reward(env).clone()


def cycle(height, progress=.08, stride=.10, foot=0):
    env, term, cfg = clearance_env()
    tick(env, term, cfg)
    stance_progress = min(progress, cfg.params["target_backstroke"])
    frames = math.ceil(cfg.params["target_push_stance"] / env.step_dt)
    for _ in range(frames):
        env.scene["robot"].data.root_link_pos_w[:, 0] += stance_progress / frames
        tick(env, term, cfg)
    settle(env, term, cfg, .06)
    reward = step(env, term, cfg, foot, height=height, stride=stride,
                  body_progress=progress - stance_progress)
    return reward, env, term, cfg


def trajectory_return(behavior):
    env, term, cfg = clearance_env()
    env._elevation_settings = env.cfg.events["reset_elevation"].params["settings"]
    env._elevation["speed"][:] = .2
    settle(env, term, cfg)
    robot = env.scene["robot"].data
    starts = robot.site_pos_w[0, :, 0].clone()
    total = 0.
    keys = ("velocity", "step_quality", "foot_participation", "slip", "action",
            "swing_clearance", "stride_length")
    with patch("pointleg.mdp.elevation.geometry_bounds", side_effect=bounds):
        for frame in range(120):
            env.common_step_counter += 1
            env.episode_length_buf += 1
            env.extras["log"] = {}
            previous = robot.site_pos_w.clone()
            speed = 0. if behavior == "standing" else .2
            robot.root_link_pos_w[:, 0] += speed * env.step_dt
            robot.root_link_lin_vel_b[:, 0] = speed
            robot.root_link_lin_vel_w[:, 0] = speed
            phase = frame % 30
            swinging = (0, 3) if phase < 15 else (1, 2)
            for foot in range(4):
                x = robot.site_pos_w[0, foot, 0].item()
                z, supported = 0., True
                if behavior == "walking" and foot in swinging:
                    if phase % 15 == 0:
                        starts[foot] = x
                    fraction = (phase % 15 + 1) / 15
                    x = starts[foot].item() + .12 * fraction
                    z = .15 * math.sin(math.pi * fraction)
                    supported = False
                elif behavior == "dragging":
                    x += speed * env.step_dt
                set_foot(env, foot, x, z, support=supported)
            robot.site_lin_vel_w[:] = (robot.site_pos_w - previous) / env.step_dt
            env.scene["elevation_contacts"].data.pos[:, ::4] = robot.site_pos_w
            collect_elevation_contacts(env)
            for key in keys:
                reward = env.cfg.rewards[key]
                function = term if key == "swing_clearance" else reward.func
                total += reward.weight * function(env, **reward.params).item() * env.step_dt
    return total


class ElevationRetentionTests(unittest.TestCase):
    def test_tracking_requires_new_forward_ground_and_positive_body_speed(self):
        env = movement_env(5)
        reward = move(env, [.004, .004, .004, 0., .004], [.2, .1, .3, .2, -.2])
        self.assertAlmostEqual(reward[0].item(), 1., places=4)
        self.assertGreater(reward[1].item(), 0.)
        self.assertLess(reward[1].item(), reward[0].item())
        self.assertLess(reward[2].item(), reward[0].item())
        self.assertEqual(reward[3].item(), 0.)
        self.assertEqual(reward[4].item(), 0.)

    def test_backtracking_and_velocity_without_displacement_cannot_repay_ground(self):
        env = movement_env()
        self.assertGreater(move(env, .004).item(), .99)
        self.assertEqual(move(env, -.004, -.2).item(), 0.)
        self.assertEqual(move(env, .004).item(), 0.)
        self.assertEqual(move(env, 0., .2).item(), 0.)
        self.assertGreater(move(env, .004).item(), .99)

    def test_velocity_reward_requires_two_supporting_feet(self):
        env = movement_env()
        env._elevation["support_count"][:] = 0.
        self.assertEqual(move(env, .004).item(), 0.)
        env._elevation["support_count"][:] = 1.
        self.assertEqual(move(env, .004).item(), 0.)
        env._elevation["support_count"][:] = 2.
        self.assertGreater(move(env, .004).item(), .99)

    def test_cached_reads_and_partial_reset_preserve_each_environment_frontier(self):
        env = movement_env(2)
        before = move(env, .004)
        frontier = env._elevation["forward_frontier"].clone()
        torch.testing.assert_close(elevation_forward_velocity_reward(env), before)
        torch.testing.assert_close(env._elevation["forward_frontier"], frontier)
        reset_elevation(env, torch.tensor([0]), env._elevation_settings, randomize=False)
        self.assertEqual(env._elevation["forward_progress"][0].item(), 0.)
        self.assertAlmostEqual(env._elevation["forward_frontier"][0].item(), 1.2)
        self.assertEqual(env._elevation["forward_frontier"][1].item(), frontier[1].item())
        reward = move(env, [.004, 0.])
        self.assertGreater(reward[0].item(), .99)
        self.assertEqual(reward[1].item(), 0.)

    def test_terminal_flags_mask_cached_positive_reward(self):
        for flag in ("failed", "complete", "invalid"):
            env = movement_env()
            self.assertGreater(move(env, .004).item(), 0.)
            env._elevation[flag][:] = True
            self.assertEqual(elevation_forward_velocity_reward(env).item(), 0.)
        reward, env, term, cfg = cycle(.15)
        self.assertGreater(reward, 0.)
        for flag in ("failed", "complete", "invalid"):
            term.gait["quality"][:] = 1.
            env._elevation[flag][:] = True
            self.assertEqual(elevation_gait_term(env, "quality").item(), 0.)
            env._elevation[flag][:] = False

    def test_nonfinite_motion_does_not_poison_frontiers_or_rewards(self):
        env = movement_env(2)
        frontier = env._elevation["forward_frontier"].clone()
        env.scene["robot"].data.root_link_pos_w[0, 0] = torch.nan
        reward = move(env, [0., .004], [torch.nan, .2])
        self.assertTrue(torch.isfinite(reward).all())
        self.assertEqual(reward[0].item(), 0.)
        self.assertEqual(env._elevation["forward_frontier"][0].item(), frontier[0].item())
        self.assertGreater(reward[1].item(), .99)
        env = movement_env()
        env._elevation["speed"][:] = torch.nan
        self.assertEqual(move(env, .004).item(), 0.)
        for std in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                elevation_forward_velocity_reward(env, std=std)
        env._elevation_settings = replace(env._elevation_settings, jump=True)
        with self.assertRaises(ValueError):
            elevation_forward_velocity_reward(env)

    def test_velocity_credit_is_invariant_to_tile_translation_and_bounded_per_second(self):
        totals = []
        for dt, shift in ((.01, 0.), (.02, 2.), (.04, 4.)):
            env = movement_env()
            env.step_dt = dt
            env._elevation["corner"][:, 0] += shift
            env.scene["robot"].data.root_link_pos_w[:, 0] += shift
            totals.append(sum(move(env, .2 * dt).item() * dt for _ in range(round(.4 / dt))))
        for total in totals:
            self.assertAlmostEqual(total, .4, places=4)

    def test_height_scores_distinguish_eight_twelve_and_fifteen_centimeters(self):
        values = [cycle(height)[0] for height in (.08, .12, .15, .20)]
        self.assertAlmostEqual(values[0] / values[2], (.08 / .15) ** 2, places=5)
        self.assertAlmostEqual(values[1] / values[2], .64, places=5)
        self.assertAlmostEqual(values[2], .08, places=5)
        self.assertAlmostEqual(values[3], values[2], places=6)
        for foot in range(4):
            self.assertAlmostEqual(cycle(.15, foot=foot)[0], .08, places=5)

    def test_splitting_distance_cannot_compensate_for_low_height(self):
        full = cycle(.15, progress=.08, stride=.08)[0]
        halves = 2 * cycle(.15, progress=.04, stride=.04)[0]
        low_halves = 2 * cycle(.08, progress=.04, stride=.04)[0]
        self.assertLessEqual(halves, full + 1.e-6)
        self.assertLess(low_halves, full)
        self.assertAlmostEqual(cycle(.15, progress=.16, stride=.16)[0], .16, places=5)
        self.assertLessEqual(cycle(.15, progress=1., stride=.16)[0], .16 + 1.e-6)

    def test_quality_threshold_uses_height_ratio_not_squared_score(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        set_foot(env, 0, 1.46, .12)
        for _ in range(5):
            tick(env, term, cfg)
        env.scene["robot"].data.root_link_pos_w[:, 0] += .08
        set_foot(env, 0, 1.5, 0., support=True)
        for _ in range(math.ceil(cfg.params["minimum_stance_time"] / env.step_dt)):
            tick(env, term, cfg)
        self.assertGreater(term.metrics["quality_steps_per_s"].item(), 0.)

    def test_supported_dragging_and_motionless_swings_do_not_earn_cycles(self):
        env, term, cfg = clearance_env()
        settle(env, term, cfg)
        set_foot(env, 0, 1.5, 0., support=True)
        env.scene["robot"].data.root_link_pos_w[:, 0] += .08
        for _ in range(5):
            tick(env, term, cfg)
            self.assertEqual(elevation_gait_term(env, "quality").item(), 0.)
        self.assertEqual(cycle(.15, progress=0.)[0], 0.)
        self.assertEqual(cycle(.15, stride=0.)[0], 0.)

    def test_walking_feedback_exceeds_standing_and_supported_dragging(self):
        walking = trajectory_return("walking")
        self.assertGreater(walking, trajectory_return("standing"))
        self.assertGreater(walking, trajectory_return("dragging"))

    def test_train_play_and_evaluation_keep_the_same_speed_and_time_limit(self):
        for play in (False, True):
            cfg = load_env_cfg("PointLeg-Elevation-v0", play=play)
            self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (.15, .25))
            self.assertEqual(cfg.episode_length_s, 60.)
            self.assertEqual(cfg.rewards["velocity"].weight, 1.)
            self.assertEqual(cfg.rewards["action"].weight, -.02)
            self.assertEqual(cfg.rewards["leg_joint_acc_l2"].weight, -1.e-7)
        for override in (None, 75.):
            options = ElevationEvaluateConfig(checkpoint_file="unused", episode_length_s=override)
            with patch("pointleg.evaluate_elevation.ManagerBasedRlEnv", side_effect=RuntimeError("capture config")) as create:
                with self.assertRaisesRegex(RuntimeError, "capture config"):
                    _run_batch("PointLeg-Elevation-v0", Path("unused"), options, 42,
                               evaluation_cases("PointLeg-Elevation-v0")[-1:], 1, "cpu")
            self.assertEqual(create.call_args.args[0].episode_length_s, 60. if override is None else override)
            self.assertEqual(options.forward_speed, .2)
        agent = load_rl_cfg("PointLeg-Elevation-v0")
        self.assertEqual(agent.algorithm.learning_rate, 3.e-4)
        self.assertEqual(agent.algorithm.schedule, "adaptive")
        self.assertEqual(agent.algorithm.entropy_coef, .003)

    def test_sixty_second_deadline_does_not_disable_short_stall_detection(self):
        env = movement_env()
        env.episode_length_buf[:] = round(35. / env.step_dt)
        self.assertGreater(move(env, .004).item(), 0.)
        self.assertFalse(env._elevation["failed"].any())
        env.episode_length_buf[:] = round(60. / env.step_dt)
        self.assertEqual(move(env, .004).item(), 0.)
        self.assertTrue(env._elevation["failed"].all())
        env = movement_env()
        for _ in range(205):
            move(env, 0., 0.)
        self.assertTrue(env._elevation["stalled"].all())


if __name__ == "__main__":
    unittest.main()
