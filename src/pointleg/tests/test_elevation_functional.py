"""Functional-posture checks using geometry metadata and tensor trajectories."""

from dataclasses import asdict
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.evaluate_elevation import functional_episode_metrics, summarize_functional_metrics
from pointleg.mdp.elevation import collect_elevation_contacts, reset_elevation
from pointleg.mdp.elevation_gait import elevation_landing_margin, elevation_surface_frame
from pointleg.robot_cfg import LEG_NAMES
from pointleg.terrains.elevation import CENTER_Y, ELEVATION_TASK_TERRAINS
from test_elevation_clearance import clearance_env, set_foot, tick
from test_elevation_gait import settle, step


def safety_env(count=1, kind=5, level=0):
    env, cycle, cycle_cfg = clearance_env(count=count, kind=kind, level=level)
    env._sim_step_counter = 0
    robot = env.scene["robot"]
    robot.find_sites = lambda name, **kwargs: ([LEG_NAMES.index(name.split('_')[0])], [name])
    robot.find_bodies = lambda name: ([0], [name])
    robot.indexing = SimpleNamespace(geom_ids=torch.tensor([2, 8]), body_ids=torch.tensor([5]))
    host = SimpleNamespace(geom_contype=np.zeros(9, dtype=int), geom_conaffinity=np.ones(9, dtype=int),
                           geom_bodyid=np.zeros(9, dtype=int), geom_type=np.zeros(9, dtype=int),
                           geom_size=np.zeros((9, 3)))
    host.geom_bodyid[8], host.geom_type[8] = 5, mujoco.mjtGeom.mjGEOM_BOX
    host.geom_size[8] = (.18, .12, .07)
    env.sim = SimpleNamespace(mj_model=host)
    robot.data.geom_pose_w = torch.zeros(count, 2, 7)
    robot.data.geom_pose_w[..., 3] = 1.
    robot.data.geom_pose_w[:, 1, :3] = env._elevation["corner"] + torch.tensor([1.5, CENTER_Y, .25])
    body_cfg = env.cfg.rewards["body_clearance"]
    body = body_cfg.func(body_cfg, env)
    touch_cfg = env.cfg.rewards["touchdown_speed"]
    touch = touch_cfg.func(touch_cfg, env)
    return env, cycle, cycle_cfg, body, body_cfg, touch, touch_cfg


def physics(env):
    env._sim_step_counter += 1
    collect_elevation_contacts(env)


def landing_event(speed=.8, dt=.02):
    bundle = safety_env()
    env, _, _, _, _, touch, cfg = bundle
    env.step_dt = dt
    set_foot(env, 0, 1.5, .05)
    env.scene["robot"].data.site_lin_vel_w[:, 0, 2] = -speed
    physics(env)
    set_foot(env, 0, 1.5, 0., support=True)
    env.scene["robot"].data.site_lin_vel_w.zero_()
    physics(env)
    env.common_step_counter += 1
    return bundle, touch(env, **cfg.params).clone()


class FunctionalLiftTests(unittest.TestCase):
    def test_lower_ground_cannot_turn_a_small_lift_into_a_high_step(self):
        env, term, cfg = clearance_env(kind=1, level=4)
        for foot in range(4):
            set_foot(env, foot, 2.96, .72, support=True)
        settle(env, term, cfg)
        set_foot(env, 0, 3.04, .74)
        for _ in range(5):
            tick(env, term, cfg)
        self.assertAlmostEqual(term.peak[0, 0].item(), .02, places=5)
        self.assertAlmostEqual(term.terrain_peak[0, 0].item(), .14, places=5)
        env.scene["robot"].data.root_link_pos_w[:, 0] += .08
        set_foot(env, 0, 3.10, .60, support=True)
        settle(env, term, cfg, .12)
        self.assertAlmostEqual(term.metrics["true_lift_height"].item(), .02, places=5)
        self.assertAlmostEqual(term.metrics["swing_peak_height"].item(), .14, places=5)

    def test_crossing_upward_early_does_not_double_count_the_step_height(self):
        env, term, cfg = clearance_env(kind=0, level=4)
        for foot in range(4):
            set_foot(env, foot, 2.94, 0., support=True)
        settle(env, term, cfg)
        reward = step(env, term, cfg, 0, height=.15, stride=.16, landing_height=.12)
        self.assertGreater(reward, .015)
        self.assertAlmostEqual(term.metrics["true_lift_height"].item(), .15, places=5)
        self.assertAlmostEqual(term.metrics["swing_peak_height"].item(), .03, places=5)

    def test_slope_tangent_motion_is_not_active_lift(self):
        for kind in (2, 3):
            env, term, cfg = clearance_env(kind=kind, level=4)
            angle = float(env._elevation["angle"][0]) * (1 if kind == 2 else -1)
            z = .2 * math.tan(angle) if kind == 2 else 1.8 * math.tan(-angle)
            for foot in range(4):
                set_foot(env, foot, 3.2, z, support=True)
            tick(env, term, cfg)
            tangent, normal = elevation_surface_frame(env, torch.full((1, 4), 3.2))
            start = env.scene["robot"].data.site_pos_w[:, 0].clone()
            env.scene["robot"].data.site_pos_w[:, 0] = start + .10 * tangent[:, 0]
            set_foot(env, 0, float(env.scene["robot"].data.site_pos_w[0, 0, 0]),
                     float(env.scene["robot"].data.site_pos_w[0, 0, 2]))
            tick(env, term, cfg)
            self.assertLess(term.peak[0, 0].item(), 1.e-6)
            env.scene["robot"].data.site_pos_w[:, 0] += .15 * normal[:, 0]
            tick(env, term, cfg)
            self.assertAlmostEqual(term.peak[0, 0].item(), .15, places=5)

    def test_tread_center_improves_the_existing_cycle_without_new_progress(self):
        values = []
        for start in (3.02, 3.205):
            env, term, cfg = clearance_env(kind=0)
            for foot in range(4):
                set_foot(env, foot, start, .04, support=True)
            settle(env, term, cfg)
            values.append(step(env, term, cfg, 0, height=.19, landing_height=.04))
            self.assertEqual(term.landing_count.item(), 1.)
        self.assertGreater(values[0], values[1] * 1.4)
        self.assertGreater(values[1], 0.)

    def test_margin_includes_top_platform_and_final_floor_without_lateral_cost(self):
        env, _, _ = clearance_env(kind=1, level=4)
        margin = elevation_landing_margin(env, torch.tensor([[2.98, 3.02, 3.16, 3.30]]))
        torch.testing.assert_close(margin, torch.tensor([[.02, .02, .16, .02]]), atol=1.e-5, rtol=0.)
        end = env._elevation["step_surfaces"][0, -1, 1].item()
        self.assertGreater(elevation_landing_margin(env, torch.tensor([[end - .01]])).item(), .05)


class FunctionalBodyTests(unittest.TestCase):
    def test_actual_box_bottom_has_nine_samples_and_a_bounded_deadband(self):
        env, _, _, term, cfg, _, _ = safety_env(count=3)
        self.assertEqual(tuple(term.points.shape), (1, 9, 3))
        env.scene["robot"].data.geom_pose_w[:, 1, 2] = torch.tensor([.25, .16, .13])
        result = term(env, **cfg.params)
        torch.testing.assert_close(result, torch.tensor([0., .25, 1.]), atol=1.e-5, rtol=0.)
        self.assertEqual(term.samples.tolist(), [1., 1., 1.])

    def test_geometry_translation_and_slope_alignment_preserve_normal_clearance(self):
        env, _, _, term, cfg, _, _ = safety_env(kind=2, level=4)
        angle = float(env._elevation["angle"][0])
        pose = env.scene["robot"].data.geom_pose_w[:, 1]
        pose[:, :3] = torch.tensor([4., CENTER_Y, math.tan(angle) + .16 / math.cos(angle)])
        pose[:, 3:7] = torch.tensor([math.cos(angle / 2), 0., -math.sin(angle / 2), 0.])
        before = term(env, **cfg.params).clone()
        self.assertAlmostEqual(term.minimum.item(), .09, places=5)
        env._elevation["corner"][:, 2] += 2.
        pose[:, 2] += 2.
        env.common_step_counter += 1
        torch.testing.assert_close(term(env, **cfg.params), before, atol=1.e-5, rtol=0.)

    def test_cache_partial_reset_and_nonfinite_box_do_not_pollute_other_envs(self):
        env, _, _, term, cfg, _, _ = safety_env(count=2)
        term(env, **cfg.params)
        term(env, **cfg.params)
        self.assertEqual(term.samples.tolist(), [1., 1.])
        term.reset(torch.tensor([0]))
        env.scene["robot"].data.geom_pose_w[0, 1, 2] = torch.nan
        result = term(env, **cfg.params)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(term.samples.tolist(), [0., 1.])
        self.assertTrue(torch.isinf(term.episode_minimum[0]))

    def test_box_front_corners_detect_an_upcoming_tread_before_the_root(self):
        env, _, _, term, cfg, _, _ = safety_env(kind=0, level=4)
        env.scene["robot"].data.geom_pose_w[:, 1, :3] = torch.tensor([2.95, CENTER_Y, .25])
        result = term(env, **cfg.params)
        self.assertGreater(result.item(), .99)
        self.assertAlmostEqual(term.minimum.item(), .06, places=5)


class FunctionalTouchdownTests(unittest.TestCase):
    def test_uses_preimpact_velocity_and_pays_each_landing_once(self):
        bundle, cost = landing_event()
        env, _, _, _, _, term, cfg = bundle
        self.assertGreater(cost.item(), 4.9)
        self.assertEqual(term.count[0, 0].item(), 1.)
        collect_elevation_contacts(env)
        torch.testing.assert_close(term(env, **cfg.params), cost)
        self.assertEqual(term.count[0, 0].item(), 1.)
        for _ in range(4):
            physics(env)
        env.common_step_counter += 1
        self.assertEqual(term(env, **cfg.params).item(), 0.)

    def test_soft_landings_and_tangent_motion_have_no_impact_penalty(self):
        self.assertEqual(landing_event(.2)[1].item(), 0.)
        for kind in (2, 3):
            env, _, _, _, _, term, cfg = safety_env(kind=kind, level=4)
            angle = float(env._elevation["angle"][0]) * (1 if kind == 2 else -1)
            set_foot(env, 0, 3.5, .5)
            env.scene["robot"].data.site_lin_vel_w[:, 0] = torch.tensor([math.cos(angle), 0., math.sin(angle)])
            physics(env)
            set_foot(env, 0, 3.5, .5, support=True)
            env.scene["elevation_contacts"].data.normal[:, 0] = torch.tensor([math.sin(angle), 0., -math.cos(angle)])
            physics(env)
            env.common_step_counter += 1
            self.assertEqual(term(env, **cfg.params).item(), 0.)

    def test_event_integral_does_not_depend_on_control_period(self):
        integrals = []
        for dt in (.01, .02, .04):
            bundle, value = landing_event(dt=dt)
            integrals.append(value.item() * dt * bundle[-1].weight)
        self.assertAlmostEqual(integrals[0], integrals[1], places=6)
        self.assertAlmostEqual(integrals[1], integrals[2], places=6)

    def test_partial_reset_failure_and_invalid_velocity_mask_old_events(self):
        env, _, _, _, _, term, cfg = safety_env(count=2)
        set_foot(env, 0, 1.5, .1)
        env.scene["robot"].data.site_lin_vel_w[:, 0, 2] = -.8
        env.scene["robot"].data.site_lin_vel_w[0, 0, 2] = torch.nan
        physics(env)
        set_foot(env, 0, 1.5, 0., support=True)
        physics(env)
        self.assertEqual(term.count[:, 0].tolist(), [0., 1.])
        term.reset(torch.tensor([0]))
        env._elevation["failed"][1] = True
        env.common_step_counter += 1
        self.assertEqual(term(env, **cfg.params).sum().item(), 0.)
        self.assertEqual(term.pending_cost.sum().item(), 0.)

    def test_contact_that_ends_before_the_control_step_is_still_latched(self):
        bundle, _ = landing_event()
        env, _, _, _, _, term, cfg = bundle
        set_foot(env, 0, 1.5, .1)
        env.scene["robot"].data.site_lin_vel_w[:, 0, 2] = -.8
        physics(env)
        set_foot(env, 0, 1.5, 0., support=True)
        physics(env)
        set_foot(env, 0, 1.5, .1)
        physics(env)
        env.common_step_counter += 1
        self.assertGreater(term(env, **cfg.params).item(), 4.9)
        self.assertEqual(term.count[0, 0].item(), 2.)

    def test_reset_of_one_env_keeps_other_pending_landings(self):
        env, _, _, _, _, term, cfg = safety_env(count=2)
        set_foot(env, 0, 1.5, .1)
        env.scene["robot"].data.site_lin_vel_w[:, 0, 2] = -.8
        physics(env)
        set_foot(env, 0, 1.5, 0., support=True)
        physics(env)
        term.reset(torch.tensor([0]))
        env.common_step_counter += 1
        value = term(env, **cfg.params)
        self.assertEqual(value[0].item(), 0.)
        self.assertGreater(value[1].item(), 4.9)


class FunctionalIntegrationTests(unittest.TestCase):
    def test_single_nonfoot_collision_is_not_diluted_and_jump_uses_old_formula(self):
        env, _, _, _, _, _, _ = safety_env()
        sensor = env.scene["elevation_body_contacts"].data
        sensor.found[0, 0], sensor.force[0, 0, 0] = 1., 10.
        collect_elevation_contacts(env)
        self.assertAlmostEqual(env._elevation["contact_cost"].item(), .75)
        env._elevation_settings = SimpleNamespace(**{**asdict(env._elevation_settings), "jump": True})
        with patch("pointleg.mdp.elevation.elevation_foot_bottom", return_value=env.scene["robot"].data.site_pos_w):
            collect_elevation_contacts(env)
        self.assertAlmostEqual(env._elevation["contact_cost"].item(), .5)

    def test_all_walking_variants_share_new_config_without_changing_interfaces(self):
        base = load_env_cfg("PointLeg-Elevation-v0")
        for task in ELEVATION_TASK_TERRAINS:
            cfg = load_env_cfg(task)
            self.assertEqual(cfg.rewards["body_clearance"].weight, -.3)
            self.assertEqual(cfg.rewards["touchdown_speed"].weight, -.2)
            self.assertEqual(cfg.observations, base.observations)
            self.assertEqual(cfg.actions, base.actions)
        for task in ("PointLeg-Jump-v0", "PointLeg-HighStep-v0", "PointLeg-Rough-v0"):
            cfg = load_env_cfg(task)
            self.assertNotIn("body_clearance", cfg.rewards)
            self.assertNotIn("touchdown_speed", cfg.rewards)

    def test_reset_clears_safety_episode_state_and_report_uses_event_weights(self):
        bundle, _ = landing_event()
        env, cycle, _, body, body_cfg, touch, _ = bundle
        body(env, **body_cfg.params)
        record = functional_episode_metrics(env, 0)
        summary = summarize_functional_metrics([{"functional_posture": record}])
        self.assertAlmostEqual(summary["feet"]["fl"]["mean_touchdown_speed_m_s"], .8)
        self.assertIsNone(summary["feet"]["fr"]["mean_touchdown_speed_m_s"])
        self.assertIsNone(summary["mean_tread_edge_margin_m"])
        reset_elevation(env, torch.tensor([0]), env._elevation_settings, randomize=False)
        self.assertEqual(touch.count.sum().item(), 0.)
        self.assertEqual(body.samples.sum().item(), 0.)

    def test_report_weights_each_event_and_preserves_missing_samples(self):
        records = []
        for count, height, velocity in ((1., .15, .2), (3., .05, .6)):
            records.append({"functional_posture": {"feet": {"fl": {
                "swing_count": count, "true_lift_sum_m": count * height,
                "touchdown_count": count, "touchdown_speed_sum_m_s": count * velocity,
                "max_touchdown_speed_m_s": velocity}}, "tread_landing_count": count,
                "tread_margin_sum_m": count * .04, "edge_landing_count": count}})
        summary = summarize_functional_metrics(records)
        self.assertAlmostEqual(summary["feet"]["fl"]["mean_true_lift_m"], .075)
        self.assertAlmostEqual(summary["feet"]["fl"]["mean_touchdown_speed_m_s"], .5)
        self.assertEqual(summary["feet"]["fl"]["max_touchdown_speed_m_s"], .6)
        self.assertIsNone(summary["minimum_body_clearance_m"])


if __name__ == "__main__":
    unittest.main()
