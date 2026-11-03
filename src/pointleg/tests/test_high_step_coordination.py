"""Check compact swing costs and real hip/knee motion without physics."""

from types import SimpleNamespace
import math
import unittest

import torch

from legacy_high_step_config import load_env_cfg
from pointleg.robot_cfg import LEG_NAMES, LEG_JOINT_NAMES, FOOT_BODY_NAMES
from pointleg.mdp.rewards import joint_pos_penalty


def coordination_env(reverse=False):
    cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["high_step_swing_retraction"]
    root = torch.tensor([[0., 0., .45]]).repeat(3, 1)
    hips = root[:, None, :] + torch.tensor([
        [.25, .15, 0.], [.25, -.15, 0.], [-.25, .15, 0.], [-.25, -.15, 0.],
    ])
    knees = hips + torch.tensor([0., 0., -.25])
    sites = hips + torch.tensor([.035, 0., -.425])
    q = torch.tensor([[0., .55, -1.125] * 4]).repeat(3, 1)
    body_names = [f"{leg}_hip_pitch_Link" for leg in LEG_NAMES]
    body_names += [f"{leg}_knee_Link" for leg in LEG_NAMES]
    site_names = [f"{leg}_foot_contact_site" for leg in LEG_NAMES]
    robot = SimpleNamespace(
        joint_names=list(LEG_JOINT_NAMES),
        find_bodies=lambda name: ([body_names.index(name)], [name]),
        find_sites=lambda name: ([site_names.index(name)], [name]),
        data=SimpleNamespace(
            root_link_pos_w=root, root_link_lin_vel_b=torch.zeros(3, 3),
            root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(3, 1),
            site_pos_w=sites, body_link_pos_w=torch.cat((hips, knees), dim=1),
            joint_pos=q.clone(), default_joint_pos=q.clone(),
        ),
    )
    force = torch.zeros(3, 4, 3)
    force[:, :, 2] = 80.
    names = list(reversed(FOOT_BODY_NAMES)) if reverse else list(FOOT_BODY_NAMES)
    data = SimpleNamespace(
        force=force, found=torch.ones(3, 4),
        current_contact_time=torch.full((3, 4), .2),
        current_air_time=torch.zeros(3, 4),
        last_contact_time=torch.full((3, 4), .2),
        last_air_time=torch.full((3, 4), .14),
    )
    sensor = SimpleNamespace(
        primary_names=names, cfg=SimpleNamespace(global_frame=True, reduce="none"),
        data=data,
        compute_first_air=lambda dt: (data.current_air_time > 0.) & (data.current_air_time <= dt),
        compute_first_contact=lambda dt: (data.current_contact_time > 0.) & (data.current_contact_time <= dt),
    )
    command = torch.tensor([[.2, 0., 0.]]).repeat(3, 1)
    env = SimpleNamespace(
        num_envs=3, device="cpu", step_dt=.02, extras={"log": {}},
        episode_length_buf=torch.full((3,), 100),
        command_manager=SimpleNamespace(get_command=lambda name: command),
        scene={"robot": robot, "feet_ground_contact": sensor,
               "base_height_sensor": SimpleNamespace(data=SimpleNamespace(
                   distances=torch.ones(3, 1), hit_pos_w=torch.zeros(3, 1, 3),
                   normals_w=torch.tensor([[[0., 0., 1.]]]).repeat(3, 1, 1),
               ))},
    )
    term = cfg.func(cfg, env)
    term(env, **cfg.params)
    return env, sensor, command, cfg, term


def begin_swing(env, sensor, cfg, term, feet=(0, 3)):
    ids = [term.contact_ids[i] for i in feet]
    sensor.data.found[:, ids] = 0.
    sensor.data.force[:, ids] = 0.
    sensor.data.current_contact_time[:, ids] = 0.
    sensor.data.current_air_time[:, ids] = env.step_dt
    term(env, **cfg.params)
    sensor.data.current_air_time[:, ids] = .12
    env.scene["robot"].data.site_pos_w[:, feet, 2] = .175


def end_swing(env, sensor, cfg, term, feet=(0, 3)):
    ids = [term.contact_ids[i] for i in feet]
    sensor.data.found[:, ids] = 1.
    sensor.data.force[:, ids, 2] = 80.
    sensor.data.current_air_time[:, ids] = 0.
    sensor.data.current_contact_time[:, ids] = env.step_dt
    env.scene["robot"].data.site_pos_w[:, feet, 2] = .025
    return term(env, **cfg.params)


class HighStepCoordinationTests(unittest.TestCase):
    def test_posture_cost_is_removed_for_motion_and_preserved_for_standing(self):
        env, _, command, _, _ = coordination_env()
        cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["static_joint_posture"]
        env.scene["robot"].data.joint_pos[:, [1, 4, 7, 10]] += .4
        torch.testing.assert_close(cfg.func(env, **cfg.params), torch.zeros(3))
        command[:] = 0.
        torch.testing.assert_close(cfg.func(env, **cfg.params), joint_pos_penalty(env, **cfg.params))
        self.assertGreater(cfg.func(env, **cfg.params).min().item(), 0.)

    def test_compact_high_lift_is_free_but_forward_and_backward_overreach_cost_equal(self):
        env, sensor, _, cfg, term = coordination_env()
        begin_swing(env, sensor, cfg, term)
        robot = env.scene["robot"].data
        hips = robot.body_link_pos_w[:, term.hip_ids]
        center = cfg.params["nominal_forward"]
        reach = cfg.params["fore_aft_margin"] + cfg.params["deviation_scale"]
        robot.site_pos_w[1, 0, 0] = hips[1, 0, 0] + center + reach
        robot.site_pos_w[2, 0, 0] = hips[2, 0, 0] + center - reach
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0., .625, .625]))
        robot.site_pos_w[:, :, 2] = .065
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))
        robot.site_pos_w[:, :, 2] = .175
        end_swing(env, sensor, cfg, term)
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))

    def test_joint_ranges_and_knee_motion_are_logged_for_completed_real_swings(self):
        env, sensor, _, cfg, term = coordination_env(reverse=True)
        begin_swing(env, sensor, cfg, term)
        robot = env.scene["robot"].data
        robot.joint_pos[:, [1, 10]] += .3
        robot.joint_pos[:, [2, 11]] -= .6
        robot.body_link_pos_w[:, [term.knee_ids[i] for i in (0, 3)], 2] += .1
        term(env, **cfg.params)
        end_swing(env, sensor, cfg, term)
        log = env.extras["log"]
        for leg in ("fl", "rr"):
            torch.testing.assert_close(log[f"Metrics/high_step_swing_hip_range_{leg}_deg"],
                                       torch.full((3,), math.degrees(.3)))
            torch.testing.assert_close(log[f"Metrics/high_step_swing_knee_range_{leg}_deg"],
                                       torch.full((3,), math.degrees(.6)))
            torch.testing.assert_close(log[f"Metrics/high_step_swing_knee_vertical_range_{leg}_m"],
                                       torch.full((3,), .1))
        self.assertNotIn("Metrics/high_step_swing_hip_range_fr_deg", log)

    def test_body_motion_does_not_look_like_knee_excursion(self):
        env, sensor, _, cfg, term = coordination_env()
        begin_swing(env, sensor, cfg, term)
        robot = env.scene["robot"].data
        translation = torch.tensor([1., 2., .1])
        robot.root_link_pos_w += translation
        robot.site_pos_w += translation
        robot.body_link_pos_w += translation
        term(env, **cfg.params)
        end_swing(env, sensor, cfg, term)
        value = env.extras["log"]["Metrics/high_step_swing_knee_vertical_range_fl_m"]
        torch.testing.assert_close(value, torch.zeros(3))

    def test_compact_cost_is_invariant_to_world_heading(self):
        env, sensor, _, cfg, term = coordination_env()
        begin_swing(env, sensor, cfg, term)
        robot = env.scene["robot"].data
        robot.site_pos_w[:, 0, 0] += .16
        before = term(env, **cfg.params)
        for name in ("root_link_pos_w", "site_pos_w", "body_link_pos_w"):
            value = getattr(robot, name)
            setattr(robot, name, torch.stack((-value[..., 1], value[..., 0], value[..., 2]), dim=-1))
        robot.root_link_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        torch.testing.assert_close(term(env, **cfg.params), before)

    def test_unanchored_air_motion_does_not_create_swing_events(self):
        env, sensor, _, cfg, term = coordination_env()
        term.reset()
        begin_swing(env, sensor, cfg, term)
        env.scene["robot"].data.site_pos_w[:, 0, 0] += .3
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))
        end_swing(env, sensor, cfg, term)
        self.assertNotIn("Metrics/high_step_swing_hip_range_fl_deg", env.extras["log"])

    def test_turning_standing_or_invalid_ground_clears_swing_tracking(self):
        env, sensor, command, cfg, term = coordination_env()
        begin_swing(env, sensor, cfg, term)
        self.assertTrue(term.tracking[:, 0].all())
        command[0] = 0.
        command[1, 2] = 1.
        env.scene["base_height_sensor"].data.distances[2] = -1.
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(3))
        self.assertFalse(term.tracking.any())

    def test_invalid_samples_and_partial_reset_do_not_leak_swing_history(self):
        env, sensor, _, cfg, term = coordination_env()
        begin_swing(env, sensor, cfg, term)
        env.scene["robot"].data.site_pos_w[0, 0] = torch.nan
        self.assertTrue(torch.isfinite(term(env, **cfg.params)).all())
        self.assertFalse(term.tracking[0, 0])
        term.reset(torch.tensor([1]))
        self.assertFalse(term.tracking[1].any())
        self.assertTrue(term.tracking[2, 0])
        term.reset()
        self.assertEqual(term.joint_max.abs().sum().item(), 0.)
        self.assertFalse(term.anchored.any())

    def test_changes_are_high_step_only_and_preserve_the_policy_interface(self):
        high = load_env_cfg("PointLeg-HighStep-v0")
        flat = load_env_cfg("PointLeg-Flat-v0")
        self.assertNotIn("high_step_swing_retraction", flat.rewards)
        self.assertIs(flat.rewards["static_joint_posture"].func, joint_pos_penalty)
        self.assertEqual(list(high.observations["actor"].terms), list(flat.observations["actor"].terms))
        self.assertEqual(list(high.observations["critic"].terms), list(flat.observations["critic"].terms))
        self.assertEqual(list(high.actions), list(flat.actions))
        self.assertGreater(high.rewards["high_step_cycle"].weight, 0.)


if __name__ == "__main__":
    unittest.main()
