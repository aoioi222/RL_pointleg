"""Check obstacle-triggered clearance and bounded recovery rewards."""

from types import SimpleNamespace
import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.obstacle_recovery import (
    FOOT_CENTER_SITES, ObstacleFootRecoveryReward, obstacle_recovery_metric,
)
from pointleg.robot_cfg import FOOT_BODY_NAMES


class Scene(dict):
    pass


def recovery_env():
    cfg = load_env_cfg("PointLeg-RoughMixed-v0")
    command = torch.tensor([[.25, 0., 0.], [-.25, 0., 0.], [.25, 0., 0.]])
    feet = torch.tensor([
        [.4, .15, .025], [.4, -.15, .025],
        [-.1, .15, .025], [-.1, -.15, .025],
    ]).repeat(3, 1, 1)
    robot = SimpleNamespace(
        find_sites=lambda name: ([FOOT_CENTER_SITES.index(name)], [name]),
        data=SimpleNamespace(
            site_pos_w=feet,
            root_link_pos_w=torch.tensor([[0., 0., .42]]).repeat(3, 1),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(3, 1),
        ),
    )
    force = torch.zeros(3, 4, 3)
    force[:, :, 2] = 40.
    contact = SimpleNamespace(
        primary_names=list(reversed(FOOT_BODY_NAMES)),
        data=SimpleNamespace(force=force, found=torch.ones(3, 4)),
    )
    hit = feet[:, :, None, :].repeat(1, 1, 3, 1)
    hit[:, :, :, 0] += command[:, 0, None, None].sign() * .04
    hit[:, :, :, 2] = 0.
    normals = torch.zeros_like(hit)
    normals[:, :, :, 2] = 1.
    scan = SimpleNamespace(data=SimpleNamespace(
        hit_pos_w=hit.reshape(3, 12, 3), normals_w=normals.reshape(3, 12, 3),
        distances=torch.full((3, 12), .35),
    ))
    scene = Scene(
        robot=robot, obstacle_foot_contact=contact, obstacle_foot_scan=scan,
    )
    scene.terrain = SimpleNamespace(
        cfg=cfg.scene.terrain, terrain_types=torch.tensor([2, 3, 4]),
    )
    env = SimpleNamespace(
        scene=scene, num_envs=3, device="cpu", step_dt=.02,
        common_step_counter=0, reset_buf=torch.zeros(3, dtype=torch.bool),
        command_manager=SimpleNamespace(get_command=lambda name: command),
        command=command,
        _mixed_obstacles={"peak_push": torch.zeros(3), "peak_tilt": torch.zeros(3)},
    )
    term_cfg = cfg.rewards["obstacle_foot_recovery"]
    return env, term_cfg.func(term_cfg, env), term_cfg


def step(env, term, cfg):
    env.common_step_counter += 1
    return term(env, **cfg.params).clone()


def block(env, foot=0):
    contact = env.scene["obstacle_foot_contact"].data
    contact.force[:, 3 - foot] = torch.tensor([20., 0., 0.])
    scan = env.scene["obstacle_foot_scan"].data.hit_pos_w.reshape(3, 4, 3, 3)
    scan[:, foot, :, 2] = .10


def arm(env, term, cfg, foot=0):
    block(env, foot)
    total = torch.zeros(3)
    for _ in range(4):
        total += step(env, term, cfg) * env.step_dt
    return total


def lift(env, term, cfg, foot=0, bottom_height=.12):
    env.scene["robot"].data.site_pos_w[:, foot, 2] = bottom_height + .025
    contact = env.scene["obstacle_foot_contact"].data
    contact.found[:, 3 - foot] = 0.
    contact.force[:, 3 - foot] = 0.
    return step(env, term, cfg)


def land(env, term, cfg, foot=0, duration=5):
    direction = env.command[:, 0].sign()
    robot = env.scene["robot"].data
    robot.site_pos_w[:, foot, 0] = term.start_foot_x[:, foot] + direction * .14
    robot.site_pos_w[:, foot, 2] = .125
    robot.root_link_pos_w[:, 0] = term.start_root_x[:, foot] + direction * .05
    contact = env.scene["obstacle_foot_contact"].data
    contact.found[:, 3 - foot] = 1.
    contact.force[:, 3 - foot] = torch.tensor([0., 0., 40.])
    return sum((step(env, term, cfg) * env.step_dt for _ in range(duration)), torch.zeros(3))


class ObstacleRecoveryTests(unittest.TestCase):
    def test_only_mixed_adds_privileged_sensors_and_keeps_policy_terms(self):
        mixed = load_env_cfg("PointLeg-RoughMixed-v0")
        rough = load_env_cfg("PointLeg-Rough-v0")
        flat = load_env_cfg("PointLeg-Flat-v0")
        sensors = {sensor.name: sensor for sensor in mixed.scene.sensors}
        self.assertEqual(sensors["obstacle_foot_contact"].reduce, "netforce")
        self.assertIsNone(sensors["obstacle_foot_contact"].secondary)
        scan = sensors["obstacle_foot_scan"]
        self.assertEqual(tuple(frame.name for frame in scan.frame), FOOT_CENTER_SITES)
        self.assertEqual(scan.include_geom_groups, (0,))
        self.assertEqual(scan.ray_alignment, "world")
        origins, directions = scan.pattern.generate_rays(None, "cpu")
        self.assertEqual(origins.shape, (45, 3))
        self.assertTrue((origins[:, 2] == .35).all())
        self.assertTrue((directions[:, 2] == -1.).all())
        for baseline in (rough, flat):
            self.assertNotIn("obstacle_foot_recovery", baseline.rewards)
            self.assertNotIn("obstacle_foot_scan", {s.name for s in baseline.scene.sensors})
        for group in ("actor", "critic"):
            self.assertEqual(list(mixed.observations[group].terms), list(rough.observations[group].terms))
        for name in ("knee_extension", "long_foot_air_time", "forward_foot_swing", "support_deficit"):
            self.assertEqual(mixed.rewards[name].weight, rough.rewards[name].weight)

    def test_collision_is_negative_and_sustained_stall_is_required(self):
        env, term, cfg = recovery_env()
        block(env)
        for _ in range(2):
            self.assertTrue((step(env, term, cfg) == 0.).all())
        reward = step(env, term, cfg)
        self.assertTrue((reward < 0.).all())
        self.assertTrue(term.active[:, 0].all())
        torch.testing.assert_close(term.target_z[:, 0], torch.full((3,), .12))
        self.assertTrue((obstacle_recovery_metric(env, "blocked_events_per_s") == 50.).all())

    def test_higher_recovery_reduces_deficit_but_extra_height_gives_no_bonus(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        rewards = []
        for height in (.01, .06, .10, .12, .20):
            lift(env, term, cfg, bottom_height=height)
            rewards.append(step(env, term, cfg))
        self.assertTrue(all((a <= b).all() for a, b in zip(rewards, rewards[1:])))
        self.assertTrue((rewards[-2] == 0.).all())
        self.assertTrue((rewards[-1] == 0.).all())
        self.assertTrue(term.lifted[:, 0].all())

    def test_loaded_foot_with_moderate_side_force_can_still_be_blocked(self):
        env, term, cfg = recovery_env()
        block(env)
        env.scene["obstacle_foot_contact"].data.force[:, 3] = torch.tensor([15., 0., 100.])
        for _ in range(4):
            reward = step(env, term, cfg)
        self.assertTrue(term.active[:, 0].all())
        self.assertTrue((reward < 0.).all())

    def test_lateral_wriggling_does_not_hide_a_forward_block(self):
        env, term, cfg = recovery_env()
        block(env)
        for _ in range(4):
            env.scene["robot"].data.site_pos_w[:, 0, 1] += .002
            step(env, term, cfg)
        self.assertTrue(term.active[:, 0].all())

    def test_clearance_uses_world_surface_height_and_physical_sphere_bottom(self):
        env, term, cfg = recovery_env()
        block(env)
        offset = torch.tensor([20., -8., 2.4])
        env.scene["robot"].data.site_pos_w += offset
        env.scene["robot"].data.root_link_pos_w += offset
        env.scene["obstacle_foot_scan"].data.hit_pos_w += offset
        for _ in range(4):
            step(env, term, cfg)
        torch.testing.assert_close(term.target_z[:, 0], torch.full((3,), 2.52))
        env.scene["robot"].data.site_pos_w[:, 0, 2] = 2.545
        env.scene["obstacle_foot_contact"].data.found[:, 3] = 0.
        env.scene["obstacle_foot_contact"].data.force[:, 3] = 0.
        step(env, term, cfg)
        self.assertTrue(term.lifted[:, 0].all())

    def test_clearance_is_twenty_mm_above_latched_top_not_world_ground(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        env.scene["obstacle_foot_scan"].data.hit_pos_w[:, :, 2] = 0.
        self.assertTrue((lift(env, term, cfg, bottom_height=.02) < 0.).all())
        self.assertFalse(term.lifted.any())
        torch.testing.assert_close(term.target_z[:, 0], torch.full((3,), .12))

    def test_bonus_requires_supported_forward_replant_and_is_paid_once(self):
        env, term, cfg = recovery_env()
        cost = arm(env, term, cfg)
        lift_reward = lift(env, term, cfg) * env.step_dt
        torch.testing.assert_close(lift_reward, torch.full((3,), .15))
        self.assertTrue((land(env, term, cfg, duration=4) == 0.).all())
        bonus = step(env, term, cfg) * env.step_dt
        torch.testing.assert_close(bonus, torch.full((3,), .10))
        self.assertTrue((cost + lift_reward + bonus < 0.).all())
        self.assertFalse(term.active.any())
        self.assertTrue((step(env, term, cfg) == 0.).all())

    def test_fore_and_hind_feet_map_correctly_despite_reversed_contact_order(self):
        for foot in range(4):
            env, term, cfg = recovery_env()
            arm(env, term, cfg, foot)
            self.assertTrue(term.active[:, foot].all())
            self.assertEqual(term.active.sum().item(), 3)
            lift(env, term, cfg, foot)
            torch.testing.assert_close(land(env, term, cfg, foot), torch.full((3,), .10))

    def test_body_lifting_without_relative_foot_lifting_cannot_earn_reward(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        env.scene["robot"].data.root_link_pos_w[:, 2] += .12
        self.assertTrue((lift(env, term, cfg) < 0.).all())
        self.assertFalse(term.lifted.any())
        self.assertTrue((land(env, term, cfg) <= 0.).all())

    def test_hanging_or_replanting_without_progress_cannot_earn_reward(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        lift(env, term, cfg)
        for _ in range(10):
            self.assertTrue((step(env, term, cfg) <= 0.).all())
        contact = env.scene["obstacle_foot_contact"].data
        contact.found[:, 3] = 1.
        contact.force[:, 3] = torch.tensor([0., 0., 40.])
        for _ in range(110):
            self.assertTrue((step(env, term, cfg) <= 0.).all())
        self.assertFalse(term.active.any())

    def test_leg_reach_without_body_progress_cannot_earn_reward(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        lift(env, term, cfg)
        land(env, term, cfg, duration=1)
        env.scene["robot"].data.root_link_pos_w[:, 0] = term.start_root_x[:, 0]
        for _ in range(8):
            self.assertTrue((step(env, term, cfg) == 0.).all())

    def test_support_may_return_after_gait_transition_without_permanent_invalidation(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        force = env.scene["obstacle_foot_contact"].data.force
        force[:, :, 2] = 0.
        self.assertTrue((lift(env, term, cfg) <= 0.).all())
        force[:, :, 2] = 40.
        lift(env, term, cfg)
        torch.testing.assert_close(land(env, term, cfg), torch.full((3,), .10))
        self.assertTrue(term.support_ok[:, 0].all())

    def test_pushing_tires_beyond_tolerance_blocks_recovery_bonus(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        lift(env, term, cfg)
        env._mixed_obstacles["peak_push"][2] = .06
        reward = land(env, term, cfg)
        torch.testing.assert_close(reward, torch.tensor([.10, .10, 0.]))

    def test_pressing_a_side_wall_is_not_a_supported_replant(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        lift(env, term, cfg)
        land(env, term, cfg, duration=1)
        env.scene["obstacle_foot_contact"].data.force[:, 3] = torch.tensor([40., 0., 10.])
        for _ in range(10):
            self.assertTrue((step(env, term, cfg) == 0.).all())

    def test_success_cannot_be_farmed_by_retreating_and_hitting_same_edge(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        lift(env, term, cfg)
        land(env, term, cfg)
        for _ in range(60):
            step(env, term, cfg)
        env.scene["robot"].data.site_pos_w[:, 0, 0] = term.start_foot_x[:, 0]
        env.scene["robot"].data.site_pos_w[:, 0, 2] = .025
        for _ in range(12):
            block(env)
            self.assertTrue((step(env, term, cfg) == 0.).all())
        self.assertFalse(term.active.any())

    def test_normal_ground_motion_no_command_and_plain_rough_do_not_arm(self):
        env, term, cfg = recovery_env()
        block(env)
        env.command[0] = 0.
        env.scene.terrain.terrain_types[1] = 0
        env.scene["obstacle_foot_scan"].data.hit_pos_w.reshape(3, 4, 3, 3)[2, 0, :, 2] = 0.
        for _ in range(8):
            self.assertTrue((step(env, term, cfg) == 0.).all())
        self.assertFalse(term.active.any())

    def test_invalid_rays_and_terminal_states_do_not_arm_or_create_nan(self):
        env, term, cfg = recovery_env()
        block(env)
        rays = env.scene["obstacle_foot_scan"].data
        rays.distances[0] = -1.
        rays.hit_pos_w[1] = torch.nan
        env.reset_buf[2] = True
        for _ in range(8):
            reward = step(env, term, cfg)
            self.assertTrue(torch.isfinite(reward).all())
            self.assertTrue((reward == 0.).all())

    def test_partial_reset_and_same_tick_calls_do_not_advance_other_worlds(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        remaining = term.remaining.clone()
        reward = term.cached_reward.clone()
        torch.testing.assert_close(term(env, **cfg.params), reward)
        torch.testing.assert_close(term.remaining, remaining)
        term.reset(torch.tensor([1]))
        self.assertFalse(term.active[1].any())
        torch.testing.assert_close(term.remaining[[0, 2]], remaining[[0, 2]])
        term(env, **cfg.params)
        torch.testing.assert_close(term.remaining[[0, 2]], remaining[[0, 2]])

    def test_recovery_bonus_cannot_exceed_blocking_cost(self):
        env, _, cfg = recovery_env()
        cfg.params["recovery_bonus"] = .20
        with self.assertRaises(ValueError):
            ObstacleFootRecoveryReward(cfg, env)

    def test_scraping_foot_is_detected_despite_forward_motion(self):
        env, term, cfg = recovery_env()
        block(env)
        for _ in range(5):
            env.scene["robot"].data.site_pos_w[:, 0, 0] += .004
            step(env, term, cfg)
        self.assertTrue(term.active[:, 0].all())
        self.assertTrue((term.metrics["stalled_feet"] == 0.).all())

    def test_upward_progress_is_dense_bounded_and_cannot_be_repaid_by_bobbing(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        first = lift(env, term, cfg, bottom_height=.06) * env.step_dt
        self.assertTrue((first > 0.).all())
        self.assertTrue((term.best_lift[:, 0] > 0.).all())
        self.assertTrue((lift(env, term, cfg, bottom_height=.03) <= 0.).all())
        self.assertTrue((lift(env, term, cfg, bottom_height=.06) <= 0.).all())
        remaining = lift(env, term, cfg, bottom_height=.12) * env.step_dt
        self.assertTrue((first + remaining <= cfg.params["lift_progress_bonus"]).all())
        self.assertTrue((lift(env, term, cfg, bottom_height=.20) == 0.).all())
        self.assertTrue((term.metrics["cleared_events_per_s"] == 0.).all())

    def test_clearance_metric_is_paid_once_at_first_actual_clearance(self):
        env, term, cfg = recovery_env()
        arm(env, term, cfg)
        lift(env, term, cfg)
        self.assertTrue((term.metrics["cleared_events_per_s"] == 50.).all())
        step(env, term, cfg)
        self.assertTrue((term.metrics["cleared_events_per_s"] == 0.).all())

    def test_swing_near_edge_receives_clearance_feedback_before_collision(self):
        env, term, cfg = recovery_env()
        scan = env.scene["obstacle_foot_scan"].data.hit_pos_w.reshape(3, 4, 3, 3)
        scan[:, 0, 1:, 2] = .10
        contact = env.scene["obstacle_foot_contact"].data
        contact.found[:, 3] = 0.
        contact.force[:, 3] = 0.
        low = step(env, term, cfg)
        self.assertTrue((low < 0.).all())
        self.assertFalse(term.active.any())
        self.assertTrue((term.metrics["approach_feet"] == 1.).all())
        env.scene["robot"].data.site_pos_w[:, 0, 2] = .145
        high = step(env, term, cfg)
        self.assertTrue((high == 0.).all())

    def test_level_ground_does_not_request_high_steps(self):
        env, term, cfg = recovery_env()
        env.scene["obstacle_foot_contact"].data.found[:, 3] = 0.
        env.scene["obstacle_foot_contact"].data.force[:, 3] = 0.
        for _ in range(10):
            self.assertTrue((step(env, term, cfg) == 0.).all())


if __name__ == "__main__":
    unittest.main()
