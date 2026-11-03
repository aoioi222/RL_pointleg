"""Regress contact frames, gradual feedback and gait acceptance without physics."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from legacy_high_step_config import load_env_cfg
from pointleg.evaluate import evaluation_command
from pointleg.high_step_evaluation import gait_accepted, snapshot_high_step
from pointleg.mdp.high_step import HighStepCycleReward
from pointleg.mdp.walking_gait import DiagonalGaitPenalty, loaded_feet
from pointleg.mdp.elevation import elevation_cost
from test_high_step import cycle_env, run_cycle
from test_elevation_clearance import clearance_env, set_foot


class WalkingAuditTests(unittest.TestCase):
    def test_high_step_sensor_and_force_reader_share_the_world_frame(self):
        env, sensor, cfg, term = cycle_env()
        self.assertTrue(sensor.cfg.global_frame)
        self.assertTrue({"normal", "tangent"}.issubset(sensor.cfg.fields))
        sensor.data.force.zero_()
        sensor.data.force[:, :, 2] = 30.
        self.assertTrue(loaded_feet(sensor, term.contact_ids, 5.).all())
        sensor.data.force.zero_()
        sensor.data.force[:, :, 0] = 30.
        self.assertFalse(loaded_feet(sensor, term.contact_ids, 5.).any())
        sensor.cfg = deepcopy(sensor.cfg)
        sensor.cfg.global_frame = False
        self.assertTrue(loaded_feet(sensor, term.contact_ids, 5.).all())
        with self.assertRaisesRegex(ValueError, "world-frame"):
            HighStepCycleReward(cfg, env)

    def test_diagonal_contacts_beat_pacing_and_prolonged_four_foot_stance(self):
        env, sensor, _, _ = cycle_env()
        cfg = SimpleNamespace(params={"sensor_name": "feet_ground_contact"})
        term = DiagonalGaitPenalty(cfg, env)
        sensor.data.found[:] = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
        for _ in range(5):
            cost = term(env, **cfg.params)
        self.assertEqual(cost.tolist(), [0., 1., 1.])
        sensor.data.found[0] = torch.tensor([0, 1, 1, 0])
        self.assertEqual(float(term(env, **cfg.params)[0]), 0.)
        term.reset(torch.tensor([1]))
        self.assertEqual(float(term.samples[1]), 0.)
        env.commands.zero_()
        self.assertEqual(term(env, **cfg.params).tolist(), [0., 0., 0.])

    def test_small_real_lift_has_bounded_feedback_and_hovering_does_not_repeat_it(self):
        env, sensor, cfg, term = cycle_env()
        feet = [0, 3]
        sensor.data.found[:, feet] = 0.
        sensor.data.current_contact_time[:, feet] = 0.
        sensor.data.current_air_time[:, feet] = env.step_dt
        term(env, **cfg.params)
        sensor.data.current_air_time[:, feet] = .08
        env.scene["robot"].data.root_link_pos_w[:, 0] += .02
        env.scene["robot"].data.site_pos_w[:, feet, 2] += .03
        env.scene["robot"].data.site_pos_w[:, feet, 0] += .04
        term(env, **cfg.params)
        first = term.feedback["lift"].clone() * env.step_dt
        self.assertTrue((first > 0).all())
        self.assertTrue((first <= .04).all())
        term(env, **cfg.params)
        self.assertEqual(term.feedback["lift"].tolist(), [0., 0., 0.])

    def test_completed_low_cycles_are_observed_separately_from_qualified_cycles(self):
        env, sensor, cfg, term = cycle_env()
        self.assertTrue((run_cycle(env, sensor, cfg, term, height=.03)[1] > 0).all())
        self.assertEqual(term.stats["cycles"][0].tolist(), [1., 0., 0., 1.])
        self.assertEqual(float(term.stats["qualified"].sum()), 0.)
        term.reset(torch.tensor([0]))
        self.assertEqual(float(term.stats["cycles"][0].sum()), 0.)
        self.assertGreater(float(term.stats["cycles"][1].sum()), 0.)

    def test_invalid_cycles_remain_in_the_evaluation_denominator(self):
        env, sensor, cfg, term = cycle_env()
        phase_cfg = SimpleNamespace(params={"sensor_name": "feet_ground_contact"})
        phase = DiagonalGaitPenalty(phase_cfg, env)
        phase(env, **phase_cfg.params)
        env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: cfg)
        run_cycle(env, sensor, cfg, term, air_time=.4)
        run_cycle(env, sensor, cfg, term)
        summary = snapshot_high_step(env, 0)
        self.assertEqual(summary["feet"]["fl"]["cycles"], 2)
        self.assertEqual(summary["feet"]["fl"]["qualified_fraction"], .5)
        self.assertFalse(summary["accepted"])

    def test_elevation_diagonal_pairing_relaxes_on_the_actual_obstacle(self):
        env, _, _ = clearance_env(kind=0)
        sensor = env.scene["elevation_contacts"]
        sensor.primary_names = [f"{leg}_foot_Link" for leg in ("fl", "fr", "rl", "rr")]
        sensor.cfg.global_frame, sensor.cfg.reduce = True, "maxforce"
        env.command_manager = SimpleNamespace(get_command=lambda name: torch.tensor([[.2, 0., 0.]]))
        env.episode_length_buf[:] = 100
        for foot in range(4):
            set_foot(env, foot, 1.4, 0., support=foot in (0, 2))
        cfg = env.cfg.rewards["diagonal_gait"]
        term = cfg.func(cfg, env)
        self.assertGreater(float(term(env, **cfg.params)), 0.)
        env.scene["robot"].data.root_link_pos_w[:, 0] = env._elevation["corner"][:, 0] + 3.1
        self.assertEqual(float(term(env, **cfg.params)), 0.)

    def test_gait_acceptance_cannot_hide_one_bad_leg_or_wrong_coordination(self):
        summary = {"target_height": .17, "target_cycle": .4, "cycle_tolerance": .08,
                   "diagonal_error": 0., "feet": {
                       leg: {"cycles": 10, "qualified_fraction": 1., "height_m": .17,
                             "cycle_s": .4, "stride_m": .08}
                       for leg in ("fl", "fr", "rl", "rr")}}
        self.assertTrue(gait_accepted(summary))
        for field, value in (("cycles", 0), ("qualified_fraction", .2),
                             ("height_m", .02), ("stride_m", .14), ("cycle_s", .8)):
            bad = deepcopy(summary)
            bad["feet"]["rr"][field] = value
            self.assertFalse(gait_accepted(bad), field)
        summary["diagonal_error"] = .5
        self.assertFalse(gait_accepted(summary))

    def test_evaluation_commands_match_the_task_training_ranges(self):
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        speed = evaluation_command("PointLeg-HighStep-v0")[0]
        self.assertLessEqual(cfg.commands["twist"].ranges.lin_vel_x[0], speed)
        self.assertGreaterEqual(cfg.commands["twist"].ranges.lin_vel_x[1], speed)
        self.assertEqual(evaluation_command("PointLeg-Rough-v0"), (.4, 0., 0.))
        self.assertEqual(evaluation_command("PointLeg-RoughMixed-v0"), (.25, 0., 0.))

    def test_elevation_uses_the_commanded_speed_and_action_cost_does_not_flatten(self):
        env, _, _ = clearance_env(kind=0)
        env._elevation["updated_tick"] = env.common_step_counter
        env._elevation["speed"][:] = .5
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .5
        self.assertEqual(float(elevation_cost(env, "velocity")), 0.)
        env.action_manager.action[:] = 1.
        first = elevation_cost(env, "action").clone()
        env.action_manager.action[:] = 2.
        self.assertTrue((elevation_cost(env, "action") > first).all())

    def test_redundant_penalties_are_disabled_without_changing_the_actor_contract(self):
        high = load_env_cfg("PointLeg-HighStep-v0")
        self.assertNotIn("long_foot_air_time", high.rewards)
        self.assertNotIn("leg_motion_penalty", high.rewards)
        elevation = load_env_cfg("PointLeg-Elevation-v0")
        for key in ("center", "heading", "time", "command_aligned_progress", "command_progress_deficit"):
            self.assertEqual(elevation.rewards[key].weight, 0.)
        flat = load_env_cfg("PointLeg-Flat-v0")
        for cfg in (high, elevation):
            self.assertEqual(cfg.observations, flat.observations)
            self.assertEqual(cfg.actions, flat.actions)


if __name__ == "__main__":
    unittest.main()
