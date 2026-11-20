"""Check ordered gait and competition geometry without physical simulation."""

import unittest
from xml.etree import ElementTree
from pathlib import Path
import math

import numpy as np
import torch
import mujoco

from mjlab.tasks.registry import load_env_cfg
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from pointleg.mdp import stepping_stones as mdp
from pointleg.mdp.fixed_stone_gait import EXIT, HOLD, FixedStoneGait, foothold_schedule
from pointleg.terrains.stepping_stones import SteppingStonesTerrainCfg, stone_centers, right_platform_start
from pointleg.tests.test_stepping_stones import fake_env, set_shin_load


class FixedGaitTests(unittest.TestCase):
    def setUp(self):
        self.gait = FixedStoneGait(2, "cpu")
        self.counts = torch.tensor([3, 6])
        self.stones = torch.tensor(np.stack([stone_centers(8, .1, size=.2)] * 2), dtype=torch.float32)
        self.exits = torch.tensor([right_platform_start(n, .1, size=.2) for n in self.counts])
        self.pos = torch.tensor([[[.72, .2, .2], [.72, -.2, .2], [.10, .2, .2], [.10, -.2, .2]]] * 2)
        self.surface = torch.zeros(2, 4, dtype=torch.long)
        self.support = torch.ones(2, 4, dtype=torch.bool)
        self.active = torch.ones(2, dtype=torch.bool)
        self.tick(16)

    def tick(self, steps=1):
        for _ in range(steps):
            self.gait.update(self.pos, self.surface, self.support, self.counts, self.stones, self.exits, self.active, .005)

    def complete_step(self):
        moving = self.gait.destinations(self.counts) != HOLD
        targets = self.gait.targets.clone()
        destinations = self.gait.destinations(self.counts).clone()
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = targets[moving]
        self.surface[moving] = destinations[moving]
        self.support[moving] = True
        self.tick(16)

    def test_six_stone_schedule_matches_requested_legs_and_stones(self):
        self.assertEqual(foothold_schedule(6), [
            (1, HOLD, HOLD, 0), (HOLD, 2, 1, HOLD),
            (3, HOLD, HOLD, HOLD), (HOLD, HOLD, HOLD, 2), (HOLD, 4, 3, HOLD),
            (5, HOLD, HOLD, HOLD), (HOLD, HOLD, HOLD, 4), (HOLD, 6, 5, HOLD),
            (EXIT, HOLD, HOLD, HOLD), (HOLD, HOLD, HOLD, 6), (HOLD, EXIT, EXIT, HOLD),
            (HOLD, HOLD, HOLD, EXIT),
        ])

    def test_right_rear_follows_the_even_stones_at_each_course_level(self):
        for count in range(3, 7):
            right_rear_targets = [row[3] for row in foothold_schedule(count) if row[3] != HOLD]
            self.assertEqual(right_rear_targets, [0, *range(2, count + 1, 2), EXIT])

    def test_short_courses_have_no_inactive_targets_and_finish_every_foot(self):
        for count in range(3, 7):
            surfaces = [0] * 4
            for row in foothold_schedule(count):
                for leg, target in enumerate(row):
                    self.assertTrue(target in (HOLD, 0, EXIT) or 1 <= target <= count)
                    if target != HOLD:
                        surfaces[leg] = target
            self.assertEqual(surfaces, [EXIT] * 4)

    def test_full_schedule_advances_once_per_confirmed_step_and_finishes(self):
        self.assertTrue(self.gait.ready.all())
        for _ in range(20):
            before = self.gait.completed.clone()
            self.complete_step()
            self.assertTrue(((self.gait.completed - before) <= 1).all())
            if self.gait.done.all():
                break
        self.assertTrue(self.gait.done.all())
        self.assertTrue((self.surface == EXIT).all())
        saved = self.gait.completed.clone()
        self.tick(30)
        torch.testing.assert_close(self.gait.completed, saved)

    def test_standing_cannot_advance_or_collect_progress(self):
        self.tick(500)
        self.assertEqual(self.gait.completed.sum().item(), 0)
        self.assertEqual(self.gait.pulses["progress"].sum().item(), 0)
        self.assertGreater(self.gait.costs["delay"].sum().item(), 0)

    def test_first_right_front_lowering_before_contact_does_not_charge_target_cost(self):
        self.complete_step()
        self.support[:, 1] = False
        self.surface[:, 1] = -1
        self.pos[:, 1, 0] = self.gait.targets[:, 1, 0] - .12
        self.pos[:, 1, 2] = .21
        self.tick()
        torch.testing.assert_close(self.gait.costs["target"], torch.zeros(2))
        self.pos[:, 1, 2] = .27
        self.tick(8)
        torch.testing.assert_close(self.gait.costs["target"], torch.zeros(2))

        self.pos[1, 1, 0] = self.gait.targets[1, 1, 0]
        self.pos[:, 1, 2] = .20
        self.tick()
        torch.testing.assert_close(self.gait.costs["target"], torch.zeros(2))
        self.assertEqual(self.gait.phase.tolist(), [1, 1])

    def test_paired_step_averages_errors_at_loaded_footholds(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2].masked_fill_(moving, .27)
        self.tick(8)
        self.pos[moving] = self.gait.targets[moving]
        self.pos[:, 0] = self.gait.origin[:, 0]
        self.pos[1, 3, 0] -= .12
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.surface[:, 0] = 0
        self.support[moving] = True
        self.tick()
        torch.testing.assert_close(self.gait.costs["target"], torch.tensor([.0025, .005]))

    def test_first_right_front_needs_clearance_through_the_stone_edge(self):
        self.complete_step()
        self.support[:, 1] = False
        self.surface[:, 1] = -1
        self.pos[:, 1] = self.gait.targets[:, 1]
        self.pos[:, 1, 0] -= .075
        self.pos[:, 1, 2] = torch.tensor([.235, .245])
        self.tick()
        self.assertGreater(self.gait.costs["clearance"][0].item(), 0.0)
        self.assertEqual(self.gait.costs["clearance"][1].item(), 0.0)

    def test_repeated_unfinished_swings_cannot_repeat_phase_progress_reward(self):
        for _ in range(6):
            self.support[:, 0] = False
            self.surface[:, 0] = -1
            self.pos[:, 0] = self.gait.targets[:, 0]
            self.pos[:, 0, 2] = .27
            self.tick(8)
            self.pos[:, 0] = self.gait.origin[:, 0]
            self.surface[:, 0] = 0
            self.support[:, 0] = True
            self.tick(16)
        self.assertTrue((self.gait.pulses["progress"] <= .5 + 1e-6).all())
        self.assertEqual(self.gait.phase.tolist(), [0, 0])

    def test_first_valid_frame_starts_when_one_rear_foot_never_loads(self):
        self.gait.reset(slice(None))
        self.support[:, 3] = False
        self.surface[:, 3] = -1
        self.pos[:, 3, 2] = .27
        self.tick(200)
        self.assertTrue(self.gait.ready.all())
        torch.testing.assert_close(self.gait.age, torch.ones(2), atol=2e-6, rtol=0)
        self.assertTrue((self.gait.costs["delay"] > 0).all())
        self.assertEqual(self.gait.phase.tolist(), [0, 0])
        observation = self.gait.observation(self.counts, torch.zeros(2, 3), torch.zeros(2))
        torch.testing.assert_close(observation[:, 12:16], torch.tensor([[1., 0., 0., 1.]] * 2))

    def test_individual_lift_receives_bounded_feedback_before_pair_overlap(self):
        self.support[:, 0] = False
        self.surface[:, 0] = -1
        self.pos[:, 0, 2] = .21
        self.pos[:, 0, 0] += .01
        self.tick(1)
        self.assertTrue((self.gait.pulses["progress"] > 0).all())
        self.assertFalse(self.gait.pair_lifted.any())
        self.pos[:, 0] = self.gait.targets[:, 0]
        self.pos[:, 0, 2] = .27
        self.tick(16)
        progress = self.gait.pulses["progress"].clone()
        self.tick(100)
        torch.testing.assert_close(self.gait.pulses["progress"], progress)
        self.assertTrue((progress <= .5).all())
        self.assertEqual(self.gait.phase.tolist(), [0, 0])

    def test_backward_swing_cannot_collect_lift_credit_and_is_penalized(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.pos[..., 0] = torch.where(moving, self.pos[..., 0] - .05, self.pos[..., 0])
        self.tick(120)
        torch.testing.assert_close(self.gait.pulses["progress"], torch.zeros(2))
        self.assertTrue((self.gait.costs["backward"] > .3).all())
        self.assertEqual(self.gait.phase.tolist(), [0, 0])

    def test_vertical_only_swing_cannot_collect_directional_lift_credit(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        torch.testing.assert_close(self.gait.pulses["progress"], torch.zeros(2))
        torch.testing.assert_close(self.gait.costs["backward"], torch.zeros(2))

    def test_planned_rearward_correction_toward_target_is_allowed(self):
        self.gait.targets[:, 3, 0] = .07
        self.support[:, 3] = False
        self.surface[:, 3] = -1
        self.pos[:, 3, 0] = .08
        self.pos[:, 3, 2] = .27
        self.tick(8)
        self.assertTrue((self.gait.pulses["progress"] > 0).all())
        torch.testing.assert_close(self.gait.costs["backward"], torch.zeros(2))

    def test_backward_foot_cost_excludes_support_feet(self):
        self.pos[:, 1, 0] -= .02
        self.tick(8)
        torch.testing.assert_close(self.gait.costs["backward"], torch.zeros(2))

    def test_one_short_rear_step_keeps_charging_for_the_missing_front_swing(self):
        self.support[:, 3] = False
        self.surface[:, 3] = -1
        self.pos[:, 3, 2] = .27
        self.tick(8)
        self.pos[:, 3] = self.gait.targets[:, 3]
        self.support[:, 3] = True
        self.surface[:, 3] = 0
        self.tick(192)
        self.assertTrue((self.gait.costs["delay"] > .1).all())
        self.assertEqual(self.gait.phase.tolist(), [0, 0])
        self.assertFalse(self.gait.pair_lifted.any())

    def test_nonfinite_start_and_failed_environment_do_not_initialize(self):
        self.gait.reset(slice(None))
        self.active[0] = False
        self.pos[1, 0, 0] = torch.nan
        self.tick(20)
        self.assertFalse(self.gait.ready.any())
        self.assertEqual(self.gait.age.sum().item(), 0)
        self.assertTrue(torch.isfinite(self.gait.targets).all())

    def test_wrong_support_leg_blocks_phase_and_rearms_after_recovery(self):
        self.support[:, 1] = False
        self.surface[:, 1] = -1
        self.complete_step()
        self.assertEqual(self.gait.phase.sum().item(), 0)
        self.assertGreater(self.gait.pulses["wrong_lift"].sum().item(), 0)
        self.assertEqual(self.gait.pulses["complete"].sum().item(), 0)

    def test_touchdown_without_airborne_and_weak_support_cannot_finish(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.tick(30)
        self.assertEqual(self.gait.phase.sum().item(), 0)
        self.support[moving] = False
        self.tick(30)
        self.assertEqual(self.gait.phase.sum().item(), 0)

    def test_landing_outside_target_or_skipping_stone_cannot_finish(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.surface[:, 0] = 3
        self.support[moving] = True
        self.tick(30)
        self.assertEqual(self.gait.phase.sum().item(), 0)
        self.assertGreater(self.gait.costs["target"].sum().item(), 0)

    def test_paired_step_requires_overlapping_swings(self):
        targets = self.gait.targets.clone()
        destinations = self.gait.destinations(self.counts).clone()
        for leg in (0, 3):
            self.support[:, leg] = False
            self.surface[:, leg] = -1
            self.pos[:, leg, 2] = .27
            self.tick(8)
            self.pos[:, leg] = targets[:, leg]
            self.surface[:, leg] = destinations[:, leg]
            self.support[:, leg] = True
            self.tick(16)
        self.assertFalse(self.gait.pair_lifted.any())
        self.assertEqual(self.gait.phase.sum().item(), 0)
        self.assertFalse(self.gait.correct_landing.any())

    def test_scraping_below_minimum_clearance_cannot_finish(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .21, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.support[moving] = True
        self.tick(16)
        self.assertEqual(self.gait.phase.sum().item(), 0)

    def test_seven_centimetres_clearance_is_required_for_swing_and_pair_qualification(self):
        moving = self.gait.destinations(self.counts) != HOLD
        destinations = self.gait.destinations(self.counts).clone()
        targets = self.gait.targets.clone()
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[moving] = targets[moving]
        heights = torch.tensor([.269, .27])[:, None].expand(-1, 4)
        self.pos[..., 2] = torch.where(moving, heights, self.pos[..., 2])
        self.tick(8)
        self.assertFalse(self.gait.lifted[0].any())
        self.assertTrue(self.gait.lifted[1, moving[1]].all())
        self.assertEqual(self.gait.pair_lifted.tolist(), [False, True])
        self.pos[moving] = targets[moving]
        self.surface[moving] = destinations[moving]
        self.support[moving] = True
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [0, 1])

    def test_pair_overlap_needs_both_swing_feet_seven_centimetres_high_together(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[:, 0, 2] = .27
        self.pos[:, 3, 2] = .26
        self.tick(8)
        self.pos[:, 0, 2] = .26
        self.pos[:, 3, 2] = .27
        self.tick(8)
        self.assertTrue(self.gait.lifted[moving].all())
        self.assertFalse(self.gait.pair_lifted.any())
        self.pos[:, [0, 3], 2] = .27
        self.tick(4)
        self.assertTrue(self.gait.pair_lifted.all())

    def test_wrong_landing_then_loaded_shift_cannot_reuse_swing_qualification(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[..., 2] = .2
        self.surface[moving] = 0
        self.support[moving] = True
        self.tick(16)
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [0, 0])
        self.assertFalse(self.gait.correct_landing.any())
        self.complete_step()
        self.assertEqual(self.gait.phase.tolist(), [1, 1])

    def test_peak_height_and_air_duration_must_belong_to_the_same_flight(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(4)
        self.pos[..., 2] = .2
        self.surface[moving] = 0
        self.support[moving] = True
        self.tick(2)
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .201, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.support[moving] = True
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [0, 0])

    def test_second_solo_flight_cannot_reuse_previous_pair_overlap(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.support[moving] = True
        self.tick(2)
        self.surface[:, 3] = -1
        self.support[:, 3] = False
        self.pos[:, 3, 2] = .27
        self.tick(8)
        self.pos[:, 3] = self.gait.targets[:, 3]
        self.surface[:, 3] = 0
        self.support[:, 3] = True
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [0, 0])
        self.assertFalse(self.gait.pair_lifted.any())

    def test_support_loss_cannot_contribute_to_legal_swing_duration(self):
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.support[:, 1] = False
        self.tick(8)
        self.support[:, 1] = True
        self.tick(4)
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.support[moving] = True
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [0, 0])

    def test_opening_platform_target_then_left_rear_shares_first_stone(self):
        self.pos[1, 1, 0] = .68
        self.gait._set_targets(self.active, self.pos, self.counts, self.stones, self.exits)
        torch.testing.assert_close(self.gait.targets[:, 3, 0], torch.tensor([.52, .48]))
        torch.testing.assert_close(self.gait.targets[:, 3, 1], self.pos[:, 3, 1])
        self.complete_step()
        self.assertEqual(self.gait.destinations(self.counts)[:, 2].tolist(), [1, 1])
        separation = torch.linalg.vector_norm(self.gait.targets[:, 2] - self.pos[:, 0], dim=-1)
        self.assertTrue((separation >= .05).all())
        offset = self.gait.targets[:, 2, :2] - self.stones[:, 0, :2]
        self.assertTrue((offset.abs() <= .04 + 1e-6).all())
        self.complete_step()
        self.assertEqual(self.gait.support_surface[:, 2].tolist(), [1, 1])

    def test_platform_landing_requires_actual_edge_margin(self):
        self.pos[:, 3, 1] = -.34
        self.gait._set_targets(self.active, self.pos, self.counts, self.stones, self.exits)
        torch.testing.assert_close(self.gait.targets[:, 3, 1], torch.tensor([-.34, -.34]))
        moving = self.gait.destinations(self.counts) != HOLD
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = self.gait.targets[moving]
        self.pos[:, 3, 1] = -.36
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.support[moving] = True
        self.tick(20)
        self.assertEqual(self.gait.phase.tolist(), [0, 0])

    def test_partial_reset_does_not_clear_other_environment(self):
        self.complete_step()
        self.gait.reset(torch.tensor([0]))
        self.assertFalse(self.gait.ready[0])
        self.assertTrue(self.gait.ready[1])
        self.assertEqual(self.gait.phase.tolist(), [0, 1])

    def test_shared_stone_target_leaves_space_for_the_holding_foot(self):
        for _ in range(3):
            self.complete_step()
        self.assertEqual(self.gait.destinations(self.counts)[:, 3].tolist(), [2, 2])
        separation = torch.linalg.vector_norm(self.gait.targets[:, 3] - self.pos[:, 1], dim=-1)
        self.assertTrue((separation >= .05).all())
        offset = self.gait.targets[:, 3, :2] - self.stones[:, 1, :2]
        self.assertTrue((offset.abs() <= .04 + 1e-6).all())
        self.complete_step()
        self.assertEqual(self.gait.phase.tolist(), [4, 4])

    def test_right_rear_leaves_second_stone_for_fourth_or_exit(self):
        for _ in range(6):
            self.complete_step()
        self.assertEqual(self.gait.destinations(self.counts)[:, 3].tolist(), [EXIT, 4])
        self.assertEqual(self.gait.support_surface[:, 3].tolist(), [2, 2])
        torch.testing.assert_close(self.gait.targets[1, 3, 0] - self.pos[1, 3, 0], torch.tensor(.60))
        separation = torch.linalg.vector_norm(self.gait.targets[1, 3] - self.pos[1, 1])
        self.assertGreaterEqual(separation.item(), .05)

    def test_left_front_support_gate_covers_all_right_rear_solo_steps(self):
        self.support[1, 0] = False
        self.surface[1, 0] = -1
        self.pos[1, 0, 2] = .27
        self.tick()
        self.assertFalse(self.gait.left_front_rear_transfer_lost_support.any())
        self.assertEqual(self.gait.left_front_rear_transfer_violations[1].item(), 0.)
        self.support[1, 0] = True
        self.surface[1, 0] = 0
        self.pos[1, 0] = self.gait.origin[1, 0]

        for number, (phase, start, target) in enumerate(((3, 0, 2), (6, 2, 4), (9, 4, 6), (11, 6, EXIT)), 1):
            for _ in range(phase - self.gait.phase[1].item()):
                self.complete_step()
            self.assertEqual(self.gait.phase[1].item(), phase)
            self.assertEqual(self.gait.support_surface[1, 3].item(), start)
            self.assertEqual(self.gait.destinations(self.counts)[1, 3].item(), target)
            front_position = self.pos[1, 0].clone()
            front_surface = self.surface[1, 0].clone()
            self.gait.clear_control(self.active)
            self.support[1, 0] = False
            self.surface[1, 0] = -1
            self.pos[1, 0, 2] += .07
            self.tick()
            self.assertEqual(self.gait.left_front_rear_transfer_lost_support.tolist(), [False, True])
            self.assertEqual(self.gait.left_front_rear_transfer_violations[1].item(), number)
            self.support[1, 0] = True
            self.surface[1, 0] = front_surface
            self.pos[1, 0] = front_position
            self.tick()
            self.gait.clear_control(self.active)
            self.assertFalse(self.gait.left_front_rear_transfer_lost_support.any())

    def test_overlapping_feet_cannot_complete_a_shared_stone_step(self):
        for _ in range(3):
            self.complete_step()
        moving = self.gait.destinations(self.counts) != HOLD
        front_anchor = self.gait.anchor[:, 1].clone()
        self.support[moving] = False
        self.surface[moving] = -1
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = self.gait.targets[moving]
        self.surface[moving] = self.gait.destinations(self.counts)[moving]
        self.support[moving] = True
        direction = self.gait.targets[:, 3] - front_anchor
        self.pos[:, 1] = front_anchor + .025 * direction / torch.linalg.vector_norm(direction, dim=-1)[:, None]
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [3, 3])
        self.assertFalse(self.gait.correct_landing.any())
        self.pos[:, 1] = front_anchor
        self.complete_step()
        self.assertEqual(self.gait.phase.tolist(), [4, 4])

    def test_failed_or_nonfinite_environment_cannot_earn_or_advance(self):
        self.active[0] = False
        self.pos[1, 0, 0] = torch.nan
        self.tick(30)
        self.assertEqual(self.gait.phase.sum().item(), 0)
        self.assertEqual(self.gait.pulses["progress"].sum().item(), 0)
        self.assertTrue(torch.isfinite(self.gait.costs["target"]).all())

    def test_phase_observation_has_targets_masks_phase_and_age(self):
        obs = self.gait.observation(self.counts, torch.zeros(2, 3), torch.zeros(2))
        self.assertEqual(obs.shape, (2, 18))
        torch.testing.assert_close(obs[:, 12:16], torch.tensor([[1., 0., 0., 1.]] * 2))


class CompetitionCurriculumTests(unittest.TestCase):
    def test_competition_curriculum_is_three_to_six_with_fixed_spacing(self):
        cfg = SteppingStonesTerrainCfg(size=(8., 8.), competition=True)
        for level in range(4):
            count, gap = cfg.count_and_gap(level / 3)
            self.assertEqual(count, level + 3)
            self.assertEqual(gap, .10)
            centers = stone_centers(count, gap, size=cfg.stone_size)
            np.testing.assert_allclose(np.diff(centers[:, 0]), .30)
            np.testing.assert_allclose(np.diff(centers[::2, 0]), .60)
        self.assertAlmostEqual(right_platform_start(6, .1, size=.2), 2.7)

    def test_physical_course_count_and_final_scene_match_competition_xml(self):
        cfg = load_env_cfg("PointLeg-SteppingStones-v0").scene.terrain.terrain_generator
        self.assertEqual((cfg.num_rows, cfg.num_cols), (4, 1))
        terrain = cfg.sub_terrains["competition_stones"]
        for level in range(4):
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name="terrain")
            output = terrain.function(level / 3, spec, np.random.default_rng(1))
            self.assertEqual(len(output.geometries), level + 6)
        scene = Path(__file__).resolve().parents[3] / "sim2sim/terrain/scene_terrain.xml"
        body = ElementTree.parse(scene).find(".//body[@name='stepping_stones']")
        expected = [body.find("geom[@name='stones_entry_platform']")]
        expected += sorted([geom for geom in body.findall("geom") if geom.attrib["name"].startswith("stone_")],
                           key=lambda geom: float(geom.attrib["pos"].split()[0]))
        expected.append(body.find("geom[@name='stones_exit_platform']"))
        for generated, reference in zip(output.geometries[1:], expected, strict=True):
            center = np.asarray(generated.geom.pos).copy()
            center[:2] -= output.origin[:2]
            np.testing.assert_allclose(center, np.fromstring(reference.attrib["pos"], sep=" "), atol=1e-12)
            np.testing.assert_allclose(generated.geom.size, np.fromstring(reference.attrib["size"], sep=" "))


class FixedGaitIntegrationTests(unittest.TestCase):
    def test_foot_side_collision_charges_force_but_excludes_top_and_ground(self):
        env = fake_env(4)
        mdp.reset_stepping_stones(env, None)
        sensor = env.scene["stepping_foot_contact"].data
        sensor.found[:, 1] = 1.0
        sensor.pos[:, 1] = torch.tensor([[.80, 0., .15], [.75, 0., .20], [.8, 0., 0.], [.80, 0., .15]])
        sensor.normal[[0, 2, 3], 1] = torch.tensor([1., 0., 0.])
        sensor.force[:, 1, 0] = torch.tensor([20., 20., 20., 2.])

        mdp.collect_stepping_stones_substep(env)
        torch.testing.assert_close(mdp.stepping_foot_side_penalty(env), torch.tensor([.25, 0., 0., 0.]))
        self.assertEqual(mdp._state(env)["legal_support"][:, 1].tolist(), [False, True, False, False])

    def setUp(self):
        self.env = fake_env(2)
        self.env.cfg = load_env_cfg("PointLeg-SteppingStones-v0")
        self.env.scene.terrain.cfg = self.env.cfg.scene.terrain
        self.env.scene.terrain.terrain_levels[:] = torch.tensor([0, 3])
        mdp.reset_stepping_stones(self.env, None)
        self.state = self.env._stepping_stones
        self.gait = self.state["fixed_gait"]
        self.pos = torch.tensor([[[.72, .2, .2], [.72, -.2, .2], [.10, .2, .2], [.10, -.2, .2]]] * 2)
        self.found = torch.ones(2, 4, dtype=torch.bool)
        self.substeps = 0
        self.tick(16)

    def tick(self, steps):
        for _ in range(steps):
            self.env.common_step_counter = self.substeps // 4
            sensor = self.env.scene["stepping_foot_contact"].data
            sensor.pos.copy_(self.pos)
            sensor.found.copy_(self.found)
            self.env.scene["robot"].data.site_pos_w.copy_(self.pos)
            mdp.collect_stepping_stones_substep(self.env)
            self.substeps += 1

    def swing(self):
        moving = self.gait.destinations(self.state["active_count"]) != HOLD
        targets = self.gait.targets.clone()
        self.found[moving] = False
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[moving] = targets[moving]
        self.found[moving] = True
        self.tick(16)

    def test_real_collector_credits_only_ordered_stable_landings(self):
        self.assertEqual(self.state["active_count"].tolist(), [3, 6])
        self.assertTrue(self.gait.ready.all())
        self.swing()
        self.assertEqual(self.gait.phase.tolist(), [1, 1])
        self.assertTrue(self.state["visited"][:, 0].all())
        self.assertFalse(self.state["visited"][:, 1:].any())
        self.assertEqual(mdp.fixed_gait_observation(self.env).shape, (2, 18))

    def test_body_sink_penalty_covers_stationary_root_drop_and_nose_dive_only_during_traversal(self):
        env = self.env
        robot = env.scene["robot"].data
        term = env.cfg.rewards["stepping_body_sink"]
        self.assertEqual(term.weight, -8.0)
        self.assertEqual(env.cfg.metrics["stepping_traverse_peak_body_sink_m"].reduce, "last")
        robot.root_link_lin_vel_w.zero_()
        robot.root_link_pos_w[:, 2] -= torch.tensor([.01, .05])
        torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0., .25]))
        self.tick(4)
        torch.testing.assert_close(mdp.stepping_metric(env, "traverse_peak_body_sink_m"), torch.tensor([.01, .05]))

        robot.root_link_pos_w[0, 2] = self.state["startup_spawn_z"][0]
        robot.root_link_quat_w[0] = torch.tensor([math.cos(.1), 0., math.sin(.1), 0.])
        self.state["stage"][1] = 3
        cost = term.func(env, **term.params)
        self.assertGreater(cost[0].item(), .5)
        self.assertEqual(cost[1].item(), 0.0)
        env.reset_terminated = torch.tensor([True, False])
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2))

    def test_left_front_hover_does_not_pay_landing_or_forward_speed_before_contact(self):
        for _ in range(2):
            self.swing()
        self.assertEqual(self.gait.phase.tolist(), [2, 2])
        self.env.scene["robot"].data.root_link_lin_vel_w[:, 0] = .25
        self.found[:, 0] = False
        self.pos[:, 0] = self.gait.targets[:, 0]
        self.pos[:, 0, 2] = .27
        self.pos[1, 0, 0] -= .10
        self.tick(52)
        self.assertEqual(self.gait.phase.tolist(), [2, 2])
        self.assertFalse(self.gait.correct_landing.any())
        torch.testing.assert_close(self.gait.best_progress[0], torch.tensor(.75))
        torch.testing.assert_close(mdp.forward_velocity_reward(self.env), torch.tensor([0., 1.]))
        delay = self.env.cfg.rewards["stepping_gait_delay"]
        torch.testing.assert_close(delay.func(self.env, **delay.params), torch.tensor([1., 0.]))

        self.pos[0, 0] = self.gait.targets[0, 0]
        self.found[0, 0] = True
        self.tick(15)
        self.assertEqual(self.gait.phase.tolist(), [2, 2])
        torch.testing.assert_close(self.gait.best_progress[0], torch.tensor(.75))
        self.tick(1)
        self.assertEqual(self.gait.phase.tolist(), [3, 2])
        torch.testing.assert_close(self.gait.pulses["progress"][0], torch.tensor(.25))

    def test_right_rear_early_lift_leaves_planted_left_front_in_place_but_blocks_handoff(self):
        for _ in range(2):
            self.swing()
        self.found[:, 0] = False
        self.pos[:, 0] = self.gait.targets[:, 0]
        self.pos[:, 0, 2] = .27
        self.tick(8)
        self.pos[0, 0] = self.gait.targets[0, 0]
        self.found[0, 0] = True
        self.found[:, 3] = False
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [2, 2])
        self.assertEqual(self.gait.lifted[:, 0].tolist(), [True, False])
        self.assertFalse(self.gait.correct_landing.any())
        self.found[:, 3] = True
        self.tick(16)
        self.assertEqual(self.gait.phase.tolist(), [3, 2])

    def test_first_policy_observation_has_targets_with_all_contact_loads_missing(self):
        self.gait.reset(slice(None))
        self.env.scene["stepping_foot_contact"].data.found.zero_()
        self.env.scene["robot"].data.site_pos_w.copy_(self.pos)
        obs = mdp.fixed_gait_observation(self.env)
        self.assertTrue(self.gait.ready.all())
        torch.testing.assert_close(obs[:, 12:16], torch.tensor([[1., 0., 0., 1.]] * 2))
        torch.testing.assert_close(self.gait.targets[:, 0], self.state["stone_centers"][:, 0])
        self.assertEqual(self.gait.age.sum().item(), 0)
        self.assertEqual(self.gait.completed.sum().item(), 0)

    def test_foot_row_alignment_uses_terrain_rows_and_penalizes_either_side_during_swing(self):
        env = self.env
        robot = env.scene["robot"].data
        env.scene.env_origins[:, 1] = torch.tensor([3., -2.])
        self.state["stone_centers"][0, 0, 1] = .23
        self.state["stone_centers"][1, 1, 1] = -.18
        row_y = torch.tensor([[.23, -.20, .23, -.20], [.20, -.18, .20, -.18]])
        robot.site_pos_w[..., 1] = row_y + env.scene.env_origins[:, 1:2]
        robot.site_pos_w[0, 2, 2] += .07
        robot.site_pos_w[1, 3, 2] += .07
        term = env.cfg.rewards["stepping_foot_row_alignment"]
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2))

        robot.site_pos_w[0, 2, 1] += .04
        robot.site_pos_w[1, 3, 1] -= .04
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2))
        robot.site_pos_w[0, 2, 1] += .03
        robot.site_pos_w[1, 3, 1] -= .03
        torch.testing.assert_close(term.func(env, **term.params) * term.weight, torch.full((2,), -1.0))

        self.state["stage"][0] = 3
        env.reset_terminated = torch.tensor([False, True])
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2))

    def test_knee_reversal_penalizes_a_single_substep_without_terminating(self):
        env = self.env
        robot = env.scene["robot"].data
        knees = self.state["knee_ids"]
        robot.joint_pos_limits = torch.tensor([-2.65, 2.65]).expand(2, 12, 2)
        robot.joint_pos[0, knees[3]] = math.radians(5.)
        robot.joint_pos[1, knees] = -math.radians(5.)
        self.tick(1)
        term = env.cfg.rewards["stepping_knee_reverse"]
        torch.testing.assert_close(term.func(env), torch.tensor([env.physics_dt / env.step_dt, 0.]))
        self.assertLess(term.weight, 0.0)
        joint_limits = env.cfg.terminations["joint_pos_out_of_limits"]
        self.assertFalse(joint_limits.func(env, **joint_limits.params).any())
        self.assertFalse(mdp.stepping_failed(env).any())

        robot.joint_pos[0, knees[3]] = 0.0
        self.tick(3)
        torch.testing.assert_close(term.func(env), torch.tensor([.25, 0.]))
        self.tick(1)
        torch.testing.assert_close(term.func(env), torch.zeros(2))

    def test_only_the_reverse_side_of_each_knee_is_exempt_from_joint_limit_reset(self):
        env = self.env
        robot = env.scene["robot"].data
        robot.joint_pos_limits = torch.tensor([-2.65, 2.65]).expand(2, 12, 2)
        joint_limits = env.cfg.terminations["joint_pos_out_of_limits"]
        for knee in self.state["knee_ids"]:
            robot.joint_pos[:, knee] = torch.tensor([2.64, -2.64])
            self.assertEqual(joint_limits.func(env, **joint_limits.params).tolist(), [False, True])
            robot.joint_pos[:, knee] = 0.0

        robot.joint_pos[0, self.state["knee_ids"][3]] = 2.64
        robot.joint_pos[:, self.state["hip_abduction_ids"][0]] = torch.tensor([2.64, -2.64])
        self.assertTrue(joint_limits.func(env, **joint_limits.params).all())

    def test_metrics_distinguish_active_first_phase_and_missing_pair(self):
        torch.testing.assert_close(mdp.stepping_metric(self.env, "gait_ready"), torch.ones(2))
        torch.testing.assert_close(mdp.stepping_metric(self.env, "gait_lifted_fraction"), torch.zeros(2))
        torch.testing.assert_close(mdp.stepping_metric(self.env, "gait_pair_ready"), torch.zeros(2))

    def test_leg_clearance_also_detects_two_rear_feet_overlapping(self):
        robot = self.env.scene["robot"].data
        robot.site_pos_w.copy_(self.pos)
        robot.site_pos_w[:, 3] = robot.site_pos_w[:, 2] + torch.tensor([.02, 0., 0.])
        robot.body_link_pos_w.copy_(robot.site_pos_w + torch.tensor([0., 0., .3]))
        self.assertTrue((mdp.stepping_leg_clearance_penalty(self.env, clearance=.05) > .1).all())

    def test_task_observation_columns_extend_actor_and_critic_at_the_end(self):
        base = load_env_cfg("PointLeg-Flat-v0").observations
        for group, expected in (("actor", 78), ("critic", 74)):
            terms = self.env.cfg.observations[group].terms
            self.assertEqual(list(terms)[-1], "fixed_gait")
            widths = [term.func(self.env, **term.params).shape[1]
                      for name, term in terms.items() if name not in base[group].terms]
            self.assertEqual(sum(widths), expected)

    def test_reward_manager_integrates_event_once_and_failure_wins(self):
        terms = {name: self.env.cfg.rewards[name] for name in ("stepping_gait_complete", "stepping_cover")}
        manager = RewardManager(terms, self.env)
        self.swing()
        torch.testing.assert_close(manager.compute(.02), torch.tensor([8., 6.]))
        self.env.reset_terminated = torch.ones(2, dtype=torch.bool)
        torch.testing.assert_close(mdp.fixed_gait_reward(self.env, "complete", event=True), torch.zeros(2))
        self.env.reset_terminated.zero_()
        self.tick(4)
        torch.testing.assert_close(manager.compute(.02), torch.zeros(2))

    def test_backward_body_velocity_cost_uses_course_direction_and_exempts_exit(self):
        term = self.env.cfg.rewards["stepping_backward_velocity"]
        self.env.scene["robot"].data.root_link_lin_vel_w[:, 0] = torch.tensor([.2, -.23])
        torch.testing.assert_close(term.func(self.env, **term.params), torch.tensor([0., 1.]))
        self.state["stage"][:] = 3
        torch.testing.assert_close(term.func(self.env, **term.params), torch.zeros(2))

    def test_constant_forward_velocity_reward_restored_with_stationary_and_failure_masks(self):
        term = self.env.cfg.rewards["stepping_forward_velocity"]
        self.env.command_manager.get_command = lambda name: torch.tensor([[.35, 0., 0.]] * 2)
        self.env.scene["robot"].data.root_link_lin_vel_w[:, 0] = torch.tensor([.35, 0.])
        torch.testing.assert_close(term.func(self.env) * term.weight, torch.tensor([5., 0.]))
        self.state["shin_support_this_step"][0] = True
        torch.testing.assert_close(term.func(self.env), torch.zeros(2))
        self.state["shin_support_this_step"].zero_()
        self.state["stage"][0] = 3
        self.state["failure"][1] = True
        self.env.scene["robot"].data.root_link_lin_vel_w[:, 0] = .35
        torch.testing.assert_close(term.func(self.env), torch.zeros(2))

    def test_rear_transfer_losing_left_front_support_blocks_only_positive_speed_for_the_control_step(self):
        for _ in range(3):
            self.swing()
        self.assertEqual(self.gait.phase.tolist(), [3, 3])
        self.env.scene["robot"].data.root_link_lin_vel_w[:, 0] = .25
        self.found[:, 3] = False
        self.pos[:, 3] = self.gait.targets[:, 3]
        self.pos[:, 3, 2] = .27
        self.tick(8)
        torch.testing.assert_close(mdp.forward_velocity_reward(self.env), torch.ones(2))
        self.found[0, 0] = False
        self.tick(1)
        torch.testing.assert_close(mdp.forward_velocity_reward(self.env), torch.tensor([0., 1.]))
        torch.testing.assert_close(
            mdp.stepping_metric(self.env, "gait_left_front_rear_transfer_violations"), torch.tensor([1., 0.])
        )

        robot = self.env.scene["robot"].data
        robot.root_link_lin_vel_w[0, 0] = -.125
        torch.testing.assert_close(mdp.forward_velocity_reward(self.env), torch.tensor([-1.25, 1.]))
        robot.root_link_lin_vel_w[0, 0] = .25
        self.found[0, 0] = True
        self.tick(3)
        self.assertTrue(self.gait.left_front_rear_transfer_lost_support[0])
        torch.testing.assert_close(mdp.forward_velocity_reward(self.env), torch.tensor([0., 1.]))
        self.tick(1)
        self.assertFalse(self.gait.left_front_rear_transfer_lost_support[0])
        self.tick(4)
        torch.testing.assert_close(mdp.forward_velocity_reward(self.env), torch.ones(2))

    def test_body_stall_cost_ramps_and_body_rocking_cannot_restart_grace(self):
        term = self.env.cfg.rewards["stepping_body_stall"]
        robot = self.env.scene["robot"].data
        start = self.state["stall_anchor_x"].clone()

        def cost():
            return term.func(self.env, **term.params) * term.weight

        for step in range(1, 31):
            self.env.common_step_counter += 1
            robot.root_link_pos_w[:, 0] = start + (.01 if step % 2 else -.01)
            actual = cost()
            torch.testing.assert_close(cost(), actual)
            if step == 15:
                torch.testing.assert_close(actual, torch.zeros(2), atol=1e-5, rtol=0)
        torch.testing.assert_close(actual, torch.full((2,), -12.))
        self.env.common_step_counter += 1
        robot.root_link_pos_w[0, 0] = start[0] + .03
        torch.testing.assert_close(cost(), torch.full((2,), -12.))
        self.env.common_step_counter += 1
        robot.root_link_pos_w[0, 0] = start[0] + .11
        torch.testing.assert_close(cost(), torch.tensor([0., -12.]))
        mdp.reset_stepping_stones(self.env, torch.tensor([1]))
        self.env.common_step_counter += 1
        torch.testing.assert_close(cost(), torch.zeros(2))

    def test_body_stall_cost_exempts_exit_zero_command_and_terminal_environments(self):
        term = self.env.cfg.rewards["stepping_body_stall"]
        self.state["stall_age"][:] = 2.
        self.env.common_step_counter += 1
        self.state["stage"][0] = 3
        self.env.reset_terminated = torch.tensor([False, True])
        torch.testing.assert_close(term.func(self.env, **term.params), torch.zeros(2))
        self.env.reset_terminated.zero_()
        self.env.command_manager.get_command = lambda name: torch.zeros(2, 3)
        torch.testing.assert_close(term.func(self.env, **term.params), torch.zeros(2))

    def test_disabled_early_failure_leaves_only_the_base_failure_cost(self):
        terms = {name: self.env.cfg.rewards[name] for name in ("stepping_failure", "stepping_early_failure")}
        manager = RewardManager(terms, self.env)
        self.env.reset_terminated = torch.ones(2, dtype=torch.bool)
        self.env.episode_length_buf[:] = torch.tensor([30, 400])
        torch.testing.assert_close(manager.compute(.02), torch.full((2,), -25.))
        self.env.reset_terminated.zero_()
        torch.testing.assert_close(manager.compute(.02), torch.zeros(2))

    def test_inactive_early_failure_formula_still_measures_remaining_time(self):
        term = self.env.cfg.rewards["stepping_early_failure"]
        self.env.reset_terminated = torch.ones(2, dtype=torch.bool)
        self.env.episode_length_buf[:] = torch.tensor([400, 450])
        remaining = term.func(self.env, **term.params) * self.env.step_dt
        torch.testing.assert_close(remaining, torch.tensor([52., 51.]))

    def test_physical_failure_overrides_latched_success_at_termination(self):
        self.state["success"][:] = True
        self.state["success_pulse"][:] = 1.
        self.env.episode_length_buf[:] = self.env.max_episode_length
        self.env.termination_manager = TerminationManager({
            "physical_failure": TerminationTermCfg(func=lambda env: torch.tensor([True, False])),
            "stepping_timeout": TerminationTermCfg(func=mdp.stepping_timeout),
            "stepping_succeeded": TerminationTermCfg(func=mdp.stepping_succeeded),
        }, self.env)
        self.env.termination_manager.compute()
        self.env.reset_terminated = self.env.termination_manager.terminated
        self.assertEqual(self.state["success"].tolist(), [False, True])
        torch.testing.assert_close(mdp.success_reward(self.env) * 40 * .02, torch.tensor([0., 40.]))
        torch.testing.assert_close(mdp.stepping_failure_reward(self.env) * -25 * .02, torch.tensor([-25., 0.]))
        self.assertEqual(list(self.env.cfg.terminations)[-1], "stepping_succeeded")

    def test_bad_orientation_cannot_keep_a_latched_success_without_a_manager(self):
        self.state["success"][:] = True
        self.state["success_pulse"][:] = 1.
        self.env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([1., 0., 0.])
        self.assertFalse(mdp.stepping_succeeded(self.env).any())
        torch.testing.assert_close(mdp.success_reward(self.env), torch.zeros(2))

    def test_repeated_partial_landings_cannot_repeat_cover_or_center_reward(self):
        manager = RewardManager({name: self.env.cfg.rewards[name]
                                 for name in ("stepping_cover", "stepping_stone_center")}, self.env)
        moving = self.gait.destinations(self.state["active_count"]) != HOLD
        self.found[moving] = False
        self.pos[..., 2] = torch.where(moving, .27, self.pos[..., 2])
        self.tick(8)
        self.pos[:, 0] = self.gait.targets[:, 0]
        self.found[:, 0] = True
        self.tick(16)
        torch.testing.assert_close(manager.compute(.02), torch.tensor([4.5, 2.5]), atol=1e-5, rtol=0)
        self.tick(4)
        for _ in range(3):
            self.found[:, 0] = False
            self.pos[:, 0, 2] = .27
            self.tick(8)
            self.pos[:, 0] = self.gait.targets[:, 0]
            self.found[:, 0] = True
            self.tick(16)
            torch.testing.assert_close(manager.compute(.02), torch.zeros(2))
        self.assertEqual(self.gait.phase.tolist(), [0, 0])

    def test_success_has_no_early_failure_cost(self):
        term = self.env.cfg.rewards["stepping_early_failure"]
        self.state["success"][:] = True
        self.env.reset_terminated = torch.ones(2, dtype=torch.bool)
        torch.testing.assert_close(term.func(self.env, **term.params), torch.zeros(2))

    def test_early_failure_cost_is_independent_of_control_period(self):
        term = self.env.cfg.rewards["stepping_early_failure"]
        self.env.reset_terminated = torch.ones(2, dtype=torch.bool)
        self.env.episode_length_buf[:] = 30
        expected = term.func(self.env, **term.params) * self.env.step_dt
        self.env.step_dt = .01
        self.env.max_episode_length = 6000
        self.env.episode_length_buf[:] = 60
        torch.testing.assert_close(term.func(self.env, **term.params) * self.env.step_dt, expected)

    def test_shin_bearing_load_disqualifies_landing(self):
        set_shin_load(self.env, 0, 10., (1., .2, .2))
        self.swing()
        self.assertEqual(self.gait.phase.tolist(), [0, 0])
        self.assertFalse(self.state["visited"].any())

    def test_full_course_reaches_exit_only_after_all_phases(self):
        for _ in range(20):
            self.swing()
            if self.gait.done.all():
                break
        self.assertTrue(self.gait.done.all())
        self.assertTrue(self.state["visited"][1, :6].all())
        self.assertTrue(self.state["visited"][0, :3].all())
        self.assertEqual(self.state["stage"].tolist(), [3, 3])
        self.tick(140)
        self.assertTrue(self.state["success"].all())


if __name__ == "__main__":
    unittest.main()
