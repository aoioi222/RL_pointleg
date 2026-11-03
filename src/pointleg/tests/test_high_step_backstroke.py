"""Check support travel incentives without running physics."""

import unittest

import torch

from mjlab.tasks.registry import load_env_cfg
from pointleg.mdp.high_step_basic import basic_backstroke_reward
from pointleg.mdp.rewards import command_orthogonal_drift_penalty
from test_high_step_basic import basic_env, contact, tick


def support_event(distance=.07, duration=.36, drag=False, dt=.02):
    bundle = basic_env(step_dt=dt)
    env, _, _, _ = bundle
    robot = env.scene["robot"].data
    tick(bundle)
    frames = round(duration / dt) - 1
    for _ in range(frames):
        if drag:
            robot.site_pos_w[:, 0, 0] -= distance / frames
            robot.site_lin_vel_w[:, 0, 0] = -distance / (frames * dt)
        else:
            robot.root_link_pos_w[:, 0] += distance / frames
        tick(bundle)
    contact(bundle, 0, False)
    tick(bundle)
    return bundle


class HighStepBackstrokeTests(unittest.TestCase):
    def test_longer_stances_pay_more_for_equal_total_progress(self):
        short = support_event(.035, .18)
        long = support_event(.07, .36)
        short_score = basic_backstroke_reward(short[0]) * short[0].step_dt
        long_score = basic_backstroke_reward(long[0]) * long[0].step_dt
        self.assertTrue((long_score > 2. * short_score).all())
        torch.testing.assert_close(long_score, torch.ones(3), atol=1.e-5, rtol=0.)
        self.assertGreater(short_score.min().item(), 0.)

    def test_dragging_stillness_short_contacts_and_failed_states_do_not_pay(self):
        for kwargs in ({"distance": 0.}, {"drag": True}, {"duration": .08}):
            bundle = support_event(**kwargs)
            self.assertEqual(basic_backstroke_reward(bundle[0]).sum().item(), 0.)
        bundle = support_event()
        env = bundle[0]
        env.termination_manager.terminated[:] = True
        tick(bundle)
        self.assertEqual(basic_backstroke_reward(env).sum().item(), 0.)

    def test_slip_reduces_reward_and_distance_saturates(self):
        bundle = support_event(.07)
        env, _, _, term = bundle
        full = basic_backstroke_reward(env).clone()
        term.ended_slip[:, 0] = .0125
        torch.testing.assert_close(basic_backstroke_reward(env), full * .5)
        term.ended_slip[:, 0] = .025
        self.assertEqual(basic_backstroke_reward(env).sum().item(), 0.)
        beyond = support_event(.14)
        torch.testing.assert_close(basic_backstroke_reward(beyond[0]), full)

    def test_takeoff_movement_is_excluded_and_diagnostics_match_support_endpoints(self):
        bundle = support_event()
        env, _, _, term = bundle
        log = env.extras["log"]
        start = log["Metrics/high_step_stance_start_x_fl_m"]
        end = log["Metrics/high_step_stance_end_x_fl_m"]
        sweep = log["Metrics/high_step_stance_sweep_fl_m"]
        torch.testing.assert_close(start - end, sweep, atol=1.e-6, rtol=0.)
        torch.testing.assert_close(sweep, torch.full((3,), .07), atol=1.e-5, rtol=0.)
        before = term.ended_backstroke.clone()
        env.scene["robot"].data.site_pos_w[:, 0, 0] -= .3
        term(env, **term.params)
        torch.testing.assert_close(term.ended_backstroke, before)
        tick(bundle)
        self.assertEqual(basic_backstroke_reward(env).sum().item(), 0.)

    def test_partial_reset_nonfinite_and_control_period(self):
        scores = []
        for dt in (.01, .02):
            bundle = support_event(dt=dt)
            env, _, _, term = bundle
            scores.append(basic_backstroke_reward(env) * dt)
            before = term.ended_backstroke.clone()
            term.reset(torch.tensor([0]))
            term(env, **term.params)
            torch.testing.assert_close(term.ended_backstroke[1:], before[1:])
            self.assertEqual(term.ended_backstroke[0].sum().item(), 0.)
            env.scene["robot"].data.root_link_pos_w[:, 0] = torch.nan
            tick(bundle)
            value = basic_backstroke_reward(env)
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value.sum().item(), 0.)
        torch.testing.assert_close(*scores, atol=1.e-5, rtol=0.)

    def test_duty_factor_and_symmetric_lateral_cost(self):
        bundle = support_event()
        env, _, _, term = bundle
        for _ in range(5):
            tick(bundle)
        contact(bundle, 0, True)
        tick(bundle)
        duty = env.extras["log"]["Metrics/high_step_stance_duty_factor_fl"]
        self.assertTrue(((duty > .6) & (duty < 1.)).all())
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        self.assertEqual(cfg.rewards["stance_backstroke"].weight, .10)
        self.assertEqual(cfg.rewards["command_orthogonal_drift"].weight, -3.)
        velocity = env.scene["robot"].data.root_link_lin_vel_b
        values = []
        for side in (-.04, .04):
            velocity[:, 1] = side
            values.append(command_orthogonal_drift_penalty(
                env, **cfg.rewards["command_orthogonal_drift"].params))
        torch.testing.assert_close(*values)
        self.assertGreater(values[0].min().item(), 0.)


if __name__ == "__main__":
    unittest.main()
