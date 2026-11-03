"""Check force deadbands and substep peaks without prescribing support balance."""

from types import SimpleNamespace
import unittest

import torch

from legacy_high_step_config import load_env_cfg
from pointleg.robot_cfg import FOOT_BODY_NAMES, LEG_NAMES


def contact_env(reverse=False):
    cfg = load_env_cfg("PointLeg-HighStep-v0").rewards["high_step_contact_force"]
    names = list(reversed(FOOT_BODY_NAMES)) if reverse else list(FOOT_BODY_NAMES)
    force = torch.zeros(4, 4, 3)
    force[:, :, 2] = 80.
    data = SimpleNamespace(
        force=force, force_history=None, found=torch.ones(4, 4),
    )
    command = torch.tensor([[.2, 0., 0.]]).repeat(4, 1)
    env = SimpleNamespace(
        num_envs=4, device="cpu", step_dt=.02, extras={"log": {}},
        episode_length_buf=torch.full((4,), 100),
        scene={"feet_ground_contact": SimpleNamespace(primary_names=names, data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    return env, data, command, cfg, cfg.func(cfg, env)


class HighStepContactForceTests(unittest.TestCase):
    def test_normal_support_and_threshold_force_have_no_penalty(self):
        env, data, _, cfg, term = contact_env()
        data.force[1, :, 2] = 150.
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(4))

    def test_single_foot_overload_is_equal_for_each_leg_and_correctly_ordered(self):
        env, data, _, cfg, term = contact_env(reverse=True)
        for i in range(4):
            data.force[i, term.ids[i], 2] = 250.
        torch.testing.assert_close(term(env, **cfg.params), torch.full((4,), .625))
        for i, leg in enumerate(LEG_NAMES):
            self.assertEqual(env.extras["log"][f"Metrics/high_step_contact_peak_{leg}_n"][i], 250.)

    def test_history_catches_a_short_peak_even_if_current_contact_has_ended(self):
        env, data, _, cfg, term = contact_env()
        data.force_history = data.force[:, :, None, :].repeat(1, 1, 4, 1)
        data.force_history[:, 2, 3, 2] = 300.
        data.force.zero_()
        data.found.zero_()
        cost = term(env, **cfg.params)
        torch.testing.assert_close(cost, torch.full((4,), 1.40625))
        self.assertEqual(env.extras["log"]["Metrics/high_step_contact_peak_rl_n"].mean(), 300.)
        data.force_history[:, 2, 3, 2] = 80.
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(4))

    def test_force_norm_is_rotation_invariant_and_nonfinite_values_are_safe(self):
        env, data, _, cfg, term = contact_env()
        data.force[:, 0] = torch.tensor([
            [250., 0., 0.], [0., 250., 0.], [0., 0., 250.], [float("nan"), float("inf"), 0.],
        ])
        cost = term(env, **cfg.params)
        torch.testing.assert_close(cost, torch.tensor([.625, .625, .625, 0.]))
        self.assertTrue(all(torch.isfinite(v).all() for v in env.extras["log"].values()))
        data.force = None
        with self.assertRaises(RuntimeError):
            term(env, **cfg.params)

    def test_standing_turn_commands_and_startup_are_excluded(self):
        env, data, command, cfg, term = contact_env()
        data.force[:, :, 2] = 250.
        command[1] = 0.
        command[2, 2] = 1.
        env.episode_length_buf[3] = 1
        torch.testing.assert_close(term(env, **cfg.params), torch.tensor([1., 0., 0., 0.]))

    def test_high_step_only_and_history_covers_a_full_control_interval(self):
        cfg = load_env_cfg("PointLeg-HighStep-v0")
        sensor = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
        self.assertEqual(sensor.history_length, cfg.decimation)
        self.assertGreater(cfg.rewards["high_step_cycle"].weight, 0.)
        for task in ("PointLeg-Flat-v0", "PointLeg-Rough-v0", "PointLeg-Elevation-v0"):
            self.assertNotIn("high_step_contact_force", load_env_cfg(task).rewards)
        flat = load_env_cfg("PointLeg-Flat-v0")
        self.assertEqual(list(cfg.observations["actor"].terms), list(flat.observations["actor"].terms))


if __name__ == "__main__":
    unittest.main()
