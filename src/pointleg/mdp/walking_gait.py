"""Contact-based diagonal coordination without an external phase clock."""

import math

import torch

from ..robot_cfg import FOOT_BODY_NAMES


def ordered_contact_ids(sensor):
    ids = []
    for body in FOOT_BODY_NAMES:
        matches = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
        if len(matches) != 1:
            raise ValueError(f"Expected one primary contact channel for {body}")
        ids.append(matches[0])
    return ids


def loaded_feet(sensor, contact_ids, minimum_force):
    """Read force components according to the sensor's declared coordinate frame."""
    data = sensor.data
    batch, primaries = data.found.shape[0], len(sensor.primary_names)
    found = data.found.reshape(batch, primaries, -1)[:, contact_ids] > 0
    force = data.force.reshape(batch, primaries, -1, 3)[:, contact_ids]
    finite = torch.isfinite(force).all(dim=-1)
    force = torch.nan_to_num(force)
    world = sensor.cfg.global_frame or sensor.cfg.reduce == "netforce"
    normal = force[..., 2].abs() if world else force[..., 0].abs()
    tangent = torch.linalg.vector_norm(force[..., :2] if world else force[..., 1:], dim=-1)
    floor = torch.ones_like(found)
    if getattr(data, "normal", None) is not None:
        normals = data.normal.reshape(batch, primaries, -1, 3)[:, contact_ids]
        floor = torch.isfinite(normals).all(dim=-1) & (normals[..., 2] < -.5)
    return (found & finite & floor & (normal >= minimum_force) & (normal >= tangent)).any(dim=2)


class DiagonalGaitPenalty:
    """Pair diagonal contacts and allow a brief four-foot support transition."""

    def __init__(self, cfg, env):
        self.ids = ordered_contact_ids(env.scene[cfg.params["sensor_name"]])
        self.overlap = torch.zeros(env.num_envs, device=env.device)
        self.cost_sum = torch.zeros_like(self.overlap)
        self.samples = torch.zeros_like(self.overlap)
        env._diagonal_gait_term = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.overlap[ids] = self.cost_sum[ids] = self.samples[ids] = 0.

    def __call__(self, env, sensor_name, command_name="twist", minimum_force=5.,
                 overlap_grace=.06, startup_grace=.6, flat_only=False):
        if not math.isfinite(minimum_force) or minimum_force <= 0:
            raise ValueError("Support force must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (overlap_grace, startup_grace)):
            raise ValueError("Gait grace periods must be finite and nonnegative")
        command = env.command_manager.get_command(command_name)
        active = ((command[:, 0] > .08) & (command[:, 1].abs() < .05)
                  & (command[:, 2].abs() < .15)
                  & (env.episode_length_buf * env.step_dt >= startup_grace))
        if flat_only:
            from .elevation import elevation_state

            state = elevation_state(env)
            x = env.scene["robot"].data.root_link_pos_w[:, 0] - state["corner"][:, 0]
            obstacle = ((env.scene.terrain.terrain_types < 4)
                        & (x >= state["entry_x"] - .5) & (x <= state["exit_x"] + .5))
            active &= ~obstacle & ~state["failed"] & ~state["complete"]
        support = loaded_feet(env.scene[sensor_name], self.ids, minimum_force)
        all_support = support.all(dim=1)
        self.overlap = torch.where(active & all_support, self.overlap + env.step_dt, 0.)
        diagonal = ((support[:, 0] != support[:, 3]).float()
                    + (support[:, 1] != support[:, 2]).float()) * .5
        stalled = all_support & (self.overlap > overlap_grace + 1.e-6)
        flight = ~support.any(dim=1)
        cost = torch.where(active, torch.maximum(diagonal, (stalled | flight).float()), 0.)
        self.cost_sum += cost
        self.samples += active.float()
        env.extras["log"]["Metrics/diagonal_contact_error"] = cost.detach()
        return cost
