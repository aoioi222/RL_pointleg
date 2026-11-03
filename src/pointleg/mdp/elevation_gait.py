"""Terrain-tangent foot workspaces and per-leg walking feedback."""

import math

import torch

from .elevation import elevation_foot_bottom, elevation_state
from ..robot_cfg import FOOT_SITE_NAMES, LEG_NAMES
from ..terrains.elevation import TILE_SIZE


def elevation_surface_frame(env, x):
    """Return forward tangents and normals without fixing a world foot height."""
    state = elevation_state(env)
    kind = env.scene.terrain.terrain_types[:, None]
    angle = torch.where(kind == 2, state["angle"][:, None],
                        torch.where(kind == 3, -state["angle"][:, None], 0.))
    angle = angle * ((x >= 3.) & (x <= 5.))
    zero = torch.zeros_like(x)
    tangent = torch.stack((angle.cos().expand_as(x), zero, angle.sin().expand_as(x)), dim=-1)
    normal = torch.stack((-tangent[..., 2], zero, tangent[..., 0]), dim=-1)
    return tangent, normal


def elevation_gait_term(env, key):
    term = env._elevation_gait_term
    term(env, **term.params)
    value = term.gait[key]
    if key == "quality":
        state = elevation_state(env)
        value = torch.where(state["failed"] | state["invalid"] | state["complete"], 0., value)
    return value


def elevation_landing_margin(env, x):
    """Measure fore/aft tread-edge distance without duplicating lateral costs."""
    state = elevation_state(env)
    treads = state["step_surfaces"]
    left = torch.zeros_like(x)
    right = treads[:, 0, 0, None].expand_as(x)
    for index in range(treads.shape[1]):
        entered = x >= treads[:, index, 0, None]
        left = torch.where(entered, treads[:, index, 0, None], left)
        right = torch.where(entered, treads[:, index, 1, None], right)
    final_drop = treads[:, -1, 2, None].abs() > 1.e-6
    final_region = x >= treads[:, -1, 0, None]
    right = torch.where(final_region & ~final_drop, TILE_SIZE[0], right)
    beyond = (x >= treads[:, -1, 1, None]) & final_drop
    left = torch.where(beyond, treads[:, -1, 1, None], left)
    right = torch.where(beyond, TILE_SIZE[0], right)
    return torch.minimum(x - left, right - x).clamp_min(0.).nan_to_num(nan=0., posinf=0., neginf=0.)


class ElevationFootWorkspacePenalty:
    """Limit fore/aft reach along the support tangent, leaving vertical motion free."""

    def __init__(self, cfg, env):
        self.foot_ids, _ = env.scene["robot"].find_sites(FOOT_SITE_NAMES, preserve_order=True)
        self.hip_ids = []
        for leg in LEG_NAMES:
            ids, _ = env.scene["robot"].find_bodies(f"{leg}_hip_pitch_Link")
            if len(ids) != 1:
                raise ValueError("Each walking foot requires one hip-pitch anchor.")
            self.hip_ids.append(ids[0])
        for key in ("forward_margin", "lateral_margin", "deviation_scale"):
            if not math.isfinite(cfg.params[key]) or cfg.params[key] <= 0.:
                raise ValueError("Workspace margins and scale must be positive.")

    def __call__(self, env, nominal_forward, nominal_outward, forward_margin, lateral_margin, deviation_scale,
                 terrain_extra_margin=0.):
        robot = env.scene["robot"].data
        feet = elevation_foot_bottom(env, self.foot_ids)
        hips = robot.body_link_pos_w[:, self.hip_ids]
        relative = feet - hips
        state = elevation_state(env)
        x = (robot.root_link_pos_w[:, 0] - state["corner"][:, 0])[:, None].expand(-1, 4)
        tangent, _ = elevation_surface_frame(env, x)
        along = (relative * tangent).sum(dim=-1)
        side = along.new_tensor([1., -1., 1., -1.])
        across = relative[..., 1] * side
        root_x = x[:, 0]
        obstacle = ((env.scene.terrain.terrain_types < 4)
                    & (root_x >= state["entry_x"] - .6) & (root_x <= state["exit_x"] + .5))
        margin = forward_margin + obstacle[:, None] * terrain_extra_margin
        excess_x = ((along - nominal_forward).abs() - margin).clamp_min(0.)
        excess_y = ((across - nominal_outward).abs() - lateral_margin).clamp_min(0.)
        cost = ((excess_x.square() + excess_y.square()) / deviation_scale ** 2).clamp_max(1.)
        cost = torch.nan_to_num(cost, nan=0., posinf=0., neginf=0.)
        active = state["initialized"] & ~state["failed"] & ~state["complete"] & (state["speed"] > .1)
        for index, leg in enumerate(LEG_NAMES):
            env.extras["log"][f"Metrics/elevation_foot_reach_{leg}_m"] = along[:, index].detach()
        return torch.where(active, .5 * cost.mean(dim=1) + .5 * cost.max(dim=1).values, 0.)
