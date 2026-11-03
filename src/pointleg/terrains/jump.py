"""Compact launch and landing platforms with an optional central wall."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


TERRAIN_NAMES = ("platform_gap", "platform_wall")
TILE_SIZE = (3.0, 1.6)
LANE_WIDTH = 0.8
CENTER_Y = TILE_SIZE[1] / 2
LAUNCH_EDGE = 1.10
PLATFORM_START = 0.08
PLATFORM_END = 2.92
SPAWN_X = 0.64
LEVELS_PER_AXIS = 5
LEVEL_COUNT = LEVELS_PER_AXIS ** 2
MIN_RATIO = 0.15


def decode_level(level):
    if not 0 <= level < LEVEL_COUNT:
        raise ValueError("Jump geometry level must be in [0, 24]")
    return divmod(int(level), LEVELS_PER_AXIS)


def level_ratio(level):
    return MIN_RATIO + (1 - MIN_RATIO) * level / (LEVELS_PER_AXIS - 1)


@dataclass(frozen=True)
class JumpProfile:
    kind: str
    height_level: int
    gap_level: int
    platform_height: float
    gap: float
    wall_rise: float
    wall_thickness: float

    @property
    def landing_edge(self):
        return LAUNCH_EDGE + self.gap

    @property
    def wall_center(self):
        return LAUNCH_EDGE + self.gap / 2

    @property
    def spawn(self):
        return np.array([SPAWN_X, CENTER_Y, self.platform_height])


def jump_profile(kind, height_level, gap_level):
    if kind not in TERRAIN_NAMES:
        raise ValueError("Unknown jump terrain")
    decode_level(height_level * LEVELS_PER_AXIS + gap_level)
    if not 0 <= height_level < LEVELS_PER_AXIS or not 0 <= gap_level < LEVELS_PER_AXIS:
        raise ValueError("Jump height and gap levels must lie in [0, 4]")
    height_ratio, gap_ratio = level_ratio(height_level), level_ratio(gap_level)
    wall = kind == TERRAIN_NAMES[1]
    return JumpProfile(kind, height_level, gap_level, .200 * height_ratio,
                       max(.080 if wall else 0., .360 * gap_ratio),
                       .300 * height_ratio if wall else 0., .080 if wall else 0.)


@dataclass(kw_only=True)
class JumpTerrainCfg(SubTerrainCfg):
    kind: str

    def function(self, difficulty, spec, rng):
        del rng
        if tuple(self.size) != TILE_SIZE or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
            raise ValueError("Jump requires a 3.0 by 1.6 meter tile and difficulty in [0, 1]")
        height_level, gap_level = decode_level(round(difficulty * (LEVEL_COUNT - 1)))
        profile = jump_profile(self.kind, height_level, gap_level)
        body = spec.body("terrain")
        geometries = [TerrainGeometry(geom=g, color=(.24, .25, .26, 1))
                      for g in make_plane(body, self.size, 0, center_zero=False)]

        def box(left, right, height, color):
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                 pos=((left + right) / 2, CENTER_Y, height / 2),
                                 size=((right - left) / 2, LANE_WIDTH / 2, height / 2))
            geometries.append(TerrainGeometry(geom=geom, color=color))

        box(PLATFORM_START, LAUNCH_EDGE, profile.platform_height, (.25, .48, .72, 1))
        box(profile.landing_edge, PLATFORM_END, profile.platform_height, (.24, .65, .38, 1))
        if profile.wall_rise:
            box(profile.wall_center - profile.wall_thickness / 2,
                profile.wall_center + profile.wall_thickness / 2,
                profile.platform_height + profile.wall_rise, (.80, .38, .20, 1))
        return TerrainOutput(origin=profile.spawn, geometries=geometries)
