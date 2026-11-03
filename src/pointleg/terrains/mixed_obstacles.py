"""Fenced rough patches and free-body annuli with planar upper surfaces."""

from dataclasses import dataclass
from functools import partial
import math

import mujoco
import numpy as np

from mjlab.entity import EntityCfg
from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


RING_HEIGHT = 0.165
RING_OUTER_RADIUS = 0.275
RING_INNER_RADIUS = 0.165
RING_MASS = 6.2
RING_ROWS = 7
RING_COLUMNS = 2
RING_SPACING = 2 * RING_OUTER_RADIUS
RING_FIRST_X = 2.8
RING_CENTER_Y = 4.0
RING_LANE_HALF_WIDTH = RING_COLUMNS * RING_SPACING / 2
RING_SECTORS = 24
RING_NAMES = tuple(f"tire_{i:02d}" for i in range(RING_ROWS * RING_COLUMNS))
FENCE_NAMES = ("fenced_random_rough", "fenced_random_bumps")
RING_TERRAIN = "movable_ring_array"
START_X = 1.2
FENCE_X = 4.0
REVERSE_START_X = 6.8
FENCE_APPROACH_DISTANCE = 1.0
RING_APPROACH_DISTANCE = 0.9
OBSTACLE_SPEED_RANGE = (0.25, 0.35)


def ring_centers():
    return np.asarray([
        (
            RING_FIRST_X + row * RING_SPACING,
            RING_CENTER_Y + (column - (RING_COLUMNS - 1) / 2) * RING_SPACING,
            RING_HEIGHT / 2,
        )
        for row in range(RING_ROWS) for column in range(RING_COLUMNS)
    ])


def parking_centers():
    return np.asarray([
        (
            -100.0 + row * 0.70,
            (column - (RING_COLUMNS - 1) / 2) * 0.70,
            RING_HEIGHT / 2,
        )
        for row in range(RING_ROWS) for column in range(RING_COLUMNS)
    ])


@dataclass(kw_only=True)
class FencedRoughTerrainCfg(SubTerrainCfg):
    background: SubTerrainCfg
    fence_height: float = 0.152
    fence_thickness: float = 0.050

    def function(self, difficulty, spec, rng):
        output = self.background.function(difficulty, spec, rng)
        if hasattr(self.background, "noise_range"):
            ground = self.background.noise_range[1] * difficulty
        else:
            low, high = self.background.grid_height_range
            ground = low + difficulty * (high - low)
        body = spec.body("terrain")
        for x, half_x, half_y in (
            (FENCE_X - FENCE_APPROACH_DISTANCE, 0.55, 0.65),
            (FENCE_X + FENCE_APPROACH_DISTANCE, 0.55, 0.65),
            (FENCE_X, 0.15, 4.0),
        ):
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, 4.0, ground - 0.10), size=(half_x, half_y, 0.10),
            )
            output.geometries.append(TerrainGeometry(geom=geom, color=(0.45, 0.60, 0.42, 1.0)))
        height = self.fence_height * difficulty
        wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(FENCE_X, 4.0, ground + height / 2),
            size=(self.fence_thickness / 2, self.size[1] / 2, height / 2),
        )
        output.geometries.append(TerrainGeometry(geom=wall, color=(0.85, 0.40, 0.12, 1.0)))
        output.origin = np.asarray([START_X, 4.0, ground])
        return output


@dataclass(kw_only=True)
class RingArrayFloorCfg(SubTerrainCfg):
    def function(self, difficulty, spec, rng):
        del difficulty, rng
        boxes = make_plane(spec.body("terrain"), self.size, 0.0, center_zero=False)
        return TerrainOutput(
            origin=np.asarray([START_X, RING_CENTER_Y, 0.0]),
            geometries=[TerrainGeometry(geom=box, color=(0.45, 0.52, 0.45, 1.0)) for box in boxes],
        )


def ring_section_dimensions():
    half_angle = math.pi / RING_SECTORS
    half_tangent = RING_OUTER_RADIUS * math.sin(half_angle)
    outer_face = RING_OUTER_RADIUS * math.cos(half_angle)
    return (outer_face - RING_INNER_RADIUS) / 2, half_tangent, (outer_face + RING_INNER_RADIUS) / 2


def ring_inertia(height):
    radial, tangent, center = ring_section_dimensions()
    xy = RING_MASS * ((radial ** 2 + tangent ** 2) / 6 + height ** 2 / 12 + center ** 2 / 2)
    z = RING_MASS * ((radial ** 2 + tangent ** 2) / 3 + center ** 2)
    return xy, xy, z


def ring_spec(position=(0.0, 0.0, RING_HEIGHT / 2), height=RING_HEIGHT):
    """Build an annulus from overlapping boxes with independently scalable height."""
    if not math.isfinite(height) or not 0.0 < height <= RING_HEIGHT:
        raise ValueError("Ring height must lie in (0, 0.165] meters.")
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ring", pos=position, gravcomp=1.0)
    body.add_freejoint(name="free")
    radial, tangent, center = ring_section_dimensions()
    for sector in range(RING_SECTORS):
        angle = 2 * math.pi * sector / RING_SECTORS
        body.add_geom(
            name=f"surface_{sector}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(radial, tangent, height / 2),
            pos=(center * math.cos(angle), center * math.sin(angle), 0.0),
            quat=(math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)),
            mass=RING_MASS / RING_SECTORS, group=0,
            friction=(0.8, 0.01, 0.001), condim=3,
            rgba=(0.18, 0.20, 0.22, 1.0),
        )
    return spec


def ring_entity_cfg(index):
    return EntityCfg(
        spec_fn=partial(ring_spec, position=tuple(parking_centers()[index])),
        init_state=EntityCfg.InitialStateCfg(pos=tuple(parking_centers()[index])),
    )


