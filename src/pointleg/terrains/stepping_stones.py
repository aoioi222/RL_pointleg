"""Fixed stepping-stone terrain used by the precision foothold task."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


MAX_STONES = 8
STONE_SIZE = 0.20
PLATFORM_SIZE = 0.80
PLATFORM_HEIGHT = 0.20
TERRAIN_ANCHOR_X = 1.00
TERRAIN_ANCHOR_Y = 4.00


def stone_count_from_difficulty(difficulty: float) -> int:
    """Map the seven terrain curriculum rows to two through eight stones."""
    return int(np.clip(round(float(difficulty) * (MAX_STONES - 2)), 0, MAX_STONES - 2)) + 2


def stone_centers(count: int) -> np.ndarray:
    """Return local top-center coordinates for the active alternating stones."""
    if not 2 <= count <= MAX_STONES:
        raise ValueError(f"stone count must be in [2, {MAX_STONES}], got {count}")
    index = np.arange(count, dtype=np.float64)
    return np.stack(
        (1.00 + 0.30 * index, 0.20 * np.where(index % 2 == 0, 1.0, -1.0), np.full(count, PLATFORM_HEIGHT)),
        axis=-1,
    )


def right_platform_start(count: int) -> float:
    """Return the local x coordinate of the right platform's near edge."""
    if not 2 <= count <= MAX_STONES:
        raise ValueError(f"stone count must be in [2, {MAX_STONES}], got {count}")
    return 0.90 + 0.30 * count


@dataclass(kw_only=True)
class SteppingStonesTerrainCfg(SubTerrainCfg):
    """An 800 mm platform, alternating 200 mm blocks, and an exit platform."""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []
        floor = make_plane(body, self.size, 0.0, center_zero=False)
        for geom in floor:
            geometries.append(TerrainGeometry(geom=geom, color=(0.36, 0.36, 0.36, 1.0)))

        anchor = np.array((TERRAIN_ANCHOR_X, TERRAIN_ANCHOR_Y), dtype=np.float64)
        green = (0.32, 0.62, 0.20, 1.0)
        platform_size = (PLATFORM_SIZE / 2, PLATFORM_SIZE / 2, PLATFORM_HEIGHT / 2)
        left = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=platform_size,
            pos=(anchor[0] + PLATFORM_SIZE / 2, anchor[1], PLATFORM_HEIGHT / 2),
        )
        geometries.append(TerrainGeometry(geom=left, color=green))

        count = stone_count_from_difficulty(difficulty)
        for center in stone_centers(count):
            stone = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(STONE_SIZE / 2, STONE_SIZE / 2, STONE_SIZE / 2),
                pos=(anchor[0] + center[0], anchor[1] + center[1], STONE_SIZE / 2),
            )
            geometries.append(TerrainGeometry(geom=stone, color=green))

        exit_x = right_platform_start(count)
        right = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=platform_size,
            pos=(anchor[0] + exit_x + PLATFORM_SIZE / 2, anchor[1], PLATFORM_HEIGHT / 2),
        )
        geometries.append(TerrainGeometry(geom=right, color=green))
        return TerrainOutput(
            origin=np.array((TERRAIN_ANCHOR_X, TERRAIN_ANCHOR_Y, 0.0)),
            geometries=geometries,
        )
