"""Curriculum stepping-stone terrain used by the precision foothold task."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


MAX_STONES = 8
STONE_SIZE = 0.15
COMPETITION_STONE_SIZE = 0.20
COMPETITION_STONE_COUNT = 6
COMPETITION_STONE_COUNTS = (2, 4, COMPETITION_STONE_COUNT)
STONE_HEIGHT = 0.20
STONE_ROW_OFFSET = 0.20
PLATFORM_SIZE = 0.80
PLATFORM_HEIGHT = 0.20
TERRAIN_ANCHOR_X = 1.00
TERRAIN_ANCHOR_Y = 4.00


def stone_count_from_difficulty(difficulty: float) -> int:
    """Map the seven terrain curriculum rows to two through eight stones."""
    return int(np.clip(round(float(difficulty) * (MAX_STONES - 2)), 0, MAX_STONES - 2)) + 2


def stone_centers(count: int, gap: float = 0.05, *, size: float = STONE_SIZE) -> np.ndarray:
    """Return local top-center coordinates for the active alternating stones."""
    if not 2 <= count <= MAX_STONES:
        raise ValueError(f"stone count must be in [2, {MAX_STONES}], got {count}")
    index = np.arange(count, dtype=np.float64)
    return np.stack(
        (PLATFORM_SIZE + gap + size / 2 + (size + gap) * index, STONE_ROW_OFFSET * np.where(index % 2 == 0, 1.0, -1.0), np.full(count, PLATFORM_HEIGHT)),
        axis=-1,
    )


def right_platform_start(count: int, gap: float = 0.05, *, size: float = STONE_SIZE) -> float:
    """Return the local x coordinate of the right platform's near edge."""
    if not 2 <= count <= MAX_STONES:
        raise ValueError(f"stone count must be in [2, {MAX_STONES}], got {count}")
    return PLATFORM_SIZE + count * (size + gap) + gap


@dataclass(kw_only=True)
class SteppingStonesTerrainCfg(SubTerrainCfg):
    """An 800 mm platform, alternating stone blocks, and an exit platform."""

    competition: bool = False
    six_stone_gap_curriculum: bool = False

    @property
    def stone_size(self) -> float:
        return COMPETITION_STONE_SIZE if self.competition else STONE_SIZE

    def gap_for_count(self, count: int) -> float:
        if self.competition:
            return 0.10
        return 0.05 + 0.05 * min(count - 2, MAX_STONES - 2) / (MAX_STONES - 2)

    def count_and_gap(self, difficulty: float) -> tuple[int, float]:
        if self.competition or self.six_stone_gap_curriculum:
            level = round(float(np.clip(difficulty, 0.0, 1.0)) * (len(COMPETITION_STONE_COUNTS) - 1))
            return COMPETITION_STONE_COUNTS[level], 0.10
        count = stone_count_from_difficulty(difficulty)
        return count, self.gap_for_count(count)

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

        count, gap = self.count_and_gap(difficulty)
        half_size = self.stone_size / 2
        for center in stone_centers(count, gap, size=self.stone_size):
            stone = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(half_size, half_size, STONE_HEIGHT / 2),
                pos=(anchor[0] + center[0], anchor[1] + center[1], STONE_HEIGHT / 2),
            )
            geometries.append(TerrainGeometry(geom=stone, color=green))

        exit_x = right_platform_start(count, gap, size=self.stone_size)
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
