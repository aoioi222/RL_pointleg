"""Straight competition-sized stair, ramp and thin-wall terrain profiles."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


TERRAIN_NAMES = ("stairs_up", "stairs_down", "slope_up", "slope_down", "wall_sequence")
FLAT_NAME = "flat"
WALKING_NAMES = (*TERRAIN_NAMES[:4], FLAT_NAME)
PAIRED_TERRAIN_NAMES = {
    "stairs": TERRAIN_NAMES[:2],
    "slope": TERRAIN_NAMES[2:4],
}
ELEVATION_TASK_TERRAINS = {
    "PointLeg-Elevation-v0": None,
    "PointLeg-Stairs-v0": "stairs",
    "PointLeg-Slope-v0": "slope",
}
WALL_NAME = TERRAIN_NAMES[4]
TILE_SIZE = (12.0, 8.0)
LANE_WIDTH = 0.8
CENTER_Y = 4.0
SPAWN_X = 1.2
STEP_DEPTH = 0.32
STAIR_FLIGHTS = 2
STEPS_PER_FLIGHT = 3
STAIR_TREAD_COUNT = STAIR_FLIGHTS * STEPS_PER_FLIGHT
STAIR_PLATFORM_RANGE = (1.2, 2.0)
EXIT_PLATFORM_LENGTH = 1.6
WALL_THICKNESS = 0.08
WALL_CENTERS = (3.0, 5.58, 8.16)


def walking_terrain_names(terrain_kind):
    """Resolve mixed, paired and single-direction walking task scopes."""
    if terrain_kind is None:
        return WALKING_NAMES
    if terrain_kind in PAIRED_TERRAIN_NAMES:
        return PAIRED_TERRAIN_NAMES[terrain_kind]
    if terrain_kind in TERRAIN_NAMES[:4]:
        return (terrain_kind,)
    raise ValueError("Unknown elevation walking terrain scope")


@dataclass(frozen=True)
class ElevationProfile:
    kind: str
    difficulty: float
    spawn: np.ndarray
    entry_x: float
    exit_x: float
    step_surfaces: tuple[tuple[float, float, float], ...]
    wall_height: float
    slope_angle: float
    top_height: float
    stair_platform_length: float
    landing_end_x: float

    def surface_height(self, x):
        x = np.asarray(x)
        if self.kind.startswith("stairs"):
            result = np.full_like(x, self.spawn[2], dtype=float)
            for left, _, height in self.step_surfaces:
                result = np.where(x >= left, height, result)
            return np.where(x >= self.landing_end_x, 0.0, result)
        if self.kind == "slope_up":
            return np.clip((x - 3.0) / 2.0, 0, 1) * self.top_height
        if self.kind == "slope_down":
            return np.clip((5.0 - x) / 2.0, 0, 1) * self.top_height
        result = np.zeros_like(x, dtype=float)
        if self.kind == FLAT_NAME:
            return result
        for center in WALL_CENTERS:
            result = np.where(np.abs(x - center) <= WALL_THICKNESS / 2, self.wall_height, result)
        return result

    def reference_pitch(self, x):
        x = np.asarray(x)
        if self.kind.startswith("slope"):
            sign = -1.0 if self.kind == "slope_up" else 1.0
            return np.where((x >= 3.0) & (x <= 5.0), sign * self.slope_angle, 0.0)
        return np.zeros_like(x, dtype=float)


def geometry_profile(
    kind: str, difficulty: float,
    stair_platform_length: float = sum(STAIR_PLATFORM_RANGE) / 2,
    exit_platform_length: float = EXIT_PLATFORM_LENGTH,
    stair_height_override: float | None = None,
    stair_height_range: tuple[float, float] | None = None,
) -> ElevationProfile:
    if kind not in (*TERRAIN_NAMES, FLAT_NAME) or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Expected an elevation terrain kind and difficulty in [0, 1]")
    step = 0.04 + 0.08 * difficulty
    if stair_height_range is not None:
        low, high = stair_height_range
        if not kind.startswith("stairs") or not math.isfinite(low + high) or not 0 < low <= high <= .12 or stair_height_override is not None:
            raise ValueError("Invalid stair height curriculum")
        step = low + (high - low) * difficulty
    if stair_height_override is not None:
        if not kind.startswith("stairs") or not math.isfinite(stair_height_override) or not 0 < stair_height_override <= .12:
            raise ValueError("Fixed stair height requires stairs and a height in (0, 0.12] meters")
        step = stair_height_override
    angle = math.radians(20 / 3 + (20 - 20 / 3) * difficulty)
    top = 2 * math.tan(angle) if kind.startswith("slope") else STAIR_TREAD_COUNT * step
    surfaces = ()
    entry, exit_x = 3.0, 5.0
    landing_end = TILE_SIZE[0]
    if kind == FLAT_NAME:
        top, angle = 0.0, 0.0
    if kind.startswith("stairs"):
        if not math.isfinite(stair_platform_length) or stair_platform_length < 1.0:
            raise ValueError("The intermediate stair platform must be at least 1 m long")
        if not math.isfinite(exit_platform_length) or exit_platform_length < 1.2:
            raise ValueError("The final stair platform must reserve at least 1.2 m")
        starts = tuple(entry + flight * (STEPS_PER_FLIGHT * STEP_DEPTH + stair_platform_length)
                       for flight in range(STAIR_FLIGHTS))
        exit_x = starts[-1] + STEPS_PER_FLIGHT * STEP_DEPTH
        landing_end = exit_x + exit_platform_length
        if landing_end > TILE_SIZE[0]:
            raise ValueError("Stair flights and final platform exceed the terrain tile")
        regions = []
        for flight, start in enumerate(starts):
            for tread in range(STEPS_PER_FLIGHT):
                left = start + tread * STEP_DEPTH
                right = left + STEP_DEPTH
                if tread == STEPS_PER_FLIGHT - 1:
                    right = starts[flight + 1] if flight + 1 < STAIR_FLIGHTS else landing_end
                ordinal = flight * STEPS_PER_FLIGHT + tread + 1
                height = (ordinal if kind == "stairs_up" else STAIR_TREAD_COUNT - ordinal) * step
                regions.append((left, right, height))
        surfaces = tuple(regions)
    elif kind == WALL_NAME:
        top = 0.0
        entry, exit_x = WALL_CENTERS[0] - 0.04, WALL_CENTERS[-1] + 0.04
    spawn_height = top if kind.endswith("down") else 0.0
    return ElevationProfile(
        kind, difficulty, np.array([SPAWN_X, CENTER_Y, spawn_height]), entry, exit_x,
        surfaces, 0.11 + 0.23 * difficulty, angle, top,
        stair_platform_length if kind.startswith("stairs") else 0.0, landing_end,
    )


@dataclass(kw_only=True)
class ElevationTerrainCfg(SubTerrainCfg):
    kind: str
    wall_height_override: float | None = None
    stair_height_override: float | None = None
    stair_height_range: tuple[float, float] | None = None
    stair_platform_range: tuple[float, float] = STAIR_PLATFORM_RANGE
    exit_platform_length: float = EXIT_PLATFORM_LENGTH

    def generated_profile(self, difficulty):
        """Use the same sampled geometry for contact and completion checks."""
        try:
            return self._generated_profiles[round(float(difficulty), 8)]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError("Elevation profiles are available only after terrain generation") from exc

    def layout_contract(self):
        contract = {"revision": 2, "flights": STAIR_FLIGHTS, "steps_per_flight": STEPS_PER_FLIGHT,
                "platform_range_m": tuple(self.stair_platform_range),
                "exit_platform_length_m": self.exit_platform_length}
        if self.stair_height_override is not None:
            contract.update(revision=3, fixed_step_height_m=self.stair_height_override)
        if self.stair_height_range is not None:
            contract.update(revision=4, step_height_range_m=tuple(self.stair_height_range))
        return contract

    def function(self, difficulty, spec, rng):
        if tuple(self.size) != TILE_SIZE:
            raise ValueError(f"Elevation terrain requires tile size {TILE_SIZE}")
        low, high = self.stair_platform_range
        if not math.isfinite(low + high) or not 1.0 <= low <= high:
            raise ValueError("Invalid intermediate stair platform length range")
        gap = rng.uniform(low, high) if self.kind.startswith("stairs") else (low + high) / 2
        profile = geometry_profile(
            self.kind, difficulty, gap, self.exit_platform_length, self.stair_height_override, self.stair_height_range,
        )
        if not hasattr(self, "_generated_profiles"):
            self._generated_profiles = {}
        self._generated_profiles[round(float(difficulty), 8)] = profile
        body = spec.body("terrain")
        geometries = [TerrainGeometry(geom=g, color=(0.35, 0.37, 0.35, 1))
                      for g in make_plane(body, self.size, 0, center_zero=False)]

        def box(left, right, height):
            if height <= 0:
                return
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=((left + right) / 2, CENTER_Y, height / 2),
                size=((right - left) / 2, LANE_WIDTH / 2, height / 2),
            )
            geometries.append(TerrainGeometry(geom=geom, color=(0.28, 0.62, 0.30, 1)))

        if self.kind.startswith("stairs"):
            if self.kind == "stairs_down":
                box(0, profile.entry_x, profile.top_height)
            for left, right, height in profile.step_surfaces:
                box(left, right, height)
        elif self.kind.startswith("slope"):
            theta = profile.slope_angle * (1 if self.kind == "slope_up" else -1)
            thickness = profile.top_height + 0.2
            normal = np.array([-math.sin(theta), 0, math.cos(theta)])
            top_center = np.array([4.0, CENTER_Y, profile.top_height / 2])
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=top_center - normal * thickness / 2,
                size=(1 / math.cos(theta), LANE_WIDTH / 2, thickness / 2),
                quat=(math.cos(theta / 2), 0, -math.sin(theta / 2), 0),
            )
            geometries.append(TerrainGeometry(geom=geom, color=(0.28, 0.62, 0.30, 1)))
            if self.kind == "slope_up":
                box(5, self.size[0], profile.top_height)
            else:
                box(0, 3, profile.top_height)
        elif self.kind == WALL_NAME:
            height = self.wall_height_override
            height = profile.wall_height if height is None else height
            if not math.isfinite(height) or not 0 <= height <= 0.5:
                raise ValueError("Wall height override must be in [0, 0.5] meters; zero is flat-jump warmup")
            for center in WALL_CENTERS:
                box(center - 0.04, center + 0.04, height)
        return TerrainOutput(origin=profile.spawn.copy(), geometries=geometries)


def make_elevation_terrains(include_flat=False, stair_height_override=None):
    names = (*TERRAIN_NAMES, FLAT_NAME) if include_flat else TERRAIN_NAMES
    return {name: ElevationTerrainCfg(
                kind=name, size=TILE_SIZE, proportion=1.0,
                stair_height_override=stair_height_override if name.startswith("stairs") else None,
            )
            for name in names}
