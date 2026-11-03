"""Check physical atlas size independently from semantic curriculum indexing."""

from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.tasks.velocity.mdp.terminations import out_of_terrain_bounds
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGenerator,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)


@dataclass
class MarkedTerrainCfg(SubTerrainCfg):
  marker: int = 0

  def function(self, difficulty, spec, rng):
    height = 1. + self.marker + difficulty
    geom = spec.body("terrain").add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, height / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, height / 2),
    )
    origin = np.array([self.size[0] / 2, self.size[1] / 2, height])
    return TerrainOutput(
      origin=origin, geometries=[TerrainGeometry(geom=geom)],
      flat_patches={"spawn": origin[None, :]},
    )


def compact_cfg(types=5, levels=10):
  return TerrainGeneratorCfg(
    seed=42, size=(2., 2.), curriculum=True, num_rows=levels,
    num_cols=types, layout_shape=(4, 4), shuffle_curriculum_columns=True,
    sub_terrains={str(index): MarkedTerrainCfg(
      marker=index, proportion=index + 1,
      flat_patch_sampling={"spawn": FlatPatchSamplingCfg(num_patches=1)},
    ) for index in range(types)},
  )


def test_sixteen_nonoverlapping_patches_keep_all_types_and_endpoints():
  generator = TerrainGenerator(compact_cfg())
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert model.ngeom == 16
  assert generator.grid_shape == (4, 4)
  assert np.unique(model.geom_pos[:, :2], axis=0).shape == (16, 2)
  assert np.unique(model.geom_pos[:, 0]).size == 4
  assert np.unique(model.geom_pos[:, 1]).size == 4
  assert np.all(np.abs(model.geom_pos[:, :2]) <= 3.)
  assert np.all(np.diff(generator.terrain_level_indices, axis=0) >= 0)
  assert np.all(generator.terrain_level_indices <= np.arange(10)[:, None])
  np.testing.assert_array_equal(generator.terrain_level_indices[0], 0)
  np.testing.assert_array_equal(generator.terrain_level_indices[-1], 9)
  for level in range(10):
    for kind in range(5):
      origin = generator.terrain_origins[level, kind]
      actual = generator.terrain_level_indices[level, kind]
      np.testing.assert_allclose(origin[2], 1. + kind + actual / 9)
      distance = mujoco.mj_ray(
        model, data, origin + [0., 0., 1.], np.array([0., 0., -1.]),
        None, 1, -1, None,
      )
      np.testing.assert_allclose(distance, 1.)
      np.testing.assert_allclose(
        generator.flat_patches["spawn"][level, kind, 0], origin
      )


def test_entity_spawning_and_resampling_keep_semantic_type_five():
  entity = TerrainEntity(TerrainEntityCfg(
    terrain_type="generator", num_envs=64, terrain_generator=compact_cfg(),
  ), device="cpu")
  assert entity.grid_shape == (4, 4)
  assert entity.terrain_types.max() == 4
  types_before = entity.terrain_types.clone()
  levels_before = entity.terrain_levels.clone()
  entity.randomize_env_origins(torch.arange(64))
  assert entity.terrain_types.max() == 4
  assert entity.terrain_levels.max() <= 9
  actual = entity.get_geometry_levels(entity.terrain_levels, entity.terrain_types)
  assert torch.all(actual <= entity.terrain_levels)
  assert (types_before != entity.terrain_types).any()
  assert (levels_before != entity.terrain_levels).any()
  torch.testing.assert_close(entity.env_origins, entity.get_terrain_origins(
    entity.terrain_levels, entity.terrain_types,
  ))


def test_compact_layout_is_reproducible_and_rejects_missing_endpoints():
  first, second = TerrainGenerator(compact_cfg()), TerrainGenerator(compact_cfg())
  for generator in (first, second):
    generator.compile(mujoco.MjSpec())
  np.testing.assert_allclose(first.terrain_origins, second.terrain_origins)
  invalid = compact_cfg()
  invalid.layout_shape = (2, 2)
  with np.testing.assert_raises_regex(ValueError, "endpoints"):
    TerrainGenerator(invalid)


def test_world_bounds_follow_physical_map_instead_of_logical_table():
  class Scene(dict):
    pass

  scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
    root_link_pos_w=torch.tensor([[3.9, 0., 0.], [4.1, 0., 0.], [0., -4.1, 0.]]),
  )))
  scene.terrain = SimpleNamespace(
    cfg=TerrainEntityCfg(terrain_type="generator", terrain_generator=compact_cfg()),
    terrain_origins=torch.zeros(10, 5, 3),
  )
  env = SimpleNamespace(scene=scene, num_envs=3, device="cpu")
  torch.testing.assert_close(
    out_of_terrain_bounds(env, margin=0.), torch.tensor([False, True, True]),
  )
