"""Resolve terrain names without interpreting random patch columns as types."""

import torch


def terrain_type_mask(env, names) -> torch.Tensor:
    """Select terrain types in a type-column grid or a single-type generator."""
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if generator is None:
        return mask
    keys = list(generator.sub_terrains)
    if len(keys) == 1:
        mask[:] = keys[0] in names
        return mask
    if not generator.curriculum:
        raise ValueError("Terrain-conditioned commands and rewards require type-column terrain generation (curriculum=True)")
    for name in names:
        if name in keys:
            mask |= terrain.terrain_types == keys.index(name)
    return mask
