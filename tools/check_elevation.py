"""Check real elevation scenes, observations, rewards and task resets."""

import argparse

import torch

import pointleg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from pointleg.terrains.elevation import STAIR_TREAD_COUNT, STAIR_PLATFORM_RANGE


def check_viser(env):
    import viser
    from mjlab.viewer import ViserPlayViewer

    server = viser.ViserServer(host="127.0.0.1", port=0, label="Elevation GUI check")
    viewer = ViserPlayViewer(
        env, lambda obs: torch.zeros(env.num_envs, 12, device=env.device),
        viser_server=server,
    )
    try:
        viewer.setup()
        viewer.sync_env_to_viewer()
        viewer.sync_env_to_viewer()
    finally:
        try:
            viewer.close()
        finally:
            server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("PointLeg-Elevation-v0", "PointLeg-Jump-v0"))
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viser-gui", action="store_true", help="Initialize and close the real Viser scene and GUI")
    args = parser.parse_args()
    for task in (args.task,) if args.task else ("PointLeg-Elevation-v0", "PointLeg-Jump-v0"):
        cfg = load_env_cfg(task, play=True)
        count = 4 if task == "PointLeg-Jump-v0" else 5
        cfg.scene.num_envs = count
        env = ManagerBasedRlEnv(cfg, device=args.device)
        try:
            obs, _ = env.reset()
            assert obs["actor"].shape == (count, 45)
            assert obs["critic"].shape == (count, 325)
            types = env.scene.terrain.terrain_types
            if task.endswith("Jump-v0"):
                assert set(types.tolist()) == {0, 1}
                assert env.scene.terrain.grid_shape == (10, 5)
                assert env.scene.terrain.terrain_origins.shape == (25, 2, 3)
            else:
                assert set(types.tolist()) == {0, 1, 2, 3, 5}
                assert env._elevation["covered"].shape == (count, STAIR_TREAD_COUNT)
            if task == "PointLeg-Elevation-v0":
                approach = env._elevation["entry_x"] - env._elevation["start_x"]
                assert bool(((approach >= 1.5) & (approach <= 2.1)).all())
                for name in ("stairs_up", "stairs_down"):
                    sub = env.scene.terrain.cfg.terrain_generator.sub_terrains[name]
                    kind = list(env.scene.terrain.cfg.terrain_generator.sub_terrains).index(name)
                    for level in env.scene.terrain.terrain_level_indices[:, kind].unique().tolist():
                        profile = sub.generated_profile(level / 4)
                        assert len(profile.step_surfaces) == STAIR_TREAD_COUNT
                        assert STAIR_PLATFORM_RANGE[0] <= profile.stair_platform_length <= STAIR_PLATFORM_RANGE[1]
            supported = False
            for _ in range(args.steps):
                obs, reward, _, _, _ = env.step(torch.zeros(count, 12, device=env.device))
                assert torch.isfinite(reward).all()
                assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
                if task.endswith("Jump-v0"):
                    supported |= bool((env._jump["source_support"].sum(dim=1) > 0).any())
                else:
                    supported |= bool((env._elevation["support_count"] > 0).any())
            if args.steps >= 64:
                assert supported, "Physical support contacts were not detected"
            if task == "PointLeg-Elevation-v0":
                assert env.scene.terrain.terrain_origins.shape == (5, 6, 3)
                assert env.scene.terrain.grid_shape == (4, 4)
            print(f"PASS {task}: {env.scene.terrain.grid_shape} scene, actor45/critic325/action12, {args.steps} finite steps")
            if args.viser_gui:
                check_viser(env)
                print(f"PASS {task}: Viser scene and command panel initialized")
        finally:
            env.close()


if __name__ == "__main__":
    main()
