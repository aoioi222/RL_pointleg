# 高程任务入口

2026-11-01：原四个单方向任务及其配置模块已经移除。当前仅保留以下三个高程行走入口：

| 任务 | 地形范围 | 实验目录 |
| --- | --- | --- |
| `PointLeg-Stairs-v0` | 上台阶、下台阶 | `logs/rsl_rl/pointleg_stairs/` |
| `PointLeg-Slope-v0` | 上坡、下坡 | `logs/rsl_rl/pointleg_slope/` |
| `PointLeg-Elevation-v0` | 上下台阶、上下坡及原平地复习 | `logs/rsl_rl/pointleg_elevation/` |

两个合并任务使用版本8的14项基础奖励，保留掉落失败与侧漂处罚；混合Elevation维持原版本7配置。合并任务的奖励和可直接resume的初始化权重及训练指令见[合并训练说明](ELEVATION_PAIRED_TRAINING.md)。

原四个方向的日志和检查点文件保留。任务入口下线后，旧任务ID无法再用于train/play/evaluate/transfer；如需复用旧模型参数，应把该文件纯权重迁移到当前任务之一，再对目标任务续训。

四种上/下方向的地形定义仍用于这三个任务的内部采样与评估，不会因删除旧任务入口而删除相应地图几何。
