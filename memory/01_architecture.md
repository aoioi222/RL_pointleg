# 架构与代码导航

2026-11-01最新入口清理：原四个单方向TaskSpec、import和配置模块已删除，`ELEVATION_TASK_TERRAINS`只含混合Elevation及Stairs/Slope两个合并任务。当前全工程11个TaskSpec、10个启用；四种地形内部定义继续用于双向/混合采样。新任务、旧混合和现有模型数据保留，旧方向ID不再注册。下文15/14及四方向兼容描述为清理前历史。

2026-11-01：新增后续使用的两个合并入口 `PointLeg-Stairs-v0`、`PointLeg-Slope-v0`，配置为 `config/stairs.py`、`config/slope.py`；共同复用 `config/elevation_basic.py` 的14项基础白名单及 `mdp/elevation_basic.py` 的独立速度/真实承重反馈。`terrains/elevation.py:walking_terrain_names`统一解析混合、双向组和单方向范围，重置采样、课程指标和评估均使用该解析；runner为合并任务保存版本8元数据，旧任务版本7保留。现有15个TaskSpec、14个启用，原四个方向入口保留兼容。新指令见 `docs/ELEVATION_PAIRED_TRAINING.md`。

核对日期：2026-10-26。入口见 [项目记忆](../MEMORY.md)。以下路径相对 `rc_mjlab` 根目录。

2026-10-30：Elevation增加四份专项配置`config/stairs_up.py`、`stairs_down.py`、`slope_up.py`、`slope_down.py`，各自提供环境/PPO工厂、独立实验目录，复用`config/elevation.py`和`mdp/elevation*.py`。`terrains/elevation.py:ELEVATION_TASK_TERRAINS`维护任务到地形范围映射，`__init__.py:TASK_SPECS`注册。`run.py`将四项评估分派至`evaluate_elevation.py`，`transfer.py`及`ElevationOnPolicyRunner`处理专项权重迁移和同任务续训校验。原混合任务保留，注册总数13、启用12；训练指南见`docs/ELEVATION_SPLIT_TRAINING.md`。

## 目录边界

| 路径 | 当前职责 |
|---|---|
| `pyproject.toml`、`uv.lock` | 包配置、命令、任务插件、本地 mjlab 与依赖锁定 |
| `src/pointleg/` | 当前点足训练、配置、MDP、评估与迁移 |
| `mjcf/pointleg.xml` | 默认点足机器人；支持用 `POINTLEG_XML` 替换 |
| `mjlab/` | 本地可编辑安装的框架源码，存在项目定制 |
| `sim2sim/pointleg_sim2sim.py` | 原生 CPU MuJoCo 的点足推理实现 |
| `sim2sim/sim2sim.py` | 点足 CPU 回放入口 |
| `sim2sim/export_onnx.py` | 当前 45 维点足 Actor 的严格导出与数值验证 |
| `mujoco_sim/` | 历史姿态、IK、动力学和 MPC 分析工具 |
| `tools/nav_tools/` | 历史导航地图、PCD、航点和路线工具 |
| `docs/` | 任务设计、训练操作、电机参数、修复记录 |
| `tests/test_suites.py` | 汇总点足源码与独立 Sim2Sim 的 unittest 测试 |
| `logs/rsl_rl/` | 实验参数、训练日志与 checkpoint，本地工件 |
| `artifacts/`、`pretrained/` | 报告、诊断输出、迁移源模型，本地工件 |

`.gitignore` 排除了环境、日志、工件、预训练目录、`*.pt` 和 `*.onnx` 等。源码存在不意味着另一个开发环境也拥有配套模型。

## 启动链路

```text
pyproject.toml
  mjlab.tasks 插件 pointleg = pointleg
  pointleg 命令 = pointleg.run:main
       |
       v
src/pointleg/__init__.py: TASK_SPECS
  启用任务 -> 构建 train/play 环境配置与 PPO 配置 -> 注册到 mjlab
       |
       v
src/pointleg/run.py: main
  list      -> 列注册任务与未启用占位项
  train     -> mjlab.scripts.train；debug/warmup 分支先调整默认配置
  play      -> mjlab.scripts.play
  evaluate  -> 通用评估或 Elevation/Jump 专用评估
  transfer  -> Elevation/Jump 的纯网络权重迁移
```

导入 `pointleg` 就会构建所有已启用任务的训练与回放配置。因此新增任务的配置错误可能使其他任务的 CLI 也无法启动。占位函数不能提前启用后返回空配置。

## 修改入口

| 修改目标 | 首先查看 |
|---|---|
| 注册、任务名、启用开关、专用 runner | `src/pointleg/__init__.py` |
| 机器人关节、默认姿态、PD、力矩与碰撞 | `src/pointleg/robot_cfg.py`、`motor_spec.py`、`mjcf/pointleg.xml` |
| 公共传感器、观测、动作、Flat/Rough/Crawl | `src/pointleg/config/env_cfgs.py` |
| 混合障碍场景与附加奖励 | `src/pointleg/config/mixed_obstacles.py` |
| 完成摆腿峰值的纯统计 | `src/pointleg/mdp/swing_metrics.py`，Mixed配置接入 |
| 楼梯、坡面 | `src/pointleg/config/elevation.py` |
| 固定触发平台跳跃 | `src/pointleg/config/jump.py`、`mdp/jump.py`、`terrains/jump.py` |
| Elevation四髋持续侧偏 | `src/pointleg/mdp/gait_bias.py`，只在行走奖励接入 |
| 梅花桩训练与任务观测 | `src/pointleg/config/stepping_stones.py` |
| PPO 网络、学习率、实验名 | `src/pointleg/config/rl_cfg.py` |
| 小规模训练参数 | `src/pointleg/config/debug.py` |
| nominal/robust 配置 | `src/pointleg/config/evaluation.py` |
| 奖励、终止、课程、命令和动作处理 | `src/pointleg/mdp/` |
| 物理场地与出生原点 | `src/pointleg/terrains/` |
| 同任务状态保存、几何校验 | `src/pointleg/rsl_rl/runners/elevation_runner.py` |

2026-10-27：Jump 独立使用 `rsl_rl/runners/jump_runner.py`、`evaluate_jump.py`，部署命令时钟为 `jump_trigger.py`。仍共享基础机器人、动作和观测构造，Elevation 行走参数未更改。Jump 的三命令槽改为触发/时间/零，详情见 `docs/JUMP_TRAINING.md`。

`env_cfgs.py` 的 Elevation、Jump、SteppingStones 工厂采用局部导入，专用配置再复用基础工厂。扩展时保留依赖方向，避免循环导入。

## 地形、规则与奖励各做什么

- `terrains/`：生成真实碰撞几何、地块出生原点及布局元数据。
- `config/`：选择地形、传感器、命令、奖励、终止、课程和回放设置。
- `mdp/`：根据真实接触与任务状态判定进入、覆盖、离开、失败和完成。

添加一个可视化障碍，不等于实现该障碍任务。涉及短暂触地或落地的规则，需要保留物理子步采集，不能只检查 50 Hz 控制帧。

## 框架定制

当前 `mjlab` 元数据版本为 1.4.0；通过根 `pyproject.toml` 的 `path="mjlab", editable=true` 使用。不要把它当作可以直接替换的无修改第三方目录。

本次工作树已显示改动的关键框架文件包括：

- `mjlab/src/mjlab/terrains/terrain_generator.py`：紧凑物理布局、逻辑课程到实际几何等级的映射。
- `mjlab/src/mjlab/terrains/terrain_entity.py`：按地形比例分配环境、保存地形等级映射并解析原点。
- `mjlab/src/mjlab/terrains/heightfield_terrains.py`：项目使用的高度场实现。
- `mjlab/src/mjlab/tasks/velocity/mdp/terminations.py`：地图边界适配实际布局。

专项测试见 `src/pointleg/tests/test_terrain_layout.py` 和 `mjlab/tests/test_compact_terrain_layout.py`。历史 `DEPENDENCIES.md` 的“只有一处本地修改”已不足以描述当前状态。修改框架目录前另读其 `AGENTS.md`。
