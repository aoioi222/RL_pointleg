# rc_mjlab

开发上下文与模块索引见 [项目记忆 MEMORY.md](MEMORY.md)，包含当前任务状态、策略接口、训练迁移、验证入口及已知文档差异。

本目录保留轮腿机器人训练与部署资料，并在 `src/pointleg/` 开发纯足式四足机器人的障碍赛训练任务。当前点足开发以本节说明和实际代码为准；文档后半部分的轮腿内容作为历史参考保留。

## 当前点足项目开发记录

### 训练入口与模块分工

点足包位于 `src/pointleg/`，通过 `pyproject.toml` 的 `mjlab.tasks` 插件入口自动注册任务。执行 `uv sync --locked` 安装后，使用 `uv run --locked pointleg` 或 `python -m pointleg.run` 启动；入口为 [`src/pointleg/run.py`](src/pointleg/run.py)，不再动态构造包或修改模块搜索路径。项目发行名、导入包名、任务插件名和命令名统一为 `pointleg`。

| 模块 | 当前职责 |
|---|---|
| [`src/pointleg/__init__.py`](src/pointleg/__init__.py) | 在 `TASK_SPECS` 中集中定义任务说明、环境工厂、PPO 工厂和启用开关 |
| [`src/pointleg/config/env_cfgs.py`](src/pointleg/config/env_cfgs.py) | 选择地形，组合机器人、观测、动作、奖励、课程、事件和终止条件；提供训练及回放配置 |
| [`src/pointleg/config/rl_cfg.py`](src/pointleg/config/rl_cfg.py) | 定义 PPO 参数及各任务的独立实验日志名 |
| [`src/pointleg/terrains/__init__.py`](src/pointleg/terrains/__init__.py) | 导出自定义地形类，供环境配置引用 |
| [`src/pointleg/terrains/competition_terrains.py`](src/pointleg/terrains/competition_terrains.py) | 生成自定义障碍的物理几何体和出生原点 |
| [`src/pointleg/mdp/`](src/pointleg/mdp/) | 存放自定义奖励、终止条件、课程、命令和动作处理等逻辑 |

### 任务状态

任务按六类核心技能组织，跳跃作为高程越障的子任务单独预留训练入口。恢复、返回启动区及快速保守模式留给后续任务调度，不要求额外注册独立模型。

| 任务 ID | 用途 | 当前状态 | 实验日志名 |
|---|---|---|---|
| `PointLeg-Flat-v0` | 平地行走与机动，后续扩展绕桩和精定位 | 已启用，已有基础环境 | `pointleg_flat` |
| `PointLeg-Elevation-v0` | 800 mm 通道内向前上下楼梯、持续坡面 | 已接入初版，待策略验收 | `pointleg_elevation` |
| `PointLeg-StairsUp-v0` | 仅上台阶，单级120 mm | 已启用，待训练与验收 | `pointleg_stairs_up` |
| `PointLeg-StairsDown-v0` | 仅下台阶，单级120 mm | 已启用，待训练与验收 | `pointleg_stairs_down` |
| `PointLeg-SlopeUp-v0` | 仅上坡，约6.67～20°课程 | 已启用，待训练与验收 | `pointleg_slope_up` |
| `PointLeg-SlopeDown-v0` | 仅下坡，约6.67～20°课程 | 已启用，待训练与验收 | `pointleg_slope_down` |
| `PointLeg-Rough-v0` | 通用崎岖迁移训练：平地、连续起伏、Perlin 起伏和低矮凸块 | 已启用，正在训练与验收 | `pointleg_rough` |
| `PointLeg-Bridge-v0` | 单木桥平衡与完整通行 | 占位，未启用 | `pointleg_bridge` |
| `PointLeg-SteppingStones-v0` | 梅花桩精准踏点与全部桩顶接触覆盖 | 已启用，待正式训练与验收 | `pointleg_stepping_stones` |
| `PointLeg-Crawl-v0` | 低姿通过，后续扩展比赛限高框架 | 已启用，已有基础环境 | `pointleg_crawl` |
| `PointLeg-Jump-v0` | 固定触发前跳，跨平台间隙及中墙，四足稳定落地 | 平台版已接入，待正式训练与验收 | `pointleg_jump` |

“已启用”表示已有环境配置并已注册，不代表已完成比赛规则适配或通过率验收。Bridge 仍为占位；高程行走与独立跳跃的迁移、前置练习、评估命令及当前限制见 [高程任务训练入口](docs/ELEVATION_IMPLEMENTATION.md)。

2026-10-30：上下台阶、上下坡已拆成四个独立配置与实验目录，原 Elevation 混合任务保留。专项训练、权重迁移、续训和评估命令见 [四项独立训练](docs/ELEVATION_SPLIT_TRAINING.md)。

2026-10-27：梅花桩已接入目标方向推进、独立的抬腿/落点事件奖励及重复取分限制，保持105/381/12接口、2～8桩和严格接地失败规则。奖励参数、指标、验证及待完成事项见 [梅花桩奖励调整](docs/STEPPING_STONES_REWARDS.md)。

2026-10-27：当前 Jump 已改为两类平台跳跃，一个策略共用，台高/间距分别从15%起步；新规格、奖励、触发时间输入、训练与评估命令见 [固定触发平台跳跃](docs/JUMP_TRAINING.md)。历史三墙、330 mm 目标墙与 `--warmup` 说明不再适用于当前 Jump。

### 点足常用命令

以下 PowerShell 命令均在 `05_software/train/rc_mjlab` 目录执行，使用已有 `.venv` 环境：

```powershell
uv sync --locked
uv run --locked pointleg list
uv run --locked pointleg train PointLeg-Flat-v0
uv run --locked pointleg train PointLeg-Rough-v0
uv run --locked pointleg train PointLeg-Crawl-v0
uv run --locked pointleg play PointLeg-Rough-v0 --checkpoint "模型文件路径"
uv run --locked --no-sync pointleg evaluate PointLeg-Rough-v0 --checkpoint-file "模型文件路径" --profile nominal --episodes-per-case 100 --output artifacts\rough_nominal.json
uv run --locked --no-sync pointleg evaluate PointLeg-Rough-v0 --checkpoint-file "模型文件路径" --profile robust --episodes-per-case 100 --output artifacts\rough_robust.json
```

回放或评估前将 `模型文件路径` 替换为与该点足任务匹配的 checkpoint。评估报告记录检查点 SHA256、固定种子、地形类型和等级、指令、成功率、终止原因、速度误差、有效进度、足滑、接触过载和能耗。`nominal` 使用确定性环境，验收线为 90%；`robust` 保留物理随机化，验收线为 80%。已激活项目环境时，可用 `pointleg` 或 `python -m pointleg.run` 替代 `uv run --locked pointleg`。`list` 同时列出已注册任务和标有“占位，未启用”的预留任务；对占位任务执行 `train` 或 `play` 会提示对应开发入口并退出。完整回归测试命令为 `uv run --locked --no-sync python -m unittest discover -v`。

当前通用崎岖训练边界、迁移模型哈希、课程门控、正式命令和验收口径见 [通用崎岖地形训练与验收](docs/ROUGH_TRAINING.md)。

### 点足独立 Sim2Sim 验证

已提供 `pointleg train PointLeg-Flat-v0 --debug` 小规模调试入口，默认 16 环境、8 步 rollout、2 次迭代，本地记录日志且不上传模型。默认 play 已关闭物理随机化、重置扰动、观测噪声与随机动作延迟。Crawl 高度奖励与终止共用局部地面测量；修复清单、随机鲁棒性评估配置及训练闭环检查见 [B 类修复说明](docs/B_FIXES.md)。

十二个腿部电机均为 N6014B-12.6，实机供电 48 V。已录入驱动峰值 31.7 N·m、制动峰值 34.4 N·m 及 60 V 空载转速 54.2 rad/s；48 V 空载转速约 43.36 rad/s 仅为估算，不是已生效的速度约束。点足训练和回放的默认对称力矩限幅为 ±25 N·m；这是仿真上限，连续扭矩尚未提供，不能据此认定实机持续能力。当前仍为位置 PD，不模拟电压或扭矩—转速包络；完整说明见 [电机参数与建模边界](docs/MOTOR_PARAMETERS.md)。

Rough 和 Flat 保留有符号奖励，Crawl 的总奖励裁剪通过启动事件配置。Rough 的地形和速度范围采用表现门控课程：连续 100 个符合条件的完成回合成功率达到 80%后，才释放下一地形等级；失败或有效进度低于目标的 40%时降级。执行器 Kp/Kd 使用联合 PD 增益随机化。Crawl 默认膝角为 `-2.50 rad`，位于软限位内，旧 Crawl checkpoint 应按当前姿态重新验证。

使用 `sim2sim/sim2sim.py` 在原生 CPU MuJoCo 验证点足 checkpoint，复用当前任务的观测和动作处理，支持 45 维观测、12 维动作和 PT/ONNX。默认地形为平地；`--task` 必须与模型对应，历史轮腿权重不能用于点足回放。

```powershell
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --check-model --headless --duration 5
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --task PointLeg-Flat-v0 --checkpoint "模型路径\model_1000.pt" --duration 0
```

窗口键位为 W/S 前后、A/D 左右、Q/E 转向、空格停止、R 重置。完整选项、无窗口验证和 ONNX 依赖见 [sim2sim/README.md](sim2sim/README.md)。

### 自定义地形与训练项目的对应关系

`terrains/` 存放从环境配置中抽离的自定义地形生成模块。地形与策略不一一对应：一个任务可以混合多种地形，同一种地形也可以被多个任务复用。普通平地、台阶和高度场可直接使用 MJLab 内置地形，不必在这里重复实现。

| 自定义地形类 | 当前实际接入 | 后续适用方向 |
|---|---|---|
| `RCWallTerrainCfg` | 已定义并导出，当前未接入通用 Rough | 高程越障及跳跃的初步跨障训练；不等同于完整比赛高墙或沟壑 |
| `RCLowBarTerrainCfg` | `PointLeg-Crawl-v0` 的 `rc_low_bar`，生成一根低横杆及两侧支柱 | 降姿、低姿行走和穿杆后恢复；比赛限高框架需另行补充 |
| `RCPyramidStairsTerrainCfg` | 已定义并导出，尚未接入当前环境 | 高程越障中的上下台阶及崎岖地形基础训练，接入前检查出生点与支撑面高度 |

当前 `Rough` 只使用 MJLab 的平地、随机连续高度场、Perlin 高度场和低矮随机凸块。正/倒台阶、高墙和大坡面不属于通用崎岖训练，后续由高程越障任务负责。

地形生成模块负责场地几何与出生位置，环境配置负责选择地形和组织训练；入口出口是否合法、是否踩完规定桩顶、是否完整通过等规则判定，需要结合真实接触与任务记忆另行实现。

### 后续开发约定

1. 在 `terrains/` 中补充目标障碍的几何生成逻辑，按需在其 `__init__.py` 导出，并在 `config/env_cfgs.py` 的目标环境函数中引用。未实现的地形类不加入已启用环境的 `sub_terrains`。
2. 实现对应环境工厂的训练与回放配置，补齐观测、任务目标、规则与阶段、奖励及终止条件。当前仍未实现的占位入口为 `bridge_env_cfg`；其他已接入任务继续补齐专项验收。
3. 检查对应 `config/rl_cfg.py` 中的 PPO 工厂，保留独立实验名，再将任务注册表 `TASK_SPECS` 中对应的 `enabled` 改为 `True`。
4. 核对 `list` 输出，检查训练与回放配置、场景生成、出生区域以及成功和失败判定，再启动训练。注册需要完整配置对象，不能让占位函数返回 `None` 后直接启用。

以下比赛专用场景仍待开发；基础噪声地形、连续矮墙和低横杆不能替代其完整实现：

| 待开发场景 | 主要接入方向 |
|---|---|
| 单木桥及两端进出区域 | `Bridge` |
| 沟壑及起落区、组合台阶和坡面 | `Elevation`、`Jump` |
| 有通道长度的比赛限高框架 | `Crawl` |
| 轮胎阵及孔洞、曲面等几何 | 待注册的 `PointLeg-RoughCompetition-v0` |
| 砖木坑及实际填料、护栏 | 待注册的 `PointLeg-RoughCompetition-v0` |
| 绕桩竖杆及有顺序的必达区域 | `Flat` |

开发依据为《点足式四足机器人障碍赛_训练与奖励函数设计思路.md》，当前保存在 `E:/OneDrive/Desktop/点足式四足机器人障碍赛_训练与奖励函数设计思路.md`。障碍尺寸和合法通过条件以原始规则及场地图为依据，未完成的几何、接触检测、规则判定和奖励功能不视为已实现。

---

## 历史参考：轮腿机器人训练与部署

从本节起，以下所有章节均保留原轮腿项目资料，涉及 `Robot-*` 任务、旧目录结构、轮部控制、模型、Sim2Sim、奖励权重与地形比例。这些描述不作为当前点足任务的配置依据；当前点足入口与任务状态以文档前半部分及 `src/pointleg/` 代码为准，历史轮腿模型不能直接视为点足任务可用模型。

基于 [mjlab](https://github.com/google-deepmind/mjlab) 框架的四轮腿混合机器人强化学习训练与部署部署项目，面向机器人竞赛场景（如越障、匍匐、斜坡、台阶等复合任务）。

> 当前目录对应 `v0.8.1`：保留比赛训练架构、后期 MuJoCo/Sim2Sim 和比赛最终 Rough 策略，并补充导航地图、打点工具、路线迭代与抽样 PCD。`model_rough.pt` 仍作为早期参考权重保留。

---

> 📄 **要为新机器人/新环境新建任务？** 见 [`NEW_ROBOT_GUIDE.md`](NEW_ROBOT_GUIDE.md)：逐文件列出需要改的参数、旧值对照、执行顺序与易漏的坑。

---

## 🛠️ 项目简介

本项目针对一台 **4 腿 × 3 关节 + 4 驱动轮（轮腿混合）** 的移动机器人，在 MuJoCo 物理引擎中利用 PPO 算法进行多任务运动控制策略训练。

系统设计特点包括：
1. **高保真动力学步进**：物理仿真计算步长设为 **`2ms` (0.002s)**，为碰撞、地面力学传递提供极高的解算频宽与稳定性。
2. **50Hz 控制决策循环**：通过在环境中设置 `decimation = 10`，策略决策周期为 `20ms` ($0.002\text{s} \times 10 = 0.02\text{s}$)，即控制决策频率为 **`50Hz`**，完全对齐真机控制周期。
3. **混合滤波执行器**：
   - 腿部 12 个位置控制关节采用位置 PD 伺服（$K_p=40, K_d=1$），并叠加截止频率为 **`5Hz`** 的低通滤波器进行动作平滑，减小高频机械抖动。
   - 轮部 4 个速度驱动关节采用阻尼速度伺服（$K_d=0.5$），叠加截止频率为 **`15Hz`** 的低通速度滤波器，保证转速响应的灵敏度。
4. **大规模并行加速**：利用 GPU 并行（通过 Warp 和 MuJoCo GPU 物理管线），支持最多 $4096$ 环境同时训练，并包含对动作变化率、关节加速度的惩罚项以平抑噪声。

---

## 📦 项目结构

```
rc_mjlab/
├── src/robot/                    # RL 训练任务包（主体代码）
│   ├── __init__.py               # 任务注册（Robot-Flat-v0 / Robot-Rough-v0 / Robot-Crawl-v0）
│   ├── robot_cfg.py              # 机器人物理参数（PD 增益、执行器上限、碰撞属性）
│   ├── config/
│   │   ├── env_cfgs.py           # 三套环境完整配置（观测、奖励、事件、地形、终止条件）
│   │   └── rl_cfg.py             # PPO 超参数（网络结构、学习率、折扣因子等）
│   ├── mdp/
│   │   ├── rewards.py            # 自定义奖励函数（速度追踪、姿态约束、接触、越障反射惩罚等）
│   │   ├── curriculums.py        # 地形关卡课程（严格速度约束版）+ 自适应速度范围
│   │   ├── lowpass_actions.py    # 低通滤波动作包装（腿 5 Hz / 轮 15 Hz IIR 滤波）
│   │   ├── disturbances.py       # 持续外力扰动（一阶低通滤波平滑随机外力/扭矩）
│   │   ├── mode_command.py       # 离散步态模式命令（保留扩展用）
│   │   └── only_positive_rewards.py  # HIMLoco 风格：每步总奖励截断为 ≥ 0，防止消极逃避
│   └── terrains/
│       └── competition_terrains.py   # 竞赛自定义地形（高墙障碍、低杆障碍）
├── sim2sim/                      # Sim2Sim 物理部署与高精度交互回放工具
│   ├── nav_sim2sim.py            # 主程序：2D Pygame 交互面板 + 全自动多地形导航追踪
│   ├── nav_route_sim2sim_check.py # ONNX 策略批量路线检查
│   ├── ik_slalom_sim2sim.py      # 纯 IK、路径跟踪与绕桩验证
│   ├── ik_compensation_sweep.py  # IK 补偿参数扫描
│   ├── export_onnx.py            # PT actor 导出与 ONNX 一致性检查
│   ├── sim2sim.py                # 简易版键盘调试工具
│   ├── interface/
│   │   └── mujoco_io.py          # MuJoCo 输入输出与传感器、低通滤波器接口
│   ├── tools/
│   │   └── math_utils.py         # 姿态重力等数学转换
│   ├── policy/                   # 保存的 pt 策略权重
│   └── terrain/
│       └── scene_terrain.xml     # 完整越障比赛场地的物理 XML 定义
├── mjcf/
│   ├── wheelleg.xml              # 机器人 MuJoCo 模型（含网格引用）
│   ├── scene.xml                 # mjlab 场景入口文件
│   └── meshes/                   # STL/OBJ 碰撞与外观网格
├── mujoco_sim/                   # 姿态、IK、动力学和 MPC 独立工具
├── tools/nav_tools/              # 地图/PCD/航点编辑、路线检查与比赛路线数据
├── model_rough.pt                # 早期 Rough 参考 checkpoint
├── model_6800.onnx               # 比赛最终 Rough 策略
├── pyproject.toml                # 项目依赖（uv 管理，含清华镜像源加速）
└── uv.lock                       # 精确依赖锁定文件
```

---

## 🚀 常用命令

### 1. 训练与回放

```bash
# 运行平地基础训练 (Robot-Flat-v0)
uv run train Robot-Flat-v0

# 运行多障碍复杂地形训练 (Robot-Rough-v0)，可从 Flat 的Checkpoint热启动
uv run train Robot-Rough-v0 --agent.resume True --agent.experiment-name robot_flat

# 运行爬坡与匍匐限高任务 (Robot-Crawl-v0)
uv run train Robot-Crawl-v0

# 使用默认 20 个并行环境回放最新 checkpoint 效果
uv run play Robot-Rough-v0
```

### 2. 交互式 Sim2Sim 自动导航仪表盘

我们提供了一个强大的 GUI 交互和全自动障碍赛追踪平台，位于 `sim2sim` 目录下：

```bash
# 启动 2D 交互导航平台
cd sim2sim
uv run python nav_sim2sim.py
```

后期 Sim2Sim 的入口、模型边界和批量检查命令见 [`sim2sim/README.md`](sim2sim/README.md)。

导航打点工具、路线快照和抽样点云说明见 [`tools/nav_tools/README.md`](tools/nav_tools/README.md)。

---

## 🖥️ 交互式自动导航平台 (sim2sim/nav_sim2sim.py)

该平台包含一个 **Pygame 2D HUD 监控面板** 和一个 **实时 MuJoCo 3D 渲染器**，支持对仿真参数和任务执行的精细控制。

### 1. 按钮面板分区与布局

面板在垂直方向进行了高紧凑性排版，避免控件重叠，并在底端留有安全间距：
* **【预设任务列表】**（按物理穿越顺序排列）：
  - **S形绕杆 (Slalom)**：绕过红蓝两色障碍杆路径。
  - **限高下蹲 (Crawl)**：降低机身高度穿过低杆障碍。
  - **砂砾碎石 (Gravel)**：平稳低速通过多颗粒非结构碎石坑。
  - **高墙越障 (Wall)**：高速度冲向障碍高墙，利用前轮攀爬反射爬越。
  - **台阶攀爬 (Stairs)**：攀越分段式台阶。
  - **斜坡木桥 (Bridge)**：穿过A坡并稳健从B坡落地。
  - **障碍赛大满贯 (Grand)**：**科技紫**圆角高亮按钮。点击后，机器人将以**顺时针**方向，自动、连贯且闭环地一次性穿越上述全部 6 个核心比赛障碍，并在木桥落地后，通过安全通道直角返航至起终点。
* **【系统与视图控制】**：
  - **清除与停止 (Stop)**：一键紧急停止并重置当前目标航点。
  - **视角居中 (Center)**：一键锁定相机随机器人机身移动。
  - **物理流速三联排 (倍速- / 标准 / 倍速+)**：在不破坏物理计算数值稳定性的前提下，实现对仿真总体时间的平滑加速与慢放（支持 `0.2x` ~ `5.0x`，可随时点击“标准”一键归位 `1.0x`）。
* **【目标微调与命令终端】**：
  - 拥有高精度航点微调发令键。
  - 底部命令行支持输入 `speed <倍率>` 更改仿真速度，或者输入 `grand` 直接开启大满贯。

---

## 📊 机器人系统规格参数

### 1. 机器人本体参数

| 参数项 | 基准数值 | 说明 |
|---|---|---|
| **物理步长 ($dt_{physics}$)** | `0.002s` (2ms) | 底层 MuJoCo 求解器步长，物理精度极高 |
| **控制决策频率 ($Freq_{ctrl}$)** | `50Hz` (20ms) | $decimation = 10$，环境每 10 个子步进行一次交互决策 |
| **单轮仿真时长** | `30.0s` | 最大决策步数上限为 $30.0 / 0.02 = 1500$ 步 |
| **腿部控制** | 位置 PD 伺服 | 目标关节角限幅 ±0.25 rad，叠加 **5Hz** 低通滤波器 |
| **轮部控制** | 阻尼速度伺服 | 目标速度限幅 ±10.0 rad/s，叠加 **15Hz** 低通滤波器 |
| **结构形式** | 4腿 × 3关节 + 4轮 | 腿：hip abduction, hip pitch, knee；轮半径 0.1m，左右轮距 0.32m |
| **关节扭矩上限** | 17.0 Nm | 关节最大输出力矩（训练时含 80%~100% 随机缩放） |
| **最大关节角速度** | 13.0 rad/s | 关节最大运动速度限制 |

### 2. 状态观测空间 (Actor Obs, 53维)

网络输入包含 $6$ 步历史数据，并在训练时注入均匀高斯噪声以提升泛化能力：

| 观测项目 | 维度 | 缩放比例 | 噪声范围 |
|---|---|---|---|
| 基座角速度 (ang_vel) | 3 | 0.25 | $[-0.2, 0.2]$ rad/s |
| 投影重力向量 (projected_gravity) | 3 | 1.0 | $[-0.05, 0.05]$ |
| 指令速度 (vx, vy, wz/heading) | 3 | 1.0 | — |
| 腿部关节相对角度 (joint_pos_rel) | 12 | 1.0 | $[-0.01, 0.01]$ rad |
| 腿部关节角速度 (joint_vel) | 12 | 0.05 | $[-1.5, 1.5]$ rad/s |
| 轮子角速度 (wheel_vel) | 4 | 0.05 | $[-1.0, 1.0]$ rad/s |
| 上一步动作缓存 (last_actions) | 16 | 1.0 | — |

> **Critic 附加观测**：包含高精度基座物理线速度、轮地实际接触状态、以及 $1.6\text{m} \times 1.0\text{m}$ 分辨率为 $0.08\text{m}$ 的高度雷达扫描网格，提供大范围越障感知。

---

## ⚖️ 奖惩体系设计 (Robot-Rough-v0)

复杂地形任务采用 **“仅正奖励截断”** 机制（即每步累加的总奖励若小于0则强制截断为0），防止机器人在困难关卡早期选择倒下自杀来规避负惩罚。

### 1. 运动追踪与状态惩罚

| 奖励/惩罚项 | 权重 (Weight) | 适用函数 / 物理意义 |
|---|---|---|
| **track_lin_vel** | `+4.5` | L1 范数水平线速度跟踪奖励，平缓高速漂移 |
| **track_ang_vel** | `+2.0` | 偏航角速度指数跟踪奖励 |
| **stand_still** | `-2.0` | 当速度指令为 0 时，严厉惩罚关节多余晃动，保持稳立 |
| **joint_pos_penalty** | `-0.8` | 当速度指令为 0 时，惩罚关节角度偏离初始对齐姿态，维持高刚度 |
| **roll_penalty** | `-1.0` | 机身横滚角 (Roll) 倾斜惩罚，抑制左右倾倒抖动 |
| **pitch_penalty** | `-1.5` | 俯仰角 (Pitch) 死区惩罚，限制仰角不超过 29 度，抑制越障瞬间前轮翘头和后翻 |
| **base_height_l2** | `-0.5` | 机身高度偏离 0.36m 惩罚（基于高度扫描均值，允许自适应高低） |

### 2. 能量正则与平滑惩罚 (平抑高频抖动)

| 奖励/惩罚项 | 权重 (Weight) | 适用函数 / 物理意义 |
|---|---|---|
| **action_rate_curriculum** | `-0.005` | 动作变化率 L2 惩罚，迫使连续两个决策步的输出动作变化平滑 |
| **joint_torques** | `-1.0e-4` | 关节输出扭矩 L2 正则，降低电机总发热和冲击性载荷 |
| **leg_joint_acc_l2** | `-2.5e-7` | 限制腿部 12 关节**角加速度**，直接抑制关节高频电磁和机械震荡 |
| **wheel_joint_acc_l2** | `-2.5e-9` | 限制 4 个驱动轮的**角加速度**，平缓轮速切换，降低打滑振荡 |
| **joint_pos_limits** | `-0.2` | 极度接近关节极限限位阻挡时的硬惩罚 |

### 3. 接触反射与安全约束

| 奖励/惩罚项 | 权重 (Weight) | 适用函数 / 物理意义 |
|---|---|---|
| **feet_contact_without_cmd** | `+0.1` | 当速度指令为 0 时，鼓励四轮保持稳定接地的正向收益 |
| **body_collision** | `-1.0` | 腿部连杆（大腿、小腿）触地碰撞惩罚，迫使抬腿跨越障碍 |
| **base_collision** | `-5.0` | 机身/底盘硬撞障碍物时的严厉惩罚，逼迫机器人学会抬起前轮支撑攀爬 |
| **is_terminated** | `0.0` | 关闭越障任务的提早终止，允许机器人跌倒后自行挣扎起立，提高生存极限 |

---

## 🌀 域随机化 (Domain Randomization)

为了使训练的控制策略具有卓越的零样本真机部署能力，在环境重置及仿真运行中注入了高强度的域随机化参数：

| 随机化项目 | 扰动操作 | 随机范围 |
|---|---|---|
| **机身质心偏移 (base_com)** | 加法 | X, Y, Z 三轴分别随机偏置 `[-0.05, 0.05]` 米 |
| **角度传感器零偏 (encoder_bias)** | 加法 | 关节传感器绝对偏置 `[-0.015, 0.015]` rad (约 $\pm 0.85^{\circ}$) |
| **几何表面摩擦力 (body_friction)** | 绝对值 | 地面及机器人碰撞几何体摩擦力在 `[0.3, 1.2]` 均匀随机 |
| **关节摩擦阻尼 (joint_friction)** | 乘法 | 所有旋转轴关节运动阻尼摩擦在原值的 `[0.7, 1.3]` 倍间随机 |
| **关节传动刚度 (actuator_stiffness)** | 乘法 | Kp 刚度系数在原值的 `[0.9, 1.1]` 对数均匀范围内随机缩放 |
| **关节传动阻尼 (actuator_damping)** | 乘法 | Kd 阻尼系数在原值的 `[0.9, 1.1]` 对数均匀范围内随机缩放 |
| **力矩输出上限 (actuator_effort_limit)**| 乘法 | 最大输出扭矩极限随机在原值的 `[0.8, 1.0]` 倍均匀缩放 |
| **负载质量 (payload_mass)** | 加法 | 在机身处添加载荷质量，扰动范围在 `[-1.0, 3.0]` kg |
| **瞬时侧向推撞 (push_robot)** | 脉冲 | 每隔 `[5.0, 10.0]` 秒，瞬间施加 X/Y 轴 `[-0.5, 0.5]` m/s 冲击速度 |
| **一阶低通持续风阻 (continuous_disturbance)** | 连续 | 机身持续叠加随机外力（±15N）与力矩（±10Nm），低通周期 0.5s |

---

## 🏆 多地形关卡难度控制 (Robot-Rough-v0)

共有 8 种子地形按照比例混合，通过自适应升级距离控制关卡难度的推进：

| 地形名称 | 混合比例 (Proportion) | 最大配置难度 |
|---|---|---|
| **平地 (flat)** | `5%` | 作为初始安定性恢复区域 |
| **金字塔台阶 (pyramid_stairs)** | `25%` | 最大阶梯高度上限 `0.30` 米，级宽 0.30m |
| **倒金字塔台阶 (pyramid_stairs_inv)** | `10%` | 最大倒台阶高度上限 `0.30` 米，级宽 0.30m |
| **随机高度网格 (random_grid)** | `10%` | 最大网格方块起伏上限 `0.30` 米 |
| **随机粗糙地形 (random_rough)** | `5%` | 地表最大颗粒随机噪声起伏 `0.06` 米 |
| **柏林噪声地形 (perlin_noise)** | `5%` | 大范围高平缓起伏最大高度 `0.06` 米 |
| **越障高墙地形 (rc_wall)** | `25%` | 自定义跳跃垂直高墙，最大墙高上限 `0.45` 米 |
| **平台斜坡地形 (sloped_terrain)** | `15%` | 最大坡度限制 `0.325` (约 $18.5^{\circ}$) |

> **地形升级规则**：当机器人朝指令方向行进距离超过当前地块的一半（4米），且实际行进距离大于速度指令对应期望距离的 45% 时，该环境关卡等级 +1。  
> **地形降级规则**：当指令速度大于 0.1m/s 但实际行进距离小于期望距离的 25%，或者实际移动不足 2.0米时，环境难度等级 -1。
