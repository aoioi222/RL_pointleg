# Sim2Sim 与辅助工具

2026-11-18 反向摆腿修复的目标方向投影与摆动足后退代价位于共享 `FixedStoneGait`。原生回放继续使用同一目标/完成逻辑，机身倒退与提前失败项为训练奖励。静态 FK 核对首步目标在前，159 项 CPU/静态原生/ONNX 检查通过。本轮只做前向位姿计算，未推进物理时间。

2026-11-18 二次审查的单次摆动资格、双摆重置和同桩避让都位于共享 `FixedStoneGait`，训练与 123 维原生回放共同采用。合法落脚在足球间距至少 5 cm 时确认，旧 105 维回放路径仍由原接口执行。151 项组合检查包含原有十项静态原生用例与新旧 ONNX 检查，未推进物理时间。

2026-11-18 起步停滞修复同步至原生回放。123 维固定步态的首份观测直接从实际足底和场景几何初始化目标，重置后重新初始化，尚未有有效接触载荷时也显示左前/右后掩码。共享状态仍通过同步和稳定承重检查整步完成。静态回归检查首次观测和重置后的掩码、计时及累计步数。142 项组合检查通过，未推进物理时间。

## 2026-11-18 梅花桩新旧接口选择

`sim2sim/pointleg_sim2sim.py` 从 PT 第一输入层或 ONNX 输入宽度识别 105 维历史模型和 123 维固定腿序模型。`configure_stepping_policy` 同步接触槽语义、固定腿序启用状态与 ObservationManager。固定腿序回放使用训练同一 `FixedStoneGait`，按场景实际几何转换赛道坐标，要求 3～6 桩，覆盖与出口整理由固定顺序门控。ONNX 导出自动选择宽度并记录固定腿序与承重观测元数据。`tools/check_stepping_stones.py` 已更新为四级、123/399/12，本轮未运行该物理诊断工具。136 项定向检查包含 10 项静态原生回放测试和 105/123 维导出校验，未推进物理时间。

2026-11-15 带 `--scene sim2sim/terrain/scene_terrain.xml` 启动后，Ctrl+F5→6 和 6→Ctrl+F5 都会在入口台顶恢复梅花桩初始关节姿态、速度、控制及观测历史，并直接按默认 0.35 m/s 开始槽 6 推理。入口重置请求消费一次，两键在同一查看器帧内排队也会处理。普通现场切换使用 1 秒平滑交接。正向速度指令限制在 0.30～0.40 m/s，Z 清零，按 1 回平地并清零速度。

2026-11-14 槽 6 首次接入 `PointLeg-SteppingStones-v0` 和 105 维 `model_29598.pt`。槽 1 是必需策略，缺失或不兼容的其他模型会给出提示。

2026-11-09 比赛场地出生点与空格跳跃：运行 `sim2sim/sim2sim.py --task PointLeg-Flat-v0 --scene sim2sim/terrain/scene_terrain.xml --multi-policy --duration 0`。出生点从 MJCF 的 site 读取世界位置和朝向，启动区 2 保留高台高度。组合键移动并清零速度，保留当前常规策略。单按 1～6 切换策略；R 在选中的出生点重置，多策略模式回到槽 1；Z 清零速度，P 暂停或继续。

| 组合键 | 出生点 |
|---|---|
| Ctrl+1 | 启动区 1 |
| Ctrl+2 | 启动区 2 高台，面向平台间隙 |
| Ctrl+3 | 单木桥短边入口 |
| Ctrl+4 | 高墙前 |
| Ctrl+5 | 梅花桩入口平台前 |
| Ctrl+F5 | 梅花桩入口平台顶，朝赛道方向 |
| Ctrl+6 | 限高框架前 |
| Ctrl+7 | 绕桩入口 |
| Ctrl+8 | 轮胎阵前 |
| Ctrl+9 | 砖木坑入口 |
| Ctrl+0 | 斜坡坡脚 |
| Ctrl+- | 台阶底部，面向上台阶 |

空格接替旧 J 槽触发跳跃，复用 `JumpTrigger`，输入为触发标志、已用时间/3、零。多策略模式一次触发最多 3 秒，按住不重复触发，结束回到触发前的常规策略；支持单策略模式下使用 `--task PointLeg-Jump-v0 --checkpoint` 交互触发。多策略跳跃槽需要 `sim2sim/policy/optical_policy/model_jump.pt` 且权重声明匹配的跳跃命令契约。当前目录没有该权重，空格提示不可用。

2026-10-27 平台 Jump 历史记录：当时 CPU Sim2Sim 仅支持速度命令链。当前模型加载仍检查 `jump_contract`，ONNX 导出附带契约。完整训练规则用 `play/evaluate` 验证。

核对日期：2026-10-26。当前实现优先于 `sim2sim/README.md` 中的旧键位和“没有策略切换”等描述。

## 点足 CPU 回放

入口为 `sim2sim/sim2sim.py`，实现为 `sim2sim/pointleg_sim2sim.py`。使用原生 CPU MuJoCo，并复用当前任务的 ObservationManager、ActionManager、机器人姿态、碰撞和 PD；不启动 GPU 并行训练。

```powershell
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --check-model --headless --duration 5

.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --task PointLeg-Flat-v0 --checkpoint "模型路径\model_1000.pt" --duration 0

.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --task PointLeg-Rough-v0 --checkpoint "模型路径\model_1000.pt" --headless --duration 20 --command 0.3 0 0 --action-delay 1
```

`--check-model` 使用零动作检查控制链路，不加载策略。`--duration 0` 只用于交互窗口，无窗口模式要求正时长。延迟参数可选 0～2，默认 0。

普通任务默认加载平地，`--task` 选择控制配置。`--scene` 只接受地形 MJCF，不得包含机器人关节、执行器或关键帧。梅花桩需要包含 `stepping_stones` 障碍的地形文件。

梅花桩回放从场景方块的顶面中心和足端接触读取相对位置、覆盖和到达进度。足端在桩顶法向力 ≥8 N、同腿小腿向上受力 <8 N 连续 20 ms 记一次覆盖。全部桩覆盖且四足分别到过右台后，前进指令归零，策略继续输出动作至站稳成功或整理超时。足端触地、整理超时或站稳成功后需手动重置或切换。场景实际为 6 根 20 cm 方桩，沿赛道中心距 30 cm。2026-11-15 比赛尺寸地形加入训练，和原 15 cm 方桩、23.3 cm 六桩中心距的地形各半使用。完整训练规则用 `play/evaluate` 验证。

`--stepping-trace <新CSV路径>` 支持梅花桩单策略和多策略模式，按物理子步记录右前、右后足端法向力、有效支撑、足端接触位置、小腿向上受力，以及足端相比右列诊断目标的赛道 X 偏差（负数为偏后）。右后小腿接触位置和终止帧触地足也会记录。目标由上一个有效支撑面推算，作为回放诊断参考。右前足在桩顶有效承重达到 20 ms 时填入 `fr_stone_landing` 和实际接触点对该桩中心的偏差 `fr_landing_offset_x_m`。独立回放记录到本回合触地或成功为止。

终点记录 `settling`、`exit_stable_time_s` 和 `exit_settle_timeout`。站稳采用机身连杆原点的速度，与训练口径一致。

Sim2Sim 与训练共用左前违规摆动判定。CSV 增加左前足端及小腿受力、`rr_transfer_active`、`lf_overlap_event` 与 `lf_landing_blocked`。左前桩顶有效承重达 20 ms 的帧记录 `lf_stone_landing`、`lf_stone_landing_eligible` 和 `lf_new_stone_covered`，区分允许计分与实际计入新覆盖。

`rr_first_step_complete` 表示右后首个完整换步是否完成。首步豁免期间 `rr_transfer_active` 为 0，之后该字段表示左前支撑限制是否已经启动。核对时槽 6 的 `model_42700.pt` 为 `new_train_16_six_gap` 权重。

## 当前窗口键位

| 按键 | 操作 |
|---|---|
| 上 / 下方向键 | 增减前后速度 |
| 左 / 右方向键 | 增减左右速度 |
| Q / E | 增减偏航角速度 |
| Z | 速度命令归零 |
| 空格 | 触发跳跃 |
| R | 在已选出生点重置机器人与动作历史 |
| P | 暂停或继续 |

每次速度变化 0.1。当前用全局键盘监听读取按键。

## 手动多策略切换

当前实现已有 `--multi-policy`，模型映射位于 `MULTI_POLICY_FILES`，目录为 `sim2sim/policy/optical_policy/`：

| 槽位 | 文件 |
|---|---|
| 1 | `model_1999.pt` |
| 2 | `model_14100.pt` |
| 3 | `model_4997.pt` |
| 4 | `model_3692.pt` |
| 5 | `model_12000.pt` |
| 6 | `model_29598.pt` |
| jump | `model_jump.pt` |

槽位 1 必须存在，其他缺失项会报告为不可用。运行后按 1～6 切换常规策略，4 使用 Crawl 接口，6 使用 105 维梅花桩接口，跳跃槽使用 Jump 接口。常规任务切换时在 1 秒内插值关节控制目标。

```powershell
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --task PointLeg-Flat-v0 --scene .\sim2sim\terrain\scene_terrain.xml --multi-policy --duration 0
```

此命令依赖本机模型文件，文件存在及兼容性需运行前核对。Jump 专用训练的任务状态、触发和落地接管不能由槽位名称自动获得。

## PT 与 ONNX

PT 要求匹配当前网络的 MJLab checkpoint，并包含 `actor_state_dict`。ONNX 回放要求一个 float32 输入和一个 12 维输出，普通点足策略输入 45 维，梅花桩输入 105 维，所需推理归一化包含在模型内。

```powershell
uv run --locked --with-requirements .\sim2sim\requirements.txt python .\sim2sim\export_onnx.py --task PointLeg-Flat-v0 --checkpoint "模型路径\model_1000.pt" --output "模型路径\policy.onnx"
```

`export_onnx.py` 复用严格 Actor 加载，验证 ONNX 结构，以及零输入、随机输入和批量输入的 PyTorch/ONNX 数值一致性，通过后才写入目标文件。当前仍固定 45 维输入；不能直接用于梅花桩，也不能把旧轮腿模型转换成点足模型。

## 历史工具边界

| 路径 | 用途与限制 |
|---|---|
| `sim2sim/legacy_sim2sim.py` | 历史轮腿回放，需要匹配的轮腿模型 |
| `sim2sim/nav_sim2sim.py` | 历史 Pygame 导航和多地形回放，默认轮腿 Rough 权重 |
| `sim2sim/nav_route_sim2sim_check.py` | 历史 ONNX 策略路线批量检查 |
| `sim2sim/ik_slalom_sim2sim.py` | 历史 IK 与差速轮绕桩 |
| `sim2sim/interface/`、`sim2sim/policy/policy_runner.py` | 历史轮腿 IO、策略与历史观测缓存 |
| `mujoco_sim/` | 姿态、动力学与 MPC；存在历史质量、姿态常量及轮模型依赖 |
| `tools/nav_tools/` | PCD 预览、航点和避障区域编辑及路线检查 |

导航工具的默认路线为 `points_20260715_120154.json`，默认场地为 `1hao.xml`，详见 [导航说明](../tools/nav_tools/README.md)。候选路线输出使用新文件，保留历史路线来源。抽样 PCD 用于预览和打点，不作为定位精度基准。

当前 `mjcf/` 不应被假定保留所有历史工具所需的 `wheelleg.xml`。先检查模型文件、质量惯性、关节顺序与观测，再复用这些脚本。完整 MPC 还需要未列入主训练依赖的 Pinocchio、OSQP 及可用 GUI 环境。
