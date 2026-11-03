# Sim2Sim 与辅助工具

2026-10-27平台Jump更新：旧CPU Sim2Sim速度命令链及手动J槽不能直接回放新Jump。新任务的三命令槽为触发/时间/零，CPU入口明确拒绝 `PointLeg-Jump-v0`，用训练环境 `play/evaluate` 验证。模型加载按任务检查 `jump_contract`，ONNX导出附带契约；`jump_trigger.py`提供独立部署命令时钟，尚未接入ROS/实机。下文历史Jump槽位名称不能作为已完成适配的依据。

核对日期：2026-10-26。当前实现优先于 `sim2sim/README.md` 中的旧键位和“没有策略切换”等描述。

## 点足 CPU 回放

入口为 `sim2sim/sim2sim.py`，实现为 `sim2sim/pointleg_sim2sim.py`。使用原生 CPU MuJoCo，并复用当前任务的 ObservationManager、ActionManager、机器人姿态、碰撞和 PD；不启动 GPU 并行训练。

```powershell
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --check-model --headless --duration 5

.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --task PointLeg-Flat-v0 --checkpoint "模型路径\model_1000.pt" --duration 0

.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --task PointLeg-Rough-v0 --checkpoint "模型路径\model_1000.pt" --headless --duration 20 --command 0.3 0 0 --action-delay 1
```

`--check-model` 使用零动作检查控制链路，不加载策略。`--duration 0` 只用于交互窗口，无窗口模式要求正时长。延迟参数可选 0～2，默认 0。

所有任务默认加载平地，`--task` 选择控制配置，不会自动重建完整训练障碍。`--scene` 只接受地形 MJCF，不得包含机器人关节、执行器或关键帧。工具不运行训练奖励、课程和正式成功判定。

当前 CLI 从已启用任务动态生成可选项，但实现仍要求 Actor 45 维；“任务名能选”不代表专用任务状态和传感器已适配，尤其不支持直接回放 105 维梅花桩。Elevation、Jump、RoughMixed 的完整任务规则需用训练环境的 play/evaluate 检查。

## 当前窗口键位

| 按键 | 操作 |
|---|---|
| 上 / 下方向键 | 增减前后速度 |
| 左 / 右方向键 | 增减左右速度 |
| Q / E | 增减偏航角速度 |
| 空格 | 速度命令归零 |
| R | 重置机器人与动作历史 |

每次速度变化 0.1。窗口需获得焦点；旧文档中的 W/S、A/D 不对应当前 `key_callback()`。

## 手动多策略切换

当前实现已有 `--multi-policy`，模型映射位于 `MULTI_POLICY_FILES`，目录为 `sim2sim/policy/optical_policy/`：

| 槽位 | 文件 |
|---|---|
| 1 | `model_1999.pt` |
| 2～5 | `model_2.pt`～`model_5.pt` |
| jump | `model_jump.pt` |

槽位 1 必须存在，其他缺失项会报告为不可用。运行后按 1～5 或 J 切换，旧新策略输出在 0.20 s 内插值。所有模型按同一个 `--task` 配置加载并共用当前观测和动作链；它是手动选择网络，不是自动切换任务配置或 ROS 任务调度。

```powershell
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --task PointLeg-Flat-v0 --multi-policy --duration 0
```

此命令依赖本机模型文件，文件存在及兼容性需运行前核对。Jump 专用训练的任务状态、触发和落地接管不能由槽位名称自动获得。

## PT 与 ONNX

PT 要求匹配当前网络的 MJLab checkpoint，并包含 `actor_state_dict`；严格加载权重，不忽略不匹配层。ONNX 要求一个 float32 45 维输入和一个 12 维输出，所需推理归一化应包含在导出模型内。

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
