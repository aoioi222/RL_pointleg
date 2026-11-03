# 点足 Sim2Sim 与历史轮腿工具

当前点足验证入口为 `sim2sim.py`，实现位于 `pointleg_sim2sim.py`，使用原生 CPU MuJoCo，不启动 GPU 并行训练环境。旧轻量轮腿入口保留为 `legacy_sim2sim.py`；导航、IK 和批量路线检查工具仍为历史轮腿实现。

## 点足验证

使用与 checkpoint 对应的 `--task`，可选 `PointLeg-Flat-v0`、`PointLeg-Rough-v0`、`PointLeg-Crawl-v0`。机器人姿态、碰撞、PD 参数、控制周期、Actor 观测和动作处理读取当前任务配置；45 维单帧观测顺序为角速度 3、重力投影 3、速度命令 3、关节相对位置 12、关节相对速度 12、上一帧原始动作 12。直接复用训练的 ObservationManager 和 ActionManager，包含动作低通、固定延迟与软限位。

电机配置为十二个 N6014B-12.6、48 V，点足回放默认仿真力矩限幅为 ±25 N·m，与训练配置一致，不是已确认的持续扭矩。启动时输出电机型号、供电记录、工作限幅及空载转速估算，并说明估算不构成限速、当前模型不模拟电压。不会将 31.7/34.4 N·m 峰值当成持续值，也不重复施加减速比；见 [电机参数与建模边界](../docs/MOTOR_PARAMETERS.md)。

```powershell
# 不需要 checkpoint，先检查模型和控制链路；零动作不代表已训练策略
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py --check-model --headless --duration 5

# 点足 checkpoint 交互验证，关闭窗口退出
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py `
  --task PointLeg-Flat-v0 --checkpoint "模型路径\model_1000.pt" --duration 0

# 无窗口运行并给定 vx、vy、wz
.\.venv\Scripts\python.exe .\sim2sim\sim2sim.py `
  --task PointLeg-Rough-v0 --checkpoint "模型路径\model_1000.pt" `
  --headless --duration 20 --command 0.3 0 0 --action-delay 1

# ONNX 回放，额外需要 onnxruntime
uv run --locked --with-requirements .\sim2sim\requirements.txt `
  python .\sim2sim\sim2sim.py --task PointLeg-Flat-v0 --checkpoint "模型路径\policy.onnx" --duration 0

# 接口回归检查
.\.venv\Scripts\python.exe -m unittest discover -s .\sim2sim\tests -v
```

窗口获得焦点后，W/S 调整前后速度，A/D 调整左右速度，Q/E 调整偏航角速度，每次按键变化 0.1；空格归零速度命令，R 重置机器人和动作历史。

默认所有任务都在平地验证，`--task` 选择对应姿态和控制配置，不生成训练的 Rough/Crawl 随机障碍。可用 `--scene "地形文件.xml"` 加载仅含地形的 MJCF；该文件不得包含机器人关节、执行器或关键帧，出生位置沿用任务默认原点。模型可通过启动前的 `POINTLEG_XML` 环境变量替换。

PT 必须是当前任务网络配置兼容的 MJLab checkpoint（含 `actor_state_dict`），严格检查全部权重；ONNX 必须内含推理归一化，输入为一个 float32 `(1,45)` 张量，输出为一个 `(1,12)` 动作张量。旧 `model_6800.onnx`、`model_rough.pt` 不作为点足策略使用。当前不支持 HIM、历史观测策略或自动任务切换；任务配置发生变化后应使用匹配的 checkpoint。

验证默认关闭观测噪声和域随机化，动作延迟固定为 `--action-delay`（默认 0，可选 0～2），不执行训练奖励和成功率判定。终端的有限状态检查与 MuJoCo 警告检查只确认数值运行正常，不等同于完成越障或稳定行走。

原生 play 也已切换为无物理随机化的 nominal 配置，但其 Rough/Crawl 仍会生成障碍地形，不能直接与本工具的默认平地轨迹比较。`python -m sim2sim.check_training --task PointLeg-Flat-v0` 可验证真实小规模 PPO 更新、恢复、导出和回放；`--pose-only` 仅做 CPU 零动作姿态诊断。使用方式与验收边界见 [B 类修复说明](../docs/B_FIXES.md)。

## 主要入口

- `nav_sim2sim.py`：Pygame 面板与 MuJoCo 多任务导航，Rough 策略优先加载根目录的 `model_6800.onnx`。
- `sim2sim.py` / `pointleg_sim2sim.py`：当前点足 PT/ONNX 验证入口。
- `legacy_sim2sim.py`：原轻量轮腿回放入口，保留作历史参考。
- `ik_slalom_sim2sim.py`：不依赖 RL 策略的 IK、差速轮、路径跟踪和绕桩测试。
- `ik_compensation_sweep.py`：批量扫描 IK 补偿参数并输出排序结果。
- `nav_route_sim2sim_check.py`：使用 ONNX 策略批量检查内置任务或外部航点路线。
- `export_onnx.py`：按指定点足任务严格加载 checkpoint，使用原生 Actor 推理模块导出 ONNX 并强制核对数值。
- `interface/mujoco_io.py`：历史轮腿 MuJoCo 模型、传感器和执行器接口。
- `policy/policy_runner.py`：历史轮腿 PT/ONNX 策略加载与历史观测缓存。

## 环境

主训练环境继续由根目录的 `uv.lock` 管理。后期 Sim2Sim 新增的 Pygame 与 ONNX Runtime 单独记录在 `sim2sim/requirements.txt`，运行时叠加，避免重新解析时改变已归档的 MuJoCo nightly 版本：

```powershell
uv run --with-requirements .\sim2sim\requirements.txt python .\sim2sim\nav_sim2sim.py
```

训练工程提供 MuJoCo、NumPy、PyTorch、Matplotlib 和 `pynput`；专用 requirements 显式补充 Pygame 与 ONNX Runtime。下面其他命令同样使用 `--with-requirements .\sim2sim\requirements.txt`。

## 常用命令

```powershell
# 比赛 Rough 策略交互回放
uv run --with-requirements .\sim2sim\requirements.txt python .\sim2sim\nav_sim2sim.py

# 历史轮腿轻量策略回放，需要恢复配套 wheelleg.xml
uv run --with-requirements .\sim2sim\requirements.txt python .\sim2sim\legacy_sim2sim.py

# 纯 IK 绕桩验证
uv run --with-requirements .\sim2sim\requirements.txt python .\sim2sim\ik_slalom_sim2sim.py --test slalom

# IK 补偿参数扫描
uv run --with-requirements .\sim2sim\requirements.txt python .\sim2sim\ik_compensation_sweep.py --top 12

# 使用内置绕桩任务做批量 Sim2Sim 路线检查
uv run --with-requirements .\sim2sim\requirements.txt python .\sim2sim\nav_route_sim2sim_check.py `
  --terrain-xml .\sim2sim\terrain\scene_terrain.xml `
  --mission slalom `
  --onnx .\model_6800.onnx

# 导出当前点足策略，必须指定对应任务和 checkpoint
uv run --locked --with-requirements .\sim2sim\requirements.txt `
  python .\sim2sim\export_onnx.py --task PointLeg-Flat-v0 `
  --checkpoint "模型路径\model_1000.pt" --output "模型路径\policy.onnx"
```

导出不再默认读取旧 `model_rough.pt`，也不手工重建旧 PolicyMLP 或跳过权重加载错误。网络及归一化采用与点足 Sim2Sim 相同的严格加载逻辑；输入 `obs` 为 45 维，输出 `actions` 为 12 维，批量大小可变。省略 `--output` 时保存为 checkpoint 同名 `.onnx`；仍支持 `--pt-path` 和 `--pt_path` 作为 `--checkpoint` 的别名。文件通过 ONNX 结构检查，以及零输入、随机输入和批量输入的 PyTorch/ONNX 数值对比后才写入目标路径；验证失败返回非零退出码，不覆盖已有输出。旧轮腿 checkpoint 会被拒绝，不能通过导出转换为点足策略。

## 模型与边界

- `../model_6800.onnx` 是 `last_not_slalom_1050` 最终真机工程使用的比赛 Rough 策略，SHA-256 为 `3C994BDD3434AD15770A52AC0E8D229F502F00D6511CDD42C2E2C742301AEF13`。
- `../model_rough.pt` 是较早阶段的参考 checkpoint，两者不是同一版本的权重。
- Crawl 模型未在本阶段归档；需要 Crawl 策略的入口会查找 `model_crawl.onnx` 或 `model_crawl.pt`。
- `nav_route_sim2sim_check.py` 依赖 `../tools/nav_tools/route_safety_check.py` 的航点和避障几何定义；默认使用 `points_20260715_120154.json` 与 `1hao.xml`。

运行时生成的日志、临时 XML、`route_check_runs/` 和批量实验输出不纳入版本库。人工打点形成的路线快照保存在 `../tools/nav_tools/points/`，大量重复仿真轨迹仍不复制。
