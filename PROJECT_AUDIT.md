# 点足项目适配审查清单

审查日期：2026-10-19。审查对象：当前 `rc_mjlab` 工作目录。

本文件保留首轮审查证据，并更新后续修复状态。A01～A10 已完成修复或依赖验证，关联的 B06、B07 一并修复。B02、B08～B11 的代码问题已处理，B03 高度基准已统一；B12 增加调试配置，B13 完成短程训练闭环。硬件标定、全身净空与长期训练验收仍按对应状态保留。

## 范围与验证结果

- 对 `src/pointleg/`、`sim2sim/`、`mujoco_sim/`、`tools/` 共 72 个自有 Python 文件进行了 UTF-8 解码、语法及遗留接口扫描，并重点检查训练配置、奖励、命令、课程、模型、策略加载及导出链路。
- 检查了 9 个 XML 的语法和文件引用，以及根配置、锁文件、README、IDE 配置、压缩快照和模型资源。
- 对随项目保存的 `mjlab/` 核查了当前任务实际调用的命令、奖励、事件、地形、执行器和训练接口；没有对整个第三方框架、虚拟环境所有依赖或二进制模型进行逐行安全审计。
- 当前统一入口执行 64 项测试全部通过，包括 A 类、6 项电机规格测试及 12 项 B 类配置／行为测试。
- `uv lock --check --offline` 通过；主环境已安装的锁内包未发现版本超出锁文件候选集合。
- 修复后 Flat、Rough、Crawl 分别以 2 个 GPU 环境完成初始化、重置和 8 次控制步；实际命令类型、PD 增益随机范围、环境实例奖励裁剪均通过断言。Actor 为 45 维，Critic 为 325 维。这不是默认 2048 环境的资源验收，也不是 PPO 更新或步态收敛验收。
- B 类修复后，三个任务各以 16 个 GPU 环境、8 步 rollout 完成两次 PPO 更新，保存／恢复策略、优化器和全局步数后继续更新一次；真实权重导出 ONNX 的最大数值误差均约 `1.19e-7`，各完成 1 秒 CPU 回放。已核对 GPU 质量／惯量同步缩放。产物位于 `artifacts/pointleg_checks/`，不代表收敛、长期显存或任务成功率验收。
- 三个任务的 nominal 回放配置各以 2 个 GPU 环境完成 reset 和 8 次步进，观测与奖励有限，Kp/Kd 保持 `50/1.5`，动作延迟为 0，关闭随机化后仍可正常构建和执行。
- 用户安装 `flatbuffers` 后，`uv pip check --python .venv/Scripts/python.exe` 已通过，113 个已安装包的依赖兼容。
- 自有 Python 文件未发现剩余编码或语法错误；这不证明全部历史入口可运行。
- 已修复 A 类与部分 B 类业务问题并更新对应测试，没有修改随项目保存的 MJLab 框架源码。B 类使用方式与验证边界见 `docs/B_FIXES.md`。

## 优先级及状态

- P1：会影响当前行为、训练含义或环境完整性，建议先修。
- P2：任务开发、联调或验收前应处理。
- P3：整理、归档和后续维护事项。
- 确认缺陷：由代码和诊断证据确认。
- 待标定／待验证：已有实现，但没有足够证据确认其适合当前机器人。
- 功能缺口：尚未实现，不意味着现有代码执行错误。
- 历史保留：不进入默认点足链路；可归档，不要求全部重写。

## A. 已确认的缺陷与接口遗漏

| 编号 | 优先级／状态 | 文件位置 | 事实、影响与修改方向 |
|---|---|---|---|
| A01 | 已修复 | `src/pointleg/mdp/commands.py` | 已实现当前 API 的 `build()`，三个任务实际创建 `UniformThresholdVelocityCommand`。站立、前向、横移和转向分组保持互斥，台阶约束和高速限制已接入；CPU 行为测试及 GPU 实例类型检查通过。 |
| A02 | 已修复 | `src/pointleg/config/env_cfgs.py` | 两个旧事件合并为 `actuator_pd_gains`，调用 `dr.pd_gains` 同时独立采样 Kp/Kd，避免分次调用重新覆盖另一项增益。三个 GPU 环境检查了实际增益：Kp 在 `[45,55]`、Kd 在 `[1.35,1.65]`，且不再固定为默认值。 |
| A03 | 已修复 | `src/pointleg/mdp/only_positive_rewards.py`；`src/pointleg/config/env_cfgs.py` | 移除全局类方法替换。Rough/Crawl 通过 startup 事件，只包装各自 RewardManager 实例的计算函数并原地裁剪总奖励；Flat 和其他实例保留有符号奖励。重复配置不会叠加包装，各奖励项日志保留原始符号。终止惩罚仍先参与求和，再按对应任务裁剪总奖励。 |
| A04 | 已修复 | `src/pointleg/mdp/rewards.py` | 镜像索引缓存按实体名及关节配对内容区分；矢状面项不会再覆盖外展项。回归验证外展奖励正确返回 `0.08`。 |
| A05 | 已修复 | `src/pointleg/robot_cfg.py`；`src/pointleg/tests/test_audit_fixes.py` | Crawl 默认膝角改为 `CRAWL_KNEE_ANGLE=-2.50 rad`，保留 `0.95` 软限位系数。三个任务的默认关节姿态均通过软限位检查。这是控制目标一致性修复，不代表完成了 B01 的物理姿态标定；旧 Crawl checkpoint 需要重新验证。 |
| A06 | 已验证解决 | `.venv/` 的 ONNX Runtime 安装 | 用户已安装 `flatbuffers`，依赖兼容性检查通过。本次没有重复安装。 |
| A07 | 已修复 | `tests/test_suites.py`；`README.md` | 根目录 `uv run --locked --no-sync python -m unittest discover -v` 当前统一运行 64 项测试，不再得到 0 项。CI 配置仍属于 F10。 |
| A08 | 已修复 | `src/pointleg/config/env_cfgs.py`；`src/pointleg/mdp/terrain_utils.py` | Rough/Crawl 回放保留 `curriculum=True` 的按类型分列布局，只冻结环境课程更新并缩小难度行数。命令和奖励共用类型掩码；单一地形的所有列均正确识别，多类型随机布局会明确报错，不再错误解释列号。 |
| A09 | 已修复，函数仍为备用 | `src/pointleg/mdp/rewards.py` | 删除错误的局部导入，使用模块已导入的 `SceneEntityCfg`。已测试默认参数调用和数值结果。 |
| A10 | 已修复切片，HIM 仍未接入 | `src/pointleg/rsl_rl/modules/him_estimator.py` | 目标编码器读取下一步 Critic 的 Actor 前缀 `[:45]`，速度标签读取其后的 `[45:48]`，并检查输入维度。测试捕获了实际目标编码器输入及速度回归标签。完整 HIM 接入仍需 F08 的其他工作。 |

## B. 点足机器人参数与训练行为的待办

| 编号 | 优先级／状态 | 文件位置 | 待办与证据 |
|---|---|---|---|
| B01 | P1，已复测，姿态仍待标定 | `src/pointleg/robot_cfg.py`；`sim2sim/check_training.py` | 已提供 `--pose-only` 诊断并用 A05 新默认姿态复测：CPU 平地零动作 5 秒后 Flat/Rough 根高度约 `0.248 m`、倾斜 `21.27°`；Crawl 高度约 `0.052 m`、倾斜 `0.94°`。不能视为稳定默认姿态，也不证明已训练策略必然失败。未无依据改变 PD、力矩或 CAD 参数。 |
| B02 | 配置差异已消除，实机高度仍待标定 | `src/pointleg/config/env_cfgs.py`；`src/pointleg/robot_cfg.py` | Flat 奖励目标与出生高度共用 `STANDING_BASE_HEIGHT=0.42`，奖励函数默认值同步，移除旧 `0.36 m` 差异。物理姿态仍受 B01 约束，这不是高度标定验收。 |
| B03 | 根高度基准已修复，全身净空仍待开发 | `src/pointleg/mdp/height.py`；`src/pointleg/mdp/rewards.py`；`src/pointleg/mdp/terminations.py` | Crawl 奖励与高度终止共用局部地面射线及有效性检查。未命中、背面、NaN 等不给高度奖励并终止；已测试不同地面／世界高度下结果一致。根高度不能代表整个机身的最高点，完整净空与穿框判定仍属于 C07。 |
| B04 | P2，电机规格已录入，动力学待标定 | `src/pointleg/motor_spec.py`；`src/pointleg/robot_cfg.py`；`mjcf/pointleg.xml` | 用户确认十二个电机均为 N6014B-12.6、48 V；已记录驱动／制动峰值 31.7／34.4 N·m、质量 0.535 kg 等规格。默认保留 ±17 N·m 临时仿真限幅，不视为额定持续扭矩；同步关节和执行器限制并检查不超过峰值上界。连杆质量／重心、惯量、摩擦、PD、持续扭矩与电流／热限制仍待标定，未把电机质量重复加入 CAD 连杆。详见 `docs/MOTOR_PARAMETERS.md`。 |
| B05 | P2，旧常量已移除，速度包络未接入 | `src/pointleg/motor_spec.py`；`src/pointleg/robot_cfg.py` | 移除未生效的 `MAX_JOINT_VEL=13.0`。已记录 60 V 空载转速 54.2 rad/s，48 V 比例估算约 43.36 rad/s，仅用于参考，不构成带载限速。用户补充的 FOC 图为 60 V 条件；左图按数值关系推断为减速前轴端，不能将 31.7 N·m 峰值当成堵转扭矩拟合曲线。当前仍为位置 PD，尚未接入经验证的 48 V 扭矩—转速包络。 |
| B06 | 已随 A01 修复 | `src/pointleg/mdp/commands.py` | 线速度和角速度使用独立死区；较小线速度不会再清掉有效的转向命令，已通过原地转向测试。 |
| B07 | 已随 A01 修复 | `src/pointleg/mdp/commands.py` | 移除平地二次覆盖采样。前向／站立样本不参与横移／转向抽样，平地两项概率可单独配置，其他地形使用通用概率；已验证互斥性及采样分布。 |
| B08 | 已修复 | `src/pointleg/mdp/curriculums.py`；`src/pointleg/tests/test_b_fixes.py` | 指令课程改为独立时间窗口和按结束回合数累计样本，到期后的首次 reset 触发更新。跳过零长度初始化回合，提前失败仍按完整回合时长归一化。异步非整除步数、不同批量回合、大间隔与范围上界测试通过。 |
| B09 | 随机化配置差异已修复 | `src/pointleg/config/evaluation.py`；`src/pointleg/config/env_cfgs.py` | 默认 play 使用 nominal：移除物理随机化、随机初态、推力、观测噪声和随机动作延迟，仅保留默认 reset 与必要奖励事件。Python 的 `evaluation_env_cfg(..., mode="robust")` 单独保留训练随机化；均冻结课程。CPU 显式固定延迟测试仍支持 0～2；地形和 CPU／GPU 数值差异仍见 D01、D08。 |
| B10 | 已修复一致性，随机化范围仍为训练假设 | `src/pointleg/config/env_cfgs.py` | `base_inertia` 通过 `dr.pseudo_inertia` 将质量与惯量按同一密度倍率 `0.85～1.45` 缩放，并合并质心各轴 `±0.05 m` 随机化。移除只改质量的旧事件及独立质心事件，避免覆盖；三个 GPU 检查验证质量与主惯量缩放比例一致。不等于完成真实惯量标定。 |
| B11 | 已修复本项说明 | `src/pointleg/mdp/rewards.py`；`src/pointleg/config/env_cfgs.py` | `upward` 已按实际公式说明为正权重的直立奖励，直立／水平／倒置分别返回 `4/1/0`，数值测试通过，公式和权重不变。同步修正 play、匍匐高度和失稳兜底相关说明。 |
| B12 | 小规模调试入口已完成，正式配置仍待验收 | `src/pointleg/config/debug.py`；`src/pointleg/run.py` | `pointleg train <task> --debug` 默认 16 环境、8 步 rollout、1 minibatch、1 epoch、2 次迭代、本地日志且不上传模型；允许原生参数覆盖且保持策略结构。三个任务真实训练检查通过；默认 2048 环境资源、各任务长期超参数和地形释放进度仍待标定。 |
| B13 | 短程训练闭环已通过，收敛与成功率仍待验收 | `sim2sim/check_training.py`；`artifacts/pointleg_checks/` | Flat/Rough/Crawl 各完成两次真实 PPO 更新，验证策略／优化器／全局步数恢复后再更新一次；各导出 ONNX 并校验至最大误差约 `1.19e-7`，完成 1 秒 CPU 回放，保留配置快照和 JSON 报告。这些是短程检查模型，不是成熟步态；完整场景、随机流与课程内部状态的精确续跑、收敛及成功率未验收。 |

## C. 比赛技能与规则功能缺口

以下均为待开发功能，不能仅将注册表的 `enabled` 改为 `True`。

| 编号 | 优先级／状态 | 文件位置 | 必须补齐的内容 |
|---|---|---|---|
| C01 | P2，功能缺口 | `src/pointleg/config/env_cfgs.py:946`；`src/pointleg/terrains/` | `Elevation`：坡面、组合台阶、高墙、沟壑、合法进出区域、成功／失败、相应观测／奖励／课程。工厂目前抛 `NotImplementedError`。 |
| C02 | P2，功能缺口 | `src/pointleg/config/env_cfgs.py:956` | `Bridge`：桥面和两端平台、横向偏差、落桥、完整通过及机身／足端接触规则。工厂目前为占位。 |
| C03 | P2，功能缺口 | `src/pointleg/config/env_cfgs.py:966` | `SteppingStones`：桩面几何、有效落足、边界余量、接触去重、覆盖记忆和完整通过。工厂目前为占位。 |
| C04 | P2，功能缺口 | `src/pointleg/config/env_cfgs.py:976` | `Jump`：起跳区、起跳／腾空／落地阶段、方向、墙／沟尺寸、落足与恢复判定。工厂目前为占位。 |
| C05 | P2，功能缺口 | `flat_env_cfg()`；`src/pointleg/terrains/` | Flat 仍是基础速度跟踪环境，未实现绕桩竖杆、有顺序的必达区域和精定位验收。 |
| C06 | P2，功能缺口 | `rough_env_cfg()`；`src/pointleg/terrains/competition_terrains.py` | Rough 仍为基础崎岖地形和连续矮墙，未实现轮胎阵孔洞／曲面、砖木坑实际填料／边界等比赛场景。 |
| C07 | P2，功能缺口 | `crawl_env_cfg()`；`RCLowBarTerrainCfg` | 当前只有低横杆，缺少有长度的比赛限高通道、全身净空、入口／出口及完整通过判定。匍匐原点高度奖励不能代替这些规则。 |
| C08 | P2，功能缺口 | `src/pointleg/mdp/`；各任务配置 | 缺少统一的任务阶段、规则记忆、成功／失败原因和阶段 reset 接口。应让每个任务维护明确状态，防止绕开障碍或重复接触刷分。 |
| C09 | P2，后续任务适配 | `src/pointleg/mdp/terminations.py:28`；基础终止配置 | 公共高度终止把无有效地面射线视为失败，并限制净空。未来跨沟和腾空任务需要阶段相关的终止条件，不能直接沿用。桥、梅花桩也需要专属落地／失足判定。 |
| C10 | P2，功能缺口 | 任务调度层尚未建立；`src/pointleg/mdp/gait_generator.py`、`mode_command.py`、`disturbances.py` | 恢复、返回启动区、保守模式、策略切换与命令交接尚未接入。参考步态、模式命令和连续扰动是备用组件，当前不会因文件存在就自动影响训练；是否使用需由对应任务明确选择。 |

## D. 新 Sim2Sim／导出的能力边界及后续完善

现有新入口已经完成基本点足适配。以下不代表它仍在使用旧轮腿导出逻辑。

| 编号 | 优先级／状态 | 文件位置 | 后续修改或验收 |
|---|---|---|---|
| D01 | P2，能力缺口 | `sim2sim/pointleg_sim2sim.py:81` | 所有任务默认只在平地验证，`--task Rough/Crawl` 不会生成该任务的训练障碍。需增加匹配的测试场景或可复用的地形导出，才能比较具体越障技能。 |
| D02 | P2，能力缺口 | `sim2sim/pointleg_sim2sim.py:99` | `--scene` 仅接受地形，出生点仍沿用任务原点；缺少与场景绑定的起点、朝向、终点和路线配置。旧比赛地图不能仅替换 XML 就假定出生和任务目标正确。 |
| D03 | P2，能力缺口 | `sim2sim/pointleg_sim2sim.py:145,281` | 目前只检查数值有限和 MuJoCo 警告，不计算任务成功、跌倒、速度误差、能耗、足滑和关节越限统计，也不输出机器可读的验收报告。需补齐；终端 `Completed` 不等于行走稳定或通过比赛。 |
| D04 | P2，后续扩展 | `sim2sim/pointleg_sim2sim.py:127,158`；`sim2sim/export_onnx.py` | 当前明确支持 45 维 Actor、12 维动作、MLP，尚不支持未来的落足目标、额外传感器、观测历史、HIM 或循环策略。新增任务观测后，应共同更新接口定义和验证，不可只改网络输入尺寸。 |
| D05 | P2，可复现性缺口 | `sim2sim/pointleg_sim2sim.py:178` | PT 加载读取当前代码中的任务网络／实体配置，没有自动恢复该次训练保存的配置快照。结构相同但激活函数、动作尺度或姿态不同，不一定能通过严格权重检查识别。需明确配置版本并与 checkpoint 绑定。 |
| D06 | P2，可复现性缺口 | `sim2sim/export_onnx.py`；模型产物 | 导出物缺少统一的任务、XML 哈希、关节顺序、观测顺序／缩放、控制周期、动作尺度、归一化及配置版本清单。ONNX 已包含原生推理归一化，但仅靠输入／输出尺寸不能证明它适合当前实体。 |
| D07 | P2，验证入口完善 | `sim2sim/pointleg_sim2sim.py:209,248` | CLI 速度命令和键盘累加值没有按任务训练范围限幅／提示，可能无意中验证到训练范围以外。需显示或约束范围，并为超范围实验提供明确选择。 |
| D08 | P2，尚未完成验收 | `sim2sim/tests/`；CPU／GPU 仿真链路 | 已验证 CPU 接口和合成 PT/ONNX 的数值一致性；尚未完成真实训练权重的 GUI 操作验收及 CPU／GPU 同输入多步轨迹比较。GPU 与 CPU 的随机化、接触及物理更新差异仍需量化。 |

## E. 尚未迁移的历史轮腿工具

若继续使用这些工具就需要适配；若不再使用，建议迁到明确的历史目录并禁用易误用的默认入口。无需为了清理名称而把所有历史算法重写一遍。

| 编号 | 优先级／状态 | 文件范围 | 遗留及处理方向 |
|---|---|---|---|
| E01 | P2，历史入口 | `sim2sim/main.py`、`nav_sim2sim.py`、`legacy_sim2sim.py` | 仍读取不存在的 `mjcf/wheelleg.xml`，部分还要求未归档的 Crawl 模型；不是当前点足入口。需迁移、显式拦截或归档。 |
| E02 | P2，历史接口 | `sim2sim/interface/mujoco_io.py`、`policy/policy_runner.py`、`tools/math_utils.py`、`tools/logger.py`、`input_dev/keyboard.py` | 旧链路仍以轮腿关节、53／318 维观测、16 维动作、轮速及旧滤波为基础。`mujoco_io.py` 与 `legacy_sim2sim.py` 的 PD／轮速参数也不一致。复用时必须经统一接口适配，不能导入旧 PolicyRunner 来运行点足模型。 |
| E03 | P2，历史验证链路 | `sim2sim/ik_slalom_sim2sim.py`、`ik_compensation_sweep.py`、`nav_route_sim2sim_check.py` | IK、差速轮、轮半径、轮接触点、旧策略和路线跟随仍有轮腿假设；默认机器人文件不存在。需重建点足运动学和任务接口后再用作点足验收。 |
| E04 | P2，历史工具集 | `mujoco_sim/` 的 13 个 Python 文件 | 控制、GUI、MPC、Pinocchio 动力学、姿态优化及扫描仍使用轮腿几何／轮心、16 关节、旧高度表和不同 PD 参数。尤其 `config.py` 指向缺失的 `wheelleg.xml`，但默认 `scene.xml` 现已包含点足，`robot.py` 又按轮关节取 ID，存在新旧模型混用。需整体归档或按点足模型系统适配。 |
| E05 | P2，导航模型适配 | `tools/nav_tools/route_safety_check.py`、`nav_map_viewer.py`、`run_route_experiments.py`、`route_candidate_optimizer.py`；旧 JSON／PCD | 默认包络根据轮心和轮宽计算，批量实验仍调用轮腿路线验证及旧模型，默认路线来自历史比赛。地图编辑和点云通用工具可保留；点足步态的动态落足包络、坐标与路线应重新确认。 |
| E06 | P2，历史地形模板 | `sim2sim/terrain/scene_terrain.xml`；`mjcf/sim2sim_temp.xml`；`tools/nav_tools/xml/` | 原地形模板引用未就地存在的 `go2w.xml` 和上级目录高度图，旧 IO 依赖字符串替换才能加载；部分临时 XML 绑定绝对机器路径。新点足 `--scene` 不执行这套替换，也不接受含机器人关节的场景。应整理为可直接加载、路径可移植的纯地形文件。 |

## F. 环境、打包、文档与维护

| 编号 | 优先级／状态 | 文件范围 | 后续处理 |
|---|---|---|---|
| F01 | P2，环境混用风险 | `.venv/`、`venv/` | 两套环境并存且版本不同：主 `.venv` 的 PyTorch 为 `2.11.0+cu128`，旧 `venv` 为 `2.14.0`。应明确只使用主锁定环境，旧环境先停用或归档，防止组员用不同解释器得出不同结果。 |
| F02 | P2，IDE 配置遗留 | `.vscode/settings.json` | 默认环境和包管理器仍设为 Conda，与当前 uv／`.venv` 工作流不一致。需设置工作区解释器并统一组内启动方式。 |
| F03 | P2，工程管理缺口 | 项目根目录 | 未找到工程级 `.gitignore`，缓存、环境、日志、模型和运行时 XML 缺少统一排除规则。应明确模型产物的归档策略，避免误提交大量运行时文件。 |
| F04 | P2，协作管理缺口 | 当前目录及其父目录 | 未发现 Git 仓库。若作为多人开发基线使用，应先建立版本控制、可复现基线和分支／合并流程。当前无法依据 Git 证明第三方框架相对基准只有哪些修改。 |
| F05 | P2，打包能力缺口 | `pyproject.toml`；`src/pointleg/robot_cfg.py` | 当前包只发现 `pointleg` Python 包；模型／网格和独立 Sim2Sim 脚本在包外。可编辑安装依赖源码目录，构建普通 wheel 后不能假定单独安装即可找到全部资源。需选择“源码工程交付”还是“独立安装包交付”，再补资源布局和打包测试。 |
| F06 | P2，可复现性缺口 | `README.md:95` | 开发依据仍指向用户桌面上的外部规则设计文档。其他成员无法仅凭项目获取依据。应将可共享的规则／场地图纳入项目并注明版本。 |
| F07 | P3，文档和注释遗留 | `DEPENDENCIES.md:7`；`README.md` 历史章节；`robot_cfg.py`、`terrains/`、`config/__init__.py`、`rsl_rl/__init__.py` | 依赖说明仍写 MuJoCo 3.8，而锁定安装为 `3.12.1.dev975126947`；主线注释仍有 `Robot-*`、`import robot.config`、`src/robot`。历史 README 已标注为历史，但最好拆出单独文档，主线注释应准确更新。发行名本身已改为 `pointleg`，不再属于未修项目。 |
| F08 | P2，备用 HIM 尚不兼容 | `src/pointleg/rsl_rl/` | 适配器引用缺失的 `him_mjlab`；HIM runner 读取旧配置键 `policy/policy_class_name/algorithm_class_name`，当前配置使用分开的 `actor/critic/algorithm`。A10 的观测切片已修复，但环境、配置、checkpoint 和导出接口仍待接入。 |
| F09 | P3，历史资源整理 | `src/27_pointleg.zip`、`src/pointleg/__init__.py.gbk_bak`、`mjcf/wheelleg.xml.bak`、`mjcf/meshes/`、根旧模型、临时 XML、日志及 egg-info | 压缩包仍含旧 `27_pointleg` 快照，共 165 个成员；点足 XML 不再引用 4 个旧膝网格和 4 个轮网格；根 PT/ONNX 仍是轮腿权重。应分清历史档案、可再生成文件和当前资源，避免解压／回退后恢复旧代码。不要在确认依赖前批量删除历史资产。 |
| F10 | P3，部分完成 | `mjlab/`；根工程配置；`.venv` 激活脚本 | 统一 unittest 入口已完成；上游本地修改清单、lint、类型检查及 CI 仍待完善。导出使用的 TorchScript ONNX 路径在当前版本可用但有弃用提示，应在升级工具链时迁移。激活提示仍显示 `robot`，只是旧虚拟环境显示名，不影响已经修正的发行元数据。 |

## 建议修改顺序

- A01～A10 已完成；后续优先处理 B01～B05、B09 和 B13，补姿态标定与训练闭环。
- 分工前处理 F01～F06，建立统一环境、配置依据和可合并的工程基线。
- 按实际分组实现 C 类任务，并同步完善 D 类验收；E 类旧工具选择迁移或归档。
- HIM 相关 A10／F08 在决定接入时处理；备用函数和历史说明可随对应模块维护。

## 已完成、不应重复列为未修的事项

- 发行名、导入包名、任务插件名及命令名已统一为 `pointleg`。
- 主包已位于 `src/pointleg/`，旧重复 `src/27_pointleg/robot/` 已移出项目。
- 当前点足模型为 12 个腿部执行器，训练入口已不要求轮部动作。
- 新 `sim2sim/sim2sim.py` 使用点足 CPU 验证实现，包含 45 维观测、12 维动作及训练动作处理。
- `export_onnx.py` 已为 UTF-8，按指定点足任务严格加载原生 Actor，保留原生归一化，明确要求 checkpoint，并在数值验证通过后保存 ONNX；不再默认导出旧 `model_rough.pt`。
- 这些完成项通过相应测试，不等于比赛任务已实现、已具备稳定步态或已通过真机验收。
