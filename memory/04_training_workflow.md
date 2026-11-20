# 环境与训练操作

## 2026-11-20 Colab 启动入口

新增 `notebooks/colab_pointleg.ipynb`，用于在 Google Colab 中克隆仓库、安装项目锁定依赖、显示 GPU 信息、确认 `pointleg list` 任务注册，并按配置启动训练。Notebook 使用 `uv sync --locked --extra cu128` 安装依赖；项目内 `mjlab/` 作为本地 editable source，因此必须完整克隆仓库后再执行安装。

训练单元默认 `START_TRAINING = False`，不会因打开或顺序执行 Notebook 而启动物理仿真或 PPO 训练。将其手动改为 `True` 后，默认训练 `PointLeg-Flat-v0`，使用 64 个环境、1000 次更新、`colab_flat` 运行名；日志通过 `--log-root` 写入 Google Drive 的 `MyDrive/pointleg_logs`。任务、并行环境数、更新次数和运行名均在训练配置单元中修改。

仓库未包含 `.pt`、`.onnx`、`pretrained/` 与历史日志；若要续训或回放，用户需自行将对应检查点放入 Google Drive，并在 Notebook 的训练命令中补充恢复参数。Colab 的 CUDA、Python、网络与可用磁盘由其运行时决定，尚未在 Colab 实际安装依赖、启动物理步进或训练验证。

## 2026-11-19 本地训练命令生成器

命令页面前两项依次为 MuJoCo 单环境和 Viser 单环境回放，其余命令按原顺序排列。

训练环境数采用 1 到 65536 的 2 的幂次下拉选项，默认 512，选中值用于续训命令。

双击 `tools/训练命令生成器.html`，粘贴 Windows 检查点路径，可复制 TensorBoard、续训、原生和 Viser 单/四环境回放及权重迁移命令。当前任务根据实验目录自动识别，也可手动选择；共用 `pointleg_rough` 目录的 Rough 和 RoughMixed 需要手动选择。续训使用输入路径的实验目录上级作为 `--log-root`，目录名和检查点使用转义后的完整正则匹配；任务与实验目录不符时禁用续训。迁移目标任务单独选择，默认为梅花桩，输出目录必须不存在。命令用于 rc_mjlab 目录下的 Windows PowerShell；训练默认 512 环境、1000 轮和运行名 train_1，迁移输出名称默认 lift5cm_pretrained，均可更改。页面无法验证文件存在、所属项目或模型接口兼容性。Node 模拟页面检查任务识别、日志根目录、多任务歧义、目录不匹配和迁移目标通过；未在浏览器实测、未执行迁移、回放或训练。

## 2026-11-18 已启动修正版 2000 轮训练

用户明确要求“改完开始给我训练自己训练2000轮”，本轮实际执行训练，覆盖默认不主动运行仿真/PPO的限制。先核对并结束旧版 `2026-11-18_14-14-44_train_2`，其 PID20740/启动器34192，配置10000次、失败窗口8s。保留旧日志和已保存 `model_0.pt`。本次未操作其他回放进程。

本次运行目录为 `logs/rsl_rl/pointleg_stepping_stones/2026-11-18_14-25-04_train_3_audit_2000`，GPU进程15928、启动器37588，工具会话78799。从已验证123/399迁移种子开始，迭代0，512环境、2000次更新、每100保存。运行保存的 env.yaml 已核对失败窗口60s、成功终止最后执行，agent.yaml确认2000次。首份 `model_0.pt` 可读并含课程状态，预期末次 `model_1999.pt`。训练进行中，14:29本地约第47轮，完整步数0。

实际启动命令为以下命令，工作目录 rc_mjlab。

```powershell
.venv/Scripts/python.exe -u -m pointleg.run train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run flat_8700_fixed_gait_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 2000 --agent.save-interval 100 --agent.run-name train_3_audit_2000 --agent.logger tensorboard --agent.upload-model False
```

2026-11-18 用户实际新运行名为 `2026-11-18_12-04-36_train_1`，配置 1000 次，从 `flat_8700_fixed_gait_pretrained/model_0.pt` 启动，保存至 `model_999.pt`。末 100 条日志均值赛道速度 −0.230413 m/s、回合约 0.636 s，完整步数与覆盖为 0。本轮修复后退奖励和提前失败代价，下一轮建议用已有迁移种子重新启动新进程。该种子 123/399、iter=0，SHA256 `a0f481a6e0bc213c427c0f62e56a712d57ef10c92172b0f28e156ab8f41d72a9`，本轮没有迁移、停止或启动训练。

2026-11-18 二次审查修复课程续训状态。`SteppingStonesOnPolicyRunner` 保存 `stepping_stone_curriculum`，完整 resume 恢复已解锁最高等级和统计窗口，Actor-only 加载保留当前课程、轮次和公共计数。较早检查点缺少课程元数据时使用当前默认课程，原 123/399 模型可直接加载。奖励资格和同桩落点修正需重启进程后采用，本轮未启动或停止训练。下方起步修复训练命令仍可使用。

## 2026-11-18 起步停滞修复后的训练命令

`2026-11-18_01-26-20_fixed_gait_30000` 日志至 8924 轮首步完成始终为 0。修复激活与单腿反馈后，使用已有 `flat_8700_fixed_gait_pretrained/model_0.pt` 重新训练。已检查种子为 Actor/Critic 123/399、iter=0，SHA256 为 `a0f481a6e0bc213c427c0f62e56a712d57ef10c92172b0f28e156ab8f41d72a9`。结束旧训练进程后，在 rc_mjlab 目录运行以下命令。此命令尚未由本轮代理执行。

```powershell
uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run flat_8700_fixed_gait_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 30000 --agent.run-name fixed_gait_start_fix --agent.logger tensorboard --agent.upload-model False
```

2026-11-15 左前支撑修正使用 `uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --competition-six-curriculum --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-15_16-59-04_new_train_16_six_gap --agent.load-checkpoint model_42700.pt --agent.max-iterations 1501 --agent.run-name new_train_18_left_front_handoff --agent.logger tensorboard --agent.upload-model False` 从六桩课程权重续训。命令已核对，本轮未启动。

2026-11-15 面对直接上 10 cm 间隙训练退化，使用 `uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --competition-six-curriculum --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-15_14-26-31_new_train_14_support_handoff --agent.load-checkpoint model_41200.pt --agent.max-iterations 1501 --agent.run-name new_train_16_six_gap --agent.logger tensorboard --agent.upload-model False` 可从退化前的检查点训练六桩间隙课程。命令已核对，本轮未启动。

2026-11-15 比赛六桩专项已运行。工作目录 `05_software/train/rc_mjlab`，命令 `uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --competition-six --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-15_14-26-31_new_train_14_support_handoff --agent.load-checkpoint model_41200.pt --agent.max-iterations 1501 --agent.run-name new_train_15_competition_six --agent.logger tensorboard --agent.upload-model False`。本次使用同等参数经本地 venv Python 后台启动。输出 `logs/rsl_rl/pointleg_stepping_stones/2026-11-15_15-55-47_new_train_15_competition_six`，预计末次 `model_42700.pt`。

2026-11-14梅花桩回放默认每次重置随机选择2～8根桩的课程等级，成功或失败重置后可看到不同等级，随机抽取可能连续重复；训练配置不受影响。

2026-11-01入口清理：原四个单方向ID及配置已下线，高程仅保留两个合并任务与混合Elevation。现有 `pointleg_stairs`、`pointleg_slope`、`pointleg_elevation` 下的初始化model_0可继续按对应任务resume；旧 `pointleg_stairs_up/down`、`pointleg_slope_up/down` 的检查点与日志文件保留，复用参数时以文件路径向现有任务transfer。下文四方向训练命令是历史记录，当前可执行命令见 `docs/ELEVATION_PAIRED_TRAINING.md`，旧专项文档已改为三入口索引。

## 2026-11-01 双向合并任务初始化

后续优先 `PointLeg-Stairs-v0`/`PointLeg-Slope-v0` 两个入口，基础奖励版本8。已把同一HighStep `2026-11-01_12-41-56_tran_4_2/model_14100.pt` 迁移至 `pointleg_stairs/highstep_tran_4_2_14100_pretrained/model_0.pt`、`pointleg_slope/highstep_tran_4_2_14100_pretrained/model_0.pt`。源SHA256仍为 `e9251beb3aee12463109f704d86ef14b1bcc3645d6ff527cf0a47b28195376a9`，两个新目标的轮次/公共计数/优化器状态/课程上限均为0；Actor/Critic全部逐张量一致，严格resume成功。目标哈希分别 `1b4f491e0d4bff10b3da148097af39287c3b3474aded67cb3ff1ab5b87d2cb29`、`db05c21fe8cdaca3753e15fa15cdcd0a67e44f624a33a6cd9422d62974d11c0b`，报告 `artifacts/elevation_paired_transfer_validation.json`。

文档 `docs/ELEVATION_PAIRED_TRAINING.md` 给出256环境各追加200次、间隔100的直接resume命令，看100和末次199；两条CLI已tyro解析并解析到实际目标model_0。没有启动训练或仿真。已有新目标直接resume，不重复transfer；旧单方向或混合范围不同，不能直接resume成双向组。旧五份model_0和原日志均保留。

## 2026-11-01 HighStep 迁移至 Elevation

用户结束 HighStep `tran_4_2`，训练进程已停止，停止前最后完整检查点为 `logs/rsl_rl/pointleg_high_step/2026-11-01_12-41-56_tran_4_2/model_14100.pt`，SHA256 为 `e9251beb3aee12463109f704d86ef14b1bcc3645d6ff527cf0a47b28195376a9`。通过 `pointleg transfer` 已创建下列五个零轮次初始化：

| 任务 | 目标检查点 |
| --- | --- |
| `PointLeg-Elevation-v0` | `logs/rsl_rl/pointleg_elevation/highstep_tran_4_2_14100_pretrained/model_0.pt` |
| `PointLeg-StairsUp-v0` | `logs/rsl_rl/pointleg_stairs_up/highstep_tran_4_2_14100_pretrained/model_0.pt` |
| `PointLeg-StairsDown-v0` | `logs/rsl_rl/pointleg_stairs_down/highstep_tran_4_2_14100_pretrained/model_0.pt` |
| `PointLeg-SlopeUp-v0` | `logs/rsl_rl/pointleg_slope_up/highstep_tran_4_2_14100_pretrained/model_0.pt` |
| `PointLeg-SlopeDown-v0` | `logs/rsl_rl/pointleg_slope_down/highstep_tran_4_2_14100_pretrained/model_0.pt` |

五项均重置为 `iter=0`，使用新优化器及地形课程；Actor/Critic 与源策略逐张量一致，接口保持45/325/12。迁移只创建检查点，没有启动训练或物理仿真。后续每个目标仅对自己的 `model_0.pt` 使用 `--agent.resume True`；五个任务之间仍属跨任务，不可直接 resume。

## 2026-11-01 tran_4_2 距离额度与超高惩罚

2026-11-01 tran_4_2 修正HighStep奖励冲突：active_lift_event +0.35及active_lift_progress +0.08的名义权重保留，二者和hip_swing_coordination +0.06改乘同一逐足承重路程额度。每控制周期沿本段前进起点固定航向刷新机身历史最远位置，新增距离封顶指令速度×dt；每足只在连续8N承重期间积累新距离，离地冻结min(该距离/7厘米,1)，新支撑清零。循环共享该额度，常见7厘米以内等高度的同总路程拆成更多小步不会多领这三项收益；静止、往返和已错过路程不补发，超速多余位移不延迟兑现。连续抬升的峰值及已支付高度封顶15厘米。增加active_lift_overshoot −0.5：摆动时相对起脚足底主动抬升超过15厘米，按5厘米尺度平方、单足封顶4，四足均值和最差足各半；不乘正奖励路程额度，原地超高摆动也扣分。short_stance_target为独立软目标0.24秒，原0.16秒有效起脚/稳定落地门槛保留。侧漂−2.5→−3.0，容许0.010米/秒/尺度0.10米/秒保持且左右等价。后扫+0.10/7厘米平方/0.30秒、姿态、PPO/PD、45/325/12和其他任务不改。当前25项奖励。新增Metrics/high_step_lift_distance_credit_{leg}、high_step_lift_overshoot_{leg}_m、high_step_new_forward_distance_m。代码入口config/high_step.py、mdp/high_step_basic.py及mdp/high_step_hip_swing.py。

42项定向配置/张量检查通过（test_high_step_basic、test_high_step_backstroke、test_high_step_lift_budget、test_high_step_coordination），覆盖8/12/15厘米评分、15厘米以上正收益封顶/负代价递增、常见等距离拆步不多领、髋摆共用额度、静止/往返/超速/腾空、重复读取/部分重置/失败/非有限、0.24秒浮点边界及控制周期积分；AST与diff --check通过。未额外启动物理验证。旧train_4已按命令行身份停止Python PID30180及对应uv启动器，最后保存model_17100.pt，模型日志保留；从用户认可原始train_3_2/model_10198.pt直接resume，源SHA256 6e38e914384558557b5585e0dfb6a1842af9880d39d52697423dcaaee785931e。启动状态另记，不把配置和张量通过等同于训练效果。

从10198追加10001次更新，每100保存，预期末次20198；先回放10300/10400观察足高、支撑和拖腿。正奖励已按路程缩放，不能直接用旧运行总分比较效果。

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-01_04-10-49_train_3_2 --agent.load-checkpoint model_10198.pt --agent.max-iterations 10001 --agent.save-interval 100 --agent.run-name tran_4_2 --agent.logger tensorboard --agent.upload-model False
```


## 2026-11-01 train_4 支撑相行程

2026-11-01 HighStep train_4 支撑行程修正：stance_backstroke 权重 +0.04→+0.10，7厘米目标由线性改为比例平方，乘 min(承重时长/0.30秒,1)；8N承重、0.16秒最低门槛、2.5厘米累计真实接触滑移质量保留。有效后扫仍取真实机身推进和相对支撑足后扫的较小值，取最后一个承重采样，避免离地后的足端移动计入。超过目标不再增加收益。command_orthogonal_drift −2→−2.5，0.010米/秒死区和0.10米/秒尺度保持，左右等价。仍24项，主动抬升目标/两项权重、姿态、45/325/12、PPO/PD及Elevation/Jump不改。新增逐足 Metrics/high_step_stance_sweep_{leg}_m、stance_body_progress_{leg}_m、stance_start_x_{leg}_m、stance_end_x_{leg}_m、stance_duty_factor_{leg}，分别表示支撑相对后扫、身体推进、落地/离地前相对机身前向位置和完整周期支撑占比，前向坐标轴在落地时固定，世界足端静止也会随身体推进产生后扫。保持旧 backstroke/stance_time/contact_slip 指标。25项配置和张量检查通过，覆盖相同总距离短步不优于长步、滑移降质、7厘米饱和、静止/拖脚/短触地/失败无收益、事件缓存/部分重置/非有限/控制周期及左右同罚；不代表训练验收。用户明确授权按固定命令实际后台续训10001次更新，不执行额外仿真测试。

单足无滑移情况下，3.5厘米/0.18秒支撑的后扫奖金约0.015，7厘米/0.36秒约0.10；相同总距离拆成两次短步合计约0.03，低于一次长支撑的0.10。延长到7厘米以后不再增加奖金，过长驻足仍受原时序代价约束。15厘米主动抬升目标和其他步态奖励继续单独计分。

验收使用四足各自的 stance_sweep、backstroke、stance_time、stance_duty_factor 及真实接触滑移，与主动抬升和回放同时比较。支撑占比是上一完整支撑时长除以支撑加摆动时长，不是触地力大小；支撑起止x是相对机身、沿落地时固定前向轴的位置。

用户指定同任务直接resume，源迭代10198，追加10001次更新，预期末次20198，保存间隔100；由用户验收。

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-01_04-10-49_train_3_2 --agent.load-checkpoint model_10198.pt --agent.max-iterations 10001 --agent.save-interval 100 --agent.run-name train_4 --agent.logger tensorboard --agent.upload-model False
```


2026-11-01最新HighStep共24项：新增lf_rr_pose_tracking +.25，在右后足真实有效摆动且主动抬升≥3厘米、机身前进>.05米/秒、至少两足承重时，左前跟踪右后当前髋摆/膝角及膝相对髋的机身竖直位置；左前自身不必已进入摆动，避免旧对角项的双足摆动门槛不给纠正信号。关节误差尺度20°/25°，膝高4厘米，两类误差各半取高斯正分；后腿仅作参考，策略仍共享。侧漂权重−1→−2，死区.015→.010米/秒，左右对称；按用户追加要求髋外展−.9→−3.6（4倍），零目标/零死区/.25弧度尺度保持。其余项、45/325/12、PPO/PD/地形及其他任务不改。静态模型审计未发现缺执行器、错绑或左前髋摆驱动失效：XML12→删除0→重建12，全部动作槽/关节对应，0.1弧度目标增量仅对应DOF产生5N·m；四髋相同静态角增量使膝上升约14.1～14.3毫米。报告artifacts/high_step_lf_actuator_audit_2026-11-01.json；未积分时间，不代表动态碰撞/带载饱和或策略已验收。一个合成张量检查通过：左前仍支撑时可获跟踪反馈、匹配高于异常姿态、失败/非有限/静止0及左右侧漂同罚。未仿真或训练；新进程同任务直接resume，当前截图未绑定到指定模型文件。

2026-11-01 HighStep承重后扫与侧漂调整：保留已有command_orthogonal_drift，权重−.5→−1.0、1.5厘米/秒死区与10厘米/秒尺度不变，对左右漂移等价，不加入只针对左侧的偏置；横向速度为机身坐标，不能替代绝对航向/全局路径控制。新增stance_backstroke +.04，每足连续8N承重≥.16秒后离地时按有效后扫/7厘米线性发放，单足每次最多+.04；有效后扫取落地到离地间机身实际前进距离与身体相对该支撑足前进距离中的较小者，负值置0。按真实接触点累计滑移乘max(1−滑移/2.5厘米,0)，无机身推进的脚底拖滑不得分；超过7厘米饱和，事件原始值除dt，由奖励管理器积分一次。现有高抬腿、对角/髋摆、姿态、滑移惩罚和支撑超时保持；总奖励23项，45/325/12、其他任务/PPO/PD不变。指标Episode_Reward/stance_backstroke、Metrics/high_step_backstroke_{fl,fr,rl,rr}_m、Metrics/high_step_stance_time_{leg}_s、Metrics/high_step_contact_slip_mean_m_s和Metrics/high_step_lateral_speed_abs_m_s，应同时观察四足抬升及回放，防止过长支撑、停步、足底滑移或高度退化。按用户要求只执行一个合成轨迹检查，23项配置、真实身体推进奖励、原地拖脚无收益、单事件、左右同等漂移通过；未仿真/Viser/训练，不能据此断言实机发力或爬阶通过。用户未指定当前较好模型路径，继续训练时固定选择其HighStep检查点直接resume，新进程才启用。

2026-11-01 HighStep加入diagonal_rear_tracking +.2，22项：左前跟右后、右前跟左后；起脚前真实承重且对角两足同时有效摆动、至少两足承重、前进指令及实际机身前向速度>.05米/秒时，以两足进入共同有效摆动时的各自髋俯仰/膝关节角为起点，比较角位移，不强制绝对关节角。后腿相对起点的髋或膝位移至少5°才发放，20°运动强度饱和；前后角位移误差按15°尺度高斯计分，两对得分相加除2，按控制周期正常积分。实际后腿位移同时参与得分，不能保证后腿策略参数被固定，也不能由后腿单独决定策略梯度；前后足不同步时该项无收益而非强制锁相。现有足高、髋外展、pitch、前足工作空间及髋摆项保持，45/325/12、PPO/其他任务不变。Metrics/high_step_diagonal_error_fl_rr_deg、_fr_rl_deg记录两关节角位移差的均方根；Metrics/high_step_diagonal_rear_motion_fl_rr_deg、_fr_rl_deg记录对应后腿相对起点最大关节位移，仅满足门槛的样本记录。只执行单个配置与张量轨迹检查，22项/权重.2，左前右后匹配高于相反、后腿未动或静止0；未仿真、Viser、训练或完整回归，右前左后由相同映射实现但未做独立轨迹测试。新进程从用户选定同任务检查点直接resume，实际效果需短训观察。

2026-11-01最新HighStep针对前足过度前伸、髋外展和前倾调整：新增front_foot_workspace −.7，分别取左右前足球心相对同侧髋俯仰连杆的机身坐标x，以.6秒EMA、1秒启动宽限追踪，平均前伸超过4厘米时按8厘米尺度平方计罚，每足封顶4后平均；仅前进时生效，不锁定瞬时摆动姿态。hip_abduction_excess −.3→−.9（默认0角、零死区/.25弧度尺度不变）；high_step_pitch −.5→−1、死区3°→2°/尺度8°不变，正负俯仰同等处理；hip_swing_coordination +.04→+.06，原有效步/稳定落地/两足支撑/身体前进资格、20°髋幅及3厘米相对髋膝上下位移饱和目标不变。主动抬升+.35/.08/15厘米和其余未提及的奖励保持，总数21，45/325/12、PPO/PD/其他任务不改。指标Metrics/high_step_front_foot_mean_x_fl_m和_fr_m，可与high_step_pitch_deg、hip_abduction_deviation_mean、分足髋摆及膝竖直幅度、分足主动抬升共同检查；EMA均值不是每步足端位置。单个配置/张量检查通过，验证前足平均前伸受罚、零前伸无罚、同周期缓存及部分重置；未仿真、回放或训练，实际改善和足高需短训验证。用户未指定截图对应检查点，续训应固定用户选定的HighStep检查点直接resume，新进程才应用奖励。

2026-10-31最新HighStep应用户要求在17项上增加3项，共20项：hip_abduction_excess −.3，以默认0弧度为中心、零死区、.25弧度尺度平方，四腿分别计分；high_step_pitch −.5，正负俯仰超过3°后按8°尺度平方惩罚；hip_swing_coordination +.04，对有效完成并稳定落地的摆动奖励髋俯仰角幅及膝相对髋在机身坐标竖直幅度，20°/3厘米各饱和，两者各占一半。只在8N支撑且摆动期间至少两足承重、原主动高度有效落地条件、实际身体前进≥1厘米时结算一次；不单独给原地抖关节奖励，正常动作仍允许偏离0，不是硬限位。原17项（足高+.35/.08、15厘米目标）保持，45/325/12及其他任务配置未改。单个无物理张量轨迹检查通过：20项配置、零死区和水平参数、髋20°/膝3厘米饱和并只结算一次；未仿真或训练。指标为Metrics/hip_abduction_deviation_mean、Metrics/high_step_pitch_deg、Metrics/high_step_swing_hip_range_{leg}_deg、Metrics/high_step_swing_knee_vertical_range_{leg}_m。

+2026-10-31最新HighStep在15项基础项上新增两项独立主动抬升奖励，当前17项：active_lift_event +.35，在足完成有效摆动并稳定承重.16秒后结算 `min(主动峰/.15,1)^2`；active_lift_progress +.08，摆动期间仅按主动峰新增加高度/.15连续支付。主动峰是足底相对起脚足底的上升，不是相对世界地面高度；无过高奖励。资格为起脚前承重≥.16秒、起脚后其余足≥2足承重、离地.10～.36秒、足前移≥2.5厘米、稳定落地≥.16秒。静止、原地点脚、短支撑、落地未稳、全足腾空、失败和非有限不领；高度不与周期/后扫/收腿/姿态质量相乘。仍用基础评估有效循环条件，高度暂报告而不作为通过硬阈值；新增主动抬升及分足指标。6200仍直接resume，建议新运行名active_lift_6200，100更新、看6250/6299；45/325/12和其他任务保持。

2026-10-31最新HighStep基础重建从用户指定2026-10-30_21-53-51/model_6200.pt直接resume，不transfer、不用7199/15498等源。完整命令见docs/HIGH_STEP_TRAINING.md首段；关键参数：--agent.resume True --agent.load-run 2026-10-30_21-53-51 --agent.load-checkpoint model_6200.pt --agent.max-iterations 100 --agent.save-interval 50 --agent.run-name basic_rebuild_6200。6200～6299，看6250/6299，输出新目录。CPU完整resume保留Actor/Critic/优化器/迭代6200/common_step_counter148968；源SHA256 4a30656c54836703bc7991c9b0ffa7f49a71bb7b9e69cff54f9d18e8d3e14b24不变，CLI精确解析通过。当前15项且没有高度收益，先确认姿态/持续换步/真实滑移，不宣称原高度必然保持；需要新进程采用，不影响已有进程或PT。未启动仿真/Viser/训练。

2026-10-31HighStep接触点滑移修正需新进程，完整300轮命令见HIGH_STEP_TRAINING.md首段：固定--agent.resume True --agent.load-run 2026-10-30_21-53-51 --agent.load-checkpoint model_7199.pt，512环境、max-iterations300、run_name=contact_slip_cadence，tensorboard/不上传。同任务不transfer。真实源iter7199/common_step_counter172944，输出新目录，更新7199～7498，对照7300/7400/末次7498，源SHA256 3170689f7ea12bde64f8c24144edad3e7a99be92b4cb3b000316f3bd2486babb保持。严格CPU加载/临时保存核对网络/优化器/计数相同，CLI已解析并精确选源；没有启动/停止仿真或训练。旧slip_velocity_mean继续为参考点速度，验收应看新的contact_slip_*及累计距离，原12.5厘米每秒不能直接与新指标比较。

2026-10-31当前版本7需新进程采用，同混合/专项可直接resume，跨专项或混合转专项仍需既有transfer。实际验证源为pointleg_elevation/highstep_fixed120_pretrained/model_0.pt，迭代/公共计数0、SHA256 be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568，严格CPU加载/临时保存后Actor/Critic/优化器/地图保持，元数据7，源hash未改。专项奖励文档给混合直接resume256环境追加200轮、run_name=functional_posture_v7，对照100与末次199；四专项沿用各自初始模型/目录，未选择或生成正式专项权重。本轮没有运行仿真/Viser/训练或停止已有进程，单元通过不代表实际爬阶改善。evaluate报告新增functional_posture及按事件计数加权的functional_posture_summary，缺事件写null，原nominal90%/robust80%按档阈值不改。

2026-10-30四项Elevation专项已可单独训练，命令见`docs/ELEVATION_SPLIT_TRAINING.md`。`pointleg train PointLeg-StairsUp-v0/StairsDown-v0/SlopeUp-v0/SlopeDown-v0`分别使用`pointleg_stairs_up/down`、`pointleg_slope_up/down`目录，默认256环境、TensorBoard、不上传模型。四份配置分别可调整，默认沿用版本6和原PPO。文档包含每项从头训练、`transfer --checkpoint-file PATH --output-run initial`、从目标目录`initial/model_0.pt`继续训练、play和专项evaluate；八组从头/续训参数已实际用tyro解析，未启动训练。现有模型进入新专项以及跨专项/回到混合任务须先transfer，只继承网络，优化器/迭代/课程重置；同专项后续resume保留这些状态。任务范围保存在检查点`elevation_settings.terrain_kind`，旧混合检查点缺失该字段视为混合，兼容同任务resume。未选择正式源模型或生成正式model_0，源路径须由用户后续选择；既有进程不热更新。

2026-10-30用户选择先在HighStep平地纠正，已改源码，完整命令见`docs/HIGH_STEP_TRAINING.md`首段。指定源`pointleg_high_step/2026-10-28_23-25-35_yaw_mild_soft_contact/model_15498.pt`本来就是HighStep，因此直接`--agent.resume True --agent.load-run 2026-10-28_23-25-35_yaw_mild_soft_contact --agent.load-checkpoint model_15498.pt`，512环境、追加`--agent.max-iterations 200`、`--agent.run-name flat_front_correction`。真实加载迭代15498/公共计数372240；周期保存15500/15600，末次15697，先对照15600和15697，不把200次更新误称model_200。保留优化器/网络/计数和45/325/12，源SHA256 `9c9879852f87b017a6e7155301d5e11b03d991c94a4127e16690dad4e9bf34b8`不变；输出新时间戳目录。CLI仅解析并核对源路径，未调用run_train，未仿真/Viser/训练或停止已有进程；需新进程采用15厘米与承重/前后慢姿态奖励。之后是否迁入Elevation由平地回放验收决定，本轮不执行第二阶段，也不新增Elevation平地开关。

2026-10-30建议下一步先做同一Elevation版本6奖励下的平地专项纠正，再恢复台阶：需要增加仅采flat的模式（尚未实现），保留同一45/325/12及几何/课程结构以避免再次更换奖励目标。现有HighStep只左右慢对称，不能把旧HighStep指令当作已采用版本6前后纠正。建议从原高抬腿基础开始、每100轮回放，以足高不退化、四足持续换步、前后慢偏差与支撑后扫改善作为切换条件；未创建新训练命令、模式或启动训练。

2026-10-30版本6已完成：同固定地图highstep_fixed120_pretrained/model_0可直接resume，不需要transfer；加载仅奖励/接近距离变化提示，网络/优化器/迭代0保持、源hash不变。推荐256环境200次更新、运行名support_push_v6，对照100/199；不从用户回放已退化的400以后继续。仅提供指令未启动/停止训练或回放。新出生设置在reset采用，旧进程不会热更新；新姿态/支撑指标不能回填旧日志。

2026-10-30版本5已实施并核对直接resume：现有highstep_fixed120_pretrained/model_0.pt，SHA256 be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568，可完整恢复网络/优化器/迭代0/课程并使用新速度0.15～0.25、60秒及15厘米奖励。地图未再变，不需要重新transfer，也不从退化600/700/1999续训。推荐256环境追加200次更新、运行名height15_low_speed_v5，默认日志/TensorBoard；框架从0编号，回放100与最终199，不能改名假称200。只交付指令，未启动或停止训练/Viser。完整uv命令与验收见专项文档；PPO学习率/熵/网络不变。

2026-10-30用户实际运行fixed120_stairs到1999后报告静止，日志确认逐步丢失高抬腿和有效循环，固定120方案尚未验收。源15498与迁移model_0逐参数一致，Actor观测/动作配置一致；原命令能启动但不代表训练方案有效。下一步建议先对齐源速度0.15～0.25与地形适配的步态保持目标，再从原基础开始，不建议原样延长1999；此为分析建议，未改配置或生成新训练命令，未启动/停止进程。

2026-10-30固定120毫米楼梯命令已提供：沿用已核实存在的`pointleg_high_step/2026-10-28_23-25-35_yaw_mild_soft_contact/model_15498.pt`，先transfer至尚未存在的`highstep_fixed120_pretrained`，成功后256环境resume其model_0追加2000轮，运行名fixed120_stairs、TensorBoard及默认日志目录。完整命令见专项文档；本轮只提供命令，没有实际迁移、启动训练或指定退化2099为源。重复执行迁移会拒绝覆盖已有输出，应在首次成功后直接使用训练命令。

2026-10-29最新：Elevation版本4已按用户要求回滚，楼梯改为固定120毫米，先前hip_swing_clearance/model_100直接resume命令已不适用于旧地图→新地图。此次楼梯布局契约实际变化，旧40～120毫米地图的Elevation或用户已有HighStep源需先用现有transfer仅迁移权重创建固定高度契约；其后新地图同任务可直接resume。不是因奖励变化强制迁移。未转换/覆盖模型，未启动/停止训练；回滚代码不会回滚已经学坏的权重，不默认从2099继续。源高抬腿模型由用户实际回放选择。

2026-10-29版本4命令示例已核对源`2026-10-29_14-35-10_compact_swing/model_100.pt`存在。依据用户“早期偶尔上阶、1000轮后蹲低”反馈建议早期100轮作为对照起点，256环境直接resume追加2000轮、运行名`hip_swing_clearance`；不宣称该源最优，不强制transfer，未启动训练。完整uv命令见`docs/ELEVATION_GAIT_REWARDS.md`。

2026-10-29 Elevation版本4只改奖励/指标/元数据，保持45/325/12及地形/课程，可从兼容Elevation检查点直接resume，保留优化器、迭代和课程，无需再transfer。新增奖励状态逐回合初始化，旧进程不会热更新；本轮没有启动、停止训练或改PT。用户指出初始model_0已有前腿大腿摆幅偏小，因此未用其作为姿态教师；髋摆幅用有效循环分级收益重新学习。跨HighStep→Elevation迁移已有`highstep_pretrained/transfer.json`，不能把源HighStep后来修改的奖励视为已经进入该Elevation运行。续训源要结合用户回放挑选，未以晚期迭代最大作为最优判据。

2026-10-29最新周期0.60秒/主循环0.38/高度宽度3厘米/收腿−0.2需新进程采用；示例固定`--agent.load-run 2026-10-29_12-53-21_compact_swing --agent.load-checkpoint model_16499.pt --agent.resume True`，512环境追加1000轮，运行名`period06_height17`。接口/网络/速度和地形保持，同任务直接resume无需transfer，原模型不覆盖；未启动或停止训练。评估必须使用新配置的0.60秒目标，旧策略回放不会因改奖励自动放慢或抬高。

2026-10-29紧凑收腿修正可从用户当前`2026-10-28_23-25-35_yaw_mild_soft_contact/model_15498.pt`直接resume，同任务45/325/12和网络不变；示例512环境追加1000轮、运行名`compact_swing`，命令见HighStep文档。需新进程才采用运动站姿掩码及摆动远伸代价；原PT不变，没有启动/停止训练。这是延续当前模型做对照的示例，不否定用户此前认为13000比13598回放更直。

2026-10-28最新温和偏航和接触力参数需新进程，继续从`2026-10-28_20-43-44_111/model_13000.pt`直接resume，512环境追加1000轮，运行名`yaw_mild_soft_contact`；命令见HighStep专项文档。瞬时偏航现−1、持续偏航尺度9，增加力阈值150N的奖励和4子步内部历史，不改45/325/12；无需transfer，不覆盖源模型。本轮未启动/停止训练，原PT回放不会因奖励改动而自动变软。

2026-10-28用户反馈13000实际Viser回放更直，最新源改为`--agent.load-run 2026-10-28_20-43-44_111 --agent.load-checkpoint model_13000.pt --agent.resume True`，追加1000轮，运行名`yaw_w2_from13000`。新偏航权重−2.0、持续偏航尺度7和抬腿0.30保留；没有再改代码或启动训练。下文13598推荐为被用户回放观察更正的历史选择。

2026-10-28最新偏航参数需新进程采用，推荐固定`--agent.load-run 2026-10-28_20-43-44_111 --agent.load-checkpoint model_13598.pt --agent.resume True`，512环境、追加1000轮、运行名`yaw_w2`，完整命令见HighStep文档。13598已从12099训练1500轮，运行保存配置抬腿0.30/持续漂移−1.5；当前新偏航−2.0及持续偏航尺度7未进入该PT。原PT不覆盖，本轮未启动/停止训练。用户追问13000只进行日志对比，没有改为从13000训练。

2026-10-28用户要求继续加强持续漂移项，当前权重−1.5。最新推荐源固定为`--agent.load-run 2026-10-28_07-46-16 --agent.load-checkpoint model_12099.pt`，同任务`--agent.resume True`，512环境、`--agent.max-iterations 1000`追加1000轮，运行名`drift_bias_w15`；命令见HighStep专项文档。该PT位于旧目录，其对应事件在`2026-10-28_14-04-52`；不要按日志目录猜PT位置或使用通配符。未启动或停止训练，原PT不覆盖。

2026-10-28二次HighStep纠偏续训示例固定最新指定的`--agent.load-run 2026-10-28_11-26-12 --agent.load-checkpoint model_7199.pt`，`--agent.resume True`，追加1000轮；完整命令见HighStep专项文档。该运行原配置为通配符且实际日志5200～7199，不能当作从8199续训的证据。只改奖励可直接恢复，原PT不覆盖；本轮未启动/停止训练。

2026-10-28 HighStep纠偏参数需新进程采用，用户指定从`logs/rsl_rl/pointleg_high_step/2026-10-28_07-46-16/model_8199.pt`直接resume，512环境；`--agent.load-run 2026-10-28_07-46-16 --agent.load-checkpoint model_8199.pt --agent.max-iterations 1000`表示在恢复迭代后追加1000轮，示例完整命令见`docs/HIGH_STEP_TRAINING.md`。同任务45/325/12和网络结构不变，不需要transfer；保存至新运行目录，不覆盖原检查点。本轮没有启动或停止训练。

2026-10-27 HighStep的`play`命令在Viser建面板时因零侧向/角速度范围触发AssertionError，已修复通用`UniformVelocityCommand.create_gui`，零轴为只读数字、单向轴不引入反方向、小幅范围不再小于Max滑条下限。模型已在报错前加载，本次不涉及模型兼容或重训。重跑原play命令即可采用修复，不需要transfer/resume训练；也未自动启动回放。uv在默认缓存目录遇到拒绝写入后改用项目内`.uv-cache`，没有安装依赖。

2026-10-27审查修复：需要新进程才采用世界坐标承重、分级抬脚和实际对角配合奖励；现有进程不会热更新，本轮没有启动/停止训练或更改检查点。HighStep和同任务Elevation仍可直接resume，维度45/325/12，奖励版本不强制transfer；优化器、迭代及兼容的课程继续恢复。HighStep通用评估默认改0.2米/秒而非0.4，增加分足足高/周期/步幅/合格比例和对角配合验收，速度误差上限0.05米/秒。外环ROS导航未实现，45维单帧策略不能保证受扰后的绝对回中。

核对日期：2026-10-26。以下命令在 `E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab` 的 PowerShell 中执行；示例模型路径需要替换为实际且匹配任务的文件。

## 本机环境快照

现有 `.venv/Scripts/python.exe` 可使用；`.venv/pyvenv.cfg` 记录 CPython 3.12.2。`venv/` 目录也存在，本文命令统一使用 `.venv`，不假定两者相同。

| 包 | 本次读取的安装版本 |
|---|---|
| pointleg | 0.1.0 |
| mjlab | 1.4.0 |
| torch | 2.11.0+cu128 |
| mujoco | 3.12.1.dev975126947 |
| mujoco-warp | 3.12.0 |
| warp-lang | 1.17.0 |
| rsl-rl-lib | 5.4.0 |
| tyro | 1.0.13 |
| onnxruntime | 1.29.0 |

这是本机安装快照，不代表其他机器的解析结果。根配置要求 Python >=3.10，本地 mjlab 还要求 <3.14；MuJoCo 和 Warp 使用项目指定源与约束。依赖以 `pyproject.toml`、本地框架配置及 `uv.lock` 为依据，不按旧 `DEPENDENCIES.md` 的 MuJoCo 3.8 描述重装。

已有环境可直接调用 Python，避免为查看任务或整理文档触发安装。需要恢复锁定环境时才使用 `uv sync --locked`；不要无意重锁或整体升级 nightly 依赖。安装进度停止时应停下报告当前状态，遵守用户要求。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run list
```

在环境已安装且无需同步时，也可使用 `uv run --locked --no-sync pointleg ...`。

用户本轮明确要求Windows原生锁定环境，日常启动优先 `uv run --locked --no-sync pointleg ...`，不切换WSL或旧 `venv`。本地框架 `mjlab/CLAUDE.md` 也要求框架开发命令经 `uv run` 执行；这里直接Python示例只指同一项目已有`.venv`，不表示允许重建另一套环境。

## 从调试到训练

2026-10-27用户更新验证偏好：今后不再自动跑仿真测试或PPO调试。下列命令保留作用户主动执行的参考；默认开发验证限于源码、配置及定向单元检查，不因历史工作流包含debug就自动启动。

先验证入口和小规模训练，再单独启动正式实验。`--debug` 默认 16 环境、每环境 8 步 rollout、2 次迭代、每次保存、TensorBoard、不上传模型，保持网络结构；CLI 可以继续覆盖默认值。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run train PointLeg-Flat-v0 --debug
.\.venv\Scripts\python.exe -m pointleg.run train PointLeg-Elevation-v0 --debug --env.scene.num-envs 4
```

共享 PPO 模板在 `config/rl_cfg.py`：Actor/Critic 隐藏层 `(512,256,128)`、ELU、关闭观测归一化；每环境 24 步 rollout、5 次学习、4 个 mini-batch、学习率 `8e-4`、entropy `0.003`、gamma `0.99`、lam `0.95`、每 100 次迭代保存。Flat 默认 10000 次迭代，共享模板默认 20000 次；Elevation/Jump 学习率为 `3e-4`、gamma 为 `0.995`，梅花桩为 `3e-4`、`0.9995`。

默认并行数不保证适合本机显存。现有专项文档给出普通 Rough 256 环境和混合任务 64 环境的实验命令，这些是起步参考，不能作为所有 GPU 的承诺。

## 从已有平地策略开始 Rough 系列实验

本机存在 `logs/rsl_rl/pointleg_rough/flat_pretrained/model_0.pt`。Rough 与 RoughMixed 都使用 `pointleg_rough` 实验名，必须用不同运行名并保留配置。

```powershell
uv run --locked --no-sync pointleg train PointLeg-Rough-v0 --env.scene.num-envs 256 --agent.resume True --agent.load-run flat_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 1000 --agent.run-name rough_from_flat --agent.logger tensorboard --agent.upload-model False

uv run --locked --no-sync pointleg train PointLeg-RoughMixed-v0 --env.scene.num-envs 64 --agent.resume True --agent.load-run flat_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 1000 --agent.run-name rough_mixed_from_flat --agent.logger tensorboard --agent.upload-model False
```

上面利用当前可完整加载的点足基线文件，不是通用的跨任务权重转换器。模型名相同、输入维度相同都不保证来自同一任务。每轮只改变一个主要实验因素，并通过固定评估挑选检查点。

## 权重迁移与续训

`pointleg transfer`当前支持Elevation及四个专项、Jump和SteppingStones。Elevation/Jump严格加载兼容网络；SteppingStones支持45/325→105/381的首层扩列或已有105/381同结构迁移，原列与其他参数严格保持，新增列置零。重新初始化优化器、迭代和公共步数，按目标任务起始课程创建`model_0.pt`和`transfer.json`；目录已存在时拒绝覆盖。

2026-10-31已实际执行用户指定`pointleg_flat/2026-10-22_09-27-04/model_8700.pt`→`pointleg_stepping_stones/flat_8700_pretrained/model_0.pt`。源SHA256`c7af6d864a0cfde0f91b382ca0854016a806879392d9327224e10d036e8e52aa`保持；目标SHA256`c6e0b9b647d7477e0f2e72095fc6e83ba3e65e01bad6513982ecdb83e7d87ade`。后续使用`uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run flat_8700_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 10000 --agent.run-name flat8700_stepping --agent.logger tensorboard --agent.upload-model False`，从0追加10000次，末编号9999，默认梅花桩日志目录。该指令已解析且目标路径实际解析正确，迁移检查点已CPU严格resume加载；未启动训练、仿真或停止已有进程。不要重复迁移到同名目录。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run transfer PointLeg-Elevation-v0 --checkpoint-file "源模型路径\model_xxx.pt" --output-run rough_pretrained
.\.venv\Scripts\python.exe -m pointleg.run train PointLeg-Elevation-v0 --env.scene.num-envs 256 --agent.resume True --agent.load-run rough_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 1000 --agent.logger tensorboard --agent.upload-model False
```

同任务、同几何续训才完整恢复旧优化器与课程。`ElevationOnPolicyRunner.load()` 会检查任务类型、墙高覆盖参数和楼梯布局；旧单段楼梯到当前两段楼梯、平台长度改变等情形必须重新做权重迁移。

2026-10-27：Jump已改为平台间隙/中墙固定跳跃，旧 `--warmup` 停用。只迁移权重时仍使用 `transfer PointLeg-Jump-v0`，会创建带新 `jump_contract` 的零迭代检查点，日志为 `pointleg_jump`：

```powershell
uv run --locked --no-sync pointleg transfer PointLeg-Jump-v0 --checkpoint-file "源模型路径/model_xxx.pt" --output-run platform_pretrained
uv run --locked --no-sync pointleg train PointLeg-Jump-v0 --env.scene.num-envs 256 --agent.resume True --agent.load-run platform_pretrained --agent.load-checkpoint model_0.pt --agent.logger tensorboard --agent.upload-model False
```

从零训练直接 `uv run --locked --no-sync pointleg train PointLeg-Jump-v0 --env.scene.num-envs 256`。两类平台地形共享一份策略，双轴课程初始15%；Jump禁用首回合随机已用时间，确保训练触发时钟从零开始。旧三墙Jump命令语义不同，不能完整续训。当前新模型同契约续训使用 `JumpOnPolicyRunner`，恢复双轴课程/优化器/迭代；修改几何、奖励权重或动作约定后需权重迁移。正式命令与接口见 `docs/JUMP_TRAINING.md`。

## 日志和检查点

2026-10-27 HighStep使用独立实验`pointleg_high_step`，已把双计数清零的平地基础原样复制至`logs/rsl_rl/pointleg_high_step/flat_pretrained/model_0.pt`，SHA256仍为F340F78849B9BFB5E7507D1175EA7E2119FACBE838859AA60648D1621B1F3F1E。无需重新清零原件或强制重置优化器。先运行`pointleg train PointLeg-HighStep-v0 --debug`验证；正式命令见`docs/HIGH_STEP_TRAINING.md`，标准resume加载基础，再训练1000轮观察。只运行了4环境两次调试更新，没有启动正式长训练。

HighStep训练和回放均保持低速向前速度命令语义。旧平地Actor/Critic可严格加载，不等于旧模型已具备14厘米抬脚或快周期步态；后续使用匹配的新任务回放和事件指标选择模型。不要用通用评估器默认0.4米/秒指令当作该任务的域内验收。

2026-10-27梅花桩奖励更新需要重启训练进程：现有105/381/12检查点结构兼容，但推进奖励和抬腿/落点回报目标已改变，续训仍需重新验收。新增曲线不会写入旧日志；本轮只运行单元测试与7环境64步GPU零动作检查，未启动PPO训练或修改检查点。详情见`docs/STEPPING_STONES_REWARDS.md`。

默认训练产物位于 `logs/rsl_rl/<experiment_name>/<时间戳及运行名>/`。保留该运行的参数文件、checkpoint、任务 ID、代码状态和评估报告；不要仅凭迭代数字判断最优策略。

用户已明确“日志就保存默认位置”。不要自行加入 `--log-root` 或改回早期的 `E:/pointleg_logs`；Rough与RoughMixed默认都写在 `logs/rsl_rl/pointleg_rough/` 下，以时间戳和运行名区分任务与修改版本。模型目录可能被用户移动，给续训命令前先核对路径。

代码、奖励或指标改动不会被已经运行的Python训练进程自动采用。先保存可用检查点、停止旧进程，再加载同任务匹配检查点续训；TensorBoard通常不必重启，刷新并选择新日志即可。新指标没有历史数据补写。本次RoughMixed峰值统计和Elevation纠偏均未改变对应45/325/12网络接口，是否可完整恢复仍要核对任务与几何；不要跨任务直接完整恢复。

2026-10-27用户更正：Elevation“小步高净空”版本2只改变奖励，不构成强制迁移理由；只要同任务、同网络接口、同地形/课程结构即可直接resume，保留优化器、迭代和课程，使用当前奖励配置。版本标签不同仅警告，先前强制transfer的检查已撤销。只有真实接口/几何/课程不兼容或用户主动重置训练状态时才transfer。直接续训命令见 `docs/ELEVATION_GAIT_REWARDS.md`；不要再因“只改奖励”要求用户从零开始。

`--agent.max-iterations N` 表示从加载的迭代位置再训练N轮，不是把总迭代数设置为N；共享PPO默认每100轮保存。只要求查看、解释或给命令，不等于授权启动长期训练。

`tools/copy_checkpoint_iteration.py` 是检查点整理工具；复制或改名不等于新训练，也不能证明已经完成对应迭代。当前模型来源快照见 [状态页](07_status_and_gaps.md)。

发生 Warp 缓存权限问题时，可在本次终端设置项目内缓存后重试：

```powershell
$env:WARP_CACHE_PATH = "E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab\artifacts\warp_cache"
```


2026-11-01 08:12实际后台启动用户指定train_4命令：运行目录 `logs/rsl_rl/pointleg_high_step/2026-11-01_08-12-22_train_4`，从固定`2026-11-01_04-10-49_train_3_2/model_10198.pt`直接resume；控制台确认加载路径，已观察迭代10215且已保存model_10200.pt，新增20条逐足支撑轨迹指标已进入TensorBoard。10001次追加更新、间隔100、512环境，预期末次20198。启动器PID 4100，TensorBoard事件记录训练PID30180；进程在后台保留，由用户后续验收，不声称训练完成或轨迹已改善。首次受限后台启动未保留进程、无训练输出；随后在沙箱外隐藏启动成功，无重复训练实例证据。源SHA256 6e38e914384558557b5585e0dfb6a1842af9880d39d52697423dcaaee785931e；checkpoint接口45/325/12和优化器字段已核对、实际resume已执行。25项张量/配置、AST及diff --check通过；启动前未做独立物理测试。MuJoCo现有MULTICCD圆柱/盒接触警告未阻止训练。启动和运行证据见artifacts/train_4_launch.json、train_4_runtime.json、high_step_train_4_preflight.json，stdout/stderr路径见launch元数据。


2026-11-01 12:41实际启动tran_4_2，运行目录`logs/rsl_rl/pointleg_high_step/2026-11-01_12-41-56_tran_4_2`，uv启动器PID48076、事件训练PID28384。控制台确认加载原train_3_2/model_10198.pt，已观察迭代10210并写出model_10200.pt；超15厘米惩罚和逐足路程额度/超高量指标已进入TensorBoard，实际采用新版配置。512环境、10001次追加更新、每100次保存，预期末次20198。后台保留供用户验收；42项张量检查、接口/优化器字段/原模型哈希、CLI、AST与diff检查通过，未做独立回放或物理测试，不声明步态已改善。旧train_4已停并保留最后model_17100.pt。启动/运行/预检证据为artifacts/tran_4_2_launch.json、tran_4_2_runtime.json、tran_4_2_preflight.json。

## 2026-11-08 台阶12000迁移至梅花桩

2026-11-12用户指定 `sim2sim/policy/optical_policy/model_1999.pt`，通过 `pointleg.transfer` 迁移到 `logs/rsl_rl/pointleg_stepping_stones/optical_1999_pretrained/model_0.pt`。源输入45/325扩为105/381，原有权重逐项一致、新增输入列为零，优化器和迭代归零。源SHA256 `2e35db3a3fa5369fdde4f8242fbb88e7e1f92b68b0418f97f4edb2b76a5e02f6`，目标SHA256 `9a52d25040b85eb01b306f7855ef5232283e048083f3c5511e0fa6c2def991e2`。未进行仿真、训练或策略回放。

按用户指定，将 `logs/rsl_rl/pointleg_stairs/2026-11-07_17-13-11_train_13_slow/model_12000.pt` 通过现有 `prepare_transfer("PointLeg-SteppingStones-v0", source, "stairs_train_13_slow_12000_pretrained")` 迁移至 `logs/rsl_rl/pointleg_stepping_stones/stairs_train_13_slow_12000_pretrained/model_0.pt`，同目录保存 `transfer.json`。训练时使用 `--agent.resume True --agent.load-run stairs_train_13_slow_12000_pretrained --agent.load-checkpoint model_0.pt`。优化器、轮次和公共步数从0开始。

迁移前核对了原观测列顺序、缩放、动作执行器与动作缩放。CPU逐张量核对原权重、Actor新增60列与Critic新增56列均置零、105/381/12接口及新检查点严格加载。源 SHA256 为 `28e87009fe6f19ff7c76361443f5130fbb4a6273d50f638cd7bd6f1269f4740b`，生成后再次核对一致。目标 SHA256 为 `554314e5c5ebbe1f1591648cbd5b941f8b62a2c43e74da8705362e64142a42b3`。验证范围限于配置、权重及CPU加载。
