# 高程任务开发完成后的训练入口

2026-10-31当前版本7已接入真实主动抬升、台阶稳定落脚边缘质量、九点机腹净空和子步触地前速度，降低单腿接触被平均稀释。混合和四专项共同采用，45/325/12及固定地图保持，同任务可resume；当前指标、200轮观察命令和验证见[专项说明](ELEVATION_GAIT_REWARDS.md)。没有自动启动训练/仿真，下文版本6及更早为历史。

2026-10-30最新版本6：在15厘米/低速/固定120毫米台阶基础上，加入承重推进质量、长驻足代价和换步速度门控，缩短出生点至入口前1.0±0.3米；前足工作区收紧、后扫区放宽，前后腿慢姿态比较在异高支撑时放宽。接口不变，固定地图model_0直接resume；当前参数、指标与support_push_v6命令见[专项说明](ELEVATION_GAIT_REWARDS.md)，以下版本5等为历史。

2026-10-30现行版本5：15厘米足底净空、0.15～0.25米/秒、60秒回合，速度项改为只有刷新历史前向位置且有支撑才能领取的正奖励，循环按新增距离和平方高度质量支付。固定120毫米台阶及45/325/12保持；从已存在highstep_fixed120_pretrained/model_0直接resume200轮，无需再次transfer，末轮编号199。当前指令与验证见[专项奖励说明](ELEVATION_GAIT_REWARDS.md)，下文旧速度/时限/8厘米参数为历史。

2026-10-29现行Elevation：已撤回版本4新增的腿部净空/大腿摆幅/平地屈腿三项奖励，恢复版本3；上、下台阶固定120毫米、踏面320毫米，取消台阶高度课程，坡面课程保留。旧课程地图检查点需先仅迁移权重至新地图契约，随后正常resume；下文涉及低台阶课程与旧检查点完整resume的说明为历史。当前边界与验证见[专项奖励说明](ELEVATION_GAIT_REWARDS.md)。

2026-10-27 Jump 已切换为固定触发的平台间隙/平台中墙任务，现行说明见 [固定触发平台跳跃](JUMP_TRAINING.md)。下文 Jump 的三墙几何、0.9～1.1 m 出生、330 mm 评估及 `--warmup` 命令均为历史记录，不能用于当前 Jump；Elevation 内容继续适用。

当前已接入 `PointLeg-Elevation-v0` 与 `PointLeg-Jump-v0` 的初版训练环境。Actor / Critic / 动作分别为 45 / 325 / 12 维；模型尚未经过正式训练和通过率验收。

2026-10-27最新Elevation奖励已改为“小步高净空、四足独立有效循环”版本2，允许同一踏面多次有效小步。同任务、同接口和同地形仅改奖励，可以直接resume；奖励版本标记不同不再强制迁移。当前参数、坡面参考及续训命令以 [步态奖励与训练](ELEVATION_GAIT_REWARDS.md) 为准；下文6厘米、2厘米和上一轮推进/踩阶权重是历史实现记录。独立平台Jump的最新任务以 `JUMP_TRAINING.md` 为准。

## 实现内容

- 正式地图为 4×4，共16块，保留上楼梯、下楼梯、上坡、下坡、三堵连续高墙五类地形。五档课程与物理行列分开，部分中间档共用较低难度地块，起步和最高难度保留。
- Elevation 只在楼梯和坡面出生，Jump 只在高墙出生；reset 前显式修正语义类型与出生原点，不依赖零比例禁用地形。
- 楼梯按真实顶面接触记录覆盖；通道外有效支撑失败。Jump 根据物理子步接触、四足过墙历史、全身离开墙体及落地接管判定单次通过。
- 每块楼梯改为同方向两段各三阶，中间水平平台净长在建图时按块随机抽取 1.2～2.0 m；末端保留 1.6 m 恢复平台。单阶仍为 40～120 mm、深 320 mm，通道宽 800 mm；两段累计高度为 240～720 mm。
- 楼梯和斜坡训练每次 reset 将出生 x 坐标随机偏移 ±0.3 m，首段接近距离为 1.5～2.1 m。Viser 回放也启用距离偏移；nominal 评估关闭，robust 保留。Jump 的原有触发距离规则不变。
- 六阶全部覆盖并完整离开第二段后才判定完成；首次踩阶每阶实际 +0.25，总额最多 +1.5。地图生成时保存实际随机布局，奖励和完成检查共用该元数据。
- 初版使用路径势函数、有限子事件、完成 +20、失败 -50 及归一化连续代价。事件在奖励函数中补偿 dt，保留负回报。
- 五类地形分别保存课程上限和成功窗口。新任务支持只迁移网络权重；同任务续训恢复课程及优化器。
- 固定评估按当前任务允许类型测试全部五档；Jump 额外测试 330 mm 目标墙。

## 开始前的检查

所有命令在 `E:/OneDrive/Desktop/RC_WheelLeg-main/05_software/train/rc_mjlab` 执行，复用项目已有环境，不需要重新安装依赖。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run list
.\.venv\Scripts\python.exe tools/check_elevation.py --steps 64
.\.venv\Scripts\python.exe -m unittest discover -v
```

如果 GPU 缓存目录受写权限限制，可在当前终端将缓存放在项目工件目录中，再执行检查或训练：

```powershell
$env:WARP_CACHE_PATH = "E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab\artifacts\warp_cache"
```

## 只看地形的 Viser 入口

不需要模型，也不会启动训练；在浏览器打开终端打印的地址，暂停后缩小视角可查看整张 4×4 地图。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run play PointLeg-Elevation-v0 --agent zero --num-envs 1 --viewer viser
```

将任务名换成 `PointLeg-Jump-v0` 可查看高墙前的出生位置。两项任务的指令面板只读，显示前向速度范围，不创建横向或转向滑块，也不允许手柄覆盖前向约束。

下面的诊断会实际创建并关闭两个任务的 Viser 场景和面板，检查完成后退出，不持续占用浏览器预览服务：

```powershell
.\.venv\Scripts\python.exe tools/check_elevation.py --steps 16 --viser-gui
```

## 从 Rough 迁移到楼梯和斜坡

将示例中的源检查点路径替换为经过评估的点足 Rough 模型。首次迁移只加载 Actor 和 Critic；优化器、迭代计数、公共步数和课程状态重新初始化。输出目录已经存在时会拒绝覆盖。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run transfer PointLeg-Elevation-v0 --checkpoint-file "源模型路径\model_xxx.pt" --output-run rough_pretrained
.\.venv\Scripts\python.exe -m pointleg.run train PointLeg-Elevation-v0 --env.scene.num-envs 256 --agent.resume True --agent.load-run rough_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 1000 --agent.logger tensorboard --agent.upload-model False
```

若只检查训练能否更新与保存，使用两次迭代的小规模入口：

```powershell
.\.venv\Scripts\python.exe -m pointleg.run train PointLeg-Elevation-v0 --debug --env.scene.num-envs 4
```

## 独立高墙 Jump 与前置练习

普通 Jump 仍使用正式的 110～340 mm 五档墙高。可先使用 `--warmup` 练习平地定向跳跃：地图规格不变，但不生成实体墙，沿虚拟墙平面检查一次离地、跨越和落地。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run transfer PointLeg-Jump-v0 --warmup --checkpoint-file "源模型路径\model_xxx.pt" --output-run rough_pretrained
.\.venv\Scripts\python.exe -m pointleg.run train PointLeg-Jump-v0 --warmup --env.scene.num-envs 256 --agent.resume True --agent.load-run rough_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 1000 --agent.logger tensorboard --agent.upload-model False
```

前置练习日志位于 `logs/rsl_rl/pointleg_jump_warmup`。挑选通过验证的模型，使用权重迁移进入正式高墙任务，重置课程；不能直接把前置练习当作正式训练续跑。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run transfer PointLeg-Jump-v0 --checkpoint-file "前置练习模型路径\model_xxx.pt" --output-run jump_pretrained
.\.venv\Scripts\python.exe -m pointleg.run train PointLeg-Jump-v0 --env.scene.num-envs 256 --agent.resume True --agent.load-run jump_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 1000 --agent.logger tensorboard --agent.upload-model False
```

直接从 Rough 开始正式高墙任务也使用 `transfer PointLeg-Jump-v0`，不要传 `--warmup`。

## 同任务续训、回放与固定评估

同任务续训时，将 `--agent.load-run` 换为该任务的实际日志目录名，`--agent.load-checkpoint` 换为选定检查点。模型目录保留实验名隔离。前置练习、正式高墙和行走任务不混用完整恢复；几何配置改变时使用权重迁移。

两段楼梯布局使用版本 2 的几何契约。原单段三阶 Elevation 检查点仍可回放或迁移网络权重，但不能完整恢复旧优化器与课程到新布局；通过 `transfer PointLeg-Elevation-v0` 创建新运行目录。中间平台范围或末端平台长度改变也需要权重迁移。固定楼梯评估当前针对两段训练布局，不代替比赛单段三阶的专项验收。

```powershell
.\.venv\Scripts\python.exe -m pointleg.run play PointLeg-Elevation-v0 --checkpoint-file "模型路径\model_xxx.pt" --num-envs 4 --viewer native
.\.venv\Scripts\python.exe -m pointleg.run evaluate PointLeg-Elevation-v0 --checkpoint-file "模型路径\model_xxx.pt" --profile nominal --episodes-per-case 100 --output artifacts/elevation_nominal.json
.\.venv\Scripts\python.exe -m pointleg.run evaluate PointLeg-Jump-v0 --checkpoint-file "模型路径\model_xxx.pt" --profile nominal --episodes-per-case 100 --wall-index 0 --output artifacts/jump_nominal.json
```

将 profile 换成 `robust` 保留物理和入口随机化。Jump 用 `--wall-index 0`、`1`、`2` 分别检查三堵墙的单次跳跃，用 `--forward-speed` 扫描经验证的入口指令范围。nominal / robust 的每用例目标为 90% / 80%，报告是否达到目标，不能把命令成功执行当作策略通过。

固定评估目前是独立技能评估，不声称完成 ROS 多策略衔接或连续三墙的整条路线验证。

## 需要标定或后续集成的边界

- `ElevationSettings` 中的 0.2～0.5 m/s 指令、0.9～1.1 m 触发根节点距离、0～0.15 m/s 初始前向速度、0.01 s 最短离地检测以及姿态、接管容差是初版仿真设置，未确认实机可行。1 N 只用作排除数值微接触的检测阈值，不是硬件允许载荷；动作变化代价使用的 10/s 也是待标定归一化参考值。
- 当前入口是默认关节姿态与小范围初始状态扰动。真实行走策略入口状态、完整动作滤波与延迟历史的采集及注入尚需结合实际部署管线集成，不能认为已验证从任意步态相位切换。
- 未取得实际标定参数的功率、机身离地范围、多余法向速度和超量落地载荷细化项未启用。实际力矩限幅、关节硬限位、姿态失控和机身接地失败仍有效。
- 实机 ROS 接口及模型切换状态机尚未接入本训练工程。现有 45 维接口可保持，但导航仍必须负责触发位置，跳跃模型不能自动发现下一堵墙。
- 需要进一步测试成功轨迹、触发容差、三墙全路线、物理随机化及硬件负载；短步检查和 PPO 两次更新不证明已学会越障。

## 本次实现验证记录

2026-10-25 已完成以下工程检查，未启动正式长时间训练：

- 全量自动化测试 185 项通过，覆盖双段几何、随机平台元数据、±0.3 m 接近距离、六阶完成和奖励上限、接触判定、课程恢复、权重迁移及前向专用 Viser 面板等。
- 两项任务均通过真实 GPU 仿真场景检查：5×5 地形、Actor 45 / Critic 325 / 动作 12 维、64 个控制步输入输出有限，并检测到物理支撑。
- 两项任务均已实际创建 Viser 场景与指令面板，验证初始化和场景同步后正常退出；修复了零横向、零转向范围与通用手柄滑块下限冲突的问题。
- 楼梯坡面、正式 Jump、平地 Jump 前置练习均完成小规模 PPO 两次更新并保存检查点；迁移后的前置练习检查点已验证可完整续训。
- 两段三阶及随机接近距离版本的 Elevation 另行完成 4 环境、2 次 PPO 更新并保存检查点；不代表策略通过越障验收。
- 固定评估入口已跑通 20 个楼梯坡面用例和 6 个 Jump 用例（包含 330 mm 目标墙）。此项使用每用例一次、0.2 s 诊断时限，只验证报告生成，不属于通过率验收；诊断报告未达到通过目标。

几何、奖励和后续验收的完整依据见 [总体设计](ELEVATION_TRAINING_DESIGN.md) 与 [奖励设计](ELEVATION_REWARD_DESIGN.md)。

## 直行侧漂与持续髋外展偏置

2026-10-27，`PointLeg-Elevation-v0` 的侧向速度、通道居中和航向代价从权重 -2、-2、-1 小幅提高为 -3、-3、-1.25。侧速容许量仍为0.015米/秒，尺度仍为0.10米/秒；横向位置容许量从0.015降至0.010米，尺度从0.12降至0.10米；航向容许角仍为2度，尺度仍为10度。代价始终左右对称，不通过只罚左侧制造反向偏置。侧向速度使用世界坐标Y轴，因此机身整体转歪后沿自身正前方行走仍会被识别为偏离任务直线。

保留原髋外展瞬时12度容许区间，另以0.5秒时间常数平滑四个髋外展关节相对默认角的有符号偏差。经过0.5秒启动宽限后，持续偏差超过4度按8度尺度平方扣分，权重为 -0.5；代价按四足平均，单足归一化分量封顶4。短暂或交替侧摆不等于持续偏置，不强制四腿瞬时镜像，也不限制正常的髋俯仰、膝关节抬腿动作。

TensorBoard 增加 `Episode_Metrics/elevation_lateral_velocity`（有符号世界侧速）、`elevation_lateral_speed_abs`（绝对侧速）、`elevation_center_error_abs`（通道中心偏差，米）及 `elevation_heading_error_deg`（航向误差，度）。四足持续偏置通过 `Metrics/hip_abduction_bias_abs_fl_deg`、`fr_deg`、`rl_deg`、`rr_deg` 分别记录，另有 `Metrics/hip_abduction_bias_max_deg`。新奖励和新曲线需重启训练、加载检查点后生效；不会让旧策略回放时立即恢复正常。

该调整仅用于 Elevation 行走任务；Jump、Flat、Rough 和 RoughMixed 的奖励配置不变。

## 摆腿低净空与台阶立面净空

2026-10-27，经用户确认接入 `swing_clearance` 代价，权重 -0.4；实现位于 `mdp/elevation_clearance.py`。此前 Elevation 只有离地时长奖励和踩阶事件，没有直接足端高度目标，课程升级不会自动增加抬腿高度奖励。

- 使用已有世界坐标接触点、法向和力判断承重；竖直立面接触不算落足。足球底按球形几何和足端姿态修正，不把旋转后的足底标记当作真实最低点。
- 平地及坡面按一次真实前向摆腿期间相对当前局部地形的峰值净空检查，目标为6厘米。先有承重、离地至少0.02秒、同足向前至少3厘米、再次承重且至少两足支撑才结算；初始落下、原地悬脚和接触抖动不直接结算。
- 低于目标时按 `max(1 - peak/0.06, 0)^2` 扣分，每足每次最多实际 -0.04。事件输出补偿控制周期，重复接地不重复扣。起脚与落脚必经的低高度不作为平地连续代价。
- 上楼梯在立面前10厘米至足球中心越过立面2.5厘米的局部窗口，额外检查摆动足底是否超过该立面实际顶面2厘米；按靠近立面的程度平滑加重。只对已起脚、非承重、合法通道内且至少两足支撑的足生效，不把支撑脚压在地面当成低净空错误，也不把下楼梯强制当成向上越阶。
- 达标后不再扣本项；超高抬腿和长时间腾空没有额外正奖励。连续立面代价每环境每秒最多 -0.4，单足最坏为 -0.1/秒。静止、拖地等仍需由推进、支撑滑移、换脚和停滞失败等已有项约束，不能仅凭本项证明没有伪策略。

目标6厘米、立面余量2厘米、8牛承重阈值和事件权重都是仿真初值，尚未通过正式续训验证。平地峰值达到6厘米不证明能越过120毫米台阶；完整楼梯仍需六阶真实覆盖与出口稳定。真实地形只用于训练反馈，不增加射线、Actor/Critic维度或部署输入。

新增曲线：`Metrics/swing_peak_height`、四足 `_fl/_fr/_rl/_rr` 和 `_relative_body`；只在符合条件的完成摆腿事件写入样本，单位为米。回合指标 `Episode_Metrics/elevation_swing_peak_height` 是本回合完成摆腿均值，无样本回合为0；`elevation_near_riser_feet`、`elevation_riser_deficit`、`elevation_low_swing_events_per_s`、`elevation_completed_swings_per_s` 用于检查触发和不足情况。相对机身曲线用于区分真正收腿与整机升高，主曲线不是跨越立面时的最小净空。

修改配置不会改变已启动的训练进程，也不会让旧检查点在回放时自动抬高足端。需要重启并续训；模型输入仍为45/325/12，历史任务或几何契约不匹配时按 `transfer` 流程迁移。

本轮验证：246项点足测试通过，Elevation 64控制步真实GPU检查通过。旧1000轮 `2026-10-26_23-25-14/model_999.pt` 的5环境160控制步短回放记录到160次有效完成摆腿，峰值净空均值约1.42厘米，新增代价与日志可触发；这是混合地形短诊断，不是正式平地统计或新模型通过率，未进行长时间训练。
