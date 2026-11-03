# 高抬腿、小步幅基础步态

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


## 2026-11-01左前姿态、侧漂与髋外展加强（当前）

2026-11-01最新HighStep共24项：新增lf_rr_pose_tracking +.25，在右后足真实有效摆动且主动抬升≥3厘米、机身前进>.05米/秒、至少两足承重时，左前跟踪右后当前髋摆/膝角及膝相对髋的机身竖直位置；左前自身不必已进入摆动，避免旧对角项的双足摆动门槛不给纠正信号。关节误差尺度20°/25°，膝高4厘米，两类误差各半取高斯正分；后腿仅作参考，策略仍共享。侧漂权重−1→−2，死区.015→.010米/秒，左右对称；按用户追加要求髋外展−.9→−3.6（4倍），零目标/零死区/.25弧度尺度保持。其余项、45/325/12、PPO/PD/地形及其他任务不改。静态模型审计未发现缺执行器、错绑或左前髋摆驱动失效：XML12→删除0→重建12，全部动作槽/关节对应，0.1弧度目标增量仅对应DOF产生5N·m；四髋相同静态角增量使膝上升约14.1～14.3毫米。报告artifacts/high_step_lf_actuator_audit_2026-11-01.json；未积分时间，不代表动态碰撞/带载饱和或策略已验收。一个合成张量检查通过：左前仍支撑时可获跟踪反馈、匹配高于异常姿态、失败/非有限/静止0及左右侧漂同罚。未仿真或训练；新进程同任务直接resume，当前截图未绑定到指定模型文件。

左前专用姿态项比较实际关节姿态及几何位置，不只比较共同摆动起点后的增量。右后真实抬脚时，即使左前仍保持落地，也会按误差获得有差别的连续反馈；不把“左前已经正确抬脚”作为学习这项奖励的前提。实际评分为exp(−0.5×关节归一化均方误差−0.5×膝相对髋高度归一化平方误差)。这是软跟踪，不是执行器硬绑定，也不能冻结右后腿动作。

指标Metrics/high_step_lf_rr_hip_pose_error_deg、high_step_lf_rr_knee_pose_error_deg、high_step_lf_rr_knee_height_error_m仅记录有效教师摆动样本。角度和膝高误差应下降，同时检查足高、拖腿、前足位置及侧速；强髋外展代价需要续训确认不会导致停步。已有进程和已有PT不会自动采用新配置。

模型审计使用当前ROBOT_XML解析的mjcf/pointleg.xml和当前HighStep实体构建流程，审计覆盖XML删除/重建、12路目标解析顺序、传动关节、50/1.5增益与±25N·m限幅；没有修改XML、执行器或电机参数。左前髋摆为零基动作槽1，膝为槽2。静态检查能排查缺通道、错绑和静态不能驱动，不能完全排除动态碰撞、真实负载和某个已经运行进程另设POINTLEG_XML的差异。

## 2026-11-01承重后扫与侧漂

2026-11-01 HighStep承重后扫与侧漂调整：保留已有command_orthogonal_drift，权重−.5→−1.0、1.5厘米/秒死区与10厘米/秒尺度不变，对左右漂移等价，不加入只针对左侧的偏置；横向速度为机身坐标，不能替代绝对航向/全局路径控制。新增stance_backstroke +.04，每足连续8N承重≥.16秒后离地时按有效后扫/7厘米线性发放，单足每次最多+.04；有效后扫取落地到离地间机身实际前进距离与身体相对该支撑足前进距离中的较小者，负值置0。按真实接触点累计滑移乘max(1−滑移/2.5厘米,0)，无机身推进的脚底拖滑不得分；超过7厘米饱和，事件原始值除dt，由奖励管理器积分一次。现有高抬腿、对角/髋摆、姿态、滑移惩罚和支撑超时保持；总奖励23项，45/325/12、其他任务/PPO/PD不变。指标Episode_Reward/stance_backstroke、Metrics/high_step_backstroke_{fl,fr,rl,rr}_m、Metrics/high_step_stance_time_{leg}_s、Metrics/high_step_contact_slip_mean_m_s和Metrics/high_step_lateral_speed_abs_m_s，应同时观察四足抬升及回放，防止过长支撑、停步、足底滑移或高度退化。按用户要求只执行一个合成轨迹检查，23项配置、真实身体推进奖励、原地拖脚无收益、单事件、左右同等漂移通过；未仿真/Viser/训练，不能据此断言实机发力或爬阶通过。用户未指定当前较好模型路径，继续训练时固定选择其HighStep检查点直接resume，新进程才启用。

这里的后扫是支撑足相对于运动机身向后扫过的距离；足端在世界坐标中近似固定也可以产生有效后扫。奖励鼓励这种支撑推进，实际是否能在台阶上产生足够的力仍需专项训练与验收。当前旧训练进程不会读取新配置。

## 2026-11-01对角腿以后腿为参考

2026-11-01 HighStep加入diagonal_rear_tracking +.2，22项：左前跟右后、右前跟左后；起脚前真实承重且对角两足同时有效摆动、至少两足承重、前进指令及实际机身前向速度>.05米/秒时，以两足进入共同有效摆动时的各自髋俯仰/膝关节角为起点，比较角位移，不强制绝对关节角。后腿相对起点的髋或膝位移至少5°才发放，20°运动强度饱和；前后角位移误差按15°尺度高斯计分，两对得分相加除2，按控制周期正常积分。实际后腿位移同时参与得分，不能保证后腿策略参数被固定，也不能由后腿单独决定策略梯度；前后足不同步时该项无收益而非强制锁相。现有足高、髋外展、pitch、前足工作空间及髋摆项保持，45/325/12、PPO/其他任务不变。Metrics/high_step_diagonal_error_fl_rr_deg、_fr_rl_deg记录两关节角位移差的均方根；Metrics/high_step_diagonal_rear_motion_fl_rr_deg、_fr_rl_deg记录对应后腿相对起点最大关节位移，仅满足门槛的样本记录。只执行单个配置与张量轨迹检查，22项/权重.2，左前右后匹配高于相反、后腿未动或静止0；未仿真、Viser、训练或完整回归，右前左后由相同映射实现但未做独立轨迹测试。新进程从用户选定同任务检查点直接resume，实际效果需短训观察。

该项是额外的正奖励，不要求前后腿静止时绝对角度一样；同一策略控制四条腿，奖励不能冻结后腿。奖励值和匹配误差要结合足高、前足位置、侧漂及回放一起看。

## 2026-11-01前足工作空间、髋外展与俯仰修正

2026-11-01最新HighStep针对前足过度前伸、髋外展和前倾调整：新增front_foot_workspace −.7，分别取左右前足球心相对同侧髋俯仰连杆的机身坐标x，以.6秒EMA、1秒启动宽限追踪，平均前伸超过4厘米时按8厘米尺度平方计罚，每足封顶4后平均；仅前进时生效，不锁定瞬时摆动姿态。hip_abduction_excess −.3→−.9（默认0角、零死区/.25弧度尺度不变）；high_step_pitch −.5→−1、死区3°→2°/尺度8°不变，正负俯仰同等处理；hip_swing_coordination +.04→+.06，原有效步/稳定落地/两足支撑/身体前进资格、20°髋幅及3厘米相对髋膝上下位移饱和目标不变。主动抬升+.35/.08/15厘米和其余未提及的奖励保持，总数21，45/325/12、PPO/PD/其他任务不改。指标Metrics/high_step_front_foot_mean_x_fl_m和_fr_m，可与high_step_pitch_deg、hip_abduction_deviation_mean、分足髋摆及膝竖直幅度、分足主动抬升共同检查；EMA均值不是每步足端位置。单个配置/张量检查通过，验证前足平均前伸受罚、零前伸无罚、同周期缓存及部分重置；未仿真、回放或训练，实际改善和足高需短训验证。用户未指定截图对应检查点，续训应固定用户选定的HighStep检查点直接resume，新进程才应用奖励。

髋外展与俯仰均为软约束，不能保证角度严格为零。前足项针对每条腿各自的时间平均前伸，不限制后足或前足瞬时前后迈步；不能凭单张图判定奖励权重是否恰当。新进程短训后应对照前足平均位置、髋外展、机身pitch、髋摆和四足抬升，并回放是否出现停步或拖腿。

## 2026-10-31髋外展、机身水平与髋摆协调（历史）

2026-10-31最新HighStep应用户要求在17项上增加3项，共20项：hip_abduction_excess −.3，以默认0弧度为中心、零死区、.25弧度尺度平方，四腿分别计分；high_step_pitch −.5，正负俯仰超过3°后按8°尺度平方惩罚；hip_swing_coordination +.04，对有效完成并稳定落地的摆动奖励髋俯仰角幅及膝相对髋在机身坐标竖直幅度，20°/3厘米各饱和，两者各占一半。只在8N支撑且摆动期间至少两足承重、原主动高度有效落地条件、实际身体前进≥1厘米时结算一次；不单独给原地抖关节奖励，正常动作仍允许偏离0，不是硬限位。原17项（足高+.35/.08、15厘米目标）保持，45/325/12及其他任务配置未改。单个无物理张量轨迹检查通过：20项配置、零死区和水平参数、髋20°/膝3厘米饱和并只结算一次；未仿真或训练。指标为Metrics/hip_abduction_deviation_mean、Metrics/high_step_pitch_deg、Metrics/high_step_swing_hip_range_{leg}_deg、Metrics/high_step_swing_knee_vertical_range_{leg}_m。

髋外展是软惩罚而不是硬锁死，零角度以当前机器人默认髋外展角0为参考。髋摆与膝部竖直幅度从真实起脚后的摆动轨迹记录，稳定落地后一次结算；不奖励绝对髋角或固定前后腿构型。机身pitch来自投影重力，有3°容许范围，防止将正常轻微起伏全部视作倾斜。上述权重是训练初值，无法单凭静态图片或张量检查保证姿态改善。

已有训练进程不会自动加载新配置。HighStep同任务、同接口的检查点可用resume在新进程续训；本轮用户未指定截图对应的检查点，不将最新文件自动假定为起点。

## 2026-10-31基础奖励重建（历史，髋姿态三项见上）

用户确认从 `2026-10-30_21-53-51/model_6200.pt` 开始，先保住前后腿姿态、稳定向前及持续换步，再接入独立的主动抬升奖励。当前HighStep显式白名单共17项，旧38项配置及本页后续15厘米/0.6秒综合评分为历史；新进程才会采用此配置，旧模型文件和已有训练进程不变。

| 项目 | 权重 | 当前口径 |
|---|---:|---|
| track_lin_vel | +2.0 | `clamp(vx/cmd,0,1)*exp(-((vx-cmd)/.10)^2)`，零速/后退/静止指令/终止/非有限不领；不乘新增前沿、换步新鲜度、直立、收腿或支撑质量 |
| command_orthogonal_drift | −0.5 | 侧速容许.015米/秒，尺度.10 |
| high_step_yaw_rate | −0.5 | 偏航角速度容许2°/秒，尺度20°/秒 |
| upright | +1.0 | 原直立评分 |
| body_ang_vel | −0.1 | 原横滚/俯仰角速度代价 |
| high_step_mean_leg_symmetry | −0.2 | 原1秒慢姿态、1.2秒宽限；左右4°，前后髋/膝8°/12°，前后系数.75；比较相对默认角偏置，不强制瞬时同角 |
| feet_slip | −0.2 | 真实接触点切向速度，8N/宽限.04秒/.02容许/.10尺度，均值与最差各半 |
| support_deficit | −2.0 | 统一8N承重，少于两足的不足数 |
| support_timing | −0.1 | 合并短承重、长驻足和长悬空；最短.16秒，驻足>.85秒/尺度.25，悬空>.36秒/尺度.18；启动.60秒宽限 |
| active_lift_event | +0.35 | 有效摆动稳定落地后，按 `min(主动峰/.15,1)^2` 一次结算 |
| active_lift_progress | +0.08 | 摆动期只对主动峰新增高度连续支付；达到15厘米累计原始分为1 |
| stand_still | −0.2 | 原静止指令默认姿态代价 |
| joint_torques | −.0002 | 原关节力矩平方 |
| action_rate | −.005 | 相邻动作变化 |
| joint_pos_limits | −10 | 原软限位代价 |
| body_collision | −1 | 原自碰撞代价 |
| is_terminated | −200 | 原非正常终止代价 |

时序分数为：持续超时的四足平均与最差各半，加上离地事件的 `.10秒/dt * mean((1-实际承重/.16)^2正部)`；短承重每次结束仅一个事件，事件分数由框架dt积分。取消.24秒软目标和.35秒/7厘米质量目标，不要求固定0.6秒周期。滑移仍使用球心线速度加角速度叉接触点偏移后投影切向，正常滚动不罚。原安全项权重保留，未按此前建议表统一降低。

主动抬升使用足底相对起脚足底的上升峰值，不使用世界地面高度；有效摆动要求起脚前真实承重≥.16秒、起脚后至少两足承重、离地.10～.36秒、足前移≥2.5厘米，并在落地后再次承重≥.16秒。静止、原地点脚、短支撑、未稳定落地、全足腾空、失败和非有限数据不领取。高度分不与周期、后扫、收腿或姿态质量相乘，超过15厘米不追加收益。

周期对称、对角步态、支撑占比、固定机身高度、髋外展/偏置、足对居中、工作区、收腿、额外俯仰/弹跳或接触力奖励仍未启用。站立样本仍10%，仅保留stand_still专项；速度保持.15～.25米/秒。复杂历史前沿防重复机制也不再用于速度收益，不能宣称此简单速度项能独立排除所有往返或拖行解。

`mdp/high_step_basic.py`独立跟踪承重和摆动，足高、步长、周期和后扫均记录。仍记录 `Metrics/high_step_peak_height_m`、`high_step_active_lift_m`、`high_step_active_lift_{fl/fr/rl/rr}_m`、`high_step_active_lift_score`、周期、承重、后扫、真实滑速及累计滑移。前两类高度分别表示相对地面的峰值和主动起脚抬升，不是关节摆幅；事件均值不能解释为每次摆动保证。接触力150N超限只移到metrics，保留逐足子步力峰和超限比例。旧push_quality、renewal及周期高度评分指标不再由当前配置生成。

评估读取basic状态，不再访问已移除的循环/对角奖励实例，也不再强制0.6秒周期。基础步态检查要求每足至少5次有效前移循环（此前承重≥.16秒、离地≤.36秒、足前移≥2.5厘米）；主动抬升作为专项指标报告，首次短训不把15厘米设为通过硬阈值。通用评估中的速度误差、稳定性和推进判定继续生效。这不是楼梯通过验收。

从6200直接resume，先追加100次更新，保存间隔50：

```powershell
cd "E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab"
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-30_21-53-51 --agent.load-checkpoint model_6200.pt --agent.max-iterations 100 --agent.save-interval 50 --agent.run-name active_lift_6200 --agent.logger tensorboard --agent.upload-model False
```

输出新时间戳目录 `pointleg_high_step/<时间>_active_lift_6200`；迭代范围6200～6299，比较6250和末次6299。先回放持续换步、前后慢姿态偏差、承重时间、真实滑移和四足主动抬升；若明显停步、拖腿或姿态恶化，不原样长训。

验证：15项基础配置的101项历史张量检查发生在本次主动抬升接入之前。当前只执行了两项针对性张量检查：8、12、15厘米主动抬升的事件分分别为比例平方，连续新增抬升分别累计为高度比；原地点脚、落地未稳定和起脚前短承重不领事件奖励。另确认当前配置17项，6200检查点仍为45/325/12。未重跑完整HighStep回归。

真实6200严格CPU resume及临时保存后Actor/Critic/优化器/迭代6200/公共步数148968完全保持，源SHA256 `4a30656c54836703bc7991c9b0ffa7f49a71bb7b9e69cff54f9d18e8d3e14b24`前后不变；完整100轮CLI及精确源路径解析通过。报告 `artifacts/high_step_basic_validation.json`。Actor/Critic/动作45/325/12，PPO/PD/地形不改；Elevation配置/核心/净空、Jump配置和PPO配置五文件前后哈希一致。未构造物理场景、启动Viser或PPO/训练，未证明实际步态改善。

## 2026-10-31接触点滑移、短支撑与距离抬升收益（历史）

用户批准按当前7199继续修正HighStep。15厘米目标、0.60±0.12秒周期、速度0.15～0.25米/秒和主循环权重0.38保持。本段覆盖下文按球心净位移计滑移、混合高度评分、每次摆动抬升奖金及周期占比0.30等历史配置；本轮仅改HighStep，Elevation版本7、Jump、PPO/PD和其他约束保持。

| 项目 | 当前实现 |
|---|---|
| 真实接触滑移 | 使用球心线速度加`角速度×(接触点−球心)`，再去掉实际接触法向分量；传感器增加pos字段，接触通道按腿名排序。球足正常滚动不因球心移动直接判为滑动。 |
| `feet_slip` | 替换为HighStep专用项，权重−0.2。实际顶面承重≥8N且连续超过0.04秒后，`clamp(max(切向滑速−.02,0)/.10,0,1)^2`；有效承重足平均与最差足各半，无有效承重时0。此归一化代价不同于原始速度平方求和，不能直接按权重绝对值判断强弱。 |
| 支撑质量 | 原2.5厘米球心最大偏移改为过触地宽限后的累计真实切向滑移距离，保留.35秒/7厘米目标与`.25+.75×质量`循环系数；滑出滑回不会清除累计，滚动不因球心净位移直接降质。身体实际推进/相对足端后扫取小值保持。 |
| `short_stance` | 软目标.16→.24秒，权重−.05→−.10；按8N真实承重结束事件结算平方不足量。主循环最低承重门槛仍.16秒，避免切断原策略恢复反馈。 |
| 周期与高度 | 周期评分占比.30→.50，0.60秒满质量评分保持；高度改为`min(峰值/.15,1)^2`，8/12/15厘米约.284/.64/1，超过15厘米仍沿用3厘米尺度的过高衰减。 |
| 辅助抬升 | 原每次摆动封顶奖金改为`0.02×新增前进距离/目标步距×Σ(有效摆动足高度分×相对收腿质量)`。目标步距=指令速度×.6秒，限制4～15厘米；返回值除dt、框架按dt积分。 |

辅助抬升的前进距离刷新历史最远位置，单周期不超过指令速度×dt；所有控制周期都更新前沿，不把未满足摆动条件的旧路程攒到以后领取。同一段新路程只支付一次，往返到旧位置不再支付。有效摆动需要此前真实承重、至少两足支撑、足端前移≥2.5厘米、机身前移≥1厘米、离地≤.36秒，保留相对机身收腿要求。原地悬停、额外原地点脚、全足腾空、纯机身弹跳和失败当步不领该辅助项；身体向前进入新位置时的有效高位摆动可持续按距离支付。本轮不宣称所有奖励都已完全消除频率影响。

本轮滑移、支撑、循环和收腿相关承重阈值均配置8N；偏航、前后慢姿态、150N接触力、工作区、10%站立指令、网络和45/325/12保持。内部新增状态按env_ids重置，同一步读取不重复积分滑移或复用前向距离；非有限接触/速度数据过滤。改变奖励需要新训练进程，旧PT及正在运行的进程不会自动改善。

指标含义：

- `Metrics/high_step_contact_slip_mean_m_s`及`high_step_contact_slip_{fl/fr/rl/rr}_m_s`：有效承重、过宽限后的真实接触点切向滑速。
- `Metrics/high_step_contact_slip_distance_{leg}_m`：承重结束时的累计真实滑移距离。
- `Metrics/high_step_settled_support_feet`：当前有效滑移统计承重足数，避免将没有有效支撑时的0滑速误作优良步态。
- `Metrics/slip_velocity_mean`保留旧足底参考点世界水平速度口径作对照，不能将原12.5厘米/秒直接等同于新真实滑速。
- 足高、承重时长、后扫及push_quality仍使用原有分足标签；高度目标仍看四足峰值和真实换步，不能只看总回报。

固定从用户指定7199直接resume，先追加300次更新，不使用通配符：

```powershell
cd "E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab"
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-30_21-53-51 --agent.load-checkpoint model_7199.pt --agent.max-iterations 300 --agent.run-name contact_slip_cadence --agent.logger tensorboard --agent.upload-model False
```

输出新时间戳目录`pointleg_high_step/<时间>_contact_slip_cadence`；原模型不覆盖。源实际迭代7199/common_step_counter172944，本次更新7199～7498，周期文件含7200/7300/7400，建议对照7300、7400和末次7498。四足峰值应向至少12厘米提高，左前差距缩小，承重/后扫增长且新真实滑速下降；不新增停步/拖腿。若只见总回报上涨而这些指标继续恶化，不原样扩展到几千轮。参数为短训实验初值，不保证实际步态改善。

实际验证：84项HighStep/walking_audit定向检查通过（6.464秒），含新增19项接触点滑移/距离收益检查。覆盖球心实际移动的正常滚动、真实拖滑、弱接触/宽限/法向过滤、通道反序、滑出滑回、缓存/部分重置/非有限、短支撑软目标、平方高度/过高衰减、距离/控制周期预算、旧前沿往返/额外摆动/超速/无效阶段不攒信用、弹跳/腾空/失败和其他任务接口。真实model_7199 CPU严格resume及临时保存后Actor/Critic/优化器、迭代和公共计数完全保持；源SHA256 `3170689f7ea12bde64f8c24144edad3e7a99be92b4cb3b000316f3bd2486babb`前后不变。完整CLI解析及精确源路径核对通过；未启动物理仿真、Viser、PPO或训练。Elevation配置/核心及Jump/PPO配置五文件修改前后哈希一致。

## 2026-10-30平地纠正（历史）

2026-10-30当前阶段：用户选择先在`PointLeg-HighStep-v0`平地纠正前腿过度前伸、短支撑和驻足退化。采用15厘米抬脚目标、0.60秒同足完整周期和0.15～0.25米/秒速度。本段覆盖下文历史17厘米目标及旧工作区；本轮不改Elevation版本6，也没有实现此前建议的Elevation仅平地开关。

| 调整 | 当前实现 |
|---|---|
| 高度与周期 | 目标17→15厘米，评分宽度3厘米；周期0.60±0.12秒、循环权重+0.38保持 |
| 承重后扫 | 连续至少8N承重，最短0.16秒，质量目标0.35秒/7厘米有效后扫；允许2.5厘米水平足位漂移，越大质量越低 |
| 循环质量 | 原落足循环评分乘`0.25+0.75×承重后扫质量`，保留低质量恢复反馈，不另加支撑奖金 |
| 持续驻足 | 连续承重超过0.85秒开始扣分，0.25秒尺度封顶，权重−0.3；原相位项关闭重复支撑惩罚，只保留过长离地代价 |
| 速度收益 | 保留原速度跟踪权重/宽度，前进时乘新前沿距离系数、正向速度比例及真实换步新鲜度；至少两足承重，静止、旧位置往返、腾空和失败当步无前进速度收益 |
| 换步新鲜度 | 每足真实落足循环峰值至少3厘米、相对收腿质量至少40%才刷新；0.6秒内为1，0.6～1.2秒线性衰减，四足平均与最差各半；不会屏蔽恢复循环本身 |
| 前后工作区 | 支撑工作区和高位摆动收腿中心改为髋后2厘米、余量14厘米，即髋前12厘米到髋后16厘米的软区间；高位收腿权重−0.2保持 |
| 前后慢姿态 | 原左右均值对称−0.2保留，增加前后均值比较系数0.75（有效权重−0.15）；1秒EMA、1.2秒宽限，髋/膝容许8°/12°，尺度15°/20° |

后扫比较身体真实向前推进与身体相对支撑足的推进，取二者较小值；足端世界坐标稳定而身体前进属于有效支撑，不是要求脚在地面向后滑。脚向后滑不能放大推进信用，离地时冻结此前支撑质量。没有独立的力越大越好或关节摆幅越大越好奖励。慢姿态比较相对默认角的时间均值，允许四足共同屈腿和正常交替，不强制前后腿瞬时同角或同相。

新指标：`Metrics/high_step_stance_time_{fl/fr/rl/rr}_s`、`high_step_backstroke_{leg}_m`、`high_step_push_quality_{leg}`和`high_step_gait_renewal`；前后慢姿态偏差沿用原有`high_step_front_rear_hip_bias_deg`/`high_step_front_rear_knee_bias_deg`。前者按离地事件记录，不能当作全回合峰值受力测量。上述0.35秒和7厘米是质量饱和目标，不是强制所有步等长。

偏航、侧漂、150N接触力约束及10%静止指令样本保持；静止指令仍保留站立速度跟踪行为，前进指令下静止不能领取前进速度收益。Actor/Critic/动作仍45/325/12，网络、PPO超参数、PD和地形不改。源模型本来就是HighStep，本阶段直接resume，不执行transfer，也不重置为model_0。

在项目训练目录执行，先追加200轮：

```powershell
cd "E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab"
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-28_23-25-35_yaw_mild_soft_contact --agent.load-checkpoint model_15498.pt --agent.max-iterations 200 --agent.run-name flat_front_correction --agent.logger tensorboard --agent.upload-model False
```

保存至新的`logs/rsl_rl/pointleg_high_step/<时间>_flat_front_correction/`，源文件不覆盖。源迭代实际15498、公共计数372240，按当前框架追加200次更新对应15498～15697；周期文件包括15500、15600，末次为`model_15697.pt`。先对照15600和15697：四足持续换步，分足峰值均值至少12厘米，稳定行走速度误差不超过0.05米/秒，前后慢姿态偏差下降且支撑后扫保持。出现停步/拖腿时，不原样扩展到2000轮。平地通过之后再决定将修正后的权重迁入Elevation，不自动执行第二阶段。

本轮65项HighStep/walking_audit定向张量单元通过；真实15498检查点在CPU严格加载、临时保存，Actor/Critic/优化器张量及迭代保持，源SHA256为`9c9879852f87b017a6e7155301d5e11b03d991c94a4127e16690dad4e9bf34b8`，前后不变。训练命令仅解析并核对源路径，未启动物理仿真、Viser、PPO或训练。检查不证明新策略已纠正，必须启动新进程续训后回放验收。

任务：`PointLeg-HighStep-v0`。日志：`logs/rsl_rl/pointleg_high_step/`。

这是在平地上训练的爬梯前置步态，不包含梯子几何、踏棍识别或实际爬梯成功判定。它独立于原来的 Flat 和 Elevation，不修改这些任务的奖励。

2026-10-29最新0.6秒周期与高度区分：按用户指定，将单足目标周期0.40→0.60秒，周期评分宽度0.08→0.12秒；主循环权重0.34→0.38，周期评分占比0.25→0.30（基础项0.50→0.45，落点项仍0.25）。高度目标仍17厘米，`height_std`0.05→0.03米；其他质量相同时，15厘米高度分由约0.867降为0.762，17厘米仍为1，因此17厘米比15厘米的分数优势从约15%增为31%。紧凑收腿惩罚−0.6→−0.2，暂不自动恢复。

主循环每次结算乘`clamp(实际周期/目标周期, 0, 1.5)`：其他质量相同时，合理周期内更快重复循环不增加每秒主循环回报，过长等待的预算也封顶。周期高斯仍明确偏好0.60秒。辅助抬升保留原每足每摆动封顶0.02的小额探索反馈，本轮没有把它变成严格的时间平均分，不声称所有正奖励的频率影响都已消除。新增`Metrics/high_step_height_score`、`high_step_period_score`及`high_step_cycle_budget`，按完成落足事件记录以便拆分观察。

配套时限：摆动上限0.24→0.36秒，支撑超时0.34→0.51秒，相位惩罚尺度0.12→0.18秒；最短离地0.10秒、最低承重0.12秒保持，允许旧策略在转向新节奏时仍获得反馈。向前速度仍0.15～0.25米/秒，因此0.60秒下同足两次落点的目标间距为9～15厘米；目标落点间距上限10→15厘米，合法落点位移上限14→18厘米，避免最高速度下所需15厘米被旧14厘米门槛判无效。18厘米只是安全余量配置，不要求足端每次向前甩18厘米，原髋参考工作区和侧向限制保持。

0.60秒描述同一只脚的完整支撑加摆动周期，并非单独摆动时间。评估器从当前奖励配置读取目标周期和宽度，验收也随之使用0.60±0.12秒；仍需同时满足逐足高度、真实承重、推进和协调条件。45/325/12、PD、速度、温和偏航−1/持续尺度9、150N接触力、静止站姿规则及其他任务保持；同任务奖励变化直接resume。

从用户当前16499继续，追加1000轮观察新目标的适应趋势：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-29_12-53-21_compact_swing --agent.load-checkpoint model_16499.pt --agent.max-iterations 1000 --agent.run-name period06_height17 --agent.logger tensorboard --agent.upload-model False
```

55项HighStep/walking_audit定向单元通过，无物理仿真、Viser、PPO或新训练。该验证证明配置和计分逻辑，不代表策略已达到17厘米/0.6秒；必须新进程续训才采用。下文0.40秒、14厘米位移上限和−0.6等为历史值，当前以本段及源码为准。

2026-10-29髋膝协同与紧凑收腿：用户反馈前腿足端很高但膝盖空间位置变化小，现保留用户已设`high_step_cycle.weight=0.34`、目标17厘米及原循环评分/结算方式，新增几何约束，而不继续增加高度权重。

`static_joint_posture`在HighStep改用`high_step_static_joint_posture`：仅平移指令范数加偏航指令绝对值不超过0.1时保留原惩罚，前进时归零；静止权重−0.15、档位8及原速度阈值保持，其他任务仍用原函数。它解除运动时髋/膝偏离默认角的代价，并不改变关节限位或PD。

新增`high_step_swing_retraction`，权重−0.6。只对真实承重后离地、摆动中始终至少两足承重、足底离地至少6厘米的足生效，启动宽限0.6秒；足端在机身坐标下相对各自髋俯仰关节的前后位置，以+3.5厘米为中心、允许±8厘米，超出后按8厘米尺度平方惩罚。每足限幅4，四足平均与最差足各占一半。支撑足、低位起落脚、静止/转向、无效地面数据不受该项惩罚。它限制高抬腿时远伸的小腿轨迹，促使策略寻找更紧凑的髋膝配合；不直接奖励关节摆动幅度，也不强制膝盖必须上抬、前后腿同角或固定步态相位。

接触确认的完整摆动（此前承重至少0.12秒、离地至少0.08秒）会写入分足诊断：`Metrics/high_step_swing_hip_range_{leg}_deg`、`high_step_swing_knee_range_{leg}_deg`为摆动期角度最大减最小，`high_step_swing_knee_vertical_range_{leg}_m`为膝相对髋在机身坐标下的竖直位置极差；避免把机身整体运动误认为抬膝。高位摆动另记录`high_step_swing_foot_x_{leg}_m`。没有完成事件时不写虚假零值，真实完成但髋未动则应记录零幅度，以便诊断。

新增实现`mdp/high_step_coordination.py`，不依赖固定时钟，训练观测/动作45/325/12和网络不变。温和偏航−1/持续偏航尺度9、150N接触力项及其他奖励保持。支持从现有同任务模型直接resume，无需transfer；以下示例延续用户当前15498以对照修正前后，并不推翻用户先前13000比13598回放更直的观察：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-28_23-25-35_yaw_mild_soft_contact --agent.load-checkpoint model_15498.pt --agent.max-iterations 1000 --agent.run-name compact_swing --agent.logger tensorboard --agent.upload-model False
```

50项HighStep/walking_audit定向单元通过，无仿真、Viser或训练。几何约束是否使前髋增加参与、是否维持足高及稳定落地，需要续训后的分足指标和回放确认；本轮没有修改主循环按事件结算的频率偏好。下文为历史记录，0.30等旧权重不代表当前源码0.34。

2026-10-28最新温和偏航与接触力约束：用户要求偏航不要加强过大，并反馈后腿落地受力似乎较大。瞬时`high_step_yaw_rate.weight`从−2.0降至−1.0，持续`high_step_drift_bias.yaw_scale_degrees`从7回调至9，总漂移权重仍−1.5；相对原始瞬时−0.75及持续尺度10，仍分别增加约33%和23%惩罚。侧漂、抬腿0.30/17厘米、足位、PD和关节限位保持。

原HighStep没有启用独立接触力项，足滑约束不等于竖直落地冲击约束。现加入`high_step_contact_force`，权重−0.5，单足接触力合力阈值150N、尺度100N，仅向前命令且启动0.6秒后生效；先计算每足超过阈值部分的归一化平方代价并限幅4，再按四足平均/最差足各一半结算。正常阈值以下支撑不罚，前后足标准相同，不强制瞬时等力或减小后足必要承重。该项限制运动中的过大接触力，不只限首次落地，也会覆盖持续过载。

HighStep足传感器`history_length`从3改为`cfg.decimation=4`，按完整20毫秒控制区间中的4个物理子步取每足力范数峰值，避免只看末帧漏掉短时冲击；当末帧已离地但历史存在力峰时仍可记录/扣分。接触通道按腿名排序，非有限输入安全过滤，左右/前后同等处理。没有历史数据时回退当前帧力。150N为训练初值而非硬件安全边界；静态XML显式质量合计16.1557kg，仅用于说明两足分担支撑时约每足80N的量级，不代表随机化环境或实际冲击测量。

新增`Metrics/high_step_contact_peak_fl_n`、`fr_n`、`rl_n`、`rr_n`和`high_step_contact_overload_fraction`。分足值是每个控制区间的历史力峰，日志聚合后通常为这些峰值的均值，不是全回合最大值或冲量。截图红箭头只能提示接触受力差异，尚未实测确认后足峰值过大，也不能区分是冲击还是正常动态承重。

仍按用户Viser反馈从13000继续，使用温和偏航与接触力奖励，追加1000轮：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-28_20-43-44_111 --agent.load-checkpoint model_13000.pt --agent.max-iterations 1000 --agent.run-name yaw_mild_soft_contact --agent.logger tensorboard --agent.upload-model False
```

41项定向单元通过，无物理仿真、Viser或训练；用户续训后需结合四足力峰、足高、真实支撑和回放调整阈值。Actor/Critic/动作45/325/12保持，接触历史是奖励使用的内部缓冲，未加入Actor。后续段落的−2.0、7度/秒为先前较强方案，当前以本段和源码为准。

2026-10-28最新偏航专项加强：用户认为足高足够但偏航严重，`high_step_yaw_rate.weight`由−0.75改为−2.0（相同误差扣分约2.67倍）；`high_step_drift_bias`总权重保持−1.5，只将其`yaw_scale_degrees`由10降至7，相同持续偏航误差在未封顶时扣分约2.04倍。持续偏航死区仍1度/秒、瞬时死区仍2度/秒；侧速分量、EMA、髋角/足位不变。当前用户已把`high_step_cycle.weight`设为0.30，本轮保留0.30及17厘米目标，不再沿用下文0.24历史值。

13598模型来自`2026-10-28_20-43-44_111`，运行配置确认从12099恢复、训练1500轮、抬腿权重0.30/持续漂移−1.5；最新偏航加强尚未进入该PT。末50轮平均足高15.88厘米，持续偏航1.39度/秒、持续侧漂1.37厘米/秒，前/后足对中心0.75/0.57厘米。13000末50轮平均足高15.70厘米、持续偏航1.38度/秒，前/后足对中心1.35/0.86厘米；13000左后髋持续外偏2.48度，13598为3.10度。先前仅据训练均值推荐13598；用户随后明确13000在Viser回放更正、偏移更小，现改为从13000续训。训练随机化下的足对中心/角速度均值不能替代回放中的实际轨迹判断，也没有据此声称完成同条件固定评估。

在工程根目录执行，使用新偏航奖励，从13000直接resume追加1000轮并保存新目录：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-28_20-43-44_111 --agent.load-checkpoint model_13000.pt --agent.max-iterations 1000 --agent.run-name yaw_w2_from13000 --agent.logger tensorboard --agent.upload-model False
```

25项HighStep定向单元通过，覆盖持续偏航加强而侧漂代价保持、左右符号对称和既有循环门控；测试中的旧0.24固定断言按用户现有0.30更新。没有启动训练、Viser或物理仿真。当前奖励约束角速度，不能保证已经转偏且停止转动的机器人回到原绝对航向；Actor45维未加航向目标误差，外环航向控制尚未实现。下文均为先前调参记录，最新数值以本段及源码为准。

2026-10-28持续漂移权重调优：用户授权将`high_step_drift_bias.weight`由−1.0改为−1.5，只提高该项50%，其余奖励、死区、EMA和接口保持。推荐从已适应当前奖励的`2026-10-28_07-46-16/model_12099.pt`直接resume；对应训练事件实际位于`2026-10-28_14-04-52`，末50轮整体足高15.15厘米、持续侧漂1.35厘米/秒、持续偏航1.51度/秒。它相对9200的足高和持续侧漂略好，不代表已经通过确定性策略验收，也不宣称优于所有历史模型。

在工程根目录执行，追加1000轮，采用新权重并保存到新运行目录：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-28_07-46-16 --agent.load-checkpoint model_12099.pt --agent.max-iterations 1000 --agent.run-name drift_bias_w15 --agent.logger tensorboard --agent.upload-model False
```

恢复路径按PT实际位置固定，避免通配符误选。继续观察四足峰值、持续侧漂和持续偏航，不能只比较总回报。以下段落为先前调参记录，其中漂移权重−1为旧值，以本段和源码−1.5为准；本轮未启动训练或物理仿真。

2026-10-28持续偏移修正：用户反馈右偏，最近指定`2026-10-28_11-26-12/model_7199.pt`对应日志末50轮，机身侧速有符号均值约+0.00082米/秒，绝对值均值0.03389米/秒。总体有符号均值接近0不能证明每个环境直行；它也不足以确认是单环境周期摆动或某个方向的共同偏差。因此增加逐环境的持续速度误差反馈，保持左右镜像等价。

`high_step_drift_bias`权重−1，以0.6秒EMA分别累计各环境的机身侧速误差和偏航角速度误差，运动启动宽限1秒；持续侧速容许0.005米/秒、尺度0.04米/秒，持续偏航角速度容许1度/秒、尺度10度/秒。先对各环境的偏差平方扣分，再由框架聚合，不让左偏和右偏互相抵消；原瞬时漂移和角速度约束保留。静止、非直行和无效采样清空历史，支持局部重置和同一步重复调用缓存。

持续髋偏置权重−0.6→−1.0，足对中心权重−1.0→−1.5；角度和足位容许值不再收紧。17厘米、循环0.40秒和抬腿权重0.24保持。新增`Metrics/high_step_lateral_bias_m_s`及绝对值`high_step_lateral_bias_abs_m_s`，偏航新增`high_step_yaw_rate_bias_deg_s`及绝对值`high_step_yaw_rate_bias_abs_deg_s`。这些是机身速度误差的时间平滑值，不是世界横向位置或绝对航向误差；固定45维策略没有这些导航误差信息。

该7199运行实际日志范围5200～7199，保存配置恢复源为通配符，不能视为由8199继续训练。本次示例固定最新用户指定7199，避免再由通配符选中其他运行；在工程根目录执行，追加1000轮：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-28_11-26-12 --agent.load-checkpoint model_7199.pt --agent.max-iterations 1000 --agent.run-name drift_bias_alignment --agent.logger tensorboard --agent.upload-model False
```

本次55项定向检查通过，未运行仿真、Viser或训练，不代表右偏已在策略回放中消除。下面8199日志和首轮纠偏参数为先前变更的依据，当前值以本段及源码为准。

2026-10-28首轮直行与外摆修正：依据`2026-10-28_07-46-16/model_8199.pt`对应日志，末50轮整体足底峰值15.93厘米，前足横向中心偏移3.24厘米。图片反映足端外张，但不能单凭截图量化关节或速度；侧漂是否改善需结合续训指标和回放判断。

| 奖励 | 原参数 | 当前参数 |
|---|---|---|
| `command_orthogonal_drift` | 权重−1.5，容许0.04米/秒，尺度0.15米/秒 | 权重−3，容许0.015米/秒，尺度0.10米/秒 |
| `high_step_yaw_rate` | 无专门死区项 | 权重−0.75，偏航角速度误差容许2度/秒，尺度20度/秒 |
| `hip_abduction_excess` | 权重−1，容许0.12弧度约6.9度 | 权重−1.5，容许4度，尺度仍0.25弧度 |
| `hip_abduction_bias` | HighStep未启用 | 当前权重−1，EMA为0.5秒，启动1秒，持续偏置容许3度、尺度8度 |
| `foot_lateral_placement` | 权重−0.5，中心容许2厘米 | 当前权重−1.5，中心容许1.5厘米；前后足对分别计算 |
| `high_step_workspace` | 单足横向容许偏差12厘米 | 横向容许偏差4厘米；权重−0.5、前后余量10厘米不变 |

单足工作区相对各自髋俯仰连杆，横向名义外伸仍10.5厘米；4厘米是围绕名义值的软余量，不是机身到足端的距离或硬限位。该项检查支撑和摆动足，针对左右对称但前后宽度不同的梯形脚位；足对中心项针对两腿共同向一侧偏移，持续髋偏置项针对单腿长期外摆。瞬时外摆项只选髋外展关节，不对髋俯仰/膝抬腿角度加罚。

本轮只改HighStep奖励，保留17厘米高度、0.40秒周期、`high_step_cycle.weight=0.24`及45/325/12接口。新增`Metrics/high_step_lateral_velocity_m_s`与`high_step_lateral_speed_abs_m_s`记录机身坐标侧向速度，`high_step_yaw_rate_error_deg_s`与`high_step_yaw_rate_error_abs_deg_s`记录偏航角速度误差；同时观察既有分足`hip_abduction_bias_abs_{leg}_deg`、前后足对中心和四足高度，避免正负漂移平均抵消。没有加入绝对世界航向/位置回正目标。

在工程根目录从指定模型直接续训，示例追加1000轮，保存到新的运行目录：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-10-28_07-46-16 --agent.load-checkpoint model_8199.pt --agent.max-iterations 1000 --agent.run-name straight_alignment --agent.logger tensorboard --agent.upload-model False
```

同任务直接resume恢复网络、优化器和迭代，使用新进程的奖励配置，无需transfer。启动中的进程不会热更新。本轮没有启动训练或物理仿真，效果待续训验证。

2026-10-27审查修复版：足接触传感器显式输出世界坐标力，配置包含normal/tangent，避免把接触坐标的切向力误当竖直承重。17厘米仍是最终评分目标；通过低高度也能得到的分级评分和有上限的抬升进展反馈改善探索，没有添加Actor维度或固定步态时钟，也没有实现自动高度等级课程。

## 初始目标

| 项目 | 当前设置 | 含义 |
|---|---|---|
| 足底抬升峰值 | 17厘米（170毫米），评分宽度5厘米 | 按真实球足中心减去2.5厘米半径测量；过低、过高都不如目标高度 |
| 单足完整周期 | 0.40秒，评分宽度0.08秒 | 前一支撑相加本次摆动相，约每足2.5次/秒，不是两条腿交替一次的时间 |
| 前进速度 | 0.15～0.25米/秒 | 只训练低速向前，10%静止样本；暂不训练后退、横移和转向 |
| 正常落脚位移目标 | 6～10厘米 | 根据速度乘0.40秒得到，指同一足相邻落点沿前进方向的距离 |
| 有效落脚位移范围 | 至少2.5厘米，水平总位移不超过14厘米 | 防止原地点脚和用长跨步代替小步 |
| 有效摆动时长 | 0.10～0.24秒 | 过滤极短接触抖动和长时间翘腿 |
| 起脚前及结算前支撑 | 至少0.12秒 | 落地后持续承重才领取奖励，防止点一下就抬脚 |
| 支撑要求 | 摆腿期间至少两足承重 | 有接触且竖向力至少5牛，不能靠四足腾空刷分 |
| 长支撑、长摆动 | 超过0.34秒支撑或0.24秒离地开始扣分 | 最差足与平均各占一半，减少某条腿一直撑着或悬着 |

这些是仿真奖励初值，不是已经训练出的测量结果，也不是硬性轨迹或安全保证。最终周期、步幅和高度必须同时查看录像、真实接触和事件指标。

## 奖励怎样改变

- 移除新任务中的原 `feet_air_time` 和 `foot_swing_height`，避免旧的长摆动窗口和0.50秒频率预算与快周期冲突。
- `high_step_cycle` 当前权重0.24（初版0.08，用户已提高）：高度得分为线性进展和目标高斯各占一半，超高额外衰减；与相对机身收腿比例及`0.5+0.25*周期评分+0.25*落点评分`相乘。低抬腿不再被多个接近零的高斯相乘压掉；结构门控仍要求向前落足、机身前移至少1厘米和落地后持续承重。
- `high_step_lift_progress`：单足单次摆动最多0.02正回报，按相对机身抬升峰值增量结算；需要身体前移、两足承重且尚未摆动超时，悬停不会反复领取。该项不能代替完整落足循环。
- `high_step_cycle_symmetry` 权重−0.4：比较左右对应足最近完成周期的足高、周期和步幅，允许15%差异；最差差异和平均各占一半。
- `diagonal_gait` 权重−0.6：左前与右后、右前与左后按实际承重接触配对；允许0.06秒四足过渡支撑，启动0.6秒宽限。该项不是硬性轨迹，也不保证最终一定学成指定步态。
- 高度还检查相对机身收腿量，单纯抬高机身不能替代关节抬腿。
- `high_step_phase` 权重−0.4：惩罚某足长期不换脚或一直悬空，静止指令不生效，回合开始有0.6秒宽限。
- `high_step_workspace` 权重−0.5：髋参考的前后软余量为±10厘米，限制过伸，不直接约束足端竖直高度。
- `high_step_body_bounce` 权重−0.5：抑制靠机身上下弹跳完成抬脚。
- `high_step_pitch` 权重−1.5：平地俯仰允许±3°小幅变化，超过后按8°尺度软惩罚，抑制持续前倾或后仰。
- `high_step_mean_leg_symmetry` 权重−0.2：仅比较左右对应腿髋俯仰和膝关节慢偏置；EMA时间0.8秒、启动宽限1秒，超过4°后扣分。不强求前后腿具有相同平均承重构型，前后偏差仍记录为诊断指标。
- `short_stance` 最低时长由0.16降至0.12秒，权重由−0.10降至−0.05；仍然处罚短促点地，但给快节奏留下空间。
- 移除重复的`long_foot_air_time`，长离地统一由`high_step_phase`处理。
- 移除`leg_motion_penalty`关节速度惩罚；保留`action_rate=−0.005`、力矩、加速度、限位和接触滑移约束。
- 平移跟踪的 `std` 从0.5收紧到0.20，让低速指令下站着不动明显不如实际跟随。

对角配合依据触地状态反馈，没有输入隐藏的绝对时间相位。左右支撑均衡、静止站姿、两足支撑、横向落脚中心和足滑约束继续保留。四腿姿态不要求每帧相等。

单张侧视图中的前后腿角度差不能证明持续不对称，也不能据此确认XML装配错误。真正的步态质量需按四足独立的足高、步幅、周期和接触配合判断。

命令使用正向范围直接均匀采样，前进分组比例设为0；这是为了避开基础生成器前进分组的0.3米/秒保底，不表示不训练前进。实际采样测试覆盖4096环境。

## 开始训练

已有的平地基础已复制到 `logs/rsl_rl/pointleg_high_step/flat_pretrained/model_0.pt`，原件未改；复制文件与原件SHA256均为：

```text
f340f78849b9bfb5e7507d1175ea7e2119facbe838859aa60648d1621b1f3f1e
```

在 `rc_mjlab` 根目录执行，先观察1000轮：

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run flat_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 1000 --agent.logger tensorboard --agent.upload-model False
```

这是使用当前新奖励续训，不是从零重新初始化；标准恢复会保留基础文件中的优化器状态。Actor/Critic/动作维度仍为45/325/12，关节默认姿态、动作缩放、5Hz滤波、50Hz控制、PD和±25牛米上限没有更改。更改奖励不构成必须清空优化器的理由；是否需要调整更新强度由后续实验决定。

已有训练进程不会自动采用本次代码；必须启动新任务。本文没有授权或声称已经替用户启动正式1000轮训练。

回放时替换实际时间戳和模型名，从0开始训练1000轮通常最终为 `model_999.pt`：

```powershell
uv run --locked --no-sync pointleg play PointLeg-HighStep-v0 --checkpoint-file ".\logs\rsl_rl\pointleg_high_step\时间戳\model_999.pt" --num-envs 1 --viewer native
```

需要浏览器时把 `native` 改为 `viser`；只在已训练的0.15～0.25米/秒前进指令内评估新风格，零指令检查站立，不能因此宣称已学会全向运动。

2026-10-27修复Viser面板零范围断言：HighStep侧向/转向轴显示为只读0，不创建初始值越界或零跨度滑条；前向滑条默认0～0.25米/秒，0用于停止。通用Max控件仍可调整非固定轴幅度，评估时保持训练速度范围。该修复仅影响GUI，可直接重跑原play命令，无需重新训练或更换检查点；实际Viser场景未在修复后启动，只执行无服务器的GUI单元检查。

TensorBoard：

```powershell
uv run --locked --no-sync python -m tensorboard.main --logdir ".\logs\rsl_rl\pointleg_high_step"
```

主要观察 `Metrics/high_step_peak_height_m`、`high_step_cycle_s`、`high_step_step_length_m`、`high_step_air_time_s` 和四足各自的周期。前四项只在记录到完整摆动时写入样本，不用没有样本的0冒充低高度；循环回报率没有样本时可以为0，须结合事件曲线判断。

还要看 `high_step_phase` 是否长期扣分、`stance_duty_*` 是否严重偏侧、`flight_fraction`、速度跟踪和滑移。一次很高的抬脚不能替代四足持续交替的小步；四足离地、拖腿或一直原地点脚都不能算达标。

新增常驻诊断：`Metrics/high_step_loaded_feet`、`high_step_takeoffs_per_s`、`high_step_landings_per_s`和`diagonal_contact_error`；分足事件新增`high_step_peak_{leg}_m`和`high_step_stride_{leg}_m`。无事件不写虚假高度零值。

`pointleg evaluate PointLeg-HighStep-v0`使用0.2米/秒、平均速度误差最多0.05米/秒。除前进和稳定外，每足至少5次已识别循环、至少80%达到高度目标的80%～120%及0.40±0.08秒周期，左右对应足平均高度/周期/步幅差异不超过20%，平均对角接触错误不超过0.15。失败落足也计入合格比例分母，避免只统计少量好动作。报告含逐足结果；判据仍需后续训练和硬件验证。

姿态相关新增 `Metrics/high_step_pitch_deg`、`high_step_front_rear_hip_bias_deg` 和 `high_step_front_rear_knee_bias_deg`，单位为度，后两者只在运动平均窗口有效时记录。平地俯仰约束不能直接当作坡地必须水平的要求。

## 后续爬梯迁移

选取实际步态合格的检查点，再接入真实梯子／楼梯环境和对应感知、接触及成功条件。高抬腿基础不证明能踩准踏棍，也不证明足端全路径不会碰撞。

迁移时重新评估17厘米目标、0.40秒周期和14厘米最大落脚位移是否适合实际梯距；窄踏面或梯子可能需要更长承重、更慢周期和不同落点范围，不要把本任务全部奖励直接锁死到爬梯环境。若改观测、动作或命令含义，要另外处理网络接口兼容。

## 代码与诊断入口

2026-10-27用户要求：今后不再自动运行仿真测试或PPO调试，只做代码检查和有针对性的单元测试；以下真实仿真命令仅作为用户主动需要时的入口，不自动执行。最新俯仰与前后腿平均姿态调整没有做仿真或训练验证。

- `src/pointleg/config/high_step.py`：独立任务配置与奖励参数。
- `src/pointleg/mdp/high_step.py`：四足独立循环、真实承重和相位超时逻辑。
- `src/pointleg/config/rl_cfg.py`、`src/pointleg/__init__.py`：独立实验名与任务注册。
- `src/pointleg/tests/test_high_step.py`：时长、落点、支撑、接触顺序、重置和真实指令采样检查。
- `tools/check_high_step.py`：短时真实物理与可选旧模型加载检查，报告 `policy_validation=false`，不等于训练验收。

```powershell
uv run --locked --no-sync python -m unittest discover -s src/pointleg/tests -p test_high_step.py -v
uv run --locked --no-sync python tools/check_high_step.py --steps 96 --checkpoint-file pretrained/flat_9000_transfer/model_0.pt
```


2026-11-01 08:12实际后台启动用户指定train_4命令：运行目录 `logs/rsl_rl/pointleg_high_step/2026-11-01_08-12-22_train_4`，从固定`2026-11-01_04-10-49_train_3_2/model_10198.pt`直接resume；控制台确认加载路径，已观察迭代10215且已保存model_10200.pt，新增20条逐足支撑轨迹指标已进入TensorBoard。10001次追加更新、间隔100、512环境，预期末次20198。启动器PID 4100，TensorBoard事件记录训练PID30180；进程在后台保留，由用户后续验收，不声称训练完成或轨迹已改善。首次受限后台启动未保留进程、无训练输出；随后在沙箱外隐藏启动成功，无重复训练实例证据。源SHA256 6e38e914384558557b5585e0dfb6a1842af9880d39d52697423dcaaee785931e；checkpoint接口45/325/12和优化器字段已核对、实际resume已执行。25项张量/配置、AST及diff --check通过；启动前未做独立物理测试。MuJoCo现有MULTICCD圆柱/盒接触警告未阻止训练。启动和运行证据见artifacts/train_4_launch.json、train_4_runtime.json、high_step_train_4_preflight.json，stdout/stderr路径见launch元数据。


2026-11-01 12:41实际启动tran_4_2，运行目录`logs/rsl_rl/pointleg_high_step/2026-11-01_12-41-56_tran_4_2`，uv启动器PID48076、事件训练PID28384。控制台确认加载原train_3_2/model_10198.pt，已观察迭代10210并写出model_10200.pt；超15厘米惩罚和逐足路程额度/超高量指标已进入TensorBoard，实际采用新版配置。512环境、10001次追加更新、每100次保存，预期末次20198。后台保留供用户验收；42项张量检查、接口/优化器字段/原模型哈希、CLI、AST与diff检查通过，未做独立回放或物理测试，不声明步态已改善。旧train_4已停并保留最后model_17100.pt。启动/运行/预检证据为artifacts/tran_4_2_launch.json、tran_4_2_runtime.json、tran_4_2_preflight.json。
