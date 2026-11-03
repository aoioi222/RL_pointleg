# 验证与排查

2026-11-03 Stairs大腿对机身姿态：`test_stairs`、`test_slope`、`test_elevation_paired`、`test_elevation_fixed_stairs`共42项通过（2.391秒），合成测试覆盖上阶区域水平大腿不罚、一条垂直大腿受罚、下阶和远离台阶不罚，并覆盖既有课程/坡面/续训；四文件py_compile通过。未进行动力学仿真或训练，不能证明姿态和成功率改善。

2026-11-03 Stairs抬脚目标随地形变化：`test_stairs`、`test_slope`、`test_elevation_paired`、`test_elevation_fixed_stairs`共41项通过（2.494秒），验证五档、上/下共10个环境各自高度目标与80%抬升一致的比例惩罚，原低抬脚排序、非失败、缓存/部分reset和固定高度其他任务回归；四文件py_compile通过。未仿真/训练。

2026-11-03 补充Stairs旧固定高度检查点兼容测试后，同一四组测试共40项通过（2.538秒）：合成旧布局checkpoint直接resume被拒绝，纯actor/critic权重加载成功。仅在临时目录处理CPU合成checkpoint，未触碰用户模型。

2026-11-03 Stairs 6～12厘米课程：`test_stairs`、`test_slope`、`test_elevation_paired`、`test_elevation_fixed_stairs`共39项通过（2.568秒），覆盖五档上下阶profile、完整地图配置、上下方向独立晋升和混合Elevation固定12厘米回归、迁移/续训检查；相关Python文件语法检查通过。验证未执行物理仿真、Viser或训练，不能证明新策略已收敛。

2026-11-03 Stairs 13厘米抬升软惩罚：`test_stairs`、`test_slope`、`test_elevation_paired`共33项通过（3.227秒），覆盖4/10/14厘米摆动落地的惩罚排序、达标零惩罚、同周期缓存、单次落地及局部重置；四文件py_compile通过。未启动物理仿真、Viser或训练，不证明策略抬脚或越阶达标。

2026-11-02 Stairs逐足落点与四足静止：`test_stairs`、`test_slope`、`test_elevation_paired`共32项通过（2.382秒），覆盖两个足独立奖励、内侧优于边缘且每级不重复支付、四足持续静止受罚及一足抬起/机身前进时清零；修改Python文件语法检查通过，未仿真/训练，不说明步态改善。

2026-11-02 取消Stairs抬升不足终止：`test_stairs`、`test_slope`、`test_elevation_paired`和`test_elevation.ElevationLogicTests`共46项通过（2.467秒），四文件py_compile通过；合成上阶不足120毫米接触保留诊断但不触发失败，覆盖出口成功、出界和坡面回归。未执行仿真、回放或训练。

2026-11-02 Stairs局部奖励调整执行`test_stairs`、`test_slope`、`test_elevation_paired`共30项通过，包含原地零推进和支撑推进一次性奖励的合成测试；未运行物理仿真、训练或课程验证。

2026-11-02 Stairs任务升级验证：执行`.venv/Scripts/python.exe -c "import sys, unittest; sys.path.insert(0, 'src/pointleg/tests'); suite=unittest.defaultTestLoader.loadTestsFromNames(['test_stairs','test_slope','test_elevation_paired']); result=unittest.TextTestRunner(verbosity=1, buffer=True).run(suite); sys.exit(not result.wasSuccessful())"`，29项通过（2.349秒）；相关Python文件`py_compile`和定向`git diff --check`通过。覆盖Stairs速度/时限/周期/髋零位迁移、全六级覆盖加四足出口承重、上阶低于120毫米硬失败，并回归Slope成功、周期和旧检查点加载行为。未初始化物理场景、未执行仿真、Viser或PPO训练，不能据此宣称真实爬阶能力已达标。

2026-11-01 Slope四足到达成功：`test_slope`、`test_elevation_paired`、`test_elevation.ElevationLogicTests`和`test_elevation.ElevationWalkingTests`共49项通过（2.590秒），六个Python文件py_compile通过。覆盖上坡顶部/下坡底部四足真实8N承重立即完成、TerminationManager终止与RewardManager单步+20结算、auto_reset配置与局部重置清状态、三足/坡上残足/错高度/弱承重/侧面/非有限拒绝、下坡出生上平台不成功、平移地块局部坐标、失败优先、256环境上下坡各128个及现有奖励/课程/旧checkpoint兼容。首次平移夹具因只移动足点未移动机身Y而触发wrong_route，补齐机身平移后通过，未放宽产品判定。只核对源码、保存的train_2环境参数和纯张量/CPU网络，未初始化物理场景、未仿真或训练，不声明已重现或定位用户所见机器人走上台阶的动态过程。

2026-11-01 Slope加速/周期/髋纠偏：执行`.venv/Scripts/python.exe -c "import sys, unittest; sys.path.insert(0, 'src/pointleg/tests'); suite=unittest.defaultTestLoader.loadTestsFromNames(['test_slope', 'test_elevation_paired']); result=unittest.TextTestRunner(verbosity=1, buffer=True).run(suite); sys.exit(not result.wasSuccessful())"`，21项通过（1.937秒）；五个本轮Python文件`py_compile`通过。覆盖0.30秒碎步比0.70秒周期受罚更多、10/20毫秒积分近似一致、长期承重/悬空不能绕过约束、单帧点地不确认落脚、同周期缓存/局部重置/失败/非有限隔离、单足内外偏同罚且髋俯仰不进入髋外展项、单足慢姿态异常及高差放宽、随机重置新速度范围、训练/回放120秒与Stairs隔离。临时合成旧revision9坡面checkpoint完整resume保留Actor、轮次123和课程上限，并在新revision10/.25～.50/120秒配置下提示奖励变化。未读取或修改生产模型文件，未启动仿真、Viser、PPO或训练；这些检查不证明碎步、偏髋或上坡能力已在策略中改善。

2026-11-01删除四个旧单方向入口后，执行 `python -m unittest test_elevation_specializations test_elevation_paired test_elevation.ElevationLogicTests test_elevation.ElevationWalkingTests test_elevation.ElevationTransferTests -q -b`，40项通过（1.639秒）。原四专项测试更新为三入口注册/旧ID拒绝检查，保留双向采样/课程/奖励/迁移及旧混合兼容回归；旧ID在配置加载、评估用例和transfer均不可用，四份配置模块不存在，当前三入口评估仍为10/10/21用例。未修改策略权重或启动仿真/训练。

## 2026-11-01 双向合并与基础奖励验证

实际执行 `.venv/Scripts/python.exe -m unittest test_elevation_paired test_elevation_specializations test_elevation.ElevationLogicTests test_elevation.ElevationWalkingTests test_elevation.ElevationTransferTests -q -b`，53项通过，5.049秒。新文件9项覆盖双向/局部重置/禁止跨组固定评估、顶部出生、双方向坡度课程独立及固定台阶、14项白名单/接口隔离、独立速度/失败和非有限正收益屏蔽、左右侧漂同罚与20度上/下坡姿态参考、实际出通道事件−50、8N真实承重及非有限接触排除、迁移/严格resume/跨任务拒绝及源hash保持。首次仅新9项时，play动作延迟与训练配置错误对比导致1项测试失败；改为同play模式比较后聚合53项全通过，产品动作定义无需改动。

另对真实HighStep14100→两个新任务执行CPU迁移、全部Actor/Critic逐张量比较、优化器/课程/轮次归零和严格resume，源哈希不变，证据 `artifacts/elevation_paired_transfer_validation.json`。两条200更新CLI已解析并通过项目路径解析器准确定位目标model_0，相关源码AST检查通过。未构建/积分物理环境、未跑Viser或训练，不将检查通过视作足高保持或实际通行率验收。

## 2026-11-01 tran_4_2 距离额度与超高惩罚

2026-11-01 tran_4_2 修正HighStep奖励冲突：active_lift_event +0.35及active_lift_progress +0.08的名义权重保留，二者和hip_swing_coordination +0.06改乘同一逐足承重路程额度。每控制周期沿本段前进起点固定航向刷新机身历史最远位置，新增距离封顶指令速度×dt；每足只在连续8N承重期间积累新距离，离地冻结min(该距离/7厘米,1)，新支撑清零。循环共享该额度，常见7厘米以内等高度的同总路程拆成更多小步不会多领这三项收益；静止、往返和已错过路程不补发，超速多余位移不延迟兑现。连续抬升的峰值及已支付高度封顶15厘米。增加active_lift_overshoot −0.5：摆动时相对起脚足底主动抬升超过15厘米，按5厘米尺度平方、单足封顶4，四足均值和最差足各半；不乘正奖励路程额度，原地超高摆动也扣分。short_stance_target为独立软目标0.24秒，原0.16秒有效起脚/稳定落地门槛保留。侧漂−2.5→−3.0，容许0.010米/秒/尺度0.10米/秒保持且左右等价。后扫+0.10/7厘米平方/0.30秒、姿态、PPO/PD、45/325/12和其他任务不改。当前25项奖励。新增Metrics/high_step_lift_distance_credit_{leg}、high_step_lift_overshoot_{leg}_m、high_step_new_forward_distance_m。代码入口config/high_step.py、mdp/high_step_basic.py及mdp/high_step_hip_swing.py。

42项定向配置/张量检查通过（test_high_step_basic、test_high_step_backstroke、test_high_step_lift_budget、test_high_step_coordination），覆盖8/12/15厘米评分、15厘米以上正收益封顶/负代价递增、常见等距离拆步不多领、髋摆共用额度、静止/往返/超速/腾空、重复读取/部分重置/失败/非有限、0.24秒浮点边界及控制周期积分；AST与diff --check通过。未额外启动物理验证。旧train_4已按命令行身份停止Python PID30180及对应uv启动器，最后保存model_17100.pt，模型日志保留；从用户认可原始train_3_2/model_10198.pt直接resume，源SHA256 6e38e914384558557b5585e0dfb6a1842af9880d39d52697423dcaaee785931e。启动状态另记，不把配置和张量通过等同于训练效果。

从10198追加10001次更新，每100保存，预期末次20198；先回放10300/10400观察足高、支撑和拖腿。正奖励已按路程缩放，不能直接用旧运行总分比较效果。

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-01_04-10-49_train_3_2 --agent.load-checkpoint model_10198.pt --agent.max-iterations 10001 --agent.save-interval 100 --agent.run-name tran_4_2 --agent.logger tensorboard --agent.upload-model False
```


2026-11-01 HighStep train_4 支撑行程修正：stance_backstroke 权重 +0.04→+0.10，7厘米目标由线性改为比例平方，乘 min(承重时长/0.30秒,1)；8N承重、0.16秒最低门槛、2.5厘米累计真实接触滑移质量保留。有效后扫仍取真实机身推进和相对支撑足后扫的较小值，取最后一个承重采样，避免离地后的足端移动计入。超过目标不再增加收益。command_orthogonal_drift −2→−2.5，0.010米/秒死区和0.10米/秒尺度保持，左右等价。仍24项，主动抬升目标/两项权重、姿态、45/325/12、PPO/PD及Elevation/Jump不改。新增逐足 Metrics/high_step_stance_sweep_{leg}_m、stance_body_progress_{leg}_m、stance_start_x_{leg}_m、stance_end_x_{leg}_m、stance_duty_factor_{leg}，分别表示支撑相对后扫、身体推进、落地/离地前相对机身前向位置和完整周期支撑占比，前向坐标轴在落地时固定，世界足端静止也会随身体推进产生后扫。保持旧 backstroke/stance_time/contact_slip 指标。25项配置和张量检查通过，覆盖相同总距离短步不优于长步、滑移降质、7厘米饱和、静止/拖脚/短触地/失败无收益、事件缓存/部分重置/非有限/控制周期及左右同罚；不代表训练验收。用户明确授权按固定命令实际后台续训10001次更新，不执行额外仿真测试。

2026-11-01最新HighStep共24项：新增lf_rr_pose_tracking +.25，在右后足真实有效摆动且主动抬升≥3厘米、机身前进>.05米/秒、至少两足承重时，左前跟踪右后当前髋摆/膝角及膝相对髋的机身竖直位置；左前自身不必已进入摆动，避免旧对角项的双足摆动门槛不给纠正信号。关节误差尺度20°/25°，膝高4厘米，两类误差各半取高斯正分；后腿仅作参考，策略仍共享。侧漂权重−1→−2，死区.015→.010米/秒，左右对称；按用户追加要求髋外展−.9→−3.6（4倍），零目标/零死区/.25弧度尺度保持。其余项、45/325/12、PPO/PD/地形及其他任务不改。静态模型审计未发现缺执行器、错绑或左前髋摆驱动失效：XML12→删除0→重建12，全部动作槽/关节对应，0.1弧度目标增量仅对应DOF产生5N·m；四髋相同静态角增量使膝上升约14.1～14.3毫米。报告artifacts/high_step_lf_actuator_audit_2026-11-01.json；未积分时间，不代表动态碰撞/带载饱和或策略已验收。一个合成张量检查通过：左前仍支撑时可获跟踪反馈、匹配高于异常姿态、失败/非有限/静止0及左右侧漂同罚。未仿真或训练；新进程同任务直接resume，当前截图未绑定到指定模型文件。

2026-11-01 HighStep承重后扫与侧漂调整：保留已有command_orthogonal_drift，权重−.5→−1.0、1.5厘米/秒死区与10厘米/秒尺度不变，对左右漂移等价，不加入只针对左侧的偏置；横向速度为机身坐标，不能替代绝对航向/全局路径控制。新增stance_backstroke +.04，每足连续8N承重≥.16秒后离地时按有效后扫/7厘米线性发放，单足每次最多+.04；有效后扫取落地到离地间机身实际前进距离与身体相对该支撑足前进距离中的较小者，负值置0。按真实接触点累计滑移乘max(1−滑移/2.5厘米,0)，无机身推进的脚底拖滑不得分；超过7厘米饱和，事件原始值除dt，由奖励管理器积分一次。现有高抬腿、对角/髋摆、姿态、滑移惩罚和支撑超时保持；总奖励23项，45/325/12、其他任务/PPO/PD不变。指标Episode_Reward/stance_backstroke、Metrics/high_step_backstroke_{fl,fr,rl,rr}_m、Metrics/high_step_stance_time_{leg}_s、Metrics/high_step_contact_slip_mean_m_s和Metrics/high_step_lateral_speed_abs_m_s，应同时观察四足抬升及回放，防止过长支撑、停步、足底滑移或高度退化。按用户要求只执行一个合成轨迹检查，23项配置、真实身体推进奖励、原地拖脚无收益、单事件、左右同等漂移通过；未仿真/Viser/训练，不能据此断言实机发力或爬阶通过。用户未指定当前较好模型路径，继续训练时固定选择其HighStep检查点直接resume，新进程才启用。

2026-11-01 HighStep加入diagonal_rear_tracking +.2，22项：左前跟右后、右前跟左后；起脚前真实承重且对角两足同时有效摆动、至少两足承重、前进指令及实际机身前向速度>.05米/秒时，以两足进入共同有效摆动时的各自髋俯仰/膝关节角为起点，比较角位移，不强制绝对关节角。后腿相对起点的髋或膝位移至少5°才发放，20°运动强度饱和；前后角位移误差按15°尺度高斯计分，两对得分相加除2，按控制周期正常积分。实际后腿位移同时参与得分，不能保证后腿策略参数被固定，也不能由后腿单独决定策略梯度；前后足不同步时该项无收益而非强制锁相。现有足高、髋外展、pitch、前足工作空间及髋摆项保持，45/325/12、PPO/其他任务不变。Metrics/high_step_diagonal_error_fl_rr_deg、_fr_rl_deg记录两关节角位移差的均方根；Metrics/high_step_diagonal_rear_motion_fl_rr_deg、_fr_rl_deg记录对应后腿相对起点最大关节位移，仅满足门槛的样本记录。只执行单个配置与张量轨迹检查，22项/权重.2，左前右后匹配高于相反、后腿未动或静止0；未仿真、Viser、训练或完整回归，右前左后由相同映射实现但未做独立轨迹测试。新进程从用户选定同任务检查点直接resume，实际效果需短训观察。

2026-11-01最新HighStep针对前足过度前伸、髋外展和前倾调整：新增front_foot_workspace −.7，分别取左右前足球心相对同侧髋俯仰连杆的机身坐标x，以.6秒EMA、1秒启动宽限追踪，平均前伸超过4厘米时按8厘米尺度平方计罚，每足封顶4后平均；仅前进时生效，不锁定瞬时摆动姿态。hip_abduction_excess −.3→−.9（默认0角、零死区/.25弧度尺度不变）；high_step_pitch −.5→−1、死区3°→2°/尺度8°不变，正负俯仰同等处理；hip_swing_coordination +.04→+.06，原有效步/稳定落地/两足支撑/身体前进资格、20°髋幅及3厘米相对髋膝上下位移饱和目标不变。主动抬升+.35/.08/15厘米和其余未提及的奖励保持，总数21，45/325/12、PPO/PD/其他任务不改。指标Metrics/high_step_front_foot_mean_x_fl_m和_fr_m，可与high_step_pitch_deg、hip_abduction_deviation_mean、分足髋摆及膝竖直幅度、分足主动抬升共同检查；EMA均值不是每步足端位置。单个配置/张量检查通过，验证前足平均前伸受罚、零前伸无罚、同周期缓存及部分重置；未仿真、回放或训练，实际改善和足高需短训验证。用户未指定截图对应检查点，续训应固定用户选定的HighStep检查点直接resume，新进程才应用奖励。

2026-10-31最新HighStep应用户要求在17项上增加3项，共20项：hip_abduction_excess −.3，以默认0弧度为中心、零死区、.25弧度尺度平方，四腿分别计分；high_step_pitch −.5，正负俯仰超过3°后按8°尺度平方惩罚；hip_swing_coordination +.04，对有效完成并稳定落地的摆动奖励髋俯仰角幅及膝相对髋在机身坐标竖直幅度，20°/3厘米各饱和，两者各占一半。只在8N支撑且摆动期间至少两足承重、原主动高度有效落地条件、实际身体前进≥1厘米时结算一次；不单独给原地抖关节奖励，正常动作仍允许偏离0，不是硬限位。原17项（足高+.35/.08、15厘米目标）保持，45/325/12及其他任务配置未改。单个无物理张量轨迹检查通过：20项配置、零死区和水平参数、髋20°/膝3厘米饱和并只结算一次；未仿真或训练。指标为Metrics/hip_abduction_deviation_mean、Metrics/high_step_pitch_deg、Metrics/high_step_swing_hip_range_{leg}_deg、Metrics/high_step_swing_knee_vertical_range_{leg}_m。

+2026-10-31最新HighStep在15项基础项上新增两项独立主动抬升奖励，当前17项：active_lift_event +.35，在足完成有效摆动并稳定承重.16秒后结算 `min(主动峰/.15,1)^2`；active_lift_progress +.08，摆动期间仅按主动峰新增加高度/.15连续支付。主动峰是足底相对起脚足底的上升，不是相对世界地面高度；无过高奖励。资格为起脚前承重≥.16秒、起脚后其余足≥2足承重、离地.10～.36秒、足前移≥2.5厘米、稳定落地≥.16秒。静止、原地点脚、短支撑、落地未稳、全足腾空、失败和非有限不领；高度不与周期/后扫/收腿/姿态质量相乘。仍用基础评估有效循环条件，高度暂报告而不作为通过硬阈值；新增主动抬升及分足指标。6200仍直接resume，建议新运行名active_lift_6200，100更新、看6250/6299；45/325/12和其他任务保持。

2026-10-31主动抬升接入后，未按用户要求重跑完整HighStep回归。只执行两项新张量检查：主动8/12/15厘米事件分严格为比例平方、连续反馈累计为高度比；原地点脚、落地不稳定和短承重均无事件收益。当前配置解析为17项，model_6200读取后仍为45/325/12、iter6200。此前15项基础版本的101项检查及完整CPU恢复记录仅适用于本次改动前；报告artifacts/high_step_basic_validation.json已标注该区分。未启动物理场景、Viser、PPO或训练。

2026-10-31HighStep滑移/距离抬升最终84项定向通过（6.464秒）：test_high_step*.py及test_walking_audit.py，新增test_high_step_slip.py的19项。覆盖接触点v+ω×r/真实滚动、拖滑/法向过滤、8N/触地宽限、排序、累计滑移回程不能消除、软短支撑与硬门槛分离、缓存/局部重置/非有限、8/12/15平方分/过高衰减、距离公式与控制周期/超速/往返/额外摆动/不攒信用、纯弹跳/全腾空/失败及任务接口隔离。原测试夹具补接触pos/normal及线/角速度，轨迹速度与位置同步；旧13厘米对height_std依赖的断言改为相等，因为新评分下std只控制过高衰减。源7199 CPU严格resume/临时保存校验Actor/Critic/优化器/迭代7199/公共计数172944全部保持，SHA256 3170689f7ea12bde64f8c24144edad3e7a99be92b4cb3b000316f3bd2486babb前后不变；300轮固定CLI及源路径解析通过。Elevation配置/核心/净空、Jump配置、共享PPO五文件前后hash一致。没有物理环境、Viser、PPO或训练，未证明实际步态改善。

2026-10-31梅花桩输入扩维迁移：6项新定向检查与11项原Elevation/专项/Jump迁移恢复检查共17项通过，均为CPU张量和网络，不构造物理场景。真实Flat8700→`flat_8700_pretrained/model_0.pt`已核对原参数、60/56新增零列、源哈希不变、优化器/迭代/计数归零及严格resume；512环境/10000轮命令已tyro解析并解析到实际model_0。报告`artifacts/stepping_stones_flat8700_transfer_validation.json`；Actor/Critic随机输入最大绝对误差分别1.90735e-6/5.24521e-6。未仿真、Viser或PPO，未重新运行历史全量检查。

2026-10-31版本7最终195项定向检查通过（19.570秒）：原Elevation106项、四专项15项、HighStep55项及新test_elevation_functional.py的19项；未执行ElevationGeometryTests或整个test_elevation.py。新检查覆盖下阶真实抬升/地形净空分离、上阶提前越棱不双计高度、正负坡面切线不算抬升、踏面中心收益/同高末平台不造假边缘、九点实际盒/坡面法向/平移/前角提前净空、首次触地使用前帧速度、短接触末帧前结束仍锁存、事件积分与控制周期无关、缓存/局部重置/失败/非有限、单腿碰撞聚合及Jump旧公式、五任务接口隔离、报告按事件加权和缺事件null。旧106先通过、新首批15通过后补4项，最终聚合195无失败。真实固定地图model_0 CPU完整resume/临时保存校验Actor/Critic/优化器及迭代0/地图契约保持，新元数据7，仅预期奖励变化提示；源hash仍be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568。未执行物理仿真、Viser、PPO或新训练，不能视为通过率验收。

2026-10-30专项拆分验证：`.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -p test_elevation_specializations.py -q -b`，15项通过（4.464秒）。覆盖四项train/play配置、独立日志/配置对象、45/325/12配置与共享奖励保持、全量/局部重置只采目标方向、固定评估拒绝跨专项、下降出生高度、固定台阶变体与坡度独立晋级、专项课程指标、原混合采样保持、evaluate分派、八组训练/续训CLI解析。CPU仅构造网络与临时检查点，验证同任务迭代/计数/坡度课程恢复、跨任务resume拒绝而权重迁移成功、四实验目录独立、不覆盖已有迁移结果、旧混合检查点兼容、专项迁回混合、源文件哈希保持；无正式模型或报告输出。另106项既有定向回归通过（18.423秒），组合为`test_elevation_alignment/clearance/gait/support/fixed_stairs/retention`、`test_walking_audit`及`test_elevation.ElevationLogicTests/ElevationWalkingTests/ElevationTransferTests`，未执行GeometryTests或物理仿真。`python -m pointleg.run list`实际列出12个启用任务及Bridge占位。共121项不同检查，不等于新任务动力学或策略通过率验收。

2026-10-30HighStep平地纠正65项定向单元通过（4.590秒）：`test_high_step*.py`加`test_walking_audit.py`，新增10项`test_high_step_correction.py`。覆盖实际循环长承重/后扫质量提升、短触地/小推进/纯竖直/滑动降质、后滑不得放大、接触顺序/航向等价、驻足衰减与循环恢复、前进静止/往返无速度收益、静止指令正常站立跟踪、缓存/局部重置/失败/无效/腾空、参数及任务隔离/配置兼容。原目标高度和工作区硬编码断言同步为当前15厘米/相对配置边界；慢姿态单元验证共同屈腿不罚、前后均值失配受罚。真实model_15498 CPU严格resume及临时保存校验Actor/Critic/优化器完全保持、迭代15498/公共计数372240，源SHA256 `9c9879852f87b017a6e7155301d5e11b03d991c94a4127e16690dad4e9bf34b8`不变。命令仅解析并精确解析源路径；没有物理环境、Viser、PPO或新训练，也没有实际纠正成功证据。

2026-10-30版本6实际105项聚合通过（16.843秒），再补1项落地奖金比较通过（.563秒）；共106项不同检查，含新文件test_elevation_support.py的13项。聚合原七文件定向与Walking/Transfer，并新增无仿真的ElevationLogicTests；没有执行原生GeometryTests或整个test_elevation.py。覆盖稳定世界足/身体推进的后扫、短触地/小后扫/滑移/纯垂直抖动区别、坡面切线、滑出再滑回、后滑不得扩大信用、驻足速度过期与换步恢复、缓存/局部重置/失败、工作区前收后放、慢前后偏差/共同屈腿/交替相位/异高支撑/非有限、近距离随机出生及任务隔离、长支撑实际循环收益高于短触地。首次扩展跑93项只有浮点积分约2e−6误差，断言精度按轨迹累计误差调整；新姿态检查发现部分reset时其他环境失败掩码遗漏，已修复后通过。

另用XML和默认关节角纯刚体变换（未编译模型/积分）核对前足球面最前缘距根点.43910095米，最靠近入口出生时距立面约.26089905米，仅证明默认姿态静态间距，不涵盖动态/噪声。真实固定地图model_0 CPU加载/临时保存核对Actor/Critic/优化器及布局保持、奖励revision6/接近距离1.0采用，源hash仍be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568。没有物理仿真、Viser、PPO或已改善策略结论。

2026-10-30版本5最终77项通过、17.788秒：聚合test_elevation_clearance/gait/alignment/fixed_stairs/retention、test_walking_audit，以及单独选取test_elevation.ElevationWalkingTests和ElevationTransferTests；未运行含原生物理ray/compile检查的GeometryTests或整个test_elevation.py。新增14项验证前进信用/速度质量、往返不重复、两足支撑/腾空屏蔽、同一步缓存/局部重置、失败完成无效掩码、非有限数据、地块平移/控制周期、8/12/15厘米评分、距离拆分不增加同质量循环收益、80%物理高度门槛、拖行/原地点脚拒绝、合成行走主要分项回报高于静止/拖行、评估60秒/显式覆盖及短停滞仍生效。组合回报仅七项主要反馈，不是物理总回报或策略验收。初版组合测试把奖励类当函数调用报错，已修正为调用已实例化项；补支撑门槛后旧速度用例缺支撑样本亦已更新，最终77均通过。

另在CPU只读加载真实highstep_fixed120_pretrained/model_0，临时保存后核对Actor/Critic张量完全相同、优化器保持、迭代/公共计数0、布局契约相同、45/325/12、奖励版本5/目标.15、低速与3000步回合；仅奖励变化提示。源SHA256 be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568 前后相同。没有物理编译/积分、Viser、PPO更新、正式训练或已改善步态的结论。

2026-10-29回滚版本4并固定120毫米台阶：聚合`test_elevation_clearance.py`、`test_elevation_gait.py`、`test_elevation_alignment.py`、`test_elevation_fixed_stairs.py`、`test_walking_audit.py`共50项通过，3.186秒；额外单独运行`test_elevation.ElevationTransferTests`6项通过，0.338秒。新增5项覆盖训练/回放配置全部等高、上下六阶/出生高度、平台变体保留、成功失败不触发台阶课程、坡面课程独立、保存状态旧台阶帽窗口清理、奖励/指标回滚、评估明确固定高度和非法高度拒绝。加载检查包含旧布局revision2高度课程拒绝完整resume但允许仅加载权重；同地图仅奖励版本不同仍直接resume。历史净空/步态夹具显式用旧合成几何检验既有逻辑，新固定高度测试明确指定.12并核对真实任务配置。只构造配置、张量/几何与临时网络检查点，没有编译/积分物理、Viser或PPO；未运行整个含仿真的test_elevation.py，未做正式策略验收。

2026-10-29 Elevation版本4聚合`test_elevation_clearance.py`、`test_elevation_gait.py`、`test_elevation_alignment.py`、`test_elevation_posture.py`、`test_walking_audit.py`，57项定向配置/张量单元检查通过，39.261秒。新增姿态文件12项覆盖四腿相同低膝代价、正常足支撑排除、腿杆靠近真实立面、地块平移、20°上下坡、平地慢参考与障碍放宽、启动宽限、缓存/局部重置/非有限/失败掩码，以及髋摆幅分级饱和、四腿等价、真实落地一次结算、拒绝地面抖动/无前进/低抬腿和任务隔离。先前35项原净空/步态/纠偏与11项新增检查亦分别通过；最终57包含补充启动检查，不是重复累加数量。夹具仅构造配置和张量/地形几何，不编译或积分MuJoCo模型。未运行含原生物理检查的整个`test_elevation.py`、Viser、PPO或正式训练；不能据此声称策略学会了新动作。

2026-10-29改0.60秒目标、周期预算、高度宽度3厘米和收腿−0.2后，聚合`test_high_step*.py`与`test_walking_audit.py`，55项通过，1.461秒。新增5项`test_high_step_period.py`覆盖其他质量相同时快周期无每秒主循环收益优势、长等待预算封顶、0.6秒17厘米优于快而低的步态、15与17厘米区分增强、相位/最高速度所需15厘米落点兼容，以及非法周期权重/步长上限拒绝。既有事件夹具改为0.36秒支撑+0.24秒摆动、0.12米同足落点；超长摆动无效样本0.30→0.40秒，与新0.36秒上限一致。未运行物理初始化、仿真、Viser、PPO或全量回归；不声称新模型已学会目标。

2026-10-29紧凑收腿修正后，聚合`test_high_step*.py`与`test_walking_audit.py`共50项通过，1.555秒；初次48项通过后补充航向不变性和未承重空摆不伪造事件两项。新增`test_high_step_coordination.py`9项覆盖运动站姿归零/静止保持、高位远伸而非抬高扣分、低位/支撑排除、反序通道、髋/膝幅度与膝部位移、整机平移不冒充抬膝、航向旋转不变、局部重置/非有限/失效地面和任务接口隔离。将旧测试写死的抬腿0.30断言改为正奖励契约，保留用户当前源码0.34。无物理初始化、Viser、PPO、长训练或全量回归，不代表已经学会目标髋膝配合。

2026-10-28温和偏航与接触力约束后，聚合`test_high_step*.py`及`test_walking_audit.py`，41项通过，1.469秒。新增6项`test_high_step_contact_force.py`覆盖正常承重/阈值不罚、四腿相同且反序通道正确、末帧已离地仍读历史峰、力范数旋转不变/非有限安全、静止/转向/启动排除和任务/接口隔离；既有偏航比较按尺度9/瞬时权重−1更新。静态XML显式质量求和16.1557kg；没有模型物理加载、仿真、Viser、PPO或训练，不证明实际后足峰值已降低。

2026-10-28瞬时偏航−2.0、持续偏航尺度7度/秒后，执行`.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -p 'test_high_step*.py' -q`，25项通过，1.383秒。增加偏航专项数值比较：相同持续误差时新/旧代价比约(10/7)^2，纯侧漂代价不变、正负偏航相同；瞬时偏航带权惩罚更强。将测试中旧抬腿0.24断言更新为用户实际0.30，未改回配置。不跑物理仿真、Viser、PPO或全量回归，实际偏航改善待用户续训。

2026-10-28持续漂移权重−1.0→−1.5后，执行现有`test_high_step_alignment.py`定向9项，全部通过，0.377秒；配置工厂同时确认新漂移权重−1.5、抬腿权重0.24及目标0.17米。仅为单参数配置与奖励单元验证，未重跑55项或全量、未创建物理环境/Viser、未训练新模型。此前55项结果对应前一权重版本。

2026-10-28二次HighStep纠偏后，通过`unittest.defaultTestLoader.discover`聚合`test_high_step*.py`、`test_walking_audit.py`、`test_translation_rewards.py`、`test_elevation_alignment.py`共55项，1.611秒。增加3项持续漂移回归：不同环境正负偏移均值为0仍逐个扣分、正常2.5Hz交替摇摆与死区允许、重复调用/局部重置/静止及无效值清空。其余52项再次通过；无物理积分、Viser、PPO或长训练，不声称右偏在实际策略中已消除。

2026-10-28 HighStep直行纠偏：`python -m unittest discover -s src/pointleg/tests -p 'test_high_step*.py' -q`通过21项（既有15项和新增6项）；另通过`unittest.defaultTestLoader.discover`聚合`test_walking_audit.py`、`test_translation_rewards.py`、`test_elevation_alignment.py`共31项。合计52项定向检查，覆盖侧漂左右等价及死区、偏航命令误差、单后腿外张与前足共同偏移、高抬腿不增加横向工作区代价、前后落点中心不互相抵消、四髋持续偏置/交替运动/重置和Flat配置隔离。只运行CPU张量与配置单元，没有创建物理环境、Viser、PPO或正式训练；没有重新运行旧324项全量。

2026-10-27 Viser零范围修复新增`test_velocity_gui.py`，通过`UV_CACHE_DIR=.uv-cache`下的`uv run --locked --no-sync python -m unittest discover -s src/pointleg/tests -p test_velocity_gui.py -q`执行4项，全部通过。使用实际HighStep play配置和模拟GUI句柄，验证零轴只读、单向符号、低于0.1的小范围、双向轴回调独立和三轴归零。没有启动Viser服务、初始化物理环境或执行仿真；不声称真实浏览器端已验收，也没有重新运行先前324项全量。默认uv缓存写入失败后改用项目内缓存，未安装依赖。

2026-10-27奖励审查修复后实际执行`.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -q`，324项通过，21.644秒。包含10项`test_walking_audit.py`回归：真实配置的力坐标契约、小幅抬腿非零反馈、悬停不重复得分、对角接触优于同侧步态、障碍区放宽、失败周期计入验收分母、分足验收不能隐藏坏腿、评估速度、动作代价不饱和及接口隔离。另已执行HighStep15项、Elevation71项定向检查。只做单元/配置/数值检查，没有物理积分、Viser、PPO、长训练或已验收策略；旧测试中的人为世界Z力不能再冒充默认接触坐标输出。专项文档记录当前版本。

2026-10-27 HighStep目标由140改为170毫米后，重新执行同一`test_high_step.py`定向检查，15项通过；测试确认170毫米比分别140毫米、低抬腿和超高抬腿评分高，最大落脚位移仍140毫米。未运行任何仿真、PPO或完整回归。

最新验证约束：2026-10-27用户要求不再自动跑仿真；今后默认只做源码、配置和定向单元检查，除非用户另行明确要求。HighStep随后新增俯仰和四腿平均构型约束，实测`test_high_step.py`15项通过；没有重跑全量、仿真或PPO，下面的310项和96步报告均为姿态修改前的历史结果。

2026-10-27 HighStep：新增11项专项测试，点足全量310项通过；覆盖高度/周期评分、时步补偿、真实支撑、点地/驻足/长跨步/弹跳门控、接触排序、局部重置和4096环境真实命令采样。`tools/check_high_step.py --steps 96 --checkpoint-file pretrained/flat_9000_transfer/model_0.pt`完成4环境真实GPU诊断，45/325/12、数值有限和物理支撑通过；旧Actor此短样本没有记录到合格起脚后的完整循环，不能据此声称新步态已学会。报告`artifacts/high_step_check.json`，`policy_validation=false`。

从相同基础执行4环境、每轮8步、两次PPO更新并保存的调试运行位于`logs/rsl_rl/pointleg_high_step/2026-10-27_19-35-00_high_step_smoke`。这是训练/保存链路验证，不是正式训练或爬梯能力验证。后续对HighStep以0.15、0.20、0.25米/秒分别重复测试，检查四足周期、峰值、落点、支撑、滑移及实际前进。

2026-10-27梅花桩奖励调整：`.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -p 'test_*.py' -q`实测282项通过，其中梅花桩23项；`tools/check_stepping_stones.py --steps 64`实测7环境覆盖2～8桩、105/381/12、真实支撑与有限观测/奖励通过，报告`artifacts/stepping_stones_reward_check.json`。首次GPU运行暴露旧`progress_ready`被替换为Python布尔值，修复并加入跨步/局部重置回归后重跑通过。未训练策略或进行正式成功率评估。

2026-10-27平台Jump入口：`pointleg evaluate PointLeg-Jump-v0 --checkpoint-file PATH --profile nominal --episodes-per-case 100 --output artifacts/jump_platform_nominal.json` 走 `evaluate_jump.py`，默认50种真实几何组合，`--final-only`只测两种满难度；robust保留物理扰动但出生姿态固定。历史 `--wall-index` 和330mm目标墙不再适用。已执行264项点足、16项Sim2Sim/导出测试、Jump 160步GPU、两次PPO更新、50用例各一次诊断和ONNX导出；调试成功率为0，不能作正式验收。具体证据见 `docs/JUMP_TRAINING.md` 和调整记录。

核对日期：2026-10-26。本页给出后续开发的验证入口。初始记忆整理只读资料；后续代码调整已实际执行回归与短仿真，结果见文末和调整记录。本轮记忆补录没有重新训练或重跑这些检查。

## 验证层次

| 检查 | 能证明什么 | 不能证明什么 |
|---|---|---|
| 配置读取、任务 list | 注册和配置构造成功 | 场景能仿真、策略有效 |
| 单元测试 | 对应几何或逻辑在测试条件下正确 | 长时间稳定、实机有效 |
| CPU `--check-model` | 模型和零动作控制链可运行 | 学会行走或越障 |
| 短步 GPU 仿真 | 指定场景短时数值、接触与接口正常 | PPO 更新有效或正式通过率 |
| `--debug` 训练 | 采样、PPO 更新与保存链路 | 策略学会任务 |
| 固定评估报告 | 指定模型、配置、用例下的结果 | 未覆盖方向、真实比赛或实机结果 |

## 回归入口

在工程根目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

根 `tests/test_suites.py` 只汇总 `src/pointleg/tests/` 与 `sim2sim/tests/`，不自动执行整个 `mjlab/tests/`。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\src\pointleg\tests -p test_stepping_stones.py -v
.\.venv\Scripts\python.exe -m unittest discover -s .\sim2sim\tests -v
```

框架目录使用 pytest；若修改了紧凑布局，额外运行其对应测试，并先确认项目环境具备 pytest：

```powershell
.\.venv\Scripts\python.exe -m pytest .\mjlab\tests\test_compact_terrain_layout.py -q
```

| 变更范围 | 优先查看的测试 |
|---|---|
| 机器人、动作顺序、电机 | `test_pointleg_model.py`、`test_pointleg.py`、`test_motor_spec.py` |
| Rough 课程与配置 | `test_rough_training.py`、`test_translation_rewards.py`、`test_turn_rewards.py` |
| 混合障碍与受阻抬脚 | `test_mixed_obstacles.py`、`test_obstacle_recovery.py` |
| 摆腿峰值统计与状态重置 | `test_swing_metrics.py` |
| 楼梯、坡面、跳跃、迁移 | `test_elevation.py` |
| Elevation侧漂、通道纠偏与四髋持续偏置 | `test_elevation_alignment.py` |
| 梅花桩 | `test_stepping_stones.py` |
| 紧凑地图 | `test_terrain_layout.py`、框架 `test_compact_terrain_layout.py` |
| PT/ONNX 与 CPU 控制链 | `sim2sim/tests/test_pointleg_sim2sim.py`、`test_pointleg_export.py` |

## 场景诊断

梅花桩七档真实仿真入口为 `.venv/Scripts/python.exe tools/check_stepping_stones.py --steps 64`，报告中的`policy_validation=false`表示它只检查接口、接触和短时数值，不是策略验收。默认报告保存在`artifacts/stepping_stones_reward_check.json`。

高程诊断入口见 `tools/check_elevation.py`，会实际创建仿真，不是只解析配置：

```powershell
.\.venv\Scripts\python.exe tools/check_elevation.py --steps 64
.\.venv\Scripts\python.exe tools/check_elevation.py --steps 16 --viser-gui
```

查看地形而不加载策略：

```powershell
.\.venv\Scripts\python.exe -m pointleg.run play PointLeg-Elevation-v0 --agent zero --num-envs 1 --viewer viser
```

`sim2sim/check_training.py` 还提供训练、恢复、导出和回放闭环检查；`--pose-only` 只是 CPU 姿态检查。执行前查看其 CLI 与 [B 类修复记录](../docs/B_FIXES.md)。

## 固定评估

```powershell
.\.venv\Scripts\python.exe -m pointleg.run evaluate PointLeg-Rough-v0 --checkpoint-file "模型路径\model_xxx.pt" --profile nominal --episodes-per-case 100 --output artifacts/rough_nominal.json
.\.venv\Scripts\python.exe -m pointleg.run evaluate PointLeg-Rough-v0 --checkpoint-file "模型路径\model_xxx.pt" --profile robust --episodes-per-case 100 --output artifacts/rough_robust.json

.\.venv\Scripts\python.exe -m pointleg.run evaluate PointLeg-Elevation-v0 --checkpoint-file "模型路径\model_xxx.pt" --profile nominal --episodes-per-case 100 --output artifacts/elevation_nominal.json
.\.venv\Scripts\python.exe -m pointleg.run evaluate PointLeg-Jump-v0 --checkpoint-file "模型路径\model_xxx.pt" --profile nominal --episodes-per-case 100 --wall-index 0 --output artifacts/jump_nominal.json
```

普通任务走 `evaluate.py`，Elevation/Jump 走 `evaluate_elevation.py`。nominal 关闭物理、重置和观测扰动；robust 保留随机化，二者都冻结课程。当前目标为每用例 nominal ≥90%、robust ≥80%。

通用评估种子为 `104729,130363,155921,181081,205759`；普通固定指令为 `(0.4,0,0)`，RoughMixed 为 `(0.25,0,0)`。报告包含 checkpoint 哈希、种子、地形、误差、进度、足滑、接触过载与能耗等。

通用固定评估主要检查正向通行，不覆盖完整全向步态验收；混合任务还需单独检查反向跨栏。Elevation 检查四类行走地形的五档；Jump 可用墙索引 0、1、2 分别检查单墙，并额外包含目标墙。正式验收前确认评估覆盖的实际几何等级，不能只读取逻辑等级。

梅花桩虽然会被 CLI 分发到通用 evaluate，但其 60 s 时限、七档课程、reset 事件与任务成功规则尚未被该评估器专门处理，不应套用上述通用结果宣称验收。

## 故障定位

| 现象 | 优先核对 |
|---|---|
| `No module named pointleg` | 当前 Python 是否为项目 `.venv`、包是否已安装；不要临时改搜索路径掩盖安装问题 |
| 任一命令在导入时失败 | 所有已启用任务的配置构造；注册发生于 `pointleg` 导入时 |
| 关节顺序异常 | XML 自然顺序、十二个关节名、动作目标顺序 |
| checkpoint shape 不匹配 | 点足与轮腿来源、梅花桩扩展观测、当前网络结构 |
| 高程续训缺课程或几何报错 | 是否应使用 `transfer`，是否跨任务、跨 warmup 或楼梯布局版本 |
| GPU 内存不足 | 显式降低环境数，核对 rollout、接触容量与附加射线开销 |
| contact / constraint 溢出 | 场景真实接触数及缓冲；特别检查满排圆环，不能直接忽略警告 |
| 修改参数后无变化 | 是否仍为旧训练进程、是否选错共享日志目录、是否已重新评估 checkpoint |
| 高度奖励与终止不一致 | `mdp/height.py` 的局部地面、扫描有效性、出生区域与模型姿态 |
| 梅花桩或跳跃漏掉短时触地 | 物理子步接触采集、接触数据字段、锁存与重置逻辑 |
| 转向或横移指标看似为零 | 对应指令是否实际被采样；条件指标无样本时也可能记零 |

## 记录实验的最小信息

记录日期、任务 ID、代码提交及未提交差异、完整命令、checkpoint 路径与 SHA256、地形和观测约定、随机种子、报告路径及失败原因。写明是配置检查、短步诊断、训练闭环还是正式评估；不能把历史文档里的测试通过数量当作当前代码结果。

## 本对话后续代码调整的实际验证

2026-10-27恢复兼容性修正：299项点足测试通过。只读加载真实Elevation `2026-10-27_08-36-41/model_2499.pt` 验证直接resume，Actor/Critic逐张量一致，迭代2499、累计步数60000、17项优化器状态及课程窗口/上限完整恢复，当前新奖励配置保留；仅奖励版本不同发提示，不拒绝恢复。未运行新的PPO更新、未覆盖检查点、未停止用户进程。任务/几何/课程实际不兼容检查仍有效。

2026-10-27最新Elevation“小步高净空”版本2：点足全量298项通过，含新增16项步态检查；Elevation 96控制步真实GPU检查通过，布局4×4、接口45/325/12保持。使用既有 `walking_pretrained_v1/model_0.pt` 严格迁移网络权重，输出 `gait_v2_check_20261027/model_0.pt`，再完成5环境两次PPO更新、有限损失和检查点保存。未启动正式长训练，不代表已经学会第三张参考图风格或通过最高台阶。新旧奖励及模型哈希见调整记录。

2026-10-27，Elevation二次纠偏和低摆腿净空代价：专项54项、全量246项点足测试通过；命令为 `.\.venv\Scripts\python.exe -m unittest discover -s src/pointleg/tests -p 'test_*.py' -q`，临时文件目录放在项目 `artifacts/test_tmp`。`tools/check_elevation.py --task PointLeg-Elevation-v0 --steps 64` 真实GPU检查通过，使用 `.venv/warp-diagnostics-cache`；验证4×4布局、45/325/12接口、观测奖励有限及物理支撑。测试覆盖局部地形参考、正常起落脚不连续误罚、上阶净空、支撑及立面区分、事件只结算一次、控制周期补偿、左右对称、状态重置和非有限输入。未启动正式长训练，不代表已改善新模型通过率。

同日用最新1000轮目录中的 `model_999.pt` 另跑5环境、160控制步旧Actor短回放：新代价有限，160次有效完成摆腿及峰值样本，平均峰值0.014182米，低于6厘米目标的事件160次，近立面足采样14个。使用临时 `python -c` 诊断，未改写检查点；模型路径和SHA256见调整记录。该统计是混合地形短采样，不是新模型正式成功率。

2026-10-26，Elevation纠偏调整后执行：

```powershell
uv --cache-dir ./.venv/uv-diagnostics-cache run --locked --no-sync python -m unittest discover -s src/pointleg/tests -p "test_*.py"
```

当时233项通过；只覆盖 `src/pointleg/tests`，不包含Sim2Sim和框架pytest全集。专项摆腿统计8项、Elevation纠偏6项分别通过。实际用 `tools/check_elevation.py --task PointLeg-Elevation-v0 --steps 64` 完成GPU检查，布局4×4、45/325/12接口、观测与奖励有限并检测到物理支撑；未做正式策略成功率验收。

摆腿统计另用旧 `model_3500.pt` 进行了5环境160控制步诊断，获得131个完成摆腿样本，临时TensorBoard事件文件写入并读回主曲线、四足及相对机身共6个即时标签。模型路径、哈希及各阶段验证结果见 [调整记录](08_change_log.md)。这些只是记录和运行链路验证，不证明足端已经稳定抬过完整高度轮胎。

### TensorBoard统计口径

- `Episode_Reward/*` 已含权重，回合累计值除最大回合时长；不能把权重当曲线数值，早结束也会改变幅度。
- `Episode_Metrics/*` 通常是时间均值；只有配置为 `reduce="last"` 的项取末值。`failure_*` 很小不等于回合失败率很低。
- `Episode_Termination/*` 是本次重置环境的原因计数，记录器还会平均这些值，不是百分比。
- `Episode_Metrics/effective_progress` 单位是米/秒；`Curriculum/terrain_levels/effective_progress` 是积分距离，单位米。
- 条件分段误差无样本时可能为0，要结合指令采样比例；Mixed的整体障碍成功均值也不是分地形、分等级正式通过率。
- `Metrics/swing_peak_height` 以米记录完成事件，0.18代表18厘米；无落足事件不填0。它与回合口径的同名统计不同，也不是相对障碍的最小净空。
- 角度沟通用度，已有带 `_deg` 或 `_deg_s` 的标签直接按度读取；未转换的原始角度或角速度需乘约57.3后再向用户解释。`Policy/mean_std` 是策略动作分布标准差，不是关节角度。


2026-11-01 08:12实际后台启动用户指定train_4命令：运行目录 `logs/rsl_rl/pointleg_high_step/2026-11-01_08-12-22_train_4`，从固定`2026-11-01_04-10-49_train_3_2/model_10198.pt`直接resume；控制台确认加载路径，已观察迭代10215且已保存model_10200.pt，新增20条逐足支撑轨迹指标已进入TensorBoard。10001次追加更新、间隔100、512环境，预期末次20198。启动器PID 4100，TensorBoard事件记录训练PID30180；进程在后台保留，由用户后续验收，不声称训练完成或轨迹已改善。首次受限后台启动未保留进程、无训练输出；随后在沙箱外隐藏启动成功，无重复训练实例证据。源SHA256 6e38e914384558557b5585e0dfb6a1842af9880d39d52697423dcaaee785931e；checkpoint接口45/325/12和优化器字段已核对、实际resume已执行。25项张量/配置、AST及diff --check通过；启动前未做独立物理测试。MuJoCo现有MULTICCD圆柱/盒接触警告未阻止训练。启动和运行证据见artifacts/train_4_launch.json、train_4_runtime.json、high_step_train_4_preflight.json，stdout/stderr路径见launch元数据。


2026-11-01 12:41实际启动tran_4_2，运行目录`logs/rsl_rl/pointleg_high_step/2026-11-01_12-41-56_tran_4_2`，uv启动器PID48076、事件训练PID28384。控制台确认加载原train_3_2/model_10198.pt，已观察迭代10210并写出model_10200.pt；超15厘米惩罚和逐足路程额度/超高量指标已进入TensorBoard，实际采用新版配置。512环境、10001次追加更新、每100次保存，预期末次20198。后台保留供用户验收；42项张量检查、接口/优化器字段/原模型哈希、CLI、AST与diff检查通过，未做独立回放或物理测试，不声明步态已改善。旧train_4已停并保留最后model_17100.pt。启动/运行/预检证据为artifacts/tran_4_2_launch.json、tran_4_2_runtime.json、tran_4_2_preflight.json。
