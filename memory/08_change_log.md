# 调整与验证记录

## 2026-11-21 将两桩左后下坠惩罚对准上出口台

用户确认回放下坠发生在两桩第 5 拍，左后从 1 号桩跨 40 cm 缝隙上出口台。先前用汇总日志推断为开场支撑，现将这轮扩展限定在左后 1 号桩→两桩出口台，并延续此前左后桩到桩的代价。权重 −8/s，低于顶面 5 cm 满额。

执行固定步态和四项定向集成检查共 42 项，全部通过。检查左后在两桩出口台近边前 1 cm 下坠计罚、越过近边 1 cm 后该项归零、开场支撑和六桩完整步序。未运行训练或物理回放。

## 2026-11-21 将左后下坠检测覆盖到开场支撑

`train_2/model_5998.pt` 所在训练后 100 轮仍为两桩，平均完整换步约 0.00004 次，左后下坠项为零。此前只在左后桩到桩时检查，现在覆盖左后摆动与支撑两种角色。按入口台、出口台和当前关卡有效桩的顶面投影判定间隙下坠。扣分仍在低于顶面时按深度施加，低 5 cm 满额 −8/s。

执行 `.venv/Scripts/python.exe -B -m unittest` 的固定步态和定向集成检查共 42 项，全部通过。覆盖两桩和六桩、首拍左后失去支撑、右后同样下落不触发该项、进入合法顶面及触底终止。未运行物理回放或训练，具体截图时的逐足接触未采集。

## 2026-11-21 足端触底失败扣分翻倍

训练终止日志中 `stepping_failed` 占主导，因此将梅花桩足端触底的终止扣分从 −25 调为 −50，其他非成功终止按原权重结算为 −25 分。实现读取终止管理器本步的 `stepping_failed`，在 `stepping_failure_reward` 乘 2。同回合多腿触底仍为一个失败事件。

执行 `.venv/Scripts/python.exe -B -m unittest` 的三项定向 CPU 检查通过，覆盖足端触底和其他失败的扣分区分、成功与失败同拍仲裁及触底取消覆盖分。未运行仿真或训练，策略行为改善待续训确认。

## 2026-11-21 右后首步目标前移以缩短首次上桩步幅

梅花桩训练和 Sim2Sim 共享的 `FixedStoneGait` 中，`FIRST_RIGHT_REAR_GAP` 从 0.20 m 调为 0.17 m。首步目标在入口台前移 3 cm，对应首次上 2 号桩的目标距离几何缩短 3 cm。核对共享引用，更新原目标断言及文档。本轮为单参数调整，未运行测试、仿真或训练。

## 2026-11-21 左后足桩间下坠

梅花桩的固定步态新增 `stepping_gait_left_rear_gap`，训练按物理子步对左后足底在起点和目标两块桩的顶面投影外、且低于桩顶的下坠深度计罚。此前仅有双摆平均的净空代价等通用项。新项低 5 cm 满额 −8/s，适用左后 1→3、3→5。

使用 `.venv/Scripts/python.exe -B -m unittest` 执行 `FixedGaitTests` 和左后下坠奖励接入、整段课程两项集成测试，40 项通过。覆盖桩间下坠、桩顶高度、起点和目标投影、横向错位及两桩直接上出口。未启动训练或物理仿真。旧模型可作为续训起点，Sim2Sim 中的实际改善待续训后的检查点验证。

## 2026-11-20 梅花桩改为 2、4、6 桩课程

默认课程、回放地形和 `--competition-six-curriculum` 使用 2、4、6 桩地形，固定步态按实际桩数分配目标。旧 3、4、5、6 桩续训上限映射到 2、4、4、6 桩，成功率窗口重新累计。读取 `2026-11-20_15-31-36_train_1/model_999.pt` 确认从两桩续训。

71 项静态及 CPU 检查中 69 项通过，包含本次课程地形、两桩完整步序、升档和检查点迁移。两项机身静止惩罚和前进奖励断言失败在切回旧桩数后同样复现。未启动训练或仿真。

## 2026-11-20 停用提前失败附加罚分

`stepping_early_failure` 权重 −4→0，管理器会跳过零权重项。基础失败 −25 和终止仍生效。同步文档与受影响的测试断言，按单一权重调整规则核对配置和引用，未运行测试或训练。

## 2026-11-20 按用户图片更新六桩步序

六桩由 14 拍改为 12 拍。第二拍改成右前到 2、左后到 1，删除左后的入口台短步。之后左后于第 5、8、11 拍分别到 3、5、出口，右前同时到 4、6、出口。右后在第 4、7、10 拍分别上 2、4、6，第 12 拍补进出口。右后首拍按距右前起步位后方 20 cm 计算。短课程沿用越过末桩即转出口的规则，支撑与同桩避让沿用现有逻辑。

执行 `.venv/Scripts/python.exe -B -m unittest` 运行 `FixedGaitTests` 加 5 项定向集成检查，42 项通过。覆盖图片指定的完整步序、三到六桩的目标、第二拍同桩避让、左右前后支撑交接和全程结束。未启动仿真或训练。

## 2026-11-20 进一步缩短右后首步目标

根据右后腿连杆与默认足位做静态几何核算，固定出生机身及足端横向坐标时，右后足在台面高度伸直最多前移约 36.57 cm。将 `FIRST_RIGHT_REAR_GAP` 从 11 cm 改为 20 cm，默认首步前迈从 39.37 cm 缩短至 30.37 cm。第 5 步目标未动，按默认台上落点与同桩避让估算，后续首次上桩步幅相应增加 9 cm。同步目标断言和文档，未运行测试或物理仿真。

## 2026-11-20 去掉摆动足 3 cm 提前落脚代价

从 `FixedStoneGait` 的目标代价删除未接触时的提前下落分支，保留有效摆动后实际错误承重才计分的逻辑。桩沿净空、真实落点容差和步序不改。下次新训练进程使用。

3 项定向 CPU 检查通过，覆盖空中下降不计目标代价、错误承重按摆动足平均、临边净空参考。未启动物理仿真或训练。

## 2026-11-20 取消梅花桩膝反曲侧的限位重置

在 `PointLeg-SteppingStones-v0` 将继承的关节限位终止替换为任务专用判定，保持原关节选择和 0.02 rad 余量，仅四膝反曲侧不触发限位重置。现行反曲惩罚 −50/s 保持，正常屈曲侧和髋关节限位、其他终止、物理关节范围与动作裁剪保持。

3 项定向 CPU 检查通过，覆盖四膝双向限位、反曲碰限同时髋碰限仍失败、单子步罚分和独立物理失败。未启动训练或仿真。

## 2026-11-20 单纯反膝保留回合并重罚

核对梅花桩配置，反膝角度本身没有终止项，独立的机械硬限位、底盘与地面触碰、姿态失败等保留。原反曲和屈膝不足共享一项并按四腿平均，现保留不足 5°的 −2/s 四腿均值项，另加反曲 −20/s，按每个物理子步最严重的膝计算，反曲 5°满额。接口不变。

3 项定向 CPU 检查通过，覆盖单子步单膝反曲计价、反曲未触发硬限位或任务失败、原屈膝余量、独立物理失败仍有效。未启动仿真或训练。

## 2026-11-19 右后首步目标后移 5 cm

第一拍右后台上目标由右前起步足位后方 6 cm 改到后方 11 cm，默认前迈从约 44.37 cm 减到 39.37 cm。调整共享步态常量 `FIRST_RIGHT_REAR_GAP`。同步目标断言和说明，按单独数值调整规则仅核对源码及参数引用。

## 2026-11-19 增加四足桩列对齐代价

新增 `stepping_foot_row_alignment`，以实际桩心确定左右列横坐标，比较各球足最低点在赛道中的 Y。横偏超过 4 cm 时按平方增加，10 cm 达到 −4/s 满额，按偏差最大的脚计分。左后空中外扩与其余各足的横向偏离都受约束，现有同桩避让目标仍在容差内。行进阶段生效，出口整理和终止豁免。

运行 2 项定向 CPU 检查并通过，覆盖不同环境原点与桩列、左右后足摆动偏离、4 cm 容差、出口和终止豁免、完整步序。未启动物理仿真或训练。

## 2026-11-19 促使左前单摆落地并处理右后提前离地

左前单摆合格抬脚后进入目标水平 5 cm 且未有效承重时，屏蔽本控制步正速度收益，连续等待 80 ms 后逐渐纳入原 −8/s 延迟项，200 ms 满额。左前单摆过程分的最后 25% 在准确落点连续承重 80 ms 后发放。若规定保持支撑的足提前离地，支撑代价和换步阻断继续生效；已经准确落地的左前保留摆动资格，等待支撑恢复后重新累计 80 ms。开场双摆规则保持。

使用 `.venv/Scripts/python.exe -B -m unittest` 执行 `FixedGaitTests` 与 4 项定向集成检查。首轮 40 项通过，悬空等待用例的满额断言跨过最后一个升罚控制周期，调整取样位置后单独复跑通过。未启动物理仿真或训练。

## 2026-11-19 左前支撑专项覆盖全部右后单摆

既有的通用支撑代价和摆动资格判定已经覆盖全部固定阶段，但左前违规屏蔽正向速度原先只识别入口台→第 2 桩。改为依据摆动足掩码判断右后是否独自摆动，覆盖入口台→2、2→4、4→6 和单独上出口台。专项违规计数随之汇总所有右后单摆。开场双摆不触发该规则。

使用 `.venv/Scripts/python.exe -B -m unittest` 运行 `FixedGaitTests` 及右后单摆速度门控集成检查，共 38 项通过，验证所列右后阶段、开场双摆放行和恢复支撑后的重新计分。未启动仿真或训练。

## 2026-11-19 纠正右后桩列与区分反膝代价

固定步态第 8 步原先把右后从第 2 桩送到异侧第 3 桩。修正为右列第 2→4→6 桩，纵向中心间距 60 cm，超出本级桩数则转出口。沿用已有同桩避让。原膝角代价在伸直时已经封顶，现对越过伸直位置的反曲追加平方代价，10°封顶，5°正向余量和名义权重 −2/s 保持。训练和 Sim2Sim 共用步态调度，需新进程加载新规则。

使用 `.venv/Scripts/python.exe -B -m unittest` 运行 `FixedGaitTests`、完整步序集成及逐腿膝角检查，共 38 项通过。覆盖各课程的右后序列、六桩第 8 步目标前移 60 cm、三桩转出口、同桩避让及反曲代价。未启动仿真或训练。
## 2026-11-19 梅花桩伪前进退化与重新训练

检查`logs/rsl_rl/pointleg_stepping_stones/2026-11-19_16-19-14_train_2`的TensorBoard事件到第1833轮：`stepping_gait_completed_steps`、`stepping_covered_fraction`、`stepping_success`、`stepping_gait_lifted_fraction`和课程成功率均为0，`stepping_gait_phase`持续为0；同时`stepping_forward_speed_mps`末5轮为约0.319至0.326 m/s，`Train/mean_reward`约−243。该运行以无效前冲后失败替代合法首拍，不能作为续训来源。

修改`src/pointleg/mdp/stepping_stones.py`的`forward_velocity_reward`：固定步态存在时，当前相位尚无任一规定摆动足取得合法抬脚资格，屏蔽正向速度收益，负向速度收益和既有安全屏蔽不变。修改`src/pointleg/config/stepping_stones.py`：前向速度权重+15/s→+5/s，步态过程奖励+2→+12，完整步奖励+4→+12，步态等待代价−2/s→−8/s。接口仍为Actor123/Critic399/动作12，已有检查点网络结构兼容，但旧回报分布不适合作为本次退化运行的续训来源。

按项目验证约束，本次奖励数值与门控修改未运行单元测试、物理仿真或回放；仅检查了修改位置及训练日志。新训练须从`flat_8700_fixed_gait_pretrained/model_0.pt`启动，实际行为和通过率待训练日志复核。

## 2026-11-19 屏蔽右后上桩时左前失撑的推进收益

第 5 步由右后独自从入口台到第 2 桩，固定步态原本要求左前留在第 3 桩。原独立正向速度项只对小腿借力屏蔽正收益，与该阶段左前失撑无联动。现仅在右后入口台至第 2 桩的目标阶段，按已有支撑判据逐物理子步锁存左前违规，阻断整控制步的正速度收益，同时保留后退扣分和原步序资格。新增逐回合阶段违规计数。最新可见训练 `2026-11-19_14-35-47_train_1` 的末轮 999 步序均值为 0、总违规约 4.68，日志没有该次回放的逐足数据。

使用 `.venv/Scripts/python.exe -B -m unittest` 运行 `FixedGaitTests` 及 2 项定向集成检查，共 36 项通过。检查一个子步失撑屏蔽整控制步、后退仍扣分、恢复支撑重新发放、其他环境和完整步序。未启动仿真或训练。

## 2026-11-19 接入行进保高并复核固定步态奖励

`PointLeg-SteppingStones-v0` 新增 `stepping_body_sink`，在行进阶段检查根节点与前髋中点相对出生位的下沉，取较大者，2 cm 死区、8 cm 满额、−8/s，不按前进速度折减。新增最大下沉指标。当前配置和旧版奖励按动作目的逐项比对，记录水平/航向、左前过伸、换脚、推进和计量差异，并核正当前速度 +15/s、停滞 −50/s 的文档。观察和动作接口保持，加载现有检查点的新训练进程采用新配置。

使用 `.venv/Scripts/python.exe -B -m unittest` 执行 `FixedGaitTests` 和 3 项定向检查，共 37 项通过。检查静止下沉、前俯、出口阶段与终止屏蔽、抬头及完整跨桩步序。未启动物理仿真或训练。

## 2026-11-19 梅花桩起步抬头、撞侧壁和抬脚净空

`PointLeg-SteppingStones-v0` 起步 1 s 内的 −6/s 项扩展到前髋支点中点，利用根姿态和角速度计算前端的高度与速度，新增前端峰值指标。新增 `stepping_foot_side` −8/s，足端法向力从 2 N 至 20 N 线性计罚。有效抬脚从 5 cm 提至 7 cm，目标桩及出口台近侧边缘前后 3 cm 的净空参考至少 4 cm。回放与训练共用固定步态阈值。接口维度保持，需新训练进程采用新奖励。

使用 `.venv/Scripts/python.exe -B -m unittest` 执行 `FixedGaitTests` 和 5 项定向集成与奖励检查，共 39 项通过。覆盖根节点不升时前端抬头、侧壁与顶面识别、桩沿净空及完整步序。未执行物理仿真、回放或训练。

## 2026-11-19 将提前下落惩罚扩展到每步

`PointLeg-SteppingStones-v0` 将仅用于第二步右前足的空中提前下落代价扩展到所有阶段的指定摆动足。按各足目标计算纵向差距，入口台、桩和出口台均覆盖。空中提前下落和错误落地共用 `stepping_gait_target`，按摆动足数平均，权重仍为 −3/s。实现位于 `src/pointleg/mdp/fixed_stone_gait.py`。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_fixed_stone_gait.FixedGaitTests`，33 项 CPU 检查通过，包括第一步双摆中一只足或两只足提前下落的计分。未启动仿真或训练。

## 2026-11-19 补充右前首次跨桩的提前下落反馈

`PointLeg-SteppingStones-v0` 第二步右前默认前移 57.77 cm 到第 2 桩中心，47.77 cm 到近侧边缘。根据地形和既有默认姿态足位复核，缺少最新回放足位数据，未测得实际失败步幅。保留桩心目标。在 `src/pointleg/mdp/fixed_stone_gait.py` 的现有落点代价中加入空中下落过早反馈，仅第二步右前足有效，沿用 −3/s 权重。步序和接口保持。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_fixed_stone_gait.FixedGaitTests`，32 项 CPU 检查通过。新增用例覆盖起抬不罚、桩前降到台面高度受罚、桩心上方下落不罚。未启动仿真或训练。

## 2026-11-18 调整梅花桩固定步态的开场三步

`PointLeg-SteppingStones-v0` 的第二步由右前与左后双摆改为右前独自上第 2 桩，第三步由右后台上短步改为左后台上前移 8 cm。第一步右后原地起脚前移 6 cm 改为落在右前起步足位后方 6 cm，按入口台边距裁剪，左前仍上第 1 桩。代码在 `src/pointleg/mdp/fixed_stone_gait.py`，训练与 Sim2Sim 共用。123/399/12 接口保持，运行中的进程需重启才读取新步序。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_fixed_stone_gait.FixedGaitTests`，31 项 CPU 检查通过，覆盖开场目标、入口台边距与完整步序。未启动物理仿真或训练。

## 2026-11-18 单纯数值调整采用源码核对

用户明确要求以后单纯小幅数值调整不跑测试。已在 AGENTS.md 记录，适用于奖励权重、距离、高度及时间阈值，仅核对修改位置、参数引用和必要的说明同步。本轮只更新项目规则与记录。

## 2026-11-18 机身前进奖励与强停滞代价

用户要求重新加入机体匀速前进奖励并加重静止代价。核对固定步态配置确实没有前进速度奖励。复用 `forward_velocity_reward` 并将权重设为+5/s。添加 `stepping_body_stall` −12/s，复用既有机身2 cm前沿进度与控制步去重计时，0.30～0.60 s线性升罚。终点整理、零指令、成功和物理终止后豁免。满额停滞损失超过等待带来的提前失败代价每秒4分下降。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b pointleg.tests.test_fixed_stone_gait pointleg.tests.test_stepping_stones`，124项通过。回归覆盖目标速度、静止、小腿借力、正负摇摆、2 cm进度清零、重复读取、局部重置与终止豁免。未启动训练或推进物理仿真。

## 2026-11-18 提高有效摆动净空至 5 cm

用户指出2 cm最低抬脚过小。将 `src/pointleg/mdp/fixed_stone_gait.py` 有效摆动峰值门槛改为5 cm，双摆同时离面高度和抬脚进度反馈采用同一常量，期望净空轨迹峰值同步。高度比较容差1e−6，避免25 cm世界高度的浮点边界误判。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b pointleg.tests.test_fixed_stone_gait`，49项通过，包含4.9 cm不足、5 cm达标、双摆高度必须同时满足的回归。另执行静态 `test_fixed_gait_policy_selects_shared_replay_targets_and_observations` 通过。未启动续训或推进物理步骤，新进程采用新门槛。

## 2026-11-18 静态模型测量跨桩目标距离

用户要求测量建模世界足端到台沿和桩心的前进距离。编译当前任务机器人并设置默认关节和入口出生姿态，静态 mj_forward 核对训练与比赛 XML 坐标。左前 x=0.72206705、右前 x=0.72228898；远沿0.80，前两桩中心1.00/1.30。首步27.7933/57.7711 cm，同侧中心距60 cm。用 FixedStoneGait._set_targets 顺序静态更新前六阶段，确认前腿目标正确，后腿台上6/8 cm为独立目标。文档记录具体坐标与公式。

读取训练 `2026-11-18_14-25-04_train_3_audit_2000` 的最终1999轮，末100条阶段0、完整步0、覆盖0，双摆资格均值0.00260736、前进速度0.00112685 m/s。未完成第一阶段，当前短抬腿不能视为小步完成。仅静态 FK、目标张量与日志读取，未改奖励或启动训练、未推进物理步骤。

## 2026-11-17 奖励右后有效跟步

用户观察到右后长期不迈、左前与另一对脚轮流前进。最新 `2026-11-17_18-58-08_restart_train_2` 末 100 次更新两桩覆盖为 100%，左前违规计数 0，终点台到达率 0。原日志没有按腿计前迈，无法独立确认右后步数。

`src/pointleg/mdp/stepping_stones.py` 将左前下一次计分与右后实际向前跨到新支撑面并承重 80 ms 关联。右后先迈可抵一次。右前、左后独立计步。左前在等待期抢抬则屏蔽该次落脚收益。配置 `stepping_right_rear_forward_step` 权重 +2，`stepping_right_rear_overdue` 权重 −2/s，宽限 1 s。日志按回合统计右后有效步和等待时间，Sim2Sim 同步覆盖判定并记录待跟步标志。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b` 运行梅花桩、迁移与九项静态 Sim2Sim 检查，共 88 项通过。末后单独复查新增的 Sim2Sim 待跟步断言通过。未启动训练或物理回放。

## 2026-11-16 缩短左前首次上台目标并保留剩余桩目标

用户反馈左前腿伸直悬空，截图中足端与台面投影重叠。原终点落脚区域覆盖 80 cm 台面的近沿后 3～77 cm，原前伸代价也未覆盖终点区。原跳桩落脚虽然屏蔽奖励，实际支撑仍会作为新目标的参考面。将首次上终点台可取范围收为近沿内 5～20 cm，单独维护按序的左前目标参考面。左前摆动正向屈膝不足 20° 纳入原 `stepping_left_front_overreach`，与空降及前伸代价逐子步取较大值。训练与 Sim2Sim 共用目标推进规则。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b` 运行梅花桩逻辑、检查点迁移和九项静态 Sim2Sim 测试，共 87 项通过。覆盖跳桩后保持原目标、回踩后恢复目标更新、靠近台沿与伸向台中央的对照、上台站稳后的豁免以及摆动近直膝。未启动训练或物理回放。

## 2026-11-16 左前足踏空前计罚

用户确认左前前伸后会踏空。原奖励按前缘越界和跳桩后的落脚计分，未覆盖偏后或横向在缝隙中下降。`src/pointleg/mdp/stepping_stones.py` 在左前足无合法顶面接触、向下运动且距目标顶面 7 cm 以内时，按偏出目标安全区的距离计罚。目标为下一根左桩或终点台，四周内缩 3 cm，偏差尺度 5 cm。和原前伸代价取较大值，最多每秒扣 4 分。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b` 运行梅花桩逻辑、检查点迁移和九项静态 Sim2Sim 测试，共 86 项通过。覆盖对准桩面、偏后和偏侧空降、抬脚、高位摆动、超前惩罚不叠加及终点台边缘。未启动训练或物理回放。

## 2026-11-16 限制梅花桩左前摆动越过下一根目标桩

根据用户报告的左前伸脚过远问题，核对原前迈按任何更靠前支撑面计分，原左前违规惩罚只针对右后支撑时机。无法从单张截图核对具体落脚面。将实际赛道前方最近左桩作为左前目标，越过前缘 2 cm 开始计罚，以 7 cm 尺度平方计分，`stepping_left_front_overreach` 权重 −4。跨过目标的落脚屏蔽左前覆盖、中心和前迈。其他腿仍可正常给该桩计覆盖。Sim2Sim 与训练共用目标判断。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b` 运行梅花桩逻辑、检查点迁移和九项静态 Sim2Sim 测试，共 84 项通过。覆盖正常落脚、摆动越界、跳桩收益屏蔽及其他腿照常计分。未启动训练或物理回放。

## 2026-11-16 左前违规抬脚每次扣 10 分

根据用户指定值，将 `src/pointleg/config/stepping_stones.py` 的梅花桩 `stepping_left_front_liftoff` 权重从 −5 调至 −10，同步奖励文档和实际扣分断言。

运行 `.venv/Scripts/python.exe -B -m unittest -q -b pointleg.tests.test_stepping_stones`，67 项 CPU 单元测试通过。未启动续训或物理回放。

## 2026-11-16 提高左前违规抬脚扣分

根据用户反馈，将 `src/pointleg/config/stepping_stones.py` 中 `stepping_left_front_liftoff` 权重从 −4 调至 −5，并同步文档和实际扣分断言。

运行 `.venv/Scripts/python.exe -B -m unittest -q -b pointleg.tests.test_stepping_stones`，67 项 CPU 单元测试通过。未启动续训或物理回放。

## 2026-11-16 提高无有效前迈与机身晃动扣分

根据用户要求，将 `src/pointleg/config/stepping_stones.py` 中梅花桩 `stepping_no_forward_step` 权重从 −5 调至 −8。超时静止每秒扣分从 1 提高到 1.6，四足在原面晃动的总扣分上限从每秒 5 提高到 8。同步文档和断言。

运行 `.venv/Scripts/python.exe -B -m unittest -q -b pointleg.tests.test_stepping_stones`，67 项 CPU 单元测试通过。未启动训练或物理回放。

## 2026-11-16 梅花桩无有效前迈与机身晃动惩罚

核对 `logs/rsl_rl/pointleg_stepping_stones/2026-11-16_10-33-58_new_train_1`。恢复自第 14 轮 `model_41200.pt`，末 200 次平均覆盖约 0.01%，右台到达率为 0，平均前进速度约 0.118 m/s，平均前进速度奖励约 +3.06 分每回合。用户要求惩罚足端留在原面而机身运动。

`src/pointleg/mdp/stepping_stones.py` 按物理子步累计，配合 `src/pointleg/config/stepping_stones.py` 的 `stepping_no_forward_step` 权重 −5。起步阶段结束或满 1 s 后，超过 0.60 s 无计分前迈，每秒扣 1 分。同一时刻四足仍在原支撑面，按机身线速度和角速度最多额外每秒扣 4 分。有效前迈清除计时，终点整理豁免。记录回合累计超时指标。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b` 运行梅花桩逻辑、检查点迁移及八项静态 Sim2Sim 测试，共 82 项通过。核对宽限、转动和平移、原地静止、同面抬放、跨面前迈、右台整理和局部重置。未启动训练或推进物理回放。

## 2026-11-16 按支撑面确认梅花桩迈步

`src/pointleg/mdp/stepping_stones.py` 和 `sim2sim/stepping_stones_replay.py` 共用进度。以入口、各桩、右台的赛道顺序判定前迈，初始支撑需要 20 ms，跨面落脚有效承重 80 ms 确认。原 15 cm 位移和离地时长门槛由支撑面转换取代。右后首个向前跨面完成后才结束开场豁免。后续在原面恢复支撑可解除左前保护，该次恢复没有前迈收益。

使用 `.venv/Scripts/python.exe -B -m unittest -q -b` 运行梅花桩逻辑、检查点迁移及八项静态 Sim2Sim 测试，共 80 项通过。覆盖同面挪动、卸力恢复、跨向前桩、落到右台、退回重复落脚、落脚承重中断及左前违规门控。未启动训练或推进物理回放。

## 2026-11-16 四足承重进度接入梅花桩策略

读取 `logs/rsl_rl/pointleg_stepping_stones/2026-11-15_23-42-47_new_train_112`。实用权重 −4，从第 14 轮 `model_41200.pt` 起训。前 50 次平均覆盖 89.32%，末 200 次 0.078%，平均前进速度 0.00845 m/s、右台到达率 0。槽 6 的 `model_42700.pt` 哈希与第 16 轮一致。

`src/pointleg/mdp/stepping_stones.py` 以四足连续有效承重替换训练观测中的二值触地。与换步判定共用合法顶面、足端 ≥8 N、小腿 <8 N 的计时，80 ms 饱和。检查点标记配对 MJLab play、Sim2Sim 和 ONNX 观测。Actor 105、Critic 381、动作 12。

使用 `.venv/Scripts/python.exe -B -m unittest -q -b` 运行梅花桩逻辑、检查点迁移及八项静态 Sim2Sim 测试，共 80 项通过。另用合成检查点导出并加载 ONNX，最大绝对推理误差约 1.2e−7。模拟启动函数验证奖励文档中的续训命令和 Tyro 配置。CPU 核对第 18 轮 `model_44200.pt` 可读且含优化器状态，SHA256 `BE21EC6B2FD53A4AC7A22F9E2291029A75DD95604E075871946567763C5965DB`。未启动续训或物理回放。

## 2026-11-15 左前违规抬脚每次扣 4 分

根据用户反馈，将梅花桩 `src/pointleg/config/stepping_stones.py` 的 `stepping_left_front_liftoff` 权重从 −3 调至 −4，并同步奖励说明和实际扣分断言。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，64 项 CPU 单元测试通过。未加载检查点、启动续训或进行物理回放。

## 2026-11-15 静止退化后恢复左前违规扣分

读取 `logs/rsl_rl/pointleg_stepping_stones/2026-11-15_21-52-31_new_train_111` 的 TensorBoard 标量和训练参数。六桩间隙 5 cm，前 50 次平均覆盖 89.77%，更新 41300 至 41349 平均覆盖 13.58%，末 200 次平均覆盖 0.01% 且右台到达率 0。前 50 次平均左前违规扣 29.51 分，覆盖、中心、前迈合计 37.52 分。与 `new_train_18_left_front_handoff` 相比，环境配置只有单次扣分从 3 提到 8 的差异，起始检查点不同。18 轮末 200 次覆盖 99.58%、右台到达 96.10%、站稳成功 0、平均左前违规 2.28 次。

`src/pointleg/config/stepping_stones.py` 将 `stepping_left_front_liftoff` 恢复为 −3。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，64 项 CPU 单元测试通过。CPU 核对 `2026-11-15_20-34-39_new_train_18_left_front_handoff/model_44200.pt` 可读，Actor 105、Critic 381、含优化器状态，SHA256 `BE21EC6B2FD53A4AC7A22F9E2291029A75DD95604E075871946567763C5965DB`。未启动续训或物理回放。

## 2026-11-15 提高左前违规抬脚扣分

针对右后失撑期间左前仍然抬脚的问题，将 `src/pointleg/config/stepping_stones.py` 中梅花桩 `stepping_left_front_liftoff` 的单次扣分从 3 分提高到 8 分。奖励文档与测试断言同步。

运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，64 项 CPU 单元测试通过，覆盖实际单次扣分和松脚重抬计罚。未加载检查点、启动续训或进行物理回放。

## 2026-11-15 首步豁免与右后失撑即时限制

用户明确希望开场首步豁免，之后右后未承重期间左前不能迈步得分。旧版首步也计罚，右后顶面低载荷和左前离面各有 20 ms 确认。共用 `LeftFrontHandoff` 改为首个右后完整换步后生效。首次右后换步需依次有效支撑 20 ms、失撑 20 ms、重新有效支撑 80 ms，避免一次接触波动消耗豁免。之后右后失撑首个子步进入限制，连续支撑 80 ms 后解除。左前离面立即单次 −3，落脚收益照旧屏蔽。Sim2Sim 增加 `rr_first_step_complete`。TensorBoard 的违规计数按新口径排除首步。

目标 106 项 CPU 单元与静态 Sim2Sim 检查通过，覆盖首步得分、首次短波动豁免、后续瞬时 6 N、左前每次违规只计一次、重置及 CSV。核对 `new_train_18_left_front_handoff` 末 200 次平均每回合 2.28 次确认重叠，地形间隙 5 cm；槽 6 权重仍为 `new_train_16_six_gap/model_42700.pt`，SHA256 `E02F1607206F3B70C1317BA6132FD912426E5DB16A64F95EFC14E7F84AF7DFBE`。本轮未启动训练或推进物理回放，未验收实际步态。

## 2026-11-15 取消右后换步期间的左前违规落脚收益

训练和 Sim2Sim 共用 `LeftFrontHandoff`。右后足有效承重 20 ms 后，离开合法顶面或连续低载荷 20 ms 进入换步，恢复有效承重 80 ms 结束。低载荷确认时从第一失撑子步累积左前离面重叠，累计满 20 ms 单次惩罚从 −1 提高到 −3，持续惩罚仍为 −12 分/秒。标记屏蔽该次左前落脚的覆盖、中心和前迈，左前需再有效承重 80 ms 才允许以后落脚计分，覆盖按新确认的合法承重事件计入。其他足和成功终止条件沿用原规则，违规不触发重置。TensorBoard 新增 `stepping_left_front_overlap_count`，CSV 增加承重、换步状态、重叠事件及左前落脚计分。

执行 `.venv/Scripts/python.exe -B -m unittest` 运行梅花桩、迁移、基础配置及六项 Sim2Sim 静态检查，共 105 项通过。覆盖左前先抬、右后顶面 6 N、短暂卸力不足 20 ms、右后先落地、同拍其他腿有效踩桩、左前恢复后下一步能计分与局部重置。已核对 `new_train_18_left_front_handoff` 的 Tyro 解析及源权重 `2026-11-15_16-59-04_new_train_16_six_gap/model_42700.pt`，Actor 105、具备优化器状态，SHA256 `E02F1607206F3B70C1317BA6132FD912426E5DB16A64F95EFC14E7F84AF7DFBE`。本轮未启动续训或物理回放。比赛场景验收按 10 cm 间隙评估 20 次，至少 18 次完整跨过六桩且重叠离面事件为 0，右台站稳率单列。

## 2026-11-15 梅花桩右台站稳后成功

此前全部覆盖且四足分别在右台接触 20 ms 后立即成功。按用户要求改为先进入停步整理，保留 105/381/12 接口。整理阶段速度指令为零，原阶段和进度观测编码第四阶段及连续站稳进度。成功须四足同时在右台有效承重，机身高度距名义站高 ≤4 cm，倾角 ≤10°，线速度 ≤0.05 m/s、角速度 ≤0.15 rad/s，每个腿关节距默认站姿 ≤0.45 rad，连续 0.70 s。失撑或姿态超界使该段计时归零，整理超时 5 s 按失败结束。新增 `stepping_exit_stability`，有效承重比例及姿态质量按回合最高值的增量给分，最多 +4。TensorBoard 记录到达率和最长站稳时长，Sim2Sim 进入整理后仍调用策略，成功或超时后结束。

102 项配置、状态逻辑、迁移及 Sim2Sim 静态检查通过，涵盖错位姿态、短暂卸力、计时重启、达标与超时。建议以 `--competition-six-curriculum` 从 `2026-11-15_16-59-04_new_train_16_six_gap/model_42700.pt` 续训为 `new_train_17_exit_stability`。已验证 CLI 参数和检查点优化器状态，SHA256 `E02F1607206F3B70C1317BA6132FD912426E5DB16A64F95EFC14E7F84AF7DFBE`。源训练末 100 次已达 10 cm 间隙，按旧判据的成功率 98.4%。本轮未启动续训或推进 Sim2Sim 物理步，尚未验收真实站稳行为。

## 2026-11-15 六桩固定间隙退化与分级恢复

读取 `new_train_15_competition_six` 的 TensorBoard 全量 1501 次更新，成功率为 0。首 50 次平均覆盖率 75.1%、前向速度 0.665 m/s，末 200 次平均覆盖率 0%、前向速度 0.089 m/s，前迈奖励为 0，起步 5 cm 奖励接近满额。确认 Sim2Sim 当前槽 6 的 `model_42700.pt` 与该次训练末检查点 SHA256 `D256D4D04DAC35CB82E601F0E00A0E25613AB3A4C5CF72080A065CB80E890A50` 一致。实际配置与源训练的差异为地形及课程。现有记录无法把退化归因到单个奖励。

`--competition-six-curriculum` 固定 6 根 20 cm 方桩，从 5 cm 边缘间隙练起，每级加 1 cm，最终达到比赛的 10 cm。采用当前最高开放等级的最近 1024 个结束回合，成功率 ≥80% 时升级，低等级成功不计入该窗口。`stepping_stone_gap_m` 记录每回合实际几何间隙。96 项 CLI、地形、课程、接口和 CPU 测试通过。核对 `new_train_16_six_gap` 可从退化前 `new_train_14_support_handoff/model_41200.pt` 恢复，源 SHA256 `D1BCA7F00A85BD45CF70AEDD62C81390AA6B385D7961AD3E8E0558D9F6617427`。本轮未启动新训练或回放。

## 2026-11-15 比赛六桩专项续训

按用户请求增加训练入口 `--competition-six`，在 `src/pointleg/config/stepping_stones.py` 固定六桩比赛尺寸地形并移除该次训练的难度升降级。走 `src/pointleg/run.py` 原训练程序和任务日志，Actor/Critic/动作 105/381/12。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg pointleg.tests.test_b_fixes -q -b`，94 项通过，包含 CLI 覆盖、地形编译、尺寸和运行时中心核对。

于 15:55 后台使用 `.venv/Scripts/python.exe -u -B -m pointleg.run train PointLeg-SteppingStones-v0 --competition-six --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-15_14-26-31_new_train_14_support_handoff --agent.load-checkpoint model_41200.pt --agent.max-iterations 1501 --agent.run-name new_train_15_competition_six --agent.logger tensorboard --agent.upload-model False` 启动。源 SHA256 `D1BCA7F00A85BD45CF70AEDD62C81390AA6B385D7961AD3E8E0558D9F6617427`。确认从指定检查点恢复，已观察到编号 41219 的更新及训练桩数 6.0000，输出 `model_41200.pt` 已验证可加载。输出位于 `logs/rsl_rl/pointleg_stepping_stones/2026-11-15_15-55-47_new_train_15_competition_six`，训练进程 PID 42576。启动和标准输出在 `artifacts/stepping_competition_six_20261115_155539_861/`，首次受限后台启动未留下训练进程或训练日志。预计末编号 42700。未等训练结束，未做独立回放或真机验证。

## 2026-11-15 Sim2Sim 入口台起步与按键顺序

原先 Ctrl+F5→6 先用平地出生姿态并在切换时保留状态、插值控制 1 秒；6→Ctrl+F5 则恢复梅花桩出生姿态并直接推理，用户观察到起步脚不同。`sim2sim/pointleg_sim2sim.py` 在非梅花桩模式使用入口台出生点后，进入槽 6 时消费一次入口重置请求，使用梅花桩初始关节姿态、速度、控制状态和观测历史。处理同一查看器帧内的两个键；普通现场策略切换沿用状态保留和过渡。

执行 `.venv/Scripts/python.exe -B -m unittest -q -b` 选取场景切换、两种入口顺序、缺少地形和两项接触跟踪测试，5 项通过。用槽 6 `sim2sim/policy/optical_policy/model_41000.pt` 做静态 CPU 推理，两种顺序首帧 105 维输入和首个 12 维动作逐元素相等，SHA256 `B61962F71D2D93F259A530E66A19A71CAE5DDAE6F2DA674FAD99E38888521720`。未启动查看器或推进物理步，未录制实际首步。

## 2026-11-15 训练加入比赛桩距并记录右前足实际落点

用户观察到 Sim2Sim 右前足的落脚逐桩向后偏，末桩踩空。核对比赛场景为 20 cm 方桩、30 cm 中心距，原六桩训练对应 15 cm 和约 23.3 cm，同侧跨距相差约 13.3 cm。槽 6 的 `model_41000.pt` 与 `new_train_14_support_handoff` 的同名训练检查点 SHA256 相同。`src/pointleg/terrains/stepping_stones.py` 和 `src/pointleg/config/stepping_stones.py` 加入比赛尺寸地形，原尺寸与比赛尺寸各占一半。比赛尺寸到六桩时边缘间隙 10 cm、中心距 30 cm，平台与 Sim2Sim 同位置。`src/pointleg/mdp/stepping_stones.py` 使用实际地形等级和类型查表，统一桩宽、中心、出口位置。Play 重置随机抽难度及地形类型。

`sim2sim/stepping_stones_replay.py` 在既有 CSV 增加右前足目标纵向误差、足端和小腿受力及有效承重满 20 ms 后实际落点的桩中心 X 偏差。执行 `.venv/Scripts/python.exe -B -m unittest` 对梅花桩逻辑、迁移、PointLeg 任务和三项 Sim2Sim 静态测试，共 84 项通过。编译两套七级地形，核对碰撞方块、运行时中心与出口位置和比赛六桩 XML，同时验证宽桩边缘区域和右前落点日志。Tyro 已核对续训入口 `new_train_15_competition_pitch`，512 环境、1501 次更新、TensorBoard，起点 `2026-11-15_14-26-31_new_train_14_support_handoff/model_41000.pt`，SHA256 `B61962F71D2D93F259A530E66A19A71CAE5DDAE6F2DA674FAD99E38888521720`，确认 Actor/Critic 105/381 和优化器状态。未启动训练或动态回放，截图显示的实际轨迹与地形差异的因果关系待验证。

## 2026-11-15 左前悬空时的支撑交接

用户澄清左前足在空中做出踩踏动作却未承重，随后右前和左后同时摆动，形成三足失撑。撤销上一版回踩原桩或后退的 `stepping_left_front_failed_step` 和旧计数。梅花桩改为 `stepping_premature_support_handoff`。左前必须在合法顶面连续有效承重 80 ms，条件为足端法向力 ≥8 N 且同腿小腿向上受力 <8 N。右前与左后同时缺少有效承重时，若左前当时尚未达到条件，重叠达到 20 ms 单次扣 3 分。双足中任一恢复有效承重后重新计次。现有低于两足承重的持续扣分继续生效。记录 `Episode_Metrics/stepping_premature_support_handoffs`。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，79 项通过。测试包括左前悬空、触顶但不足 8 N、同腿小腿承重、左前承重不足 80 ms 即起脚、稳固承重前后、单腿摆动、事件重新计次及局部重置。Tyro 已核对 `new_train_14_support_handoff` 续训入口，使用 `2026-11-15_13-03-57_new_train_13/model_39800.pt`，SHA256 `A41E75512554F03852AB0007745E409EF54BB3662C39ABF6A0A64A8AD206C67D`，512 环境、1501 次更新、TensorBoard。本轮未启动训练或回放。

## 2026-11-15 左前足空迈落脚惩罚

用户确认起步上窜已改善，左前足仍有多余一步。`src/pointleg/mdp/stepping_stones.py` 按子步保存左前足最后有效承重的支撑面和起脚位置。由桩上离开至少 20 ms 后，重新在同一合法支撑面累计有效承重 20 ms 时判定是否完成一次空迈。若回到原桩且前移不足 5 cm，或落到更靠后的支撑面，`stepping_left_front_failed_step` 单次扣 1.5 分，独立于右后足状态。`src/pointleg/config/stepping_stones.py` 注册奖励和回合计数 `stepping_left_front_failed_steps`。Actor/Critic/动作仍为 105/381/12。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，79 项通过。验证同桩回落、后退落脚、同桩有效前挪、换到前桩、擦碰及载荷不足、事件仅扣一次和部分重置。Tyro 解析已核对，推荐用 `new_train_13/model_39800.pt` 启动 1501 次更新的 `new_train_14_left_front`，512 环境、TensorBoard、本地保存。检查点 CPU 读取确认优化器与 105 维 Actor，SHA256 `A41E75512554F03852AB0007745E409EF54BB3662C39ABF6A0A64A8AD206C67D`。未启动训练或物理回放。

## 2026-11-15 梅花桩起步机身上窜

用户观察到出发瞬间机身明显升高后回落。已有机身高度奖励乘前向速度比例，静止时为零。`src/pointleg/mdp/stepping_stones.py` 从每环境实际重置高度起监测 1 秒，子步累计机身上升超过 2 cm 或向上速度超过 0.10 m/s 的代价。分别使用 4 cm、0.30 m/s 平方尺度，取较大值封顶 1。`src/pointleg/config/stepping_stones.py` 注册 `stepping_startup_body_rise`（−6 分/秒）及回合指标 `stepping_startup_peak_body_rise_m`。Actor/Critic/动作 105/381/12。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，78 项通过。验证原地上窜、短时速度峰值、公差、回落和超高、1 秒边界、部分重置以及峰值保留。CPU 读取 `2026-11-15_11-36-11_new_train_12/model_39800.pt` 确认优化器存在且 Actor 输入 105，SHA256 `4381F1AD24BB5C9896BF698124B67506E49CA500F472D1CFEEFF5790B8105F0F`。未启动新训练或回放，尚无优化后的实际步态数据。

## 2026-11-15 加强左前足支撑并惩罚三足失撑

用户观察到右后换步时左前足短暂空摆，随后右前和左后起脚形成三足失撑。`src/pointleg/mdp/stepping_stones.py` 将左前足计罚终点从右后首次接触合法顶面延长至连续有效承重 80 ms，短时卸力或小腿借力重置右后承重计时。`stepping_left_front_support` 权重 −2→−12 分/秒，`stepping_left_front_liftoff` 在该阶段左前足连续 20 ms 未接触合法顶面时事件 −1 分。`stepping_low_support` 在回合已有至少两足曾接触合法顶面后，只要当前有效承重不足两足就按 −12 分/秒计罚。该项也覆盖右后落稳后的下一拍。观测和动作维度仍为 105/381/12。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，76 项通过。覆盖右后弱接触和短时卸载、左前短暂虚踩与单次扣分、右后落稳后仅一足支撑、恢复两足时停止计罚。已用 PyTorch CPU 加载 `2026-11-15_12-30-05_new_train_12_resume/model_39800.pt`，确认能读取优化器且 Actor/Critic 输入 105/381，SHA256 `4381F1AD24BB5C9896BF698124B67506E49CA500F472D1CFEEFF5790B8105F0F`。未启动新训练或回放。

## 2026-11-15 右后腿摆动时左前足支撑

用户观察到右前足向前空摆后回撤，希望右后腿迈步时左前足继续支撑机身。梅花桩由 `src/pointleg/mdp/stepping_stones.py` 在物理子步累计右后足离开既有合法支撑且左前足失去有效承重的时长，`src/pointleg/config/stepping_stones.py` 按每秒 −2 加权。左前足沿用合法顶面、足端法向力 ≥8 N 和同腿小腿受力 <8 N 的条件。右后足接触合法顶面即恢复左前足自由换步。已有前进速度奖励继续跟踪赛道 X 方向，右前腿不新增专属项。Actor/Critic/动作仍为 105/381/12。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，75 项通过。覆盖左前足合法承重、接触不足 8 N、小腿借力、右后足收步、右前足独立摆动及控制步累计。未启动仿真或续训，右前足回撤是否减少待新策略回放。

## 2026-11-15 梅花桩四足独立迈步

用户观察到左前与右后连续双摆，先后只有一足踏上桩。移除 `src/pointleg/config/stepping_stones.py` 中的对角姿态 +0.6、右后首步 +3、等待别足的重复迈步 −1 和后腿专属跟步 +1.5。由 `src/pointleg/mdp/stepping_stones.py` 对四足独立计奖，同一足离开合法支撑至少 80 ms、重新在合法顶面连续有效承重 80 ms，较起脚位置和上次计奖点均至少前进 15 cm 时给 +0.75。保持 5 cm 起步机身推进与小腿借力相关计奖。Actor/Critic/动作仍为 105/381/12。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，74 项通过。验证任意脚可先迈、同一脚连续有效迈步、对角脚先后承重独立得分、短步和退后再踩旧点不刷分。续训入口 `uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-14_21-05-56_new_train_9 --agent.load-checkpoint model_29598.pt --agent.max-iterations 1501 --agent.run-name new_train_11_free_gait` 参数已用 Tyro 解析。源 SHA256 `14DC6AB183008326D9F82899591B95A70D09030AFEC0CEFE81635BD002A51791`，CPU 读取确认输入维度及优化器字段。选择与 `new_train_10` 相同的源检查点便于对照。未启动续训、仿真或回放。

## 2026-11-14 梅花桩足端承重与小腿借力

`src/pointleg/mdp/stepping_stones.py` 与 `config/stepping_stones.py` 按腿增加借力事件惩罚。向上受力 ≥8 N 连续 20 ms 或单子步 ≥25 N 扣 1 分，低于 8 N 连续 20 ms 后重新武装。保留 −8 的小腿接触力时间惩罚。足端法向力 ≥8 N 且同腿小腿受力 <8 N 持续 20 ms 才记入覆盖和中心落脚，后腿跟步使用 80 ms 连续有效承重和 15 cm 步幅。借力的控制步取消正向速度分，仍扣后退分，借力不结束回合。Actor/Critic/动作维度为 105/381/12。

`sim2sim/stepping_stones_replay.py` 和 `sim2sim/pointleg_sim2sim.py` 接入逐子步右后足和小腿的力及接触位置 CSV。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg sim2sim.tests.test_pointleg_sim2sim -q -b`，90 项通过。使用 `sim2sim/sim2sim.py --task PointLeg-SteppingStones-v0 --command 0.35 0 0 --headless --duration 12 --stepping-trace <CSV>` 对照训练几何六桩与 `sim2sim/terrain/scene_terrain.xml`。检查点 `2026-11-14_21-05-56_new_train_9/model_29598.pt`，SHA256 `14DC6AB183008326D9F82899591B95A70D09030AFEC0CEFE81635BD002A51791`。报告在 `artifacts/stepping_support_model_29598/`。训练几何 4.090 s 成功。比赛场景 3.365 s 因右前足触地失败，右后足距下一右侧桩中心偏后 45.9 cm，法向力为 0，右后小腿仍在上一桩承受 123.5 N。本次未观察到右后小腿在失败前卸力。两场景同时存在尺寸和接触参数差异，本次回放只对既有权重做对照，未续训或验证新奖励下的策略。

## 2026-11-14 Sim2Sim 槽 6 接通 105 维梅花桩

`sim2sim/pointleg_sim2sim.py` 按槽位任务加载 Actor。槽 6 从只尝试 45 维改为严格加载 `PointLeg-SteppingStones-v0` 的 105 维 `model_29598.pt`。使用 `sim2sim/stepping_stones_replay.py` 从场景几何、航向和四足碰撞数据生成附加 60 维，沿用训练观测顺序和缩放，桩顶和右台接触按 20 ms 确认。切换保留位姿和速度，1 秒平滑接管，默认 0.35 m/s。新增 Ctrl+F5 左台顶出生点，Ctrl+5 仍为台前。Sim2Sim 场景有 6 根 20 cm 方桩，训练用 15 cm 方桩。

执行 `.venv/Scripts/python.exe -B -m unittest -q` 选取 `PointLegSim2SimTests` 的切换、场景缺失、接触累计三项及既有观测、动作滤波、旧权重拒绝三项，6 项通过。实际比赛场景做 45→105→45 观测切换及真实 checkpoint 单次前向，输出 12 个有限动作，未调用物理步进。槽 6 模型 SHA256 为 `14DC6AB183008326D9F82899591B95A70D09030AFEC0CEFE81635BD002A51791`。未启动窗口、动态回放、训练或真机。

## 2026-11-14 Sim2Sim 多策略不兼容槽位排查

用户报告加载第5个策略后启动失败。检查实际权重，槽6 `model_29598.pt` Actor首层输入105维，当前Sim2Sim观察固定45维；槽7为45维。将多策略加载封装为 `load_available_policies`，非首槽不兼容时提示文件、输入维度并跳过，首槽仍为必需；没有伪造梅花桩观测或修改策略。通过真实CPU权重加载确认槽1～5、7可加载，槽6明确跳过，jump文件缺失；执行 `.venv/Scripts/python.exe -B -m unittest sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_old_wheel_leg_checkpoint_and_missing_weights_are_rejected -q` 1项通过。未启动物理仿真、回放或实机；105维梅花桩的观测采集和部署仍待实现。

## 2026-11-14 增强梅花桩膝部正向屈曲惩罚

用户反馈瞬间共线或轻微反曲，要求加大惩罚。`stepping_knee_min_angle` 权重−1改为−8，正向屈膝目标与尺度均5°、逐腿封顶后四腿平均不变。原项按控制步采样，可能漏掉子步瞬间，软惩罚不构成硬限位。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，49项CPU测试通过；未仿真训练，须重启进程并验证。

## 2026-11-14 四膝正向屈膝至少5度

用户澄清目标是正向屈膝大于5°，而非反向超过5°才惩罚。撤销本轮未交付的反向角新项，将既有 `stepping_knee_min_angle` 的有方向正常屈膝量−q最低阈值和尺度从15°改为5°，权重−1不变；反向弯折仍扣分，软惩罚非硬限位。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，73项CPU测试通过；未仿真训练，重启进程生效。

## 2026-11-14 增强梅花桩小腿受力惩罚

用户说明续训后小腿仍撑桩边缘。`new_train_8` 的 `Episode_Reward/stepping_shin_stone` 近期非零，说明训练中该项有触发，但不能断言图片对应接触被检测。将小腿碰桩及平台惩罚权重−2.5提高至−8，保留2 N死区、20 N满额和子步积分。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，49项CPU单元测试通过；未仿真训练，需重启续训，观察是否仍撑腿及是否出现卡住。

## 2026-11-14 梅花桩平台边缘小腿受力

用户反馈小腿撑平台边缘前进。原 `stepping_shin_stone` 只匹配桩体接触点，增加左右平台顶面和侧面判定，仍用四小腿geom接触传感器、2 N起罚、20 N满额、−2.5权重和子步时间积分；地面及足端接触不计。最新 `2026-11-14_14-14-33_new_train_8` 参数中有原桩体项，但是否为用户图片对应模型未确认。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，73项CPU测试通过，覆盖两平台、桩和地面；未仿真训练，需重启训练进程。

## 2026-11-14 梅花桩小腿受力惩罚

用户要求小腿不碰桩承重。增加四条小腿碰撞geom与地形接触传感器，最多保留每条腿4个最大力接触；按每环境有效桩中心和接触点位置筛选桩体，2 N死区、18 N尺度线性封顶（20 N满额），物理子步取最大并积分，`stepping_shin_stone` 权重−2.5；仅软惩罚，不更改足端接触和终止。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，73项CPU测试通过，覆盖桩体与平台区别、力阈值和子步累计；未仿真或训练，传感器实际动力学表现需重启进程验证。

## 2026-11-14 梅花桩回放随机课程

用户要求回放能看到不同等级并参考Slope。`make_stepping_stones_env_cfg(play=True)` 为重置事件指定 `randomize_levels=True`，每次回合重置均匀采样0～6级，更新地形原点后重置机器人并按等级生成目标位置；训练不传此参数，原课程保持。保留可选固定 `terrain_level` 参数用于外部指定，二者不可同时启用。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，72项CPU单元测试通过，覆盖七级采样、原点和桩位；未启动回放或仿真。

## 2026-11-14 梅花桩边缘和前后腿净空惩罚调整

用户要求桩四周边缘惩罚由1 cm扩大到3 cm，`stepping_leg_clearance` 权重由−2改为−4，距离阈值5 cm保持。桩面15 cm见方，边缘以实际接触点判断，中心9 cm见方；保留已有边缘权重−3.5和机身高度权重+1，更新滞后的测试期望。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，71项CPU测试通过，含四边、边界与重复落脚；未仿真训练，重启生效。

## 2026-11-14 梅花桩中心落脚与边缘惩罚

用户要求倾向踩中心、桩面边缘1 cm惩罚。新增 `stepping_stone_center` +2，连续有效顶面接触20 ms时按接触点到该等级实际桩中心的平面距离以4 cm尺度高斯计分，边缘无中心奖；每足每桩仅首次有效落脚可领。新增 `stepping_stone_edge` −3，桩顶接触点距任一边不超过1 cm的有效落脚每次计罚。保留原覆盖及成功逻辑与观测维度。现有摆动高度配置已为−5，更新旧测试期望以反映当前配置，本次未修改该配置。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，70项CPU测试通过；未仿真训练，需重启训练进程。

## 2026-11-13 梅花桩摆动足高度7厘米

用户反馈抬足过高，要求7 cm。增加 `stepping_swing_height` −2：行进阶段无地形接触的足，球足最低点相对平台/桩顶高于0.07 m起罚，超出部分按0.07 m尺度平方、逐足封顶1并求和；不对低于目标高度给悬停奖励。保留其余规则和接口。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，69项CPU测试通过；未仿真训练，需重启进程观察落桩和净空。

## 2026-11-13 梅花桩对角姿态奖励

用户反馈后腿过伸和前腿上台姿态异常，要求对角姿态一致奖励。增加 `stepping_diagonal_posture` +0.6，对FL/RR和FR/RL，比较同接触相位的髋摆和膝角，30°高斯尺度，乘正向速度比例；无符合条件对角组或静止时为零。现有四膝最小屈曲惩罚保留，未加入后足相对前足位置限制，软奖励无法保证不伸直或越位。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，68项CPU测试通过；未仿真训练，需重启进程。

## 2026-11-13 梅花桩前后腿距离阈值调整

按用户要求将 `stepping_leg_clearance` 的中心/线段距离阈值从0.08 m改为0.05 m，权重−2不变。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，43项CPU测试通过；未仿真或训练，需重启进程生效。

## 2026-11-13 梅花桩前后足与小腿净空

用户反馈后腿追前足同一桩时碰腿。机器人碰撞几何contype=0、conaffinity=1，自身不产生物理接触，新增 `stepping_leg_clearance` −2：所有前后足配对，取足球心间和足球心至对方小腿膝至球心线段距离最小值，低于8 cm按缺口比例平方惩罚，取最坏一对；全回合生效。不改变接触物理和接口，未检测大腿和两条小腿之间的碰撞。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，67项CPU测试通过，含足足及足对小腿近距离；未仿真训练，重启进程后观察。

## 2026-11-13 梅花桩逐足到达右台成功

按用户要求，终点由全部桩覆盖后四足同时右台支撑且机身稳定1秒，改为全部桩覆盖且四足各自在右台顶面连续有效接触20 ms即成功重置；各足顺序不限且锁存，可在桩覆盖前记录。沿用接触位置和法向，无承重力阈值，足端触地失败优先。行进指令保持至成功，原停稳进度观测改为右台已接触足数比例，Actor/Critic/动作105/381/12不变但观测语义变化。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，66项CPU单元测试通过，覆盖逐足和漏桩、重置；未仿真训练，需重启训练进程。

## 2026-11-13 梅花桩倾斜惩罚与起步机身前移

用户希望抑制倾斜且先将机身前移约5 cm。新增 `stepping_body_tilt` −2，整体俯仰横滚倾角超过8°后以15°尺度平方封顶，保留原水平奖励；增加四足左台支撑后机身沿X向5 cm新增进度最多+2，未达到时任一足连续失撑80 ms一次−1并结束起步奖励。软奖励不强制支撑或改变出生位置，保持15 cm步幅目标、旧任务接口。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，66项CPU测试通过；未仿真训练，重启进程并观察起步卡住与倾斜情况。

## 2026-11-13 梅花桩迈步门槛15厘米

用户要求迈步至少15 cm。将换足解锁从X向绝对位移2 cm改为X向净前移15 cm，后腿跟步奖从单次和历史各8 cm改为各15 cm，右后首步奖从3 cm改为15 cm。足完成离地/落地确认后即进入等待，即使步幅不足15 cm；不足15 cm不能解锁其他足。其他时间、权重、终止和观测维度不变；约束为奖励逻辑而非硬动作限制。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，64项CPU测试通过，覆盖14 cm无奖、16 cm有奖、短步不解锁和重置。未仿真或训练，需新进程验证15 cm可达性及是否卡住。

## 2026-11-13 梅花桩换足顺序软约束

用户反馈后足反复小步，希望其他足完成一次迈步支撑后才继续。增加逐足换步状态：合法支撑起步、离地80 ms、合法顶面落地80 ms及前后位移≥2 cm记有效换步，完成后该足等其他足有效换步。已有桩覆盖后，等待期间再次离地80 ms单次−1；后足在等待期间起步不予后腿跟步奖励。保留软约束、自由选择其他足、任务判定和观测维度，不直接锁住动作。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，64项CPU测试通过，包含连续同足惩罚、其他足落脚解锁和重置；未仿真或训练，须新进程验证。

## 2026-11-13 梅花桩后腿跟步阈值

用户反馈左后腿小步反复点地。`stepping_hind_forward_step` 对两后足的最小单次前移及相对历史获奖位置前移由4 cm提高至8 cm，落地确认由40 ms提高至80 ms；离地确认80 ms、权重+1.5和任务终止均不变。这是奖励门槛，不是禁止短步。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，39项CPU测试通过，涵盖6 cm无奖、9 cm获奖和原地重复落脚无奖；未仿真训练，需新进程验证。

## 2026-11-12 梅花桩膝关节方向修正

用户指出反向屈膝不应满足防伸直目标。将 `stepping_knee_min_angle` 的膝角绝对值改为有方向屈膝量−q，模型四膝轴均为+Y，任务正常站姿膝关节约−1.4 rad；正常屈膝量低于15°时惩罚，反向弯折不再得免罚。权重−1、尺度15°及逐腿封顶平均保持，软惩罚不构成硬限位。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，39项CPU单元测试通过；未仿真或训练，需重启进程。

## 2026-11-12 梅花桩四膝防伸直

按用户要求增加 `stepping_knee_min_angle` 权重−1，四膝关节角绝对值小于15°时按15°尺度平方逐腿封顶并平均。膝关节零角近似大小腿同向共线，正负屈曲均满足条件，并非硬角度限制。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，63项CPU单元测试通过；未仿真或训练，重启进程生效。

## 2026-11-12 梅花桩前进间隙课程

用户要求前进间隙5～10厘米课程。保留原七级及2～8桩课程，在同一等级沿赛道X方向台边到首桩、相邻桩及末桩到出口台的边缘间隙从0.05线性增至0.10米。原中心间距0.30米及固定平台间距已替换，桩面0.15米、横向中心±0.20米保持；几何、按环境等级生成的桩位、接触判断和左右平台观测同步，观测维度保持105/381/12。旧checkpoint可按结构加载但几何分布改变，需新进程。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，62项CPU测试通过，含七级地形和运行时坐标一致性；未进行物理仿真或训练。

## 2026-11-12 增强梅花桩机身水平奖励

用户反馈机身倾斜，希望保持俯仰和横滚。原 `stepping_body_level` 已按投影重力评估整体倾角，故不重复添加同目标项，将权重+0.5提高至+1.5；12°尺度和按正向速度门控不变。腿部伸直无单独修改。运行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，37项CPU单元测试通过；未仿真或训练，需重启进程生效。

## 2026-11-12 增强梅花桩髋展目标约束

用户希望加强髋展目标约束，`stepping_hip_abduction_target` 权重由−0.5调整为−1.0，左右目标±0.15 rad、容差0.05 rad、尺度0.15 rad及对称奖励均不变。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones -q -b`，37项CPU单元测试通过；未仿真或训练，需重启训练进程生效。

## 2026-11-12 梅花桩机身高度奖励

用户反馈机器人通过抬高机身跨桩。增加 `stepping_body_height` +0.5，目标为出生时机身原点高出台面0.362米，容差±0.03米，超出按0.08米尺度高斯衰减，乘前进速度比例，静止、终止、终点停稳无奖励。软奖励不构成高度硬约束，不改变接口。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，61项CPU测试通过，未仿真或训练，重启进程生效。

## 2026-11-12 梅花桩髋展轻度目标

用户希望保持适度外展并检查左右方向。模型 `mjcf/pointleg.xml` 四髋展轴均为+X，左右外展应采用正、负镜像；增加 `stepping_hip_abduction_target` 权重−0.5，四腿分别跟踪+0.15/−0.15 rad，死区0.05、尺度0.15、逐腿封顶1后取均值。保留左右对称项；软目标不限制调整、不改变接口。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，60项CPU测试通过，覆盖镜像符号、死区和反向；未仿真或训练，需重启进程。

## 2026-11-12 梅花桩后腿向前跟步奖励

用户反馈前腿踩桩时后腿固定在原地、伸腿支撑。增加 `stepping_hind_forward_step` +1.5：桩已覆盖后，两只后足分别从已检测合法支撑起步、离地0.08秒、重新合法顶面支撑0.04秒，落脚相对起脚及上次获奖位置均前进至少4厘米，结算一次；失败回合当步不发奖。无对角同步硬要求，不更改首步规则和接口。`.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b` 59项CPU测试通过，未仿真或训练，需重启进程验证。

## 2026-11-12 右后首步离地和落地延长至0.7秒

根据用户要求，梅花桩右后腿首步奖的离地和落地确认时间均从0.1秒改为0.7秒。其他三足连续失撑20 ms限制、3 cm步幅、一次性+3及其他判定不变。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，58项CPU单元测试通过，含0.7秒阈值；未运行物理仿真或训练，需重启训练进程生效。

## 2026-11-12 延长梅花桩右后首步确认时间

按用户要求，右后足首步奖励的离地和落地确认时间各由20 ms改为100 ms，避免短促离地或触碰计为完整步；其他足连续失撑20 ms的门槛和每回合最多一次+3保持不变。执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，58项CPU测试通过，覆盖不足100 ms离地和不足100 ms落地；未仿真或训练，新进程生效。

## 2026-11-12 梅花桩右后腿首步奖励

按用户要求在 `PointLeg-SteppingStones-v0` 加入 `stepping_first_rr_step` 权重+3，一回合最多一次。四足左平台顶面接触后检测右后足起步：离开合法支撑20 ms、重新在合法顶面接触20 ms、相对起脚前进至少3 cm；其他三足连续失撑达到20 ms则取消本回合奖励资格。右后足首次确认落地后无论是否达到前进距离均结束首步奖励，不改变覆盖与成功判定或后续腿序。保留105/381/12接口，已运行进程需重启。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，57项CPU测试通过，覆盖右后足落脚奖励、其他足失撑和重复领取；未进行物理仿真或训练。

## 2026-11-12 梅花桩加入机身侧漂惩罚

按用户要求参考Slope任务，为 `PointLeg-SteppingStones-v0` 增加 `stepping_lateral_velocity`，复用 `deadband_cost` 计算机身坐标系Y向速度，容差0.01 m/s、尺度0.10 m/s、平方封顶1，权重−3，与Slope一致。该项全回合生效；原路线对齐奖励中的赛道Y向速度衰减仍保留。奖励数10→11，观察和动作接口保持105/381/12。已有训练进程需重启才能使用。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，55项CPU单元测试通过，覆盖左右速度对称、死区和封顶。未执行物理仿真或训练，实际收敛表现待验证。

## 2026-11-12 梅花桩机身匀速前进奖励调整

用户反馈触碰首桩后机身停滞，要求直接奖励机身匀速前进。`src/pointleg/mdp/stepping_stones.py` 的 `forward_velocity_reward` 从截断正向速度比乘高斯跟踪改为 `2r−r²`，其中 `r` 为世界赛道X向机身速度与命令速度之比，截断到−1～2。保持权重+2.5及0.30～0.40 m/s指令，静止为零、目标速度得1、后退扣分；保留10项奖励、终止逻辑及105/381/12接口。原训练进程需重启才能应用新奖励。

在 `rc_mjlab` 执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，54项单元测试通过，检查静止、低速、目标速度、超速、后退及终点情况。未运行物理仿真或训练，收敛效果待验证。

2026-11-09 点足 Sim2Sim 比赛场地键位：依据赛规 V1.0 图 1-2 及当前场景 XML，将原单一场地出生点扩展为 Ctrl+0～9、Ctrl+- 共 11 个预设，启动区 2 加高台面偏移，斜坡、台阶和砖木坑按各自入口设置朝向。按组合键保留常规策略、清零速度并重置位姿，R 重置到已选位置。旧 J 槽改为空格，接入 `JumpTrigger` 和 `PointLeg-Jump-v0`，一次触发到 3 秒回原槽，兼容性检查同时校验跳跃命令语义。检查通过：Python 语法、XML/11 个 site 世界坐标、模拟键盘回调及 150 步命令时钟，现有 1～6 号权重 CPU 加载。未启动物理仿真或 GUI，当前无 `model_jump.pt`，未验证实际跳跃或障碍通过率。入口 `sim2sim/sim2sim.py`，重启进程后生效。

2026-11-04 随后修正旧六级成功数据污染新三级晋升窗口：同几何Stairs checkpoint完整resume时，若保存的completion与当前不同，只清空上下台阶的课程成功率窗口，保留已解锁上限、网络、优化器和迭代并发警告；不同成功标准不混合计算。新增合成旧契约checkpoint测试，五组64项测试通过（2.631秒），未触碰用户实际模型或仿真。

2026-11-04 用户要求Stairs完成前三级且四足在中间平台承重即成功，不再需要后三级。成功检查切换为前3个踏面均实际承重接触、四足同时在第三阶后中间平台≥8N/高度误差≤8毫米/通道内；上行、下行均按前进方向定义。漏阶判定在整机后缘越过中间平台末端且缺前3阶时失败；出界、跌倒、停滞及失败优先保持。旧第三阶profile合并中间平台，平台边界按第三阶起点+0.32至第二段起点计算；继续保留六级物理几何。新增末值`stairs_middle_loaded_feet`并改`stairs_covered_treads`为前三阶末值，评估覆盖数同步；旧exit_loaded_feet作为中间平台兼容别名。runner奖励契约21→22，几何契约不变，旧同几何resume保留优化器/课程上限并警告语义变化；旧成功率窗口清空。63项单元测试（3.144秒）和语法检查通过，未仿真或训练。

2026-11-04 用户希望摆动时增加膝关节角度以提高足端。按模型默认膝−1.125弧度假设为朝更负方向增加屈膝幅度，新增Stairs `swing_knee_flexion`+0.2；−1.125到−1.6弧度线性增分后饱和，每足只在真实离地、起脚后足底高≥1.5厘米、至少两足8N支撑、正向命令且实际前速>0.05米/秒时计分，速度因子到0.15米/秒饱和，持续离地超0.45秒不计，四足平均。逐足日志`Metrics/stairs_swing_knee_{leg}_deg`。该奖励不能保证足端越阶，不限制饱和以上屈膝；runner契约20→21，旧同几何checkpoint可resume并提示变化。45项单元测试及四文件py_compile通过；无仿真或训练验收。

2026-11-04 用户反馈大腿过伸近水平，希望比台阶坡度稍大。Stairs `thigh_body_alignment`由机身轴0～45°免罚改为世界水平无向大腿角相对台阶等效坡度`atan(单级高度/0.32)`加15°的±10°范围免罚；超出按30°尺度平方、每腿封顶1、四腿均值，权重−0.35和上阶附近激活区不变。6/12厘米目标约25.6°/35.6°；不规定膝前后、不保证倾斜朝向，非直接坡面法向。保留原机身角度日志，增加世界角度和目标日志。runner契约19→20，同几何checkpoint可resume并提示奖励变更。44项单元测试通过（2.515秒），语法检查通过；未仿真/训练，无法断言过伸已改善。

2026-11-04 用户反馈前腿前后姿态并要求增加足端高度权重。仅将Stairs的`swing_lift_deficit`惩罚权重−1→−2（两倍），目标仍按当前单级台阶高度计算，保持连续/落脚扣分与不触发失败，几何/其他奖励不变；训练契约18→19，同几何checkpoint可resume并提示变化。43项定向单元测试及四文件py_compile通过，未仿真或训练。抬升峰值软惩罚不能保证越过立面时净空，也不能凭单帧断定前足偏置成因。

2026-11-04 用户希望修正Stairs左前腿前伸，参考右后腿姿态。新增`left_front_rear_tracking`−0.5：正向命令、至少两足8N支撑且左前与右后接触状态一致时，比较髋俯仰/膝角，25°死区、30°尺度、平方上限1并均值；足高差>3厘米时乘0.25，避免不同台阶高度强行同步。记录逐关节角差；共享策略使右后腿也可能受到梯度影响。契约17→18，同几何检查点可resume并提示奖励变化，需重启。四组43项测试通过（2.835秒）、语法检查通过；未仿真/训练，不证明姿态改善。

2026-11-03 用户参考轮腿机器人爬阶姿态，希望大腿更接近机身倾角。仅Stairs新增`thigh_body_alignment`−0.35，上台阶入口前0.4米至出口后0.4米，计算四足各自髋到膝连杆在机身坐标XZ平面与机身前后轴的无向夹角，45°容许、30°尺度、平方单足封顶1、四足均值；非强制平行并允许膝在前或后，适用于完整连杆姿态，保持既有地形、步态及成功规则。日志逐足`Metrics/stairs_thigh_body_angle_{leg}_deg`。runner契约16→17，同课程几何旧checkpoint可resume并提示奖励变化。42项单元测试（2.391秒）及py_compile通过；未仿真/训练。大腿近水平可能影响0.42米默认站高，不保证参考轮腿机器人姿态适合本点足模型。

2026-11-03 用户要求抬腿惩罚目标与当前台阶高度一致。Stairs的`swing_lift_deficit`由固定13厘米改为逐环境`top_height/6`：实际台阶6/7.5/9/10.5/12厘米时分别要求相对起脚位置抬升相同数值。保留−1权重、0.08秒连续欠高惩罚与有效落脚事件代价；高度不足仍非失败，其他奖励/地形不变。记录目标指标`Metrics/stairs_target_lift_m`，runner契约15→16，同几何checkpoint可resume并提示变化。四组41项单元测试通过（2.494秒）、四文件py_compile通过；未仿真/训练。

2026-11-03 补充合成旧固定高度检查点验证：直接resume按几何契约拒绝，纯actor/critic权重加载成功；四组单元测试共40项通过（2.538秒），未读写真实模型内容或运行仿真。

2026-11-03 用户反馈Stairs训练退化静止，要求6～12厘米高度课程。仅Stairs的上下台阶由固定0.12米改为5档0.06/0.075/0.09/0.105/0.12米，使用既有每方向独立100回合/80%解锁课程，成功升档失败降档，六级覆盖与四足出口成功不变。Stairs地图4×4紧凑布局改完整5×6以确保每个难度有实际物理地形；布局契约revision4记录高度范围，旧地图checkpoint完整resume被拒绝，需纯权重transfer且清空优化器/课程/迭代。评估case记录各档实际目标高度。其他任务固定高度不变，13厘米软抬升仍在。39项定向单元通过（2.568秒）及语法检查通过，未仿真或训练；课程达标和足高保持尚未验证。

2026-11-03 用户反馈训练逐渐降低足高，要求13厘米欠高惩罚。Stairs新增`swing_lift_deficit`权重−1，以8N最后承重足底世界高度为起点记录逐足摆动峰值，目标0.13米；欠高比例平方在摆动超过0.08秒连续计罚、有效落地（至少0.04秒离地）再计0.10秒等效事件代价，四足平均；记录逐足落地峰值。旧120毫米硬终止仍关闭；无净空硬保证。runner契约14→15，同任务旧checkpoint可resume并提示配置变化。执行`test_stairs`、`test_slope`、`test_elevation_paired`33项通过（3.227秒）及四文件py_compile，未运行仿真/训练。

2026-11-02 针对Stairs左前足仅踩台阶边缘、右前足迈空和四足静止，增加`tread_foot_placement`+0.5，四足各自对每个踏面8N确认0.10秒，以接触点到踏面前后边缘最小距离/8厘米截断后平方评分，每足每级仅发一次；落脚信号为软奖励，不能强行保证足端到达。增加`four_foot_standstill`−5，正向命令且四足承重、机身向前位移不足2厘米超过1秒才开始惩罚，0.5秒线性升满，任一足离地或推进2厘米重置，避免惩罚正常短暂四足支撑，存在短促离地重置计时的限制。Stairs训练契约13→14，旧同任务resume会提示奖励变化，需重启训练。32项定向测试通过（2.382秒）和py_compile通过，未进行仿真或训练。

2026-11-02 用户确认删除可能误判的上阶抬升不足120毫米立即失败。`update_elevation_state`仅对terrain_kind=stairs屏蔽该失败条件及原因10，保留旧诊断状态与原地形接触检测；其他失败条件、其他任务不变。故当前Stairs未强制足端达到120毫米。runner契约revision12→13并记录无净空终止，旧同任务检查点可续训并提示配置变化。46项单元测试和语法检查通过，未运行仿真/回放/训练；需重启训练进程才能生效。

2026-11-02 用户同意后扫与密集进展方案，本次完成局部实现：Stairs新增`stairs_progress`，历史前沿进度最高2/秒、每足有效后扫7厘米目标与支撑时长/滑移质量折减单次最多0.3、每级踏面内至少0.1秒8N承重首次奖励1；步态时序权重−0.6改−0.2。原固定地形、120毫米峰值硬判定、停滞终止和成功条件保持；窗口停滞、越阶瞬间净空、净空软奖励、身体跟进奖励和阶段课程尚未实现。30项相关测试通过（2.694秒），首次测试发现上下地形随机长度不同导致旧夹具坐标错误，已改为各环境终点；未运行仿真或训练。

2026-11-02 实施`PointLeg-Stairs-v0`台阶训练任务revision12：速度0.15～0.30米/秒、回合60→120秒，基础14项上迁入Slope五项约束，奖励共19项。`forward_stall`抑制静止，`prolonged_stalled_swing`抑制停滞单足悬空，`hip_abduction_zero`为四髋外展默认0度零死区/−7.2，`step_timing`要求0.70±0.10秒周期、最短.30秒承重并惩罚>.55秒悬空，`mean_leg_posture`惩罚慢髋俯仰/膝偏差。上台阶时摆动足从低支撑面落在更高踏面前，足端相对起脚峰高不足.12米立即失败；六个踏面均须至少一次真实承重接触；上行四足8N承重在最高出口平台、下行四足8N承重在最低出口平台即成功，掉出通道仍失败。新增`mdp/stairs.py`及Stairs指标，通用时序项支持按任务前缀记录。执行`test_stairs`、`test_slope`、`test_elevation_paired`共29项通过（2.349秒），`py_compile`及定向diff检查通过；未启动仿真、Viser或训练，不宣称策略已经学会爬阶。

2026-11-01 修复`PointLeg-Slope-v0`训练出现的低趴静止、单足原地悬空微动局部最优：只修改Slope，不影响Stairs、混合Elevation、HighStep和Jump。新增`forward_stall`−1：正向指令且机身前向速度<0.05米/秒连续0.60秒后开始扣分，0.40秒内升至1；新增`prolonged_stalled_swing`−1：仅停滞期间累计每足真实8N地面承重缺失，0.35秒后按0.20秒尺度升至1并取最差足，移动中的正常摆腿不累计。新增`hip_abduction_zero`−3.6，四髋外展相对默认0弧度无死区、0.25弧度尺度平方扣分，不约束髋俯仰/膝关节。新增Metrics/elevation_stall_duration_s及Metrics/elevation_max_stalled_swing_s。Slope奖励共14→17项，训练契约revision 8→9，旧模型完整resume会按既有机制提示奖励配置改变，奖励状态从零开始。执行`test_elevation_paired`共12项通过并通过四个修改文件语法检查；未启动仿真、Viser或训练，未宣称步态已改善。

2026-11-01 修复`PointLeg-Slope-v0`与`PointLeg-Stairs-v0`首次环境重置崩溃：框架首次`reset()`先调用课程管理器，原`ElevationTerrainCurriculum.__call__`却直接读取只会在后续重置事件中写入的`env._elevation_settings`，因此触发`AttributeError`。现在缺少该运行时字段时从`cfg.events["reset_elevation"].params["settings"]`读取同一份`ElevationSettings`；正常重置后仍使用环境缓存设置。新增`test_curriculum_initializes_before_the_first_reset_event`覆盖该时序。执行`test_elevation_paired`共10项通过，且两处改动已通过`py_compile`；未启动仿真、Viser或训练。奖励、地形、课程参数、观测、动作与检查点接口均未改变。

2026-11-01最新HighStep共24项：新增lf_rr_pose_tracking +.25，在右后足真实有效摆动且主动抬升≥3厘米、机身前进>.05米/秒、至少两足承重时，左前跟踪右后当前髋摆/膝角及膝相对髋的机身竖直位置；左前自身不必已进入摆动，避免旧对角项的双足摆动门槛不给纠正信号。关节误差尺度20°/25°，膝高4厘米，两类误差各半取高斯正分；后腿仅作参考，策略仍共享。侧漂权重−1→−2，死区.015→.010米/秒，左右对称；按用户追加要求髋外展−.9→−3.6（4倍），零目标/零死区/.25弧度尺度保持。其余项、45/325/12、PPO/PD/地形及其他任务不改。静态模型审计未发现缺执行器、错绑或左前髋摆驱动失效：XML12→删除0→重建12，全部动作槽/关节对应，0.1弧度目标增量仅对应DOF产生5N·m；四髋相同静态角增量使膝上升约14.1～14.3毫米。报告artifacts/high_step_lf_actuator_audit_2026-11-01.json；未积分时间，不代表动态碰撞/带载饱和或策略已验收。一个合成张量检查通过：左前仍支撑时可获跟踪反馈、匹配高于异常姿态、失败/非有限/静止0及左右侧漂同罚。未仿真或训练；新进程同任务直接resume，当前截图未绑定到指定模型文件。

2026-11-01 HighStep承重后扫与侧漂调整：保留已有command_orthogonal_drift，权重−.5→−1.0、1.5厘米/秒死区与10厘米/秒尺度不变，对左右漂移等价，不加入只针对左侧的偏置；横向速度为机身坐标，不能替代绝对航向/全局路径控制。新增stance_backstroke +.04，每足连续8N承重≥.16秒后离地时按有效后扫/7厘米线性发放，单足每次最多+.04；有效后扫取落地到离地间机身实际前进距离与身体相对该支撑足前进距离中的较小者，负值置0。按真实接触点累计滑移乘max(1−滑移/2.5厘米,0)，无机身推进的脚底拖滑不得分；超过7厘米饱和，事件原始值除dt，由奖励管理器积分一次。现有高抬腿、对角/髋摆、姿态、滑移惩罚和支撑超时保持；总奖励23项，45/325/12、其他任务/PPO/PD不变。指标Episode_Reward/stance_backstroke、Metrics/high_step_backstroke_{fl,fr,rl,rr}_m、Metrics/high_step_stance_time_{leg}_s、Metrics/high_step_contact_slip_mean_m_s和Metrics/high_step_lateral_speed_abs_m_s，应同时观察四足抬升及回放，防止过长支撑、停步、足底滑移或高度退化。按用户要求只执行一个合成轨迹检查，23项配置、真实身体推进奖励、原地拖脚无收益、单事件、左右同等漂移通过；未仿真/Viser/训练，不能据此断言实机发力或爬阶通过。用户未指定当前较好模型路径，继续训练时固定选择其HighStep检查点直接resume，新进程才启用。

2026-11-01 HighStep加入diagonal_rear_tracking +.2，22项：左前跟右后、右前跟左后；起脚前真实承重且对角两足同时有效摆动、至少两足承重、前进指令及实际机身前向速度>.05米/秒时，以两足进入共同有效摆动时的各自髋俯仰/膝关节角为起点，比较角位移，不强制绝对关节角。后腿相对起点的髋或膝位移至少5°才发放，20°运动强度饱和；前后角位移误差按15°尺度高斯计分，两对得分相加除2，按控制周期正常积分。实际后腿位移同时参与得分，不能保证后腿策略参数被固定，也不能由后腿单独决定策略梯度；前后足不同步时该项无收益而非强制锁相。现有足高、髋外展、pitch、前足工作空间及髋摆项保持，45/325/12、PPO/其他任务不变。Metrics/high_step_diagonal_error_fl_rr_deg、_fr_rl_deg记录两关节角位移差的均方根；Metrics/high_step_diagonal_rear_motion_fl_rr_deg、_fr_rl_deg记录对应后腿相对起点最大关节位移，仅满足门槛的样本记录。只执行单个配置与张量轨迹检查，22项/权重.2，左前右后匹配高于相反、后腿未动或静止0；未仿真、Viser、训练或完整回归，右前左后由相同映射实现但未做独立轨迹测试。新进程从用户选定同任务检查点直接resume，实际效果需短训观察。

2026-11-01最新HighStep针对前足过度前伸、髋外展和前倾调整：新增front_foot_workspace −.7，分别取左右前足球心相对同侧髋俯仰连杆的机身坐标x，以.6秒EMA、1秒启动宽限追踪，平均前伸超过4厘米时按8厘米尺度平方计罚，每足封顶4后平均；仅前进时生效，不锁定瞬时摆动姿态。hip_abduction_excess −.3→−.9（默认0角、零死区/.25弧度尺度不变）；high_step_pitch −.5→−1、死区3°→2°/尺度8°不变，正负俯仰同等处理；hip_swing_coordination +.04→+.06，原有效步/稳定落地/两足支撑/身体前进资格、20°髋幅及3厘米相对髋膝上下位移饱和目标不变。主动抬升+.35/.08/15厘米和其余未提及的奖励保持，总数21，45/325/12、PPO/PD/其他任务不改。指标Metrics/high_step_front_foot_mean_x_fl_m和_fr_m，可与high_step_pitch_deg、hip_abduction_deviation_mean、分足髋摆及膝竖直幅度、分足主动抬升共同检查；EMA均值不是每步足端位置。单个配置/张量检查通过，验证前足平均前伸受罚、零前伸无罚、同周期缓存及部分重置；未仿真、回放或训练，实际改善和足高需短训验证。用户未指定截图对应检查点，续训应固定用户选定的HighStep检查点直接resume，新进程才应用奖励。

2026-10-31最新HighStep应用户要求在17项上增加3项，共20项：hip_abduction_excess −.3，以默认0弧度为中心、零死区、.25弧度尺度平方，四腿分别计分；high_step_pitch −.5，正负俯仰超过3°后按8°尺度平方惩罚；hip_swing_coordination +.04，对有效完成并稳定落地的摆动奖励髋俯仰角幅及膝相对髋在机身坐标竖直幅度，20°/3厘米各饱和，两者各占一半。只在8N支撑且摆动期间至少两足承重、原主动高度有效落地条件、实际身体前进≥1厘米时结算一次；不单独给原地抖关节奖励，正常动作仍允许偏离0，不是硬限位。原17项（足高+.35/.08、15厘米目标）保持，45/325/12及其他任务配置未改。单个无物理张量轨迹检查通过：20项配置、零死区和水平参数、髋20°/膝3厘米饱和并只结算一次；未仿真或训练。指标为Metrics/hip_abduction_deviation_mean、Metrics/high_step_pitch_deg、Metrics/high_step_swing_hip_range_{leg}_deg、Metrics/high_step_swing_knee_vertical_range_{leg}_m。

## 2026-10-31 HighStep主动抬升奖励

用户要求接入抬高腿奖励并略提高权重以加快收敛。在15项基础奖励上增加active_lift_event +.35与active_lift_progress +.08，共17项；目标15厘米。high_step_basic.py记录足底相对起脚足底的主动上升峰：高度新增时按新增高度/.15连续支付，稳定落地后按min(峰/.15,1)^2结算。一次完整15厘米有效摆动的原始累计反馈为1，事件原始分为1，按对应权重形成+.08与+.35总收益；8/12/15厘米事件收益比例约.284/.64/1。超过15厘米饱和，不额外鼓励。

资格只防止刷分：起脚前连续承重≥.16秒、起脚后余下至少两足承重、离地.10～.36秒、足前移≥.025米、落地后承重≥.16秒。高度不与周期、后扫、收腿、工作区、姿态质量或速度收益相乘；静止、原地点脚、短支撑、落地未稳、全足腾空、失败和非有限不付款。足高相对地面峰与主动起脚峰分开记录；基础评估仍只验收有效换步，不把15厘米设为本轮短训硬阈值。

实际验证按用户要求缩减为：py_compile通过；两项新张量检查通过（8/12/15厘米排序与公式、无效摆动门控）；当前配置解析17项，6200读取确认45/325/12和iter6200。不重跑101项完整回归，也未启动物理、Viser、PPO或训练。此前101项和完整CPU状态保存验证适用于接入前15项基础版本，不能当作本次奖励的全量回归。源模型、PPO/PD、地形及Elevation/Jump未改。训练使用active_lift_6200新目录，从6200直接resume100更新、保存6250/6299。

+2026-10-31最新HighStep在15项基础项上新增两项独立主动抬升奖励，当前17项：active_lift_event +.35，在足完成有效摆动并稳定承重.16秒后结算 `min(主动峰/.15,1)^2`；active_lift_progress +.08，摆动期间仅按主动峰新增加高度/.15连续支付。主动峰是足底相对起脚足底的上升，不是相对世界地面高度；无过高奖励。资格为起脚前承重≥.16秒、起脚后其余足≥2足承重、离地.10～.36秒、足前移≥2.5厘米、稳定落地≥.16秒。静止、原地点脚、短支撑、落地未稳、全足腾空、失败和非有限不领；高度不与周期/后扫/收腿/姿态质量相乘。仍用基础评估有效循环条件，高度暂报告而不作为通过硬阈值；新增主动抬升及分足指标。6200仍直接resume，建议新运行名active_lift_6200，100更新、看6250/6299；45/325/12和其他任务保持。

## 2026-10-31 HighStep从6200重建基础奖励

用户批准前一轮精简方案，实际修改config/high_step.py为白名单15项，38→15；新增mdp/high_step_basic.py实现无循环依赖的向前速度、统一8N不足支撑、真实点滑移和合并support_timing。速度2.5/宽度.20及旧前沿/renewal/承重门控→+2/宽度.10/仅实际正向比例与指数跟踪；侧漂−3→−.5，偏航−1→−.5且删除角速度正奖励和持续偏航/侧漂。保留慢姿态−.2及前后系数.75，直立+1/body_ang−.1；保留安全/力矩/动作原权重。删除高度/辅助抬升/对角/周期对称/支撑平衡/精确周期/承重后扫质量/外展/工作区/收腿/固定高度/额外俯仰/弹跳等当前奖励配置；接触力−.5改metrics，保留150N逐足峰与超限统计。support_timing−.1合并最短承重.16、驻足.85、悬空.36及启动.6秒；短事件按.10秒/dt与四足均值积分，持续超时均值最差各半，取消.24软目标及.35秒/7厘米质量目标。首次基础阶段不发高度收益，不保证原高度保持。

承重/滑移/摆动状态独立逐环境缓存、重置和非有限过滤，保留足底地形净空峰、周期、承重、后扫及真实滑移作为指标，不参与速度评分。high_step_evaluation.py支持basic，移除对已删除奖励实例的依赖；每足至少5次有效前移循环，不强制15厘米/0.6秒，通用速度/稳定/推进继续验收。旧算法代码未挂当前任务，8份历史测试显式使用新增legacy_high_step_config.py冻结夹具；新增test_high_step_basic.py验证当前15项，避免把旧算法回归混称当前配置验证。

实际执行：101项HighStep/walking_audit张量配置检查通过（10.039秒），17当前+84历史；当前覆盖奖励/指标管理器初始化、计分及局部重置，速度/高度解耦、零速/反向/超速/失败/非有限，正常滚动/真实滑移、短支撑单事件/dt、长驻足/悬空、足高独立观察和基础评估。CPU真实2026-10-30_21-53-51/model_6200.pt严格resume+临时保存，Actor/Critic/优化器/iter6200/common_step_counter148968完全相同，源SHA256 4a30656c54836703bc7991c9b0ffa7f49a71bb7b9e69cff54f9d18e8d3e14b24前后不变；100更新/间隔50/固定6200的完整CLI解析通过。报告artifacts/high_step_basic_validation.json。前后哈希核对确认Elevation配置/核心/净空、Jump及共享PPO五文件未改；45/325/12、PD、平地与随机化保持。未启动物理环境、Viser、PPO或训练。用户报告6200姿态更好，不是本轮回放结论。同步docs/HIGH_STEP_TRAINING.md及memory03/04/06/07与索引；后续先100轮看6250和6299，若停步/拖行不原样长训。

## 2026-10-31实施HighStep接触点滑移、短支撑和距离抬升方案

交付前本轮8个Python文件AST解析及尾空白检查通过，`git -c core.safecrlf=false diff --check`通过。静态、定向张量和真实检查点加载验证全部完成。

用户批准完整计划，固定从2026-10-30_21-53-51/model_7199.pt同任务续训。此前12.5厘米每秒为接触足底参考点速度，不能直接证明真实滑动；当前周期约.34～.35秒、支撑约.17～.18秒、足峰整体10.47/LF8.29厘米，作为修正依据而非动力学因果证明。本轮只改HighStep配置/循环/支撑及相关夹具/定向测试、文档，保留用户其他未提交改动。

- config/high_step.py足接触fields增加pos，保持global_frame；feet_slip换HighStep专用函数、权重−.2。high_step_support.py按腿名解析球心/足底参考site及接触通道，用v+ω×(pos−center)并去接触法向分量计算实际切向滑速。8N真实顶面支撑且持续>.04秒后，容许.02/尺度.10米每秒、平方限1，有效支撑足均值/最差各半；非法坐标系拒绝、非有限数据过滤，无有效支撑0，不改通用feet_slip或其他任务。
- 支撑质量的drift改累计过宽限后的真实滑速×dt，2.5厘米质量下降尺度、.35秒/7厘米目标及.25恢复底座保持；滑出再滑回不消掉累计，正常球足滚动不再因球心净位移直接降质。身体实际推进和相对足端后扫取较小值保持。缓存阻止重复积分，部分reset保留其他环境同周期结束事件。
- short_stance函数换为HighStep实际承重结束事件，软目标.16→.24秒/权重−.05→−.10，平方缺口除dt后由框架积分；主循环有效最低.16秒保持。相关支撑、主循环和收腿承重配置统一8N，其他姿态/偏航/力项保持。
- high_step.py高度分由线性+目标高斯混合改min(峰/.15,1)^2，保留>15厘米以.03尺度衰减；周期占比.30→.50，.60±.12秒/主循环.38保持。原paid_lift每摆动预算移除，改冻结前向轴维护回合历史前沿：新增距离按cmd×dt封顶，每控制周期都消费该段路程，不把无效摆动路程攒到以后。辅助积分=.02×新距离/目标步距×Σ(有效足高度分×相对收腿质量)，目标步距cmd×.6限制4～15厘米；要求真实起脚/至少两足支撑/足前移≥2.5厘米/身前移≥1厘米/离地≤.36秒。往返、重复原地点脚、静止、全腾空、纯机身弹跳、失败和非有限无辅助收益；身体前进的新路程可支持高位摆动持续按距离计分，不宣称所有奖励已无频率影响。
- 新日志high_step_contact_slip_mean_m_s、分足contact_slip_*_m_s、分足contact_slip_distance_*_m及settled_support_feet；旧slip_velocity_mean保留原足底参考点世界XY速度作对照。旧值不能与新真实接触滑速直接作数值下降对比。Actor/Critic/动作45/325/12、网络、PPO/PD/地图保持，奖励缓冲新进程重新初始化。

实际验证：旧65项在核心修改后先通过；新18项首批通过，补实际球心移动的滚动及额外摆动旧路程不复用后，新文件19项，最终84项聚合通过（6.464秒）。范围test_high_step*.py+test_walking_audit.py，覆盖真实接触点、弱触地/法向/宽限、累计和回程、短支撑、周期/高度/过高、距离预算/控制周期/超速/无效信用、缓存/局部reset/非有限/失败、其他任务接口及原约束。夹具补位姿/速度/角速度/接触点且轨迹同步，原低于目标时height_std影响断言改为相等，未用放宽断言掩盖新机制失败。

真实7199只在CPU构造网络严格resume和临时保存，Actor/Critic/优化器逐张量完全保持，iter7199/common_step_counter172944；源SHA256 3170689f7ea12bde64f8c24144edad3e7a99be92b4cb3b000316f3bd2486babb前后不变。完整512环境/300更新/固定load-run及checkpoint/运行名contact_slip_cadence的CLI解析与路径核对通过，末次7498；建议7300/7400/7498回放。源文件未覆盖，未启动/停止物理仿真、Viser、PPO或训练。Elevation配置/核心/净空、Jump配置、共享PPO五文件修改前后SHA256一致。同步HighStep专项及MEMORY任务/操作/验证/状态页；参数为实验初值，实际步态和足高改善待用户短训验收。

## 2026-10-31实施版本7功能姿态与落地缓冲方案

交付前本轮8个Python文件AST/尾空白检查通过，`git -c core.safecrlf=false diff --check`通过。配置、张量和CPU检查点检查全部完成；未启动物理环境或训练。

用户批准完整计划，选择功能和安全姿态，要求保留15厘米/现有承重后扫/推进循环，只增加机腹净空与首次触地速度两项惩罚，其余修正原评分。实施范围为混合Elevation及四个上下阶/坡专项；保留既有未提交修改，不改HighStep/Jump、PPO/网络/PD/45/325/12和地形契约。不恢复已撤回的绝对大腿角/膝位置模板，也未启动/停止仿真、Viser或训练。

- elevation_clearance维护最后承重足点和局部法向，以固定起脚参考计算真实主动峰值进入原比例平方高度分和15厘米目标，保留相对机身收腿门槛、临阶实际立面＋3厘米、承重后扫、参与度和距离预算。原地形净空峰单独保留原指标含义，增加逐足主动抬升指标/累计；下阶地面下降不再提高主动峰，上阶提前越棱不需将立面重复叠加。纯机身弹跳仍被相对收腿条件拒绝。
- elevation_gait增加真实踏面前后边缘余量查询，含初始平台/最终平台；下阶末踏面0高度与平地连续处不生成假棱边。循环稳定确认期间保存最小余量，乘.5+.5×clamp(余量/.05)，保留边缘落脚恢复反馈，不新增独立奖金，不重复横向边缘罚，同阶有效小步仍可领取。非有限余量归零，逐足独立状态和缓存/局部重置保持。
- 新mdp/elevation_safety.py：body_clearance−.3从实际模型中筛选base启用碰撞盒，每盒底面3×3采样，世界姿态旋转后按各点实际地形/坡法向计算最低净空。12厘米死区、6厘米尺度平方代价封顶1；记录逐周期与回合最小值，缓存防重复，按env_ids重置。九点采样不等于连续碰撞检测保证。
- touchdown_speed−.2在原collect_elevation_contacts物理子步回调中锁存前一子步球心速度，首次有效顶面接触按primary→terrain法向计算正接近速度；.30米每秒死区/.50尺度，逐足事件归一化代价≤1、事件.10秒，控制率返回值除dt确保积分尺度一致（每足单次最大扣.02）。即使同一控制步内触地又离地仍保留事件；持续静态承重不重复收费，物理计数/控制计数各自缓存。无前帧有效速度的初始化不伪造落地速度，非有限数据拒绝，失败/完成不支付旧事件；重置清空对应足缓冲而保留其他环境事件。
- contact_cost仍−.2，walking改.5×(body接触均值+foot非支撑接触均值)+.5×两组最差值再封顶，降低单腿被平均稀释；Jump显式保持原body均值+foot均值公式。reset_elevation对安全项显式局部reset，RewardManager的再次reset幂等。
- runner奖励元数据revision6→7、gait_style=functional_clearance_and_soft_landing，记录起脚法向参考、5厘米余量、新参数/权重/接触聚合。几何契约不变，同专项resume仍恢复网络、优化器、迭代及课程，仅提示奖励改变；跨专项既有transfer要求保持。新增指标elevation_true_lift_*_m、landing_edge_margin_m、body_min_clearance_m、touchdown_speed_*_m_s，旧swing_peak_height仍为地形净空。
- evaluate每回合加入functional_posture，逐足主动峰、触地速度均值/最大值、机腹最小净空和踏面余量/近边落脚计数；每档functional_posture_summary按真实事件次数加权，缺事件写null，避免将缺少换步误写为零冲击。原nominal90%/robust80%通过率判据保持，功能质量需并行人工验收。

实际检查：原106项先通过，新首批15项通过后补4项；最终195项聚合通过（19.570秒），组成原Elevation106+四专项15+HighStep55+新功能19。覆盖真实/被动净空、上下坡切线、提前越棱、中心/棱边收益、底面角点/旋转/平移/局部重置、前帧触地/短时接触/事件尺度、失败非有限、单腿碰撞与Jump隔离、报告加权/null。未执行含原生编译/射线的ElevationGeometryTests或整个test_elevation.py，也未做动力学验收。

真实源logs/rsl_rl/pointleg_elevation/highstep_fixed120_pretrained/model_0.pt只在CPU完整resume并临时保存，Actor/Critic/优化器逐张量、迭代0/common_step_counter0及地图契约保持；仅1条预期奖励变化提示，临时元数据7，源SHA256 be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568前后不变。没有写回源文件或创建正式专项模型。同步专项奖励/四专项入口/三个历史设计入口以及MEMORY任务/操作/验证/状态页；200轮示例从已验证混合model_0直接resume，四专项继续使用各自实际初始模型。所有参数为实验初值，需新进程后回放/正式评估，不能承诺奖励修改即达到图片姿态或通过率。

## 2026-10-30将Elevation拆分为四个独立专项训练入口

用户要求将`src/pointleg`中的Elevation拆分出上台阶、下台阶、上坡、下坡四项，供后续训练。本轮实际新增四任务并保留原混合任务；未选择训练源模型、未迁移正式权重、未启动或停止训练/仿真/Viser，保留原有未提交改动。

- 新配置`config/stairs_up.py`、`stairs_down.py`、`slope_up.py`、`slope_down.py`各自提供环境与PPO工厂，注册`PointLeg-StairsUp-v0`、`PointLeg-StairsDown-v0`、`PointLeg-SlopeUp-v0`、`PointLeg-SlopeDown-v0`，实验目录分别为`pointleg_stairs_up`、`pointleg_stairs_down`、`pointleg_slope_up`、`pointleg_slope_down`，使用TensorBoard且不上传模型。注册项9→13，启用8→12，Bridge仍占位。
- `ElevationSettings.terrain_kind`从隐含混合范围扩展为显式`None/四类地形名`，原工厂默认None不变，专项工厂传固定名称。`mdp/elevation.py:select_elevation_terrain`对全量和局部重置都只分配该专项，固定评估亦拒绝错误地形；原混合的四类障碍加flat采样不变。物理地图保留原六定义、4×4布局及语义编号，未重写MDP判定或改变几何契约。
- 四项初始继承Elevation版本6全部奖励、传感器、终止与PPO；速度.15～.25、时限60秒、足高目标.15米、Actor/Critic/动作45/325/12保持。台阶单级.12/踏面.32/宽.8米、两段各三级、随机中间平台和下降顶面.72米保持；台阶无高度晋级，坡面约6.67～20度课程保持。专项课程日志只输出对应地形，内部仍保留原六槽状态。没有新增Actor地形观测。
- `terrains/elevation.py:ELEVATION_TASK_TERRAINS`维护混合/专项任务映射，`run.py`将专项evaluate分派至`evaluate_elevation.py`；`evaluation_cases`从混合21个用例收窄为每专项对应五档，保留nominal/robust语义。`transfer.py:prepare_transfer`接受四新任务，并在迁移元数据保存任务/地形范围。
- `ElevationOnPolicyRunner`保存`terrain_kind`，同专项resume继续恢复网络、优化器、计数、迭代及课程；跨专项或混合/专项互转的完整resume会拒绝，需仅权重transfer重新建立目标状态。旧混合检查点无新字段等价于None，仍可同混合任务resume。`flat_rehearsal`契约在专项为False、原混合仍True；布局契约与奖励revision6未变。未来选一专项迁回原混合继续训练可行，但四教师蒸馏/自动权重合并未实现。

实际验证日期2026-10-30：新`test_elevation_specializations.py`15项通过（4.464秒），包括8组从头/续训CLI参数解析、注册/配置隔离、全量/局部地形采样、固定评估越界拒绝、上下出生几何、固定台阶与坡度课程、同专项resume/跨专项迁移及旧混合兼容。临时合成检查点仅CPU网络加载/保存，不含物理仿真；比较Actor/Critic逐参数、初始优化器/迭代/课程、源文件SHA256前后相同，不生成正式模型或报告。

已有回归实际执行`.venv/Scripts/python.exe -c "import sys, unittest; sys.path.insert(0, 'src/pointleg/tests'); names=['test_elevation_alignment','test_elevation_clearance','test_elevation_gait','test_elevation_support','test_elevation_fixed_stairs','test_elevation_retention','test_walking_audit','test_elevation.ElevationLogicTests','test_elevation.ElevationWalkingTests','test_elevation.ElevationTransferTests']; suite=unittest.defaultTestLoader.loadTestsFromNames(names); result=unittest.TextTestRunner(verbosity=1, buffer=True).run(suite); sys.exit(not result.wasSuccessful())"`，106项通过（18.423秒）。没有执行`ElevationGeometryTests`、物理积分、场景/Viser初始化或PPO。新旧共121项不同单元检查通过；`python -m pointleg.run list`实际列出12项启用及Bridge占位，不能等同于专项策略已收敛或实机验收。

新增`docs/ELEVATION_SPLIT_TRAINING.md`，提供四项从头/迁移/续训/回放/评估命令、共享实现边界及后续迁回混合说明，并同步README、MEMORY及架构/任务/操作/验证/状态页。配置需新进程采用，现有PT和训练进程不自动变化；尚无四专项的动力学或正式通过率验证。

交付前13个相关Python文件AST解析通过，相关源码/项目记忆/新指南无尾空白，`git -c core.safecrlf=false diff --check`通过。首次全文件尾空白扫描发现README历史正文有一行既有尾空白，保留原内容后按修改范围检查通过；未清理无关改动。

## 2026-10-30按用户指定修改HighStep作为平地纠正第一阶段

交付前本轮8个Python文件AST解析/尾空白检查及`git -c core.safecrlf=false diff --check`通过。训练CLI按实际`mjlab.TYRO_FLAGS`解析通过，源路径准确指向用户指定15498；仅检查解析与路径，没有调用训练入口。

用户指定`pointleg_high_step/2026-10-28_23-25-35_yaw_mild_soft_contact/model_15498.pt`作为起点，要求修改HighStep并给训练指令。本次落实用户选择，覆盖此前仅建议的Elevation同任务平地模式；没有实现该Elevation开关，也不改Elevation版本6。保留其他未提交改动、网络/PPO超参数、PD、地形与45/325/12。

- `config/high_step.py`目标高度.17→.15，原.03评分宽度、.60秒周期/.12宽度/循环权重.38保留；short_stance最低.12→.16秒。主循环仍按实际周期预算计分，不额外加推进或支撑奖金。
- 新`mdp/high_step_support.py`跟踪四足世界足位与机身水平推进，真实连续8N承重最短.16秒、质量目标.35秒/有效后扫.07米、足位最大漂移容差.025米。后扫取机身实际推进与机身相对足推进较小者；脚世界坐标不动/机身前移有效，后滑不得放大收益，滑出滑回保留最大漂移。离地冻结质量，原循环评分乘.25+.75质量，保留恢复反馈。
- 连续驻足>.85秒按.25秒尺度加−.3代价，平均/最差各半；原相位项stance_weight=0，保留离地>.36秒惩罚，避免重复短支撑偏好。速度项使用原本地XY/upright跟踪函数，不引入额外竖向速度误差；仅前进时乘历史新前沿比例、正向速度比例和四足换步新鲜度，至少两足承重，静止/旧前沿往返/全腾空/失败不领前进速度收益。10%显式站立指令仍保持原站立跟踪。
- 每足成熟落地循环峰≥目标20%（3厘米）、相对收腿质量≥40%刷新年龄；.6秒内1、.6～1.2线性到0，四足平均/最差各半。该衰减只门控速度项，不阻断恢复循环。支持同周期缓存和按env_ids重置，非有限数据和终止掩码过滤。
- 支撑工作区及高位摆动收腿统一髋参考中心−.02/余量.14米，即髋前12/后16厘米软边界，收腿权重−.2保持。慢关节均值原左右约束−.2保留，新增前后比较系数.75（有效−.15），EMA.8→1秒/宽限1→1.2秒，髋/膝容许8/12°、尺度15/20°；允许共同屈腿和正常交替，不直接奖励关节摆幅或规定瞬时同角。非有限关节状态屏蔽。偏航/侧漂/150N过载及其余奖励保持。
- 新逐足指标high_step_stance_time_*_s、backstroke_*_m、push_quality_*及整体gait_renewal，原前后髋/膝慢偏差指标继续记录。支撑指标按离地事件统计，不能据此声称实际机械功或力峰已改善。

实际验证：65项定向单元全部通过（4.590秒，test_high_step*.py及test_walking_audit.py），新增10项纯张量轨迹检查，含长承重实际循环得分高于短触地、小推进/纯弹跳/滑移降质、后滑不能增信用、接触排序/航向等价、驻足/换步、静止/往返/失败/无效/腾空、缓存/局部重置/配置隔离。既有目标/工作区断言按新配置调整。真实源model_15498只在CPU构造网络严格加载并临时保存，Actor/Critic/优化器及迭代/公共计数完全保留；源SHA256 `9c9879852f87b017a6e7155301d5e11b03d991c94a4127e16690dad4e9bf34b8`前后不变。源实际迭代15498、common_step_counter=372240，配置学习率仍8e−4；没有启动物理仿真、Viser、PPO或训练。

训练命令仅完成CLI解析/精确源路径检查，交付同任务直接resume512环境、追加200轮、run_name=flat_front_correction；不transfer、不从退化Elevation400继续、不重置model_0，源文件不覆盖。按框架计数检查15600与末次15697，周期文件另有15500。平地验收需四足持续换步、分足均值≥12厘米、稳定速度误差≤.05米/秒以及前后慢姿态偏差下降；失败不原样延长至2000轮。后续迁入Elevation需先验收，本轮不执行。同步专项文档与MEMORY/任务/操作/验证/状态页，新进程才采用，不能把单元通过写成策略已改善。

2026-10-30用户询问先平地纠正前腿位置是否有必要。本轮只读核对HighStep配置和慢对称实现，确认其前后差仅记指标、实际只左右比较；Elevation版本6才有本次承重后扫/换步门控/前后慢姿态。给出先在同一Elevation奖励下仅平地纠正、达标再恢复固定120毫米台阶的建议，平地仍退化则先修训练目标，不把平地通过当爬梯保证。仅平地模式尚未实现，没有修改训练源码、配置或开始训练，已在操作/状态记忆区分建议与实现。另阅读原始研究https://arxiv.org/abs/2505.16084，其采用平地运动先验后用分层残差适应复杂地形；只作为分阶段思路参考，不声称本项目已采用该架构或取得同等效果。

## 2026-10-30版本6承重后扫、换步、前后慢姿态及近距离出生

交付前本轮10个Python文件AST解析与尾空白检查通过，`git -c core.safecrlf=false diff --check`通过。

用户反馈版本5在400附近出现足端固定、身体前探后停止，支撑期短、前腿前伸/膝位置靠前，要求改奖励和环境。先只读核对运行`2026-10-30_15-47-37_height15_low_speed_v5`，实际agent.max_iterations=1300、事件到1299，源仍highstep_fixed120_pretrained/model_0；0～49/400～449整体足峰.15297/.12886米、质量步6.48661/3.50526每秒、前足reach .16192/.18468→.14292/.17791米，后足约−.00028/−.02429米。不能从一帧或接触占空比直接证明实际支撑力；新增连续承重/后扫指标用于后续确认。

仅修改Elevation及相关无仿真检查，保留其他工作树改动。参数前后与实现：

- `elevation_clearance.py`维护每足连续8N顶面承重时长、支撑起始足/根位置与地形切线、最大切向漂移；离地冻结推进质量。最短有效推蹬.16秒、目标.35秒/有效后扫.07米、漂移容差.025米，质量为三者归一化乘积；有效后扫取身体实际切向推进与相对足端推进较小值，防止靠脚后滑增加信用。循环质量再乘.25+.75*该质量，保留低质量恢复反馈，不增加独立支撑奖金；原15厘米、距离奖金及落足/收腿/新前沿约束保持。纯垂直弹跳、原地抖动不独立得高推进质量；支撑身体世界Z变化仅记录诊断。
- 稳定落地确认.06→.10秒；short_stance最短.08→.16秒、权重−.02→−.05。增加stance_overstay−.3，连续真实承重>.85秒起罚，.25秒尺度归一化封顶，最差足/平均各半。
- 增加每足换步年龄；真实循环峰值至少当次目标20%且相对收腿质量≥40%才刷新。年龄≤.6秒系数1、.6～1.2线性至0，平均/最差各半。现有速度奖励调用同周期缓存步态项后乘该系数，固定四足前探不能持续领速度正收益；其他循环参与度逻辑保持。
- `elevation_gait.py`工作区中心+.035→−.02米、前后余量.16→.14米，即前界+.195→+.12、后界−.125→−.16；障碍入口前.6至出口后.5米再放宽.04米。侧向规则与足端竖向自由保持。
- `elevation_support.py`新增mean_leg_posture−.15：髋俯仰/膝相对默认姿态的1秒EMA，至少两足支撑并满1.2秒后比较，前后均值/左右差各半。髋/膝容许8/12°，尺度15/20°；承重足高差>.03米时前后容许×2.5、全项代价×.25。共同屈腿、交替相位允许，不强制瞬时同角或复刻旧源姿态。部分reset同周期其他环境失败掩码遗漏已通过新测试发现并修复。
- `ElevationSettings.walking_approach_distance=1.0`，walking楼梯/坡面reset根点改为入口−1.0米，继续±.3扰动；原入口3/地块spawn1.2约1.8米改为.7～1.3米，名义根点2。下阶仍顶面出生、纯平地复习和Jump出生不变。原spawn_table/地图几何未改，因此布局契约仍revision3，版本元数据5→6并记录接近距离，仍允许同固定地图直接resume。
- 记录逐足elevation_stance_time_*_s、backstroke_*_m、support_body_rise_*_m、push_quality_*，整体gait_renewal和前后髋/膝慢偏差度数。15厘米/低速/60秒、120毫米楼梯、随机平台、坡面课程、PPO、PD/力矩及45/325/12均保持，未恢复旧腿杆净空或绝对膝后移项。

实际验证：105项聚合通过（16.843秒），再补1项实际循环落地收益比较通过（.563秒），共106个不同检查；新支撑文件13项涵盖长/短支撑收益、真实固定脚/身体前移、纯垂直/滑移、小后扫、坡面、后滑/滑回拒绝、驻足速度过期/换步恢复、缓存/部分重置/失败、工作区、慢姿态/异高放宽/非有限和近出生/任务隔离。原Logic/Walking/Transfer选择性运行，未执行含原生物理检查的GeometryTests或整个test_elevation.py。轨迹积分断言按约2e−6累积误差调整，未以放宽断言掩盖机制失败；没有仿真、Viser、PPO或新策略验收。

用XML和默认关节角做纯刚体变换（没有编译/积分）得到前足球面最前缘距根点.43910095米；最近根点距入口.7米，默认姿态前足静态余量.26089905米，不证明动态/关节噪声安全。真实固定地图model_0 CPU加载/临时保存再次核对Actor/Critic/优化器、迭代0、45/325/12和地图契约保持，奖励版本6/接近距离1采用；源hash仍`be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568`，未改源文件。

交付200次更新指令run_name=support_push_v6，从原固定地图model_0直接resume，不再transfer，不从已退化400后继续；建议先看100和末轮199的足高、连续承重/后扫、四足换步及真实上阶。未替用户启动或停止现有进程，旧进程不会热更新。同步专项及三个历史设计入口提示、MEMORY和任务/操作/验证/状态记忆；参数是训练初值，不能承诺单元通过即修复实际步态。

## 2026-10-30实施用户批准的Elevation版本5主动迈步与15厘米方案

交付前静态检查：本轮10个Python文件AST解析、尾空白检查通过，`git -c core.safecrlf=false diff --check`通过；已确认撤回的elevation_posture.py没有被恢复。

用户完整批准实施此前计划，并已选择15厘米抬脚和对齐基础0.15～0.25米/秒低速。本轮只改Elevation及其定向测试/文档，不恢复版本4髋角/膝部/腿杆姿态项，不改固定120毫米地图、45/325/12、网络、PD/力矩或PPO学习率/自适应调度/熵系数。保留其他未提交改动，没有启动、停止训练/回放或执行物理仿真。

实际参数与代码：

- `config/elevation.py`基础target_clearance .08→.15米；`mdp/elevation_clearance.py`高度质量从比例到比例平方，临阶仍取基础与真实立面+3厘米较大值。用于高质量计数的80%仍判原高度比例（12厘米），防止误变成约13.42厘米。低摆腿继续有连续小额反馈及原参与度刷新；不新增髋角模板或固定节奏。
- step_bonus .04→.16，reward_progress_scale .08→.16米；在16厘米预算内每8厘米新的机身前沿进度满质量+0.08，单足单次最多+0.16。保留足端3厘米/机身2厘米新前沿、真实承重起脚、离地至少.08秒、至少两足支撑、落地.06秒、最短周期.4秒、收腿和长跨步质量；原地、往返、拖行、全足腾空、纯机身弹跳不能独立刷循环。正事件读取与生成均增加完成掩码，与失败/无效统一。
- `mdp/elevation.py`增加训练内部forward_frontier/forward_progress，以局部出生位置初始化、回合重置按env_ids独立清零，由缓存的update_elevation_state每控制周期计算刷新历史前沿的新增距离。新elevation_forward_velocity_reward替换walking velocity负代价，外层+1、std=.1；为新增距离/(cmd*dt)封顶1 × 正向体速/cmd封顶1 × exp(-((v-cmd)/.1)^2)。要求初始化、有限正指令、至少两足支撑，失败/完成/无效清零；静止、只有速度而没位移、返回旧位置不得分，不另加command_aligned_progress等重复奖励。
- action权重−.05→−.02、leg_joint_acc_l2−2.5e−7→−1e−7；参与度、滑移、侧漂、偏航、接触力及完成/失败项保持。实际配置工厂速度从.2～.5到.15～.25，训练/回放回合35→60秒。原1秒停滞宽限和3秒净推进窗口不变，60秒不代表允许长期原地不动。
- `evaluate_elevation.py`forward_speed默认.35→.20；先保存任务配置时限，再调用通用评估并恢复时限，避免通用configure_evaluation的1e9覆盖60秒；显式时限覆盖继续生效。`ElevationOnPolicyRunner`奖励revision3→5、style=terrain_height_retention，记录速度范围/15厘米目标；几何契约仍revision3，奖励标签差异仅提示，不强制transfer。

实际检查：旧50项先通过，首批13项新增检查通过；补组合轨迹时奖励类调用方式错误已修为已实例化项，补两足支撑门槛后旧速度用例显式设置支撑样本。最终六文件定向加ElevationWalkingTests/ElevationTransferTests共77项通过，17.788秒，无跳过或失败。新增14项覆盖前进信用、静止/往返、两足支撑、缓存/局部重置、终止掩码、非有限、平移/控制周期、8/12/15厘米、距离拆分预算、12厘米合格门槛、拖行拒绝、组合轨迹、时限与短停滞。组合轨迹使用真实奖励函数的速度/循环/参与/滑移/动作/净空/步长七项，不是实际动力学总回报，也不能证明策略已经改善。没有运行整个含原生ray/compile的test_elevation.py。

真实源加载检查只在CPU构造网络/临时检查点：`highstep_fixed120_pretrained/model_0.pt`完整resume成功、迭代/公共计数0、优化器保持、Actor/Critic张量完全一致、原布局/45/325/12保持，新配置速度.15～.25、时限60秒/3000控制步、奖励元数据5/目标.15采用。仅出现预期奖励变化提示。源SHA256 `be58e94aa5082877b4bb7ea0dde344fe8b949d15f19b611b69430b7acdc4e568`前后不变，未写回源文件。

交付指令从已存在固定地图model_0直接resume256环境、追加200次更新、run_name=height15_low_speed_v5；不再transfer、不从600/700/1999恢复。核对框架0起始保存命名后，将计划中的“100、200”说明为实际model_100和末轮model_199（200次更新），不改名伪造轮数。日志默认目录，先由用户回放；平地/接近段逐足峰值均值至少12厘米、稳定速度平均误差≤.05米/秒且主动换步；明显静止/拖腿不原样加至2000。同步专项及实现/历史设计提示、MEMORY和任务/操作/验证/状态页。参数仍为实验初值，没有正式训练或通过率验收。

2026-10-30用户明确model_700开始不迈步、只静止。只读确认fixed120_stairs运行600/700/800检查点存在，取TensorBoard50条窗口核对退化时间：0～49整体足峰0.162355/左前0.158755米、step_quality0.053656；600～649为0.092828/0.072582米、step_quality0.076005、质量步4.996720/秒；650～699左前0.065772米；700～749整体0.079133/左前0.049249米、step_quality0.066400、质量步3.680460/秒。退化早于用户700回放失效点逐步发生，600也不是未退化起点；训练多环境聚合不能代替该PT的Viser无动作观察。当前步态项在足峰下降时仍高于起始，需审查目标保持，不能据此称静止总回报更高。同步状态记忆，本轮未改训练源码、模型或启动/停止任何训练/回放，没有新增测试。

2026-10-30用户反馈固定高度训练后连迈步都不会，并明确只会静止。本轮只读`2026-10-30_12-13-08_fixed120_stairs`配置、TensorBoard事件与迁移检查点，未改训练源码或模型。记录0～1999；开头/末尾各50条足峰16.2355→3.6741厘米、前足FL2.4937/FR2.6709厘米，有效循环6.4777→0.3865/秒、停滞聚合0.079084→0.514851、mean_reward−56.476→−84.651，台阶上下始终固定0.12。按每100条阶段看足高0～99约15.855、100～199约14.052、300～399约12.041、500～599约9.837、1000～1099约7.115、1900～1999约3.418厘米，退化发生在训练更新过程中，不能据一帧图归因机械结构。

核对源`2026-10-28_23-25-35_yaw_mild_soft_contact/model_15498.pt`与`highstep_fixed120_pretrained/model_0.pt`，Actor及Critic所有state_dict张量均完全相同；保存env.yaml的Actor观测、动作配置完全相同、decimation均4。源速度0.15～0.25而Elevation0.2～0.5米/秒、基础目标仍8厘米，源HighStep循环奖励没有随网络迁入，当前有效循环奖励依赖足/机身推进和落地结算。源迁移正确不等于新任务会保持步态。实际Policy/mean_std0.3505→0.5759、末50条学习率4e−5，不把配置3e−4或某项惩罚直接当作已证实主因。建议下一步对齐低速和持续步态目标后从源基础开始，未实施、未启动/停止进程，截图具体播放文件未知，没有仿真、Viser或新的测试。同步操作和状态记忆。

2026-10-30用户请求固定120毫米台阶训练指令。只读确认此前HighStep15498源文件存在、固定高度配置为.12且输出highstep_fixed120_pretrained未创建；提供现有transfer迁移至该目录，再256环境resume其model_0追加2000轮、运行名fixed120_stairs的完整uv命令。同步专项文档与操作记忆；未执行迁移、训练、仿真或测试，未改源模型/参数，日志保持默认位置。

## 2026-10-29按用户要求回滚版本4并取消台阶高度课程

用户明确要求把最近一次代码改回去，且不要低台阶，已有高抬腿基础直接学习比赛台阶。已按上一轮变更边界撤回Elevation版本4，而不是执行此前未落地的15厘米平地目标方案；没有使用git整仓回滚，没有触碰其他未提交改动、模型或日志。

- 回滚`config/elevation.py`三项hip_swing（+1/每足最多+0.02）、leg_clearance（−0.8）、flat_hip_fold（−0.2），删除`mdp/elevation_posture.py`及`test_elevation_posture.py`，从`elevation_clearance.py`移除25°髋摆幅奖励/状态/指标、恢复原函数参数与三类循环反馈，`elevation_gait.py`恢复原正奖励掩码。Runner奖励元数据revision4→3、gait_style恢复已从旧model_100元数据核实的graded_contact_steps。足底基础8厘米、临阶3厘米余量、其他奖励及45/325/12保持。
- 台阶原40～120毫米高度梯度改为实际Elevation配置固定120毫米，踏面320毫米/宽800毫米保持。`terrains/elevation.py`增加可选stair_height_override，所有训练/回放行生成相同单级高度，累计六阶720毫米、下阶出生同步在顶面；两段三级、1.2～2.0米随机中间平台、末平台1.6米、起步±0.3米、物理块布局保持。通用构造器默认保留历史梯度供旧合成测试使用，不代表现行Elevation仍训练低台阶。
- `mdp/elevation.py`固定高度台阶不再累积成功率窗口或按成功/失败升降；重置随机行只选择等高平台变体，评估固定case行不被覆盖。旧状态加载时清理台阶帽/窗口，坡面帽/窗口正常恢复，坡度原课程保持。台阶日志从_cap换为_fixed_height_m=.12；`evaluate_elevation.py`case记录fixed_step_height_m，避免把五个行变体误读为五档台阶高度。
- 楼梯layout_contract revision2→3并记录fixed_step_height_m，旧梯度地图完整resume仍被拒绝，但网络仅权重加载正常。旧HighStep/旧Elevation源要通过既有transfer生成当前固定高度契约后训练，之后同地图可直接resume；这是实际几何变化，与仅改奖励允许resume的规则一致。本轮没有替用户迁移文件或启动/停止训练，也没有因回滚源码而恢复已退化权重，不默认从2099继续。

实际验证：五文件定向50项通过（3.186秒），额外ElevationTransferTests6项通过（0.338秒）；9个本轮Python文件AST与尾空白检查通过，已确认两个版本4专用文件删除，git diff --check通过（仅既有LF/CRLF提示）。5项固定高度测试检查上下台阶各行高度/平台变体/下阶出生、成功失败不降阶、坡面独立、旧台阶课程清理、奖励移除/评估契约与非法高度；6项加载检查包含旧高度课程布局不允许冒充新地图resume、允许仅权重加载、优化器/迭代/其他课程保持与迁移清零。合成夹具只构造几何/张量，未编译或积分物理；临时加载测试无PPO更新，没有Viser、仿真、正式训练或通过率验收。同步专项及三个历史设计文档提示、项目索引和任务/操作/验证/状态记忆。低台阶并未被证明为此前退化唯一原因，固定120毫米也不是学习成功保证。

2026-10-29用户反馈版本4膝部略抬但足端高度下降。本轮只读最新`2026-10-29_19-42-14_hip_swing_clearance`的env/agent与TensorBoard事件，末步2099、检查点model_2099存在。开头/末尾各50条指标均值：整体足峰0.146937→0.054600米；FL0.139620→0.037091、FR0.151323→0.048966、RL0.153623→0.071789、RR0.143599→0.061093。前髋摆幅FL15.149→12.665°、FR15.236→7.351°；高质量步6.89675→1.72411/秒，停滞指标0.012872→0.263752，训练mean_reward−54.218→−79.394，课程帽全0。确认并非仅脚高与膝高交换，整体训练有退化迹象；不能将终止/停滞聚合指标当正式分地形成功率。运行配置仍足底目标8厘米、髋收益需高度和收腿质量均80%，高度8厘米以上饱和；上次实现未提供保留源模型约15厘米高抬腿的目标。提出提高平地目标至15厘米、髋奖励从硬80%门槛改连续质量缩放作为下一步方案，尚未修改源码或启动实验，不因日志代价量级直接断言因果。同步状态记忆。

2026-10-29按用户请求提供Elevation版本4续训命令：只读列举确认`2026-10-29_14-35-10_compact_swing/model_100.pt`等检查点存在，源agent记录为`highstep_pretrained/model_0.pt`。依据用户早期100轮比1000轮蹲低退化前的描述，建议以100轮作为修正实验起点，256环境直接resume追加2000轮，运行名`hip_swing_clearance`，默认日志目录。同步专项文档/操作记忆；没有启动、停止训练或修改检查点，没有把该起点写成最优模型。

## 2026-10-29 Elevation地形相关腿部净空与有效大腿摆动

用户授权加入此前分析的膝/腿杆保护、轻量平地屈腿参考，并要求鼓励大腿关节摆动；指出基础model_0已存在前腿大腿摆幅小的问题。只调整Elevation奖励，不模仿旧模型姿态、不强制前后腿相同角。未启动/停止已有训练，未修改PT或其他任务。

修改前Elevation没有腿部接近地面的连续惩罚，也没有有效循环髋俯仰摆幅收益。版本4参数与实现：

- `config/elevation.py`的`hip_swing`由无到+1外层权重；`elevation_clearance.py`在真实摆动中跟踪髋俯仰角极差，目标25°、每足有效循环额外最多+0.02。仅在原循环前进、收腿、合法承重和落地0.06秒成熟后支付，净空和相对收腿质量均至少80%；继承新落点/机身前沿与0.4秒最短周期，原地摆腿、无前进、低净空不独立得分。`elevation_gait.py`对新正项沿用同周期失败/无效掩码。
- `mdp/elevation_posture.py`增加`leg_clearance`−0.8，膝包络余量0.06米、腿杆0.025米、包络半径0.025米、启动0.6秒。检查膝、大腿中点、膝至球足中心25/50/70%点，沿前后包络查询真实阶面并按坡面法向修正；不罚小腿末端30%的正常足接触。最差采样点/平均和最差腿/平均各占一半。离散近似不保证完整扫掠净空，这些是训练初值。
- `flat_hip_fold`由无到−0.2，平地支撑样本EMA0.5秒、累计支撑0.6秒后，对膝沿切向落后髋不足0.08米按0.08米尺度软惩罚。仅实际承重时更新并计分，离地保留慢参考不计分；障碍入口前0.5米至出口后0.5米内放宽，允许上/下阶和坡面调整，不设置固定世界膝高度或固定髋角。
- 新增逐足`elevation_hip_swing_*_deg`、`elevation_knee_clearance_*_m`、`elevation_link_clearance_*_m`、`elevation_knee_setback_*_m`。摆幅指标只对真实已完成且有限的样本写入，净空指标剔除非有限几何。
- `ElevationOnPolicyRunner`奖励契约revision3→4，gait_style改`terrain_clearance_hip_swing`；保留标签差异仅提示、直接resume优化器/迭代/课程的既有逻辑。45/325/12、平地足底8厘米局部目标、速度、地形、课程结构、PD/力矩和其他任务不变。

实际验证：既有Elevation净空/步态/纠偏35项通过；新姿态11项先通过，补启动宽限成为12项。最终聚合五文件57项通过，39.261秒（命令如下）；本次7个Python文件AST解析与尾空白检查通过，`git diff --check`通过（现有文件LF/CRLF提示不构成失败）。配置/张量夹具不编译或积分物理模型，没有Viser、PPO、正式训练或策略通过率验收。同步专项文档及项目索引/任务/操作/验证/状态记忆。新进程续训才采用新奖励，旧模型和旧运行不会因源码变化自动改善。

```powershell
.\.venv\Scripts\python.exe -c 'import unittest; loader=unittest.TestLoader(); suite=unittest.TestSuite(loader.discover("src/pointleg/tests", pattern=p) for p in ("test_elevation_clearance.py", "test_elevation_gait.py", "test_elevation_alignment.py", "test_elevation_posture.py", "test_walking_audit.py")); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not result.wasSuccessful())'
```

更新日期：2026-10-29。记录本对话已实施的变更及后续维护要求；当前参数仍以源码和对应进程的启动配置为准。没有精确分项时间的历史变更按主题补录，不声称刚刚重新执行了训练或验证。

## 2026-10-29用户指定0.6秒周期、高度收益区分与弱化收腿

用户明确要求目标周期改0.6秒、稍提高周期奖励、增加15/17厘米收益差距、暂削减收腿惩罚。`config/high_step.py`主循环目标0.40→0.60秒、周期高斯宽度0.08→0.12秒，权重0.34→0.38；新增可配置`timing_weight=0.30`（原周期系数0.25），落点系数仍0.25、基础项0.50→0.45。高度目标保持0.17米，宽度0.05→0.03米；其他质量相同时15厘米高度分由约0.867→0.762，17厘米为1。收腿权重−0.6→−0.2，未设置自动恢复日程。

`mdp/high_step.py::HighStepCycleReward`主循环分数乘实际完成周期/目标周期预算，范围0～1.5，减少短周期重复落足的次数优势，长等待不能无限领取预算；周期高斯继续偏好0.60秒。辅助抬升仍为每足每摆动封顶0.02的小额反馈，本轮未严格归一它，不夸大为消除所有频率影响。新增`Metrics/high_step_height_score`、`high_step_period_score`及`high_step_cycle_budget`按实际落足事件记录。

为与0.60秒一致，最大摆动0.24→0.36秒、支撑超时0.34→0.51秒、相位尺度0.12→0.18秒；最小离地0.10和最小承重0.12秒保持，保留旧策略转节奏时的奖励反馈。速度仍0.15～0.25米/秒，其同足相邻落点间距需9～15厘米，故目标间距上限10→15厘米（新增`max_target_step_length`），合法落点位移上限14→18厘米；18厘米是余量上限而非要求每次向前甩18厘米。原髋参考工作区/侧向限制、真实支撑/合法推进、静止站姿、温和偏航−1/尺度9、150N接触力、PD、45/325/12、地形及其他任务保持。验收器从奖励配置动态读目标/周期容差，随之改用0.60±0.12秒，无固定相位输入。

示例从用户当前`2026-10-29_12-53-21_compact_swing/model_16499.pt`直接resume，SHA256=`0a5b2a5fbc379cdf9e493177e2cd2b54cb42f56c711155df052348ffe935a2b5`，512环境追加1000轮，运行名`period06_height17`。同任务奖励变化无需transfer，原模型不覆盖；没有启动/停止训练。旧16499动作不会随奖励源码变化自动变慢/变高，须新进程训练并重新观察。

实际验证：HighStep/walking_audit聚合55项通过，1.461秒。新增5项周期/高度数值检查，覆盖计分频率公平、长等待封顶、目标节奏与高度优于快速低步态、15/17区分、相位/速度步幅一致及非法参数。原夹具改为0.36秒支撑+0.24秒摆动、12厘米落点；非法长摆动样本改0.40秒以匹配新0.36秒上限。没有物理仿真、Viser、PPO、长训练或全量回归；单元结果不证明新策略已达到17厘米或0.6秒。同步专项文档、主索引、任务/操作/验证/状态页。
## 2026-10-29高位摆动紧凑收腿与静止站姿隔离

用户在两张回放截图中观察到前腿足端较高而膝空间位置变化小、后腿有明显上下运动，要求按前述分析修改奖励。没有把膝位置不动当成膝关节角不动，也未据截图确认唯一原因；本轮不自动运行仿真。

新增`mdp/high_step_coordination.py`：`high_step_static_joint_posture`复用原关节姿态代价，但仅在平移指令范数与偏航指令绝对值之和≤0.1时保留；HighStep前进时返回0，静止保持−0.15权重、原档位和姿态函数行为。Flat/Rough/Elevation等仍用原函数，未改PD或关节限位。

`HighStepSwingRetractionPenalty`以腿名解析接触/足球心/髋俯仰与膝body及关节，用真实承重后离地跟踪摆动，支持局部重置和无效输入排除。配置项`high_step_swing_retraction`−0.6，仅向前命令、启动0.6秒后且足底高≥6厘米、摆动全程至少两足有力支撑时，惩罚足端相对各自髋俯仰参考点在机身坐标的过大前后偏移；名义+3.5厘米、余量±8厘米、尺度8厘米，每足限幅4，平均与最差各半。不惩罚支撑足或低位起落脚，不直接奖励髋/膝乱摆、不锁角度或固定前后腿轨迹。期望通过几何收腿引导髋膝配合，但效果需训练验证。

真实完整摆动（此前承重≥0.12秒、离地≥0.08秒）记录`high_step_swing_hip_range_{leg}_deg`、`high_step_swing_knee_range_{leg}_deg`及`high_step_swing_knee_vertical_range_{leg}_m`；膝位移使用膝相对髋的机身坐标，避免机身平移/弹跳冒充抬膝。高位摆动记录`high_step_swing_foot_x_{leg}_m`。无事件不写虚假0，真实不动关节的0幅度照实记录。指标不加入Actor/Critic。

保留用户已将抬腿权重设为0.34及17厘米目标、原循环事件评分/结算、温和偏航−1/持续尺度9、150N接触力项、地形和45/325/12。没有在本轮修复主循环按次数结算的频率取舍，避免与收腿目标混为同一改动。测试中旧0.30固定断言改为正奖励契约，避免调参时把用户设置改回旧值。

示例从用户当前`2026-10-28_23-25-35_yaw_mild_soft_contact/model_15498.pt`直接resume，SHA256=`9c9879852f87b017a6e7155301d5e11b03d991c94a4127e16690dad4e9bf34b8`，512环境追加1000轮、运行名`compact_swing`。同任务奖励变化无需transfer，原模型不覆盖；当前源用于修正前后对照，不否定用户13000比13598回放更直的历史判断。本轮只提供指令，没有启动或停止训练。

实际检查：初次48项通过后补充两项，最终HighStep与walking_audit聚合50项通过，1.555秒，包含新增9项协调/几何测试。覆盖静止隔离、真实摆动/承重、支撑低位排除、方向对称、旋转/平移不变、分足角幅与膝位移、重置/失效地面/非有限及接口隔离。没有物理仿真、Viser、PPO、完整回归或新策略验收；50项不证明前髋已增加参与。同步专项文档、主索引和任务/操作/验证/状态页。

## 2026-10-28温和偏航与过大接触力约束

用户要求偏航不要调大过多，并反馈后足落地受力看似很大。截图不能量化力峰或区分瞬时冲击与正常动态承重；没有擅自运行仿真测量。核对源码，HighStep继承Flat但未启用独立接触力/软落地项，现有足滑项不能直接约束竖直重落地。

`config/high_step.py`只将瞬时偏航`high_step_yaw_rate.weight`−2→−1，持续偏航`high_step_drift_bias.yaw_scale_degrees`7→9，总漂移−1.5和侧速分量保持；相对早先−0.75/尺度10仍为温和增强。抬腿0.30/17厘米、周期、PD、动作滤波、限位、地形、其他任务及45/325/12保持。

加入`HighStepContactForcePenalty`，配置名`high_step_contact_force`、权重−0.5、阈值150N、尺度100N，前进命令和启动0.6秒后生效。按足名重排通道，取每足力范数在整个控制区间的峰值，超阈归一化平方、每足限幅4，再按平均/最差各半扣分。阈值以下支撑不罚，前后/左右相同，不强行均衡动态支撑；该项也作用于持续超阈承重，不仅首次落地。HighStep足力历史3→`cfg.decimation`（当前4）以覆盖全部物理子步，末帧离地不抹掉历史冲击；无历史时回退当前帧，非有限量安全过滤。新增分足`Metrics/high_step_contact_peak_{leg}_n`及`high_step_contact_overload_fraction`，记录控制区间峰值而非全回合最大值或冲量。150N只是初始训练超载阈值，静态XML显式质量合计16.1557kg用作量级参考，不是硬件安全值或实际冲击实测。

仍推荐固定`2026-10-28_20-43-44_111/model_13000.pt`resume，SHA256沿用已核实`1a2c3e4a3b324ab1fce0991c1378a661e1ea2f951cf3d3d64d3db05c47715385`；512环境追加1000轮，运行名`yaw_mild_soft_contact`。新进程采用新奖励/历史缓冲，原PT不覆盖，用户须续训才能改变策略；没有自动启动或停止训练。

实际验证：聚合HighStep及walking_audit共41项单元通过，1.469秒。新增6项接触力测试覆盖阈值、腿序/前后等价、短时历史峰、旋转/非有限、命令与启动门控及任务接口隔离；既有偏航测试按9度/秒尺度和−1权重更新。不声称已跑物理仿真、Viser、PPO、长训练或完整回归；41项不代表后足冲击已改善。同步主索引、专项文档、任务/操作/验证/状态页。

## 2026-10-28按用户Viser观察改从13000继续

用户明确反馈13000在Viser中比13598更正、偏移更小。先前仅按训练日志均值推荐13598的依据不足以判断实际轨迹，现续训源改为`logs/rsl_rl/pointleg_high_step/2026-10-28_20-43-44_111/model_13000.pt`，SHA256=`1a2c3e4a3b324ab1fce0991c1378a661e1ea2f951cf3d3d64d3db05c47715385`。使用同任务直接resume追加1000轮，运行名`yaw_w2_from13000`，完整命令已同步HighStep文档；不覆盖原PT，不启动训练。

实际只读取文件哈希和当前源码，确认新偏航权重−2.0、持续偏航尺度7和抬腿0.30保持，没有再次改奖励或网络，不需要为更换命令重复执行奖励单元检查。用户回放观察与训练日志分别记录，未假定用户做了同指令/同初态控制实验，也未独立运行仿真、Viser或正式评估。同步主索引、操作/状态页和专项文档。

## 2026-10-28偏航专项加强与13000/13598比较

用户先提供`2026-10-28_20-43-44_111/model_13598.pt`要求分析，随后打断读取回合并明确要求足高足够、进一步加强偏航。当前回合又追问13000；已只读比较两个模型附近50轮，同时完成已授权的奖励修改，不启动回放或训练。

`config/high_step.py::high_step_env_cfg`只改偏航两处：`high_step_yaw_rate.weight`−0.75→−2.0，死区2度/秒和尺度20度/秒不变，瞬时带权代价约2.67倍；`high_step_drift_bias.params.yaw_scale_degrees`10→7，总权重−1.5和持续死区1度/秒保持，未封顶同误差约(10/7)^2=2.04倍。侧漂分量、EMA0.6秒、启动1秒、髋外偏/足位奖励均保持。核对源码和13598运行配置，用户已经把抬腿循环权重设为0.30，本轮保留0.30与17厘米，不按旧记忆恢复0.24。未改网络/45/325/12、地形、限位、滤波或其他任务。Actor未加绝对航向误差，角速度项不能保证停止转向后自动回到原航向。

13598从12099直接恢复训练1500轮，末50轮整体峰值15.883厘米、持续偏航1.388度/秒、持续侧漂1.369厘米/秒、前/后足对中心0.752/0.566厘米。13000整体15.696厘米、持续偏航1.382度/秒、持续侧漂1.388厘米/秒、前/后足对中心1.350/0.857厘米。13000左后髋持续外偏2.480度、13598为3.102度；13598四足高度范围约0.53厘米，13000约0.83厘米，推荐13598作为新偏航参数续训起点，不声称它各项都优或已正式验收。

推荐固定源`logs/rsl_rl/pointleg_high_step/2026-10-28_20-43-44_111/model_13598.pt`，SHA256=`c6ea8d500639ca32b5bc7d7de7e94cabb92f063ce8a12cbdfaef6fb01fa560d2`；`--agent.resume True --agent.load-run 2026-10-28_20-43-44_111 --agent.load-checkpoint model_13598.pt --agent.max-iterations 1000`，512环境，运行名`yaw_w2`。只提供命令，原PT不改不覆盖，需新训练进程采用参数。

实际验证：`python -m unittest discover -s src/pointleg/tests -p 'test_high_step*.py' -q`25项通过，1.383秒；新增1项偏航加强且侧漂不变的数值检查，旧抬腿权重断言同步用户0.30。不写新的纯参数镜像测试；没有物理仿真、Viser、PPO、正式训练或全量回归。同步专项文档、主索引和任务/操作/验证/状态页。下一步需由续训/回放判断偏航是否下降及高度是否保持。

## 2026-10-28持续漂移权重提高50%并固定12099续训源

用户明确要求实施此前建议并给出恢复点和指令。`src/pointleg/config/high_step.py::high_step_env_cfg`中只将`high_step_drift_bias.weight`从−1.0改为−1.5；其他奖励、死区、EMA、17厘米高度、0.24抬腿权重、观测/动作45/325/12、动力学和其他任务均保持。原因是12099末50轮整体足高15.15厘米但持续侧漂仍1.35厘米/秒、持续偏航1.51度/秒，相对9200改善有限；这是单参数实验，不保证已改善策略。

推荐从`logs/rsl_rl/pointleg_high_step/2026-10-28_07-46-16/model_12099.pt`直接resume，文件存在且SHA256=`ce1423df3ae6a74415d6945a483528b79524d76218217cf0edac727436b07fba`。该模型已适应原持续纠偏奖励，足高略高于9200，作为延续本实验的起点；不声称它通过正式验收或优于所有历史模型。它对应的续训事件日志在`2026-10-28_14-04-52`，PT已放到旧运行目录，恢复路径按实际PT位置指定。固定`--agent.load-run 2026-10-28_07-46-16 --agent.load-checkpoint model_12099.pt --agent.resume True`，512环境，`--agent.max-iterations 1000`追加1000轮，运行名`drift_bias_w15`。完整命令见HighStep文档；不使用通配符，保存新目录，不覆盖源PT。用户要求的是改权重和提供命令，本轮未启动/停止训练。

实际验证：配置工厂确认漂移权重−1.5、抬腿0.24、目标0.17米，现有`test_high_step_alignment.py`9项通过，0.377秒。没有写镜像配置的新测试、没有重跑55项或全量、没有物理积分/Viser/PPO或新策略验收。同步主索引、任务/操作/验证/状态页和专项文档；新权重需新进程采用，后续看持续侧漂/偏航下降及四足高度是否保持。

## 2026-10-28二次HighStep持续偏移修正

用户反馈纠偏后仍向右偏移。只读最新指定`2026-10-28_11-26-12/model_7199.pt`日志末50轮，机身侧速有符号均值约+0.000820米/秒，但绝对值均值0.033893米/秒；前足对中心绝对偏移0.023304米。整体均值可能掩盖各环境或各时刻差异，不据此确认单环境右偏的唯一原因。该运行实际记录5200～7199，参数恢复源使用通配符；不能认为它从此前指定8199继续，也不能将足高降低单独归因于纠偏项。

仅对HighStep在`config/high_step.py`加入`high_step_drift_bias`−1，实现在`mdp/high_step.py::HighStepDriftBiasPenalty`：各环境分别对机身侧速命令误差和偏航角速度命令误差做0.6秒EMA，运动宽限1秒，持续侧速容许0.005米/秒、尺度0.04米/秒，持续偏航容许1度/秒、尺度10度/秒。两个分量各自平方并限制最大4后求和，再交给奖励管理器逐环境汇总；左右严格等价，避免跨环境正负抵消。相同控制步不重复更新，静止/非直行/无效采样清空历史并支持局部重置。保留已有瞬时漂移/偏航约束，不引入固定向左或向右修正，也不加入Actor缺少的绝对位置/世界航向误差。

持续髋偏置权重−0.6→−1，足对中心权重−1→−1.5；容许值和其他工作区不再收紧。高度17厘米、循环0.40秒、循环权重0.24、45/325/12接口及其余任务保持。新增持续侧速及偏航速率有符号/绝对值指标，详见专项文档。没有修改PT、控制滤波、地形或网络。

本轮默认续训示例固定最新用户指定`--agent.load-run 2026-10-28_11-26-12 --agent.load-checkpoint model_7199.pt`，文件SHA256=`f278533c2f9fdeae8dbde277cafe36fd1092a5e6cead274c72f13ac524c6163b`。同任务奖励变化直接resume，新进程采用新参数，原PT不覆盖；没有启动或停止训练。

实际执行HighStep全部模式、walking_audit、translation_rewards和elevation_alignment聚合定向单元共55项，通过，1.611秒；新增3项覆盖相反环境漂移不抵消、正常交替摇摆允许、同一步缓存及局部重置/无效值/静止清空。9个修改文件语法/空白检查通过；没有创建物理环境、Viser、PPO或运行策略验收。同步主索引、任务/操作/验证/状态页和HighStep专项文档。后续需训练验证持续偏移是否下降且四足高度不退化。

## 2026-10-28 HighStep侧漂、髋外展及梯形脚位修正

原因：用户反馈8199模型足高足够但有侧向无效速度、左后髋外转和前足侧偏。只读训练日志末50轮四足峰值约15.93厘米，前足对横向中心偏离3.24厘米；截图不能确定关节角或侧速数值，故不据此宣称已量化漂移方向。

适用任务仅`PointLeg-HighStep-v0`。`config/high_step.py`中`command_orthogonal_drift`权重−1.5→−3、死区0.04→0.015米/秒、尺度0.15→0.10米/秒；新增`high_step_yaw_rate`−0.75，角速度误差死区2度/秒、尺度20度/秒。`hip_abduction_excess`−1→−1.5，死区0.12弧度约6.9度→4度，尺度保留0.25弧度；复用`gait_bias.HipAbductionBiasPenalty`，权重−0.6、EMA0.5秒、启动宽限1秒、持续偏置容许3度和尺度8度。`foot_lateral_placement`−0.5→−1、中心容许2→1.5厘米；`high_step_workspace`横向余量12→4厘米，髋参考名义横向外伸10.5厘米、前后余量10厘米和权重−0.5保留。单足工作区防对称外张的梯形，足对中心防共同侧移，持续髋偏置防单腿长期偏转；不强制瞬时四腿同相。

`mdp/high_step.py`增加偏航速率误差死区代价和机身侧向速度/偏航误差的有符号、绝对值指标。高度目标17厘米、循环0.40秒及用户已设循环权重0.24保留；不改变关节限制、髋俯仰/膝动作幅度、动作滤波、地形、45/325/12接口、网络或其他任务。没有实现绝对世界航向/横向位置回正。

用户指定恢复源`logs/rsl_rl/pointleg_high_step/2026-10-28_07-46-16/model_8199.pt`，SHA256=`7371ad7cd33adc73ddbe018d23fe8d230975b5f1f9f2e710b60623f0eb6a1f4a`。已核实源文件存在、原运行512环境及循环权重0.24；直接`--agent.resume True --agent.load-run 2026-10-28_07-46-16 --agent.load-checkpoint model_8199.pt`，示例`--agent.max-iterations 1000`追加1000轮并在新目录保存，不覆盖源PT。只提供命令，未启动/停止用户训练或加载PT进入物理环境；奖励配置须新进程采用。

实际检查：HighStep模式定向21项和walking_audit/translation_rewards/elevation_alignment聚合31项通过，共52项；新增6项`test_high_step_alignment.py`。只做配置/张量单元，不运行仿真、Viser、PPO或全量回归。后续需验证侧速绝对值、偏航误差、逐髋持续偏置和前后足对中心下降，同时四足峰值维持约17厘米、真实步态不退化。同步专项文档、任务页、训练页、验证页、状态页和主索引。

## 维护要求

用户要求所有调整及时写入 memory。后续每次交付前，同步任务页、操作页或验证页，并在这里追加日期、原因、适用任务、修改前后值、代码入口、执行过的验证和待验证事项。保留历史结果的适用范围，不用后一次测试数量覆盖前一次实验的含义。

## RoughMixed 的障碍训练占比与有效时间

适用任务：`PointLeg-RoughMixed-v0`，不覆盖独立 Rough、Flat 或 Jump。

| 项目 | 调整前 | 当前值 |
|---|---|---|
| 普通崎岖合计 | 15% | 5%：随机起伏0.026、凸块0.024 |
| 带栏崎岖合计 | 30% | 35%：带栏起伏0.182、带栏凸块0.168 |
| 可移动轮胎阵 | 55% | 60% |
| 最大训练、固定评估回合时长 | 45秒 | 30秒 |
| 障碍指令速度大小 | 0.15～0.35米/秒 | 0.25～0.35米/秒 |
| 机身至围栏的接近距离 | 约2.8米 | 约1米，正向局部X=3、反向X=5 |
| 机身至首排圆环前缘 | 约1.025～1.625米 | 名义0.9米，随机偏移±0.3米后为0.6～1.2米 |
| 卡住后退出 | 依赖其他失败或超时 | 连续8秒未取得至少3厘米新有符号前进进度则失败 |

比例控制并行环境分配，有限环境数受取整和每类型保底分配影响，不等于面积、时间或完成回合占比。缩短回合同时提高障碍速度下限、移近出生点，完整2列7排阵列最远出生点按最低指令速度理想通过约需23.1秒；这是路程可行性估算，不是实测通过时间。

围栏最终高152毫米、厚50毫米；轮胎最终高165毫米、外径约550毫米、孔径约330毫米、每环6.2公斤，均未因本次提效改变。轮胎可移动并在reset恢复，完整内部布局保持2列7排，正对2列短边沿正X进入。圆环地块不加围栏，地面为平地。障碍每回合固定方向：围栏正反各半，圆环只正X。

代码：`src/pointleg/config/mixed_obstacles.py`、`terrains/mixed_obstacles.py`、`mdp/mixed_obstacles.py`。围栏出生支撑平台随近距离出生移动。`mdp/curriculums.py` 对障碍以真实完整通过和稳定离开作为成功依据，不再叠加普通长地块的进度比例要求；普通崎岖的进度判定和100有效回合、80%升级门槛保留。

## RoughMixed 受阻抬脚反馈

原因：旧日志中受阻恢复事件非常稀少，持续受阻时缺少可用的抬脚过程反馈。不能把奖励能运行等同于已经学会高抬腿。

| 参数或机制 | 调整前 | 当前值 |
|---|---|---|
| 持续侧向接触时长 | 0.06秒 | 0.04秒 |
| 水平受力门槛 | 10牛、竖直力绝对值的10% | 6牛、竖直力绝对值的5% |
| 近乎静止才能触发 | 要求前后足速≤0.05米/秒 | 触发不再要求近乎静止；≤0.08仅保留为受阻速度诊断 |
| 恢复窗口 | 1.2秒 | 2秒 |
| 未达到目标的惩罚系数 | 0.5 | 4 |
| 碰撞前的净空反馈 | 无独立连续项 | 障碍边缘摆动足或侧向碰撞足净空不足，系数2 |
| 抬升过程反馈 | 主要依赖最终恢复 | 每取得新最大抬升进度即返还，单次窗口最多0.15分 |
| 碰撞代价、成功落足返还 | −0.15、+0.10 | −0.30、+0.10；加上抬升返还仍无正碰撞循环收益 |
| 冷却 | 1秒 | 0.4秒 |
| 支撑换相 | 一次短暂支撑不足可能永久取消恢复资格 | 奖励结算要求当时另有至少两足承重，短暂换相不永久取消 |

足底由球心减25毫米计算，目标是锁存的障碍顶面以上20毫米，并要求相对机身实际抬脚至少20毫米。恢复落足还要求同足沿指令前移至少0.12米、机身前移至少0.03米、重新承重连续0.10秒。轮胎推移或倾斜超容差不发抬升、落足正奖励。长期悬脚、反复升降或把整机抬高不能当作完整恢复。

新增阶段指标包括 `obstacle_foot_approach_feet`、`obstacle_foot_lift_progress`、`obstacle_foot_cleared_events_per_s`，与原受阻、恢复事件率配合。实现位于 `mdp/obstacle_recovery.py`。Actor没有新增地形或接触输入，仍为45/325/12接口；这是奖励引导，不是运行时硬抬腿控制器。

## 所有已启用任务的4×4地图

用户明确指地图块数，不是单块尺寸或轮胎阵内部排列。所有已启用PointLeg任务的物理地图统一为 `layout_shape=(4,4)`，16块；单块尺寸、类型和逻辑课程接口保留。

物理地块按类型平衡分配，保留各类型的起步及最高难度，中间部分逻辑档位映射到同类型可用的较低难度。不能只改 `num_cols`，原课程生成器会按类型数重建列数。`TerrainGenerator`、`TerrainEntity.get_geometry_levels()`、地图边界和高程实际几何元数据已联动调整。高程奖励、出生和判定读取真实生成档位，评估结果增加实际 `geometry_level`，避免逻辑档位与物理楼梯不一致。

代码：`mjlab/src/mjlab/terrains/terrain_generator.py`、`terrain_entity.py`、`tasks/velocity/mdp/terminations.py`；任务配置在 `config/env_cfgs.py`、`elevation.py`、`stepping_stones.py`，Mixed继承Rough布局。此处不修改第三方Go1/G1示例任务。

## RoughMixed 摆腿峰值曲线

用户要求接入 `Metrics/swing_peak_height`，只增加统计，没有接入Flat的10厘米摆腿奖励，也没有增加策略观测或射线。

实现：`src/pointleg/mdp/swing_metrics.py:FootSwingPeakHeightMetric`；接入：`config/mixed_obstacles.py`。

| 标签 | 含义 |
|---|---|
| `Metrics/swing_peak_height` | 完成摆腿样本的平均足底峰值，相对最近起脚承重面，米 |
| `Metrics/swing_peak_height_fl`、`fr`、`rl`、`rr` | 左前、右前、左后、右后足各自的完成摆腿峰值 |
| `Metrics/swing_peak_height_relative_body` | 相对起脚时足底至机身高度差的最大增加量，米 |
| `Episode_Metrics/swing_peak_height` | 各环境本回合完成摆腿的平均值，在回合结束时统计 |

足球心减25毫米得到实际足底；先有承重基准、离开承重至少0.08秒、重新承重才纳入。承重要求竖直合力绝对值≥8牛且不小于水平合力。侧壁接触、初始化落下、短时接触抖动、一直悬空未落足不结算。主曲线及分足曲线保留每次落足样本，按事件加权；无完成样本不填0。回合指标则在无完成摆腿的回合记0，两种口径不能混淆。整体升高可以增加支撑面相对峰值，但相对机身曲线可区分是否真正收腿。

旧进程不会热加载，新曲线需重启训练后产生；旧日志不会补写。该曲线不是相对障碍顶面的最小净空，也不是越高越好。

## Elevation 直行侧漂和四髋偏置纠正

用户提供截图后明确确认任务为 `PointLeg-Elevation-v0`，不是RoughMixed；前后腿都有偏置，不能只处理左后腿。截图本身不能证明某个关节是唯一因果来源。

| 调整项 | 调整前 | 当前值 |
|---|---|---|
| `lateral` 权重、容许侧速、尺度 | −0.5、0.03、0.15米/秒 | −2、0.015、0.10米/秒 |
| `center` 权重、容许偏移、尺度 | −1、0.02、0.18米 | −2、0.015、0.12米 |
| `heading` 权重、容许角、尺度 | −0.5、3°、12° | −1、2°、10° |
| `hip_abduction_bias` | 无持续偏置专用项 | 权重−0.5；0.5秒EMA、0.5秒宽限、容许4°、归一化尺度8° |

瞬时髋外展12°容许惩罚保留，额外对四个髋外展关节相对默认角的有符号偏差做时间平滑，再按超出4°的量平方扣分。代价按四足平均，单足归一化分量封顶4。短暂或正负交替摆动不等于持续偏置，不强制瞬时镜像，不更改正常髋俯仰、膝关节范围、PD或机器人几何。

侧速使用世界Y轴，配合世界通道中心和零航向，识别机身转歪后沿自身前方行走产生的侧漂。修改位于 `config/elevation.py:configure_walking_rewards`、`mdp/elevation.py:elevation_cost`；持续偏置实现在 `mdp/gait_bias.py`。共享函数保留旧默认参数，Jump不采用本次权重和新惩罚，Flat、Rough、RoughMixed也未接入。

新增回合指标：`Episode_Metrics/elevation_lateral_velocity`、`elevation_lateral_speed_abs`、`elevation_center_error_abs`、`elevation_heading_error_deg`。新增即时指标：`Metrics/hip_abduction_bias_abs_fl_deg`、`fr_deg`、`rl_deg`、`rr_deg`、`Metrics/hip_abduction_bias_max_deg`。完整标签需要将相同前缀与各后缀组合，角度曲线直接使用度。

## 本对话实际执行的验证

| 阶段 | 实际检查 | 结果与边界 |
|---|---|---|
| 混合受阻反馈与提效 | `uv run --locked --no-sync python -m unittest discover -s src/pointleg/tests -p "test_*.py"`，使用项目诊断缓存 | 当时208项通过；不是当前以后永远固定的测试数量 |
| 紧凑地图 | 点足211项回归；框架紧凑地图4个函数检查 | 通过；当时环境未安装pytest，通过runpy直接调用4个测试函数，未声称跑完框架pytest全集 |
| 紧凑地图GPU | Mixed 10环境200步；Elevation、Jump各64控制步 | 观测和奖励有限，实际布局4×4；不代表策略通过 |
| 摆腿曲线 | 专项8项测试、当时点足227项全量回归 | 通过；未重复写入原训练日志 |
| 摆腿曲线GPU及TensorBoard | 5环境160控制步，旧模型3500；临时写入并读回事件文件 | 得到131个完成摆腿样本，主曲线、四足和相对机身共6个即时标签可读；不构成高抬腿验收 |
| Elevation纠偏 | 专项6项测试、点足233项全量回归 | 通过；覆盖正负漂移对称、世界坐标侧速、静态偏离回正、四髋、交替摆动、状态重置和NaN处理 |
| Elevation纠偏GPU | `tools/check_elevation.py --task PointLeg-Elevation-v0 --steps 64`，项目内Warp缓存 | 4×4、45/325/12接口、64步有限及物理支撑检查通过；未训练出纠偏后的新策略 |

诊断加载的模型：`logs/rsl_rl/pointleg_rough/2026-10-26_10-30-32/model_3500.pt`。本轮记忆补录重新读取文件并核对SHA256：`175DAE0B70724DF1D797BDAAA0331B8CFB3DAC7FA10CCFDA390398EA5DC3A3E7`。先前短仿真严格加载了Actor、Critic及优化器，优化器状态17项；没有覆盖源模型。

诊断使用项目`.venv`，命令统一可经 `uv --cache-dir ./.venv/uv-diagnostics-cache run --locked --no-sync ...` 运行；Warp缓存位于 `.venv/warp-diagnostics-cache`。`.venv`内临时诊断脚本及缓存不是受版本管理的正式入口，优先复用 `src/pointleg/tests/` 和 `tools/check_elevation.py`。本对话未替用户启动新正式长训练，也未停掉用户已有训练；无法据代码变更判断旧进程已采用新参数。

## 尚未实现或尚未证明

- 曾建议单独进行8、12、16、19厘米摆腿课程，再迁移单栏、单排和完整阵列；这是建议，尚未编写或启动该专项课程。
- 尚未证明当前奖励能把四足稳定训练到完整轮胎要求的18.5厘米净空目标；事件触发、记录高度、训练轮数与策略成功率不是同一件事。
- 静态运动学曾检查：机身高42厘米、髋俯仰65°、膝角−130°时，四足球底约22.4～22.6厘米且位于软限位内。这只证明示例姿态的几何可达，不证明带载动态抬腿、稳定或实机可行。
- 四足相对障碍顶面的最小净空、分足净空成功率、膝关节角范围和按地形等级分组的正式通过率仍需单独增加或评估。
- Elevation纠偏后的策略尚未完成正式续训及固定种子验收；新增奖励不使旧模型仅回放就自动纠正。

## 2026-10-27 Elevation 二次纠偏与摆腿低净空

用户报告1000轮后仍略向左、平地摆腿很低，并通过本轮问答明确同意同时加入低净空惩罚。仅调整 `PointLeg-Elevation-v0`；不修改Jump、Flat、Rough、RoughMixed，Actor/Critic/动作保持45/325/12，不增加传感器或更改PD、关节限位。

| 参数 | 调整前 | 调整后 |
|---|---|---|
| 世界Y侧速 | 权重 -2，容许0.015米/秒，尺度0.10 | 权重 -3，容许及尺度不变 |
| 通道居中 | 权重 -2，容许1.5厘米，尺度12厘米 | 权重 -3，容许1厘米，尺度10厘米 |
| 航向误差 | 权重 -1，容许2°，尺度10° | 权重 -1.25，容许及尺度不变 |
| 四髋持续偏置 | 权重 -0.5、0.5秒平滑、容许4° | 不变，避免继续压制正常换脚与抬腿 |
| 足端高度反馈 | 无直接高度目标；仅有0.05权重离地时长和踩阶事件 | 新增 `swing_clearance`，权重 -0.4，局部地形相对峰值目标6厘米，上台阶立面余量2厘米 |

纠偏保持左右对称，不以单罚左侧造成反向偏置。低净空实现在 `mdp/elevation_clearance.py`；共用 `mdp/elevation.py:elevation_foot_bottom` 处理球足旋转后的真实最低点。平地在完成有效前向摆腿后才结算低峰值代价，足端前移至少3厘米、已承重后离地至少0.02秒、再次承重且至少两足支撑；单足每次最多实际 -0.04，除dt后交奖励管理器。上阶立面前10厘米至中心过立面2.5厘米内，对非承重摆动足追加顶面以上2厘米的连续净空不足代价，近立面逐渐加重。不对平地起落脚低点持续扣分，不奖励超高、长悬空或反复点地。8牛承重阈值与上述高度、系数均是待训练验证的仿真初值。

新增Elevation即时 `Metrics/swing_peak_height`、分足及相对机身曲线，只有有效完成样本才写入；另有回合均值和近立面足数、缺口、低摆腿事件率。当前源码之前没有直接高度激励，课程升级不会自动补出该奖励；平地6厘米目标也不证明能通过120毫米台阶。

实际验证：专项54项通过；全量 `python -m unittest discover -s src/pointleg/tests -p 'test_*.py' -q` 共246项通过，`tools/check_elevation.py --task PointLeg-Elevation-v0 --steps 64` GPU短仿真通过，保持4×4布局、45/325/12接口及有限奖励/观测。没有进行正式续训或固定种子通过率验收，也未修改或停止用户的训练进程。

只读检查的本地最新1000轮日志为 `logs/rsl_rl/pointleg_elevation/2026-10-26_23-25-14`；检查点 `model_999.pt` SHA256 为 `3383A31D610CE08C140D11DD3586515FB5099CB4A0A8C23972086EA06FA29D04`。日志最后30次记录的回合居中误差均值约0.0893米、绝对侧速约0.0607米/秒，各障碍课程上限仍为0；这些混合回合指标不是分地形成功率，也不能当作用户必然正在回放的模型。新增配置需要重启续训，旧检查点回放本身不会获得更高抬腿能力。

另用该 `model_999.pt` 严格加载Actor，只读执行5环境、160控制步GPU短回放，未更新参数或覆盖源文件。观测、奖励和新代价有限；采到160次满足前向位移及承重条件的完成摆腿，160次低于6厘米目标，事件峰值净空均值0.014182米；累计14个近立面摆动足采样，新代价和峰值日志均能实际触发。统计混合了楼梯、坡面和平地的短轨迹，不是单独平地正式评估，也不是最小立面净空或新策略验收。

## 2026-10-27 Jump 改为固定触发的平台前跳

用户要求触发后立即单次前跳、固定出生、跳前前脚接触点至跳后后脚接触点净距大于400mm且小于500mm，两种地形训练同一个策略。用户明确确认该测距含义、墙顶高出平台300mm，以及墙厚始终80mm、带墙起步间距至少80mm。适用任务仅 `PointLeg-Jump-v0`；Elevation行走的奖励、楼梯和坡面未更改，其他未提交改动保留。

| 项目 | 调整前 | 调整后 |
|---|---|---|
| 物理地形 | 12×8m三墙，墙高110～340mm，4×4混合地图 | 每块3×1.6m，两个等高平台，间隙或间隙中墙，50块10×5布局 |
| 最终规格 | 三堵地面薄墙 | 平台高200mm、净间距360mm；墙厚80mm、墙顶高出台面300mm |
| 起步 | 随机墙前0.9～1.1m、随机速度/朝向 | 相对基座x=.64m、y=.8m、固定零朝向/零初速/默认关节；按球面真实最低点+.5mm校准出生高度 |
| 课程 | 五档墙高、每地形单上限 | 高度和间距各五档15%～100%，两类各维护双轴上限；带墙间距最低80mm、墙厚不缩放 |
| 成功 | 单次越墙和两足支撑接管0.12s | 有效整机腾空后落到对面，前后脚所有配对净跨度严格400～500mm，四足连续稳定0.5s |
| 失败 | 某些轻微擦碰仅扣分 | 墙、地面、平台侧面等任意非法接触子步锁存；身体/腿杆地形接触、错距、助跑、迟跳、重复跳等立即失败 |
| 时限 | 6s | 3s；有效起跳须在0.6s内，允许下蹲蹬地，前支撑足漂移限4cm、基座限8cm |
| Actor三命令槽 | 前向/侧向/转向速度 | 触发标志、触发后时间/3、零；总尺寸45/325/12保留，语义不兼容旧策略 |
| runner与评估 | 共用Elevation runner/评估及warmup | `JumpOnPolicyRunner`、`evaluate_jump.py`、独立双轴课程；warmup停用 |
| 奖励 | 越墙事件、完成+20、失败-50等 | 势函数5、起跳+1、净空峰值增量总额+2、首次落地+3、四足稳定+30、失败-50及轻量控制/姿态/落稳代价 |

代码入口为 `config/jump.py`、`terrains/jump.py`、`mdp/jump.py`、`rsl_rl/runners/jump_runner.py`、`evaluate_jump.py`、`jump_trigger.py`，注册仍用Jump任务ID与 `pointleg_jump` 默认日志目录。五档高度×五档间距编码为25逻辑档，要求每种地形有完整25块真实几何，拒绝框架紧凑布局把不同组合别名到低难度地块。高度轴同时缩放平台高度和墙的相对高度；两个等高平台只增加共同高度本身不增加跃升要求，本文没有把它写成跳高课程已有效。

测距锁定前足最后一次起跳台接触与后足首次落地台接触，检查最小/最大配对跨度并排除400/500mm边界；落地后再滑到区间外也失败。前足漂移基准在首次真实平台接触建立，避免足端site与球面接触点差异被误判。四足承重以1N为阈值，非法接触以found为准不忽略轻擦碰；子步失支撑清零落稳计时。四足落稳还要求姿态角<12°、线速<.10m/s、角速<.50rad/s。阈值均是训练初值，需正式训练和实机标定。

奖励保留负值，单次事件有dt补偿，势函数/PPO折扣均.995。删除旧Jump行走速度跟踪，不约束正常起跳的竖直速度、不用普通步态支撑不足或长腾空代价。失败当步屏蔽正事件，重复跳不给再次事件奖励并失败。训练器禁用首回合随机episode已用时间，保证触发时钟从0开始。部署端提供防重复触发的 `JumpTrigger`，但没有写入ROS/实机控制器或验证模型切换滤波状态；旧CPU速度命令回放入口拒绝新Jump，导出ONNX附带触发时钟契约。

TensorBoard新增 `Episode_Metrics/jump_*` 跳距最小/最大/均值、距离有效、起跳延迟、四足落稳时间、净空峰值、落地速度、各类失败标志、实际台高/间距/墙高与双轴档位；`Curriculum/jump_levels/{platform_gap,platform_wall}_success_rate`按各类最近100个有效终止回合统计成功率，另记录height/gap上限。没有后足落地的距离为0，未起跳延迟记.6s删失值，必须结合有效标志解读，不能当作成功回合测距均值。

实际验证命令均在现有Windows uv锁定环境执行，没有安装依赖：

- `uv run --locked --no-sync python -m unittest discover -s src/pointleg/tests -q`：最终264项通过，其中Jump专项18项；50种实际几何射线、严格距离边界、非法接触锁存、四足连续稳定、球底出生校准、双轴解锁/保存恢复均有覆盖。
- `uv run --locked --no-sync python -m unittest discover -s sim2sim/tests -q`：16项通过，保留其他任务导出/回放功能。
- `tools/check_elevation.py --task PointLeg-Jump-v0 --steps 160`：真实GPU4环境160控制步有限，观测/动作45/325/12，地图10×5、原点表25×2；该检查在后续球底校准前执行，校准后重新跑了下述PPO和评估。
- 校准后 `pointleg train PointLeg-Jump-v0 --debug --env.scene.num-envs 8 --agent.num-steps-per-env 64 --agent.max-iterations 2 --agent.run-name jump_platform_final_check`：两次PPO更新和保存完成。模型 `logs/rsl_rl/pointleg_jump/2026-10-27_09-04-57_jump_platform_final_check/model_1.pt` SHA256为 `f18edac041c67c2dd6810a9e143cbef04cbb2a2d72a7be5d5534a8baaf4370fa`，真实球足校准偏移为+.000812644546m。TensorBoard实际有67种标量，跳距和分类型成功率均有记录；成功率0、课程仍起始档。
- 从该model_1用相同8环境/64步配置、`--agent.resume True --agent.load-run 2026-10-27_09-04-57_jump_platform_final_check --agent.load-checkpoint model_1.pt --agent.max-iterations 1 --agent.run-name jump_platform_final_resume` 实际续训一轮通过，输出 `logs/rsl_rl/pointleg_jump/2026-10-27_09-06-49_jump_platform_final_resume`。此前更早检查点也完成过一次恢复。
- 当前代码 `pointleg evaluate PointLeg-Jump-v0 --checkpoint-file logs/rsl_rl/pointleg_jump/2026-10-27_09-04-57_jump_platform_final_check/model_1.pt --episodes-per-case 1 --output artifacts/jump_platform_final_evaluation.json`：50用例报告生成，0/50成功，全部runup失败，accepted=false；只有一组种子/每用例一次，不是正式验收。评估实现已修复inference_mode生成张量与手动reset的冲突，使用no_grad。
- 更早调试检查点 `logs/rsl_rl/pointleg_jump/2026-10-27_08-55-49_jump_platform_verified/model_1.pt` SHA256为 `7e75c650699e10c5e8426575a5a83ce9644ff22eeeb834886226f0510ae83bed`，诊断ONNX为 `artifacts/jump_platform_smoke.onnx`，实测与Actor最大绝对误差8.1956387e-8，非最新球底校准模型。该模型的50用例早期报告为 `artifacts/jump_platform_smoke_evaluation.json`，同样0/50。
- 本次相关已跟踪文件 `git diff --check` 无空白错误。存在此前用户未提交改动和删除项，未回滚、提交或覆盖。

尚未执行正式长训练、多种子100回合通过率、起跳延迟/漂移阈值调优、真实部署触发与落地接管或实机验收，不能声称已经训练出符合400～500mm要求的模型。旧三墙、warmup或Rough/Elevation权重只能用transfer初始化新训练，不能完整续训；相同45维不表示语义兼容。文档和所有相关项目记忆页已同步，配置变化不会自动进入其他已运行训练进程。

## 2026-10-27 梅花桩借鉴目标推进和首次落足奖励

用户在对照`jindadu00/legged_robot_competition`后明确同意加入值得借鉴的内容。适用任务仅`PointLeg-SteppingStones-v0`；Flat、Rough、RoughMixed、Elevation、Jump配置未改，保留其他未提交改动。参考仓库README描述目标方向速度、全目标完成和边缘风险，但其默认master的GO2代码没有接入这三项自定义奖励，本次没有宣称直接复现对方策略。

| 项目 | 调整前 | 调整后 |
|---|---|---|
| 连续速度引导 | `stepping_forward_velocity`，机体前向速度指数跟踪，权重1 | `stepping_goal_velocity`，世界水平速度向机身目标投影，正收益上限1，权重1，登台倍率0.3 |
| 机身目标 | 无未覆盖目标引导 | 入口平台中心、最前未覆盖桩对的纵向中点、出口中心；横向场地中心，奇数尾桩仍居中；不强制踩踏顺序 |
| 重复速度取分 | 仅速度跟踪 | 每目标历史最近距离门控；退后再走回旧位置不重复拿正分，切目标首步不发，负向速度保留负值 |
| 落点与抬腿 | 两种质量先取最大值，再同时结算 | 落点只按2厘米边缘余量；抬腿只按有效摆腿足底净空，两者独立，每桩各取最佳足一次 |
| 抬腿饱和 | site峰值24～28厘米递增，且可由居中落点替代 | 球足真实最低点20～24厘米递增，24厘米饱和，不奖励超高或更久离地 |
| 有效抬腿 | 由离地转接触，缺少起脚支撑及最短摆腿检查 | 先有合法支撑、离地至少20毫秒、摆腿期间至少两只其他足合法支撑、合法落足且未失败；25毫米球足旋转修正 |
| 失败同周期正事件 | 之前子步累积的正事件可能保留 | 任一子步失败清空本周期正事件，框架其他失败终止也在读取时屏蔽正事件 |
| 固定路径势函数 | `progress_ready`首步后误赋Python `True`，第二步和局部reset可能报错 | 保留逐环境布尔张量、首步零势差语义；权重10、gamma=.9995不变 |
| 指标 | 成功、失败、覆盖率、桩数 | 额外目标距离、低净空比例、有效落足样本数、边缘接触比例 |

代码入口：`src/pointleg/mdp/stepping_stones.py`、`config/stepping_stones.py`；回归入口`src/pointleg/tests/test_stepping_stones.py`；真实仿真入口`tools/check_stepping_stones.py`。已同步`docs/STEPPING_STONES_REWARDS.md`、根README梅花桩占位描述、`01_doc/梅花桩训练环境与奖励函数设计.md`实现状态、项目主索引及任务/操作/验证/缺口记忆页。

未改变场地尺寸、2～8桩课程、出生15～35厘米、指令0.15～0.35米/秒、严格接地失败与全覆盖出口停稳规则。覆盖/落点/抬腿/成功/失败/边缘权重仍30/10/1/100/-150/-0.3；Actor/Critic/动作仍105/381/12。未引入主动跳跃、长腾空正奖励、总奖励非负裁剪或额外策略观测。

实际验证使用已有Windows `.venv`，没有安装依赖：

- `.venv/Scripts/python.exe -m unittest pointleg.tests.test_stepping_stones -v`在修复路径势函数旧问题前22项通过；补充跨步/局部reset回归后该文件共23项，全部包含在以下最终全量回归中。
- `.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -p 'test_*.py' -q`：最终282项通过；覆盖独立落点/净空质量、饱和、时长无额外收益、支撑/接触抖动门控、重复事件、同周期失败优先、球足旋转、目标组/奇数尾桩/非活动桩、世界水平投影、幂等、往返限制和局部重置。
- `.venv/Scripts/python.exe tools/check_stepping_stones.py --steps 64`：首次运行暴露上述旧`progress_ready`类型错误，修复后重新执行成功；7环境分别2～8桩，真实GPU `cuda:0`，105/381/12，64控制步观测与奖励有限、检测到真实足端支撑。零动作报告`artifacts/stepping_stones_reward_check.json`，SHA256为`08f0ac30eaca0f98638e0cbc2b045ee9540a1e00f7a9d79bca50d9f83ca72c0f`，`policy_validation=false`。
- `git diff --check`无空白错误；运行仅有已有LF/CRLF转换提示。新增Python文件为UTF-8，代码注释未使用编号或图形符号。

本次没有启动PPO训练、创建或覆盖策略检查点、停掉用户现有进程，也未做正式通过率/实机验收。当前抬腿奖励仍是峰值净空，不是全路径最小扫掠净空；现有接触found/pos归属接口、专用正式评估器、课程窗口持久化和部署缺口仍需后续处理。奖励变化需要重启训练才生效；已有105/381/12模型结构兼容不等于已经学会新目标，45维平地模型也不能直接原样加载。

## 2026-10-27 Elevation 小步高净空与四足独立循环版本2

原因：用户报告前腿大幅向前伸、后腿被拖动和右后腿持续擦地，希望接近参考图的小步高净空风格，并允许同一踏面多次踩踏。旧首次覆盖奖励仅去重，并未禁止重复踩踏；实际缺口是旧净空不足只在完成摆腿后扣分，持续不抬脚的腿可能避开，且前后伸腿余量及直接推进偏强。图片本身不证明单个关节是唯一原因，本轮未宣称已复现MIT控制器或训练出参考风格。

仅改Elevation行走奖励与该任务runner契约，不改场地、速度命令范围、PD、限位、力矩和45/325/12接口，不改独立平台Jump、Rough、Mixed或梅花桩。完整机制及启动命令见 `docs/ELEVATION_GAIT_REWARDS.md`。

| 项目 | 调整前 | 调整后 |
|---|---|---|
| 额外直接推进权重 | 1.5 | 0.5，路径势函数5、完成+20、失败-50保留 |
| 首次踏面事件权重 | 0.5，六阶总额1.5 | 0.2，六阶总额0.6；另发不依赖新踏面编号的有效循环反馈 |
| 前后工作区 | 全机身坐标，前后余量30厘米，权重-0.15 | 髋参考点支撑切向±16厘米，权重-0.8，不约束法向足高 |
| 离地时长正奖励、旧接触时长惩罚 | 0.05、-0.25 | 均为0，由四足独立合格循环取代 |
| 平地峰值目标、上阶余量、前方窗口 | 6厘米、2厘米、10厘米 | 法向8厘米、3厘米、16厘米；最高台阶局部约15厘米抬升目标 |
| 滑移 | 标记点线速度平均，权重-0.5 | 实际接触点刚体速度，最差足与平均各半，权重-1 |
| 四足循环、长跨步 | 无独立限额反馈 | 每足实际奖励最多+0.04且按机身新进度缩放；16厘米以内不扣步幅，26厘米不发质量奖励；长期不合格单足持续扣分 |

有效循环需合法承重后离地至少0.08秒、期间至少两足承重、同足新前向落点至少3厘米且机身新增至少2厘米、重新承重0.06秒、结算间隔至少0.4秒。高度和相对支撑法向收腿质量合格后才重置该足参与计时；超过0.9秒开始扣分，1.8秒饱和，启动宽限0.6秒，最差足与平均各半。仅前腿循环、原地点脚、身体弹跳、全足腾空、旧前沿位置往返不能替代四足真实换步。同一踏面可多次结算有新前移的小步；不强制腿序、固定对角步态或每阶仅迈一次。

身体俯仰参考拟合真实承重点，数量/跨度不足时退回已知坡面；坡面允许10°相对偏差，楼梯15°、平地5°。脚的位置不设固定世界高度，也不统一向世界负Z压低。净空与相对收腿改为支撑面法向口径；新旧坡面曲线不能不加区分地直接比较。近楼梯、坡面的速度误差参考上限0.3米/秒，实际指令范围不变。仅改奖励初值，不能代替真实动态和硬件验证。

代码：`config/elevation.py`、`mdp/elevation.py`、`mdp/elevation_clearance.py`、新增 `mdp/elevation_gait.py`；新增测试 `tests/test_elevation_gait.py`。当时将步态标签升为版本2，并错误地要求旧模型强制transfer；该限制已在后续“恢复兼容性修正”中撤销，仅改奖励可直接resume。网络权重形状不变；新增分足循环年龄、合格事件率、髋切向前后位置以及支撑俯仰、步幅和目标高度指标。

实际验证：

- 全量 `.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -p 'test_*.py' -q` 共298项通过，包含新增16项步态回归；临时文件使用 `artifacts/test_tmp`。
- `.venv/Scripts/python.exe tools/check_elevation.py --task PointLeg-Elevation-v0 --steps 96` 通过：4×4、45/325/12、观测/奖励有限且存在物理支撑；Warp缓存为 `.venv/warp-diagnostics-cache`。
- 从 `logs/rsl_rl/pointleg_elevation/walking_pretrained_v1/model_0.pt` 严格迁移Actor/Critic，源SHA256为 `A323AF39825D4014D831FAD6875ECB0CE17EA8F996B35A1C6D054C543913D33F`，输出新目录 `gait_v2_check_20261027/model_0.pt`，未覆盖源文件。
- 加载该迁移模型进行5环境、两次PPO调试更新，日志 `logs/rsl_rl/pointleg_elevation/2026-10-27_17-11-50_gait_v2_smoke`；value loss约0.0738、0.1174，已保存model_0和model_1，model_1 SHA256为 `F5DB25F62CC8D511EF49161897B24EDE0BF0845F6FE5B121364ACF643A2292CA`。

源目录存在并不表示源模型已验收或最佳。未启动正式长训练、未修改或停止用户既有训练进程，未证明新策略已解决拖腿或通过最高台阶。两个调试更新只验证迁移、训练和保存链路；正式验收要同时检查四足循环、接触滑移、实际净空、动作风格及分地形通过率。

## 2026-10-27 用户更正：仅改奖励允许直接resume

用户指出同任务改奖励可以直接resume。核对本地MJLab恢复实现后确认：Actor/Critic、优化器、迭代、common_step_counter和课程会恢复，奖励项由当前环境配置创建且回合状态在reset时初始化；旧奖励配置不会覆盖当前奖励。因此奖励标签从1变为2并不构成训练状态不兼容，先前的硬拦截过严。

修正 `ElevationOnPolicyRunner.load()`：`elevation_walking` 标签或奖励/停滞设置不同只发提示，不再强制transfer；保留任务、楼梯几何和课程维度/窗口检查，网络形状仍严格加载。真正跨任务/接口/地形/课程不兼容，或用户主动重置训练状态时才需要transfer。没有修改本次步态奖励、观测、地形或平台Jump。

实际验证：299项点足全量回归通过。直接只读加载 `logs/rsl_rl/pointleg_elevation/2026-10-27_08-36-41/model_2499.pt` 到当前runner，严格核对Actor和Critic所有张量；成功恢复迭代2499、累计步数60000、17项非空优化器状态和六类课程上限 `[0,0,0,0,0,0]` 及窗口。新 `step_quality` 权重1、8厘米净空目标保留，标签差异仅出现一次警告。源模型SHA256仍为此前记录的 `291B25A4FD916DEC62D9C40E0143B67FC393532A0597AC107F2B32700DD526CB`，未写回模型或启动长训练。

已同步当前训练文档、任务/操作/验证/缺口记忆与主索引；默认命令改为直接resume上述2499轮检查点，再更新指定轮数，日志仍使用默认目录的新时间戳运行。

## 2026-10-27 独立HighStep平地基础与后续姿态修正

用户要求为后续爬梯训练一个高抬腿、小步幅、快周期基础步态。新增独立`PointLeg-HighStep-v0`，不覆盖Flat或改Elevation。配置在`src/pointleg/config/high_step.py`，奖励在`mdp/high_step.py`；注册和独立PPO实验名位于包入口与`config/rl_cfg.py`。日志为`pointleg_high_step`，说明为`docs/HIGH_STEP_TRAINING.md`。

| 项目 | 原Flat | HighStep初值 |
|---|---|---|
| 抬腿奖励 | 峰值10厘米、0.50秒频率预算 | 球足底峰值14厘米、评分宽度5厘米，周期目标0.40秒、宽度0.08秒 |
| 摆动时长正奖励 | `feet_air_time`权重0.4 | 移除；完整循环需0.10～0.24秒离地、重承重0.12秒 |
| 落脚位移 | 无对应完整循环目标 | 同足前进目标6～10厘米、至少2.5厘米、水平不超过14厘米，机身前移至少1厘米 |
| 速度采样 | 全向及后退分组 | 向前0.15～0.25米/秒、10%静止，禁止横移和转向采样 |
| 相位超时 | 单足悬空0.80秒后罚 | 高抬腿长相位权重−0.4，支撑0.34秒或离地0.24秒后罚；原悬空门槛改0.28秒 |
| 短支撑 | 0.16秒、权重−0.10 | 0.12秒、权重−0.05 |
| 运动及动作代价 | leg_motion−0.02、action_rate−0.01 | 均−0.005；力矩/加速度/滑移保留 |
| 跟踪std | 0.5 | 0.20，减少低速下原地不动的局部收益 |
| 足端工作区及弹跳 | 无专用项 | 髋前后软余量±10厘米、权重−0.5；垂直机身速度权重−0.5 |

完整循环逐足结算，要求先承重再离地、至少两足5牛以上竖向支撑，最高点包含相对机身收腿量，落地后持续承重只领取一次。拒绝点地、原地摆腿、长跨步、纯机身抬升和全足腾空刷分。不设固定腿序或对角相位。实际采样检查发现框架前进组有0.3米/秒下限，因此新任务前进分组比例设0、用正向范围直接均匀采样；同时降低deadband为0.08并清除关闭航向后的heading范围，原生成器未修改。

保留45/325/12、默认站姿、动作缩放、5Hz滤波、50Hz、PD及±25牛米；基础原件未改，把`pretrained/flat_9000_transfer/model_0.pt`原样复制至`logs/rsl_rl/pointleg_high_step/flat_pretrained/model_0.pt`。两者SHA256为`F340F78849B9BFB5E7507D1175EA7E2119FACBE838859AA60648D1621B1F3F1E`。

姿态修正前实际验证：

- 点足全量310项通过，包含11项新增HighStep检查；真实4096环境指令采样验证0.15～0.25米/秒及静止、无横向/转向，未被0.3米/秒下限覆盖。
- `tools/check_high_step.py --steps 96 --checkpoint-file pretrained/flat_9000_transfer/model_0.pt`完成4环境真实GPU数值、接口及支撑诊断，严格加载原Actor/Critic且原文件哈希不变。报告`artifacts/high_step_check.json`，SHA256 `C29DB6A0394D163C28F4A18AC8A40DFD8C60179015559B5BC94BE4FFA4EF1B4B`，`policy_validation=false`。旧Actor此短采样没有合格起脚后的完整周期事件，不能证明新目标已学会。
- 使用现有Windows锁定环境执行`pointleg train PointLeg-HighStep-v0 --debug --env.scene.num-envs 4 --agent.resume True --agent.load-run flat_pretrained --agent.load-checkpoint model_0.pt --agent.run-name high_step_smoke`，实际两次PPO、每轮8步，运行目录`logs/rsl_rl/pointleg_high_step/2026-10-27_19-35-00_high_step_smoke`；model_1 SHA256为`C8F9EDFE0F2F8A1B267CC6E2C566745C5FD9E625C7F456EEFC58697131428B93`。这是采样/更新/保存链路，不是正式训练。可选GUI初始化尝试没有取得通过结果并已停止；不记录GUI通过。

用户随后发侧视图，反馈前腿膝夹角大、后腿髋夹角大和机身前倾，并明确要求以后不要再跑仿真测试。已将默认验证约束同步至项目AGENTS、主索引和操作/验证记忆；只做源码和定向单元检查，除非用户另行明确要求，不自动启动CPU/GPU物理、Viser或PPO。

图片不能确认持久均值，也不能据此归因XML；本轮明确在刚新增的HighStep基础任务改软约束，不改其他任务。新增`high_step_pitch`权重−1.5，俯仰容许±3°、超限尺度8°；新增`high_step_mean_leg_symmetry`权重−0.8，髋俯仰/膝相对默认角采用0.8秒EMA、启动宽限1秒，前后和左右慢偏置差容许4°、尺度8°。共同屈曲允许，正常摆动相和支撑相不强制同角。新增俯仰和前后髋/膝偏差度数指标，无有效平均窗口不写空样本。

最新实际执行`.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -p test_high_step.py -q`，15项通过，新增正反俯仰等价、持续偏置、共同行为不罚、交替摆腿和局部重置检查。没有在最新姿态项上执行仿真、PPO或完整回归，也没有正式步态、真实爬梯或实机验收。需要新启HighStep训练进程才采用这些奖励；已运行进程不热更新。

## 2026-10-27 HighStep足底目标提高至170毫米

用户指定抬腿目标足端高度170毫米。仅将`config/high_step.py`中`high_step_cycle.params.target_height`和`mdp/high_step.py:HighStepCycleReward.__call__`默认值从0.14米改为0.17米；评分宽度仍0.05米。0.40秒周期、0.15～0.25米/秒命令、6～10厘米目标步幅、14厘米最大落脚位移、承重门控、姿态约束、网络和动力学均不变；Flat、Elevation没有改动。

同步了专项文档、测试样本和项目记忆。实际执行`.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -p test_high_step.py -q`，15项通过，明确检查170毫米优于旧140毫米目标、低抬腿及过高抬腿，且最大步幅仍140毫米。没有运行仿真、PPO、完整回归或修改检查点。需要重启训练并续训才采用新奖励；现有进程和保存模型不会自动变化，实际170毫米动态可达性及步态尚未验收。


## 2026-10-27 奖励审查修复与逐足验收

用户要求按本轮审查修复问题代码。仅改HighStep、Elevation、共用评估及相关测试/文档，未改Flat/Rough/Jump动力学、地形、默认站姿、模型文件或策略输入45/325/12；没有启动/停止训练，没有执行物理仿真或PPO。

HighStep修复接触坐标系接口：原传感器global_frame=False却按force[...,2]判断竖直承重；现fields增加normal/tangent、global_frame=True，循环构造器拒绝非世界坐标契约。loaded_feet按声明帧读取并排除侧面接触。旧测试人为Z力掩盖错误，已改为读取实际任务传感器配置。

HighStep最终目标仍0.17米、周期0.40秒、速度0.15～0.25米/秒。原高度/收腿/周期/步长全乘评分改为分级高度乘收腿比例及加权节奏评分，并增加单足每次摆动最多0.02回报的历史抬升进展；需要身体前移、两足承重、未超时，悬停不反复支付。新增diagonal_gait权重−0.6，实际接触配对，四足过渡宽限0.06秒、启动0.6秒；cycle_symmetry权重−0.4比较左右足高/周期/步幅，容许15%。删除重复long_foot_air_time与leg_motion_penalty；慢髋/膝对称−0.8改−0.2且仅左右比较，前后偏差仍记录。没有隐藏相位或自动高度等级课程。

Elevation奖励元数据版本2改3，兼容直接resume的逻辑不变。command_aligned_progress从0.5置0、command_progress_deficit从−3置0、time从−0.05置0；center从−3置0、heading从−1.25置0，保留边界和世界偏差指标。lateral改机身侧速，新增yaw_rate权重−0.75；angular不再重复罚偏航。速度目标取消障碍区暗中0.30米/秒截断，与Actor命令统一。平地diagonal_gait权重−0.2，障碍区入口前0.5米至出口后0.5米放宽。高度质量改peak/target，取消2厘米以下零区和平方；有效小步刷新参与计时，高质量周期另计。动作代价由截断平方改log1p，避免大动作统一饱和。

通用评估对HighStep从0.4改0.2米/秒，平均速度误差0.25改0.05米/秒。每足至少5次已识别循环、至少80%达到目标高度80%～120%和0.40±0.08秒周期；左右平均足高/周期/步幅相差不超20%，对角接触错误均值不超0.15。所有已识别失败落足也进分母，报告保留逐足统计，不能用少量好步隐藏坏腿。

实际检查：HighStep15项、Elevation71项定向单元通过；最终`.venv/Scripts/python.exe -m unittest discover -s src/pointleg/tests -q`共324项通过，用时21.644秒，包含10项审查回归。git diff --check通过。没有物理积分、Viser、PPO、正式模型验收或硬件能力验证。新代码需新进程续训才生效；保留同任务直接resume，不强制transfer。绝对回中仍需外环导航信息，未接ROS闭环。


## 2026-10-27 HighStep回放Viser零范围断言修复

用户粘贴的play日志指向`pointleg_high_step/2026-10-27_21-05-02/model_5900.pt`，模型已加载且45/325/12正常，失败位于通用速度命令GUI：侧向/转向最大值0被用作最小值0.1的Max滑条初值。修复`mjlab/src/mjlab/tasks/velocity/mdp/velocity_command.py:create_gui`，零范围轴改只读数字并保持三轴索引；非零轴按两端绝对幅度创建合法Max控件，兼容小于0.1的范围，保留单向/非对称方向比例并包含停止值0；调整幅度后裁剪当前GUI值。同步framework changelog与HighStep说明。

新增`test_velocity_gui.py`，用实际HighStep play配置和模拟GUI验证，无服务器或物理场景。默认uv缓存写入失败后使用项目内`.uv-cache`，执行`uv run --locked --no-sync python -m unittest discover -s src/pointleg/tests -p test_velocity_gui.py -q`，4项通过。没有安装依赖、启动Viser/物理仿真/训练、修改模型或奖励，没有重跑先前324项全量。可重跑原play命令，无需重训或更换检查点。

## 2026-10-31 Flat8700迁移至梅花桩及训练指令

用户明确要求将`logs/rsl_rl/pointleg_flat/2026-10-22_09-27-04/model_8700.pt`实际transfer到梅花桩，并提供512环境、10000轮训练命令。源文件确实存在，6267957字节，Actor45、Critic325、动作12，迭代8700；源保存的Actor/critic观测顺序和scale已与当前目标公共前缀核对。原迁移工具仅支持Elevation/专项/Jump，不能直接加载到梅花桩105/381维，因此本轮仅扩展迁移工具，不改任何奖励、地形、PPO默认或机器人控制参数。

修改`src/pointleg/transfer.py`，为`PointLeg-SteppingStones-v0`使用真实训练所用`MjlabOnPolicyRunner`和无物理场景的105/381观测桩；加载源Actor/Critic时保留前45/325列，增加60/56列零权重，其余权重/偏置/探索噪声参数严格完整加载。支持已有105/381网络同结构权重迁移，拒绝未知输入维度、隐藏层不匹配及其他严格加载不匹配；沿用拒绝覆盖已有输出目录。仅加载网络，优化器/迭代/公共步数重新初始化为0，不带入源8700轮计数或环境状态。

实际执行：

```powershell
.\.venv\Scripts\python.exe -m pointleg.run transfer PointLeg-SteppingStones-v0 --checkpoint-file "E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab\logs\rsl_rl\pointleg_flat\2026-10-22_09-27-04\model_8700.pt" --output-run flat_8700_pretrained
```

输出`logs/rsl_rl/pointleg_stepping_stones/flat_8700_pretrained/model_0.pt`和同目录`transfer.json`。源SHA256为`c7af6d864a0cfde0f91b382ca0854016a806879392d9327224e10d036e8e52aa`，迁移后不变；目标SHA256为`c6e0b9b647d7477e0f2e72095fc6e83ba3e65e01bad6513982ecdb83e7d87ade`。没有覆盖源模型或已有目标目录。

验证：新增`test_stepping_stones_transfer.py`的6项CPU检查覆盖输入前缀/scale、旧权重保持、新列全零、优化器及公共步数归零、严格resume、初始MLP数值一致、已有105/381迁移、不覆盖/非法维度拒绝与训练参数解析。连同`test_elevation.ElevationTransferTests`、`test_elevation_specializations.SpecializedCheckpointTests`、`test_jump.JumpCurriculumAndCheckpointTests`共17项通过。原专项测试采用发现模式的`from test_elevation import ...`，初次限定包路径调用遇到导入错误；在测试目录加入模块搜索路径后原类和新类全部通过，未改原专项测试源码，也未声称初次命令通过。

另对实际新model_0进行CPU严格resume，优化器state为空、迭代与公共步数0；全部原网络参数逐项相等，新列均零。32组随机原观测和随机任务扩展输入的MLP对照最大绝对误差Actor1.9073486328125e-6、Critic5.245208740234375e-6；源哈希再次核对不变。报告为`artifacts/stepping_stones_flat8700_transfer_validation.json`。该数值检查只说明初始化输出保持，不代表机器人已会通过梅花桩。

交付训练指令已tyro解析且实际检查点路径解析正确：`uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run flat_8700_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 10000 --agent.run-name flat8700_stepping --agent.logger tensorboard --agent.upload-model False`。从0追加10000次更新，最后编号9999；日志在默认`logs/rsl_rl/pointleg_stepping_stones/<时间戳>_flat8700_stepping/`，不用重新transfer。完整命令与模型记录同步`docs/STEPPING_STONES_REWARDS.md`、MEMORY及接口/操作/验证/缺口页。

遵守当前用户验证约束，本轮未运行CPU/GPU物理积分、Viser、PPO调试或正式训练，未启动/停止用户已有进程。主任务仍2～8桩课程，接口105/381/12；课程持久化、正式评估与部署适配等既有缺口未由本次迁移自动解决。




## 2026-11-01 train_4 支撑行程与轻度侧漂修正

2026-11-01 HighStep train_4 支撑行程修正：stance_backstroke 权重 +0.04→+0.10，7厘米目标由线性改为比例平方，乘 min(承重时长/0.30秒,1)；8N承重、0.16秒最低门槛、2.5厘米累计真实接触滑移质量保留。有效后扫仍取真实机身推进和相对支撑足后扫的较小值，取最后一个承重采样，避免离地后的足端移动计入。超过目标不再增加收益。command_orthogonal_drift −2→−2.5，0.010米/秒死区和0.10米/秒尺度保持，左右等价。仍24项，主动抬升目标/两项权重、姿态、45/325/12、PPO/PD及Elevation/Jump不改。新增逐足 Metrics/high_step_stance_sweep_{leg}_m、stance_body_progress_{leg}_m、stance_start_x_{leg}_m、stance_end_x_{leg}_m、stance_duty_factor_{leg}，分别表示支撑相对后扫、身体推进、落地/离地前相对机身前向位置和完整周期支撑占比，前向坐标轴在落地时固定，世界足端静止也会随身体推进产生后扫。保持旧 backstroke/stance_time/contact_slip 指标。25项配置和张量检查通过，覆盖相同总距离短步不优于长步、滑移降质、7厘米饱和、静止/拖脚/短触地/失败无收益、事件缓存/部分重置/非有限/控制周期及左右同罚；不代表训练验收。用户明确授权按固定命令实际后台续训10001次更新，不执行额外仿真测试。

源运行末50轮10149～10198：四足有效后扫3.709/3.842/4.155/3.929厘米，承重时长0.173/0.179/0.201/0.199秒（FL/FR/RL/RR）；主动抬升均值19.346厘米、周期0.329秒、绝对侧速0.04171米/秒。属于训练日志统计，不是当前模型独立回放结果。代码入口 src/pointleg/config/high_step.py、mdp/high_step_basic.py；测试 test_high_step_basic、test_high_step_backstroke。


2026-11-01 08:12实际后台启动用户指定train_4命令：运行目录 `logs/rsl_rl/pointleg_high_step/2026-11-01_08-12-22_train_4`，从固定`2026-11-01_04-10-49_train_3_2/model_10198.pt`直接resume；控制台确认加载路径，已观察迭代10215且已保存model_10200.pt，新增20条逐足支撑轨迹指标已进入TensorBoard。10001次追加更新、间隔100、512环境，预期末次20198。启动器PID 4100，TensorBoard事件记录训练PID30180；进程在后台保留，由用户后续验收，不声称训练完成或轨迹已改善。首次受限后台启动未保留进程、无训练输出；随后在沙箱外隐藏启动成功，无重复训练实例证据。源SHA256 6e38e914384558557b5585e0dfb6a1842af9880d39d52697423dcaaee785931e；checkpoint接口45/325/12和优化器字段已核对、实际resume已执行。25项张量/配置、AST及diff --check通过；启动前未做独立物理测试。MuJoCo现有MULTICCD圆柱/盒接触警告未阻止训练。启动和运行证据见artifacts/train_4_launch.json、train_4_runtime.json、high_step_train_4_preflight.json，stdout/stderr路径见launch元数据。


2026-11-01 12:12 train_4末段分析：实际读取TensorBoard截至16586，末50轮16537～16586，与源train_3_2的10149～10198比较；读取后最新保存model_16600.pt，日志仍增长，本轮未停止/重启训练或改代码。四足平均支撑0.18812→0.19500秒，有效后扫3.909→4.099厘米，未达到7厘米/0.30秒目标。主动抬升FL/FR/RL/RR由20.96/18.30/19.17/18.94变为22.85/18.31/16.19/25.26厘米，左右后腿差从0.22增至9.08厘米；右后膝相对髋竖直摆幅3.16→0.98厘米。对角FR/RL角位移误差4.65→14.15度，慢姿态惩罚−0.01131→−0.26219。绝对侧速4.171→4.013厘米/秒，接触点滑速2.879→3.019厘米/秒，均回报101.78→99.78，回合长度976.77→987.51，不能用存活/总分证明步态质量。抬升两项日志收益合计4.2263，后扫0.2082，相差约20.3倍；只说明回报构成，不等价于梯度贡献。源码high_step_basic.py:245连续抬升收益未在15厘米截断，只有height_event平方项饱和；先前“无过高收益”的描述不适用于连续项。当前事件抬升仍按每次有效摆动计分，持续鼓励高频，与后扫平方/时长系数的目标存在冲突；判断该调整未达成支撑目标且动作不对称加重，建议暂停原样延长并从10198重新设计距离预算/饱和与短支撑反馈，尚未实施。后扫端点计量本次改为最后承重样本，与源多含一个起脚采样有小口径差异；指标为每轮日志均值再平均，非独立回放。证据artifacts/train_4_latest_analysis.json，无仿真或训练验收。


## 2026-11-01 tran_4_2 距离额度与超高惩罚

2026-11-01 tran_4_2 修正HighStep奖励冲突：active_lift_event +0.35及active_lift_progress +0.08的名义权重保留，二者和hip_swing_coordination +0.06改乘同一逐足承重路程额度。每控制周期沿本段前进起点固定航向刷新机身历史最远位置，新增距离封顶指令速度×dt；每足只在连续8N承重期间积累新距离，离地冻结min(该距离/7厘米,1)，新支撑清零。循环共享该额度，常见7厘米以内等高度的同总路程拆成更多小步不会多领这三项收益；静止、往返和已错过路程不补发，超速多余位移不延迟兑现。连续抬升的峰值及已支付高度封顶15厘米。增加active_lift_overshoot −0.5：摆动时相对起脚足底主动抬升超过15厘米，按5厘米尺度平方、单足封顶4，四足均值和最差足各半；不乘正奖励路程额度，原地超高摆动也扣分。short_stance_target为独立软目标0.24秒，原0.16秒有效起脚/稳定落地门槛保留。侧漂−2.5→−3.0，容许0.010米/秒/尺度0.10米/秒保持且左右等价。后扫+0.10/7厘米平方/0.30秒、姿态、PPO/PD、45/325/12和其他任务不改。当前25项奖励。新增Metrics/high_step_lift_distance_credit_{leg}、high_step_lift_overshoot_{leg}_m、high_step_new_forward_distance_m。代码入口config/high_step.py、mdp/high_step_basic.py及mdp/high_step_hip_swing.py。

42项定向配置/张量检查通过（test_high_step_basic、test_high_step_backstroke、test_high_step_lift_budget、test_high_step_coordination），覆盖8/12/15厘米评分、15厘米以上正收益封顶/负代价递增、常见等距离拆步不多领、髋摆共用额度、静止/往返/超速/腾空、重复读取/部分重置/失败/非有限、0.24秒浮点边界及控制周期积分；AST与diff --check通过。未额外启动物理验证。旧train_4已按命令行身份停止Python PID30180及对应uv启动器，最后保存model_17100.pt，模型日志保留；从用户认可原始train_3_2/model_10198.pt直接resume，源SHA256 6e38e914384558557b5585e0dfb6a1842af9880d39d52697423dcaaee785931e。启动状态另记，不把配置和张量通过等同于训练效果。

从10198追加10001次更新，每100保存，预期末次20198；先回放10300/10400观察足高、支撑和拖腿。正奖励已按路程缩放，不能直接用旧运行总分比较效果。

```powershell
uv run --locked --no-sync pointleg train PointLeg-HighStep-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run 2026-11-01_04-10-49_train_3_2 --agent.load-checkpoint model_10198.pt --agent.max-iterations 10001 --agent.save-interval 100 --agent.run-name tran_4_2 --agent.logger tensorboard --agent.upload-model False
```



2026-11-01 12:41实际启动tran_4_2，运行目录`logs/rsl_rl/pointleg_high_step/2026-11-01_12-41-56_tran_4_2`，uv启动器PID48076、事件训练PID28384。控制台确认加载原train_3_2/model_10198.pt，已观察迭代10210并写出model_10200.pt；超15厘米惩罚和逐足路程额度/超高量指标已进入TensorBoard，实际采用新版配置。512环境、10001次追加更新、每100次保存，预期末次20198。后台保留供用户验收；42项张量检查、接口/优化器字段/原模型哈希、CLI、AST与diff检查通过，未做独立回放或物理测试，不声明步态已改善。旧train_4已停并保留最后model_17100.pt。启动/运行/预检证据为artifacts/tran_4_2_launch.json、tran_4_2_runtime.json、tran_4_2_preflight.json。


2026-11-01 13:52 tran_4_2最新足端分析：读取截至11776轮，末50轮11727～11776；最近保存model_11700.pt，训练日志仍在更新，本轮未改代码或停止训练。逐足主动抬升FL/FR/RL/RR为14.84/14.64/15.23/14.95厘米，源10198末50轮为20.96/18.30/19.17/18.94厘米；全足合并事件均值14.916厘米，超高和四足差异明显收敛。承重时长为0.1824/0.1839/0.1943/0.1949秒，四足平均0.1889秒，源0.1881秒，仍未达0.24软短支撑目标或0.30时长质量满分。有效后扫3.70/3.69/3.90/3.96厘米，平均3.81厘米，源3.91厘米；足端相对机身原始支撑后扫4.22/4.42/4.70/4.59厘米，均未到7厘米有效目标。支撑占比约58.5～61.7%，周期0.3290→0.3127秒。侧速绝对值4.171→3.429厘米/秒，真实接触滑速2.879→2.688厘米/秒，偏航速率绝对值7.883→6.520度/秒；右后膝竖直摆幅3.16→3.72厘米，右前左后对角误差4.65→4.15度。抬升与一致性修正有效，但不能认为支撑推进已经改善；训练总回报因计分口径变化不直接横比。证据artifacts/tran_4_2_latest_analysis.json，属于日志迭代均值，不是独立模型回放验收。


2026-11-01 15:22 TensorBoard链接现场核对：localhost:6006/data/environment实际data_location为pointleg_high_step/2026-11-01_12-41-56_tran_4_2，run="."，非其他实验；链接钉选feet_contact_without_cmd不在该次HighStep奖励/日志中。事件读取截至13677，末50轮13628～13677，最近保存model_13600.pt，训练仍运行，本轮未改奖励或停止/重启。FL/FR/RL/RR主动抬升15.33/15.19/14.82/15.25厘米，承重0.1886/0.1877/0.1972/0.1973秒，有效后扫3.79/3.82/4.01/3.96厘米；距0.24/0.30秒及7厘米目标仍明显不足。对比11727～11776：侧速绝对值3.429→3.339厘米/秒，接触滑速2.688→2.759厘米/秒，回报66.98→71.19，回合长度986.73→995.67；当前同版奖励内可比，存活/速度跟踪改善不代表后扫达标。前后髋慢偏差5.68→11.22度，左右前足平均前伸由7.68/5.26变9.26/3.81厘米，右前左后对角误差4.15→6.19度，慢姿态惩罚−0.00734→−0.06197，说明姿态分化重新加重；不应仅按回报上升认定策略更好。原始统计和上次11776窗口同表保存在artifacts/tran_4_2_tensorboard_analysis.json；仅为训练日志均值，建议11700和13600独立回放对照，未执行回放验收。

## 2026-11-01 HighStep tran_4_2 权重迁移至 Elevation

- 原因：用户结束 HighStep `tran_4_2`，要求将当前策略作为混合 Elevation 和四个上下台阶/坡面专项的后续训练初始化，并将目标轮次清零。
- 操作：确认 HighStep 训练已停止；停止前最后完整检查点为 `logs/rsl_rl/pointleg_high_step/2026-11-01_12-41-56_tran_4_2/model_14100.pt`，SHA256 `e9251beb3aee12463109f704d86ef14b1bcc3645d6ff527cf0a47b28195376a9`。依次执行 `pointleg transfer` 到 `PointLeg-Elevation-v0`、`PointLeg-StairsUp-v0`、`PointLeg-StairsDown-v0`、`PointLeg-SlopeUp-v0`、`PointLeg-SlopeDown-v0`；输出运行名均为 `highstep_tran_4_2_14100_pretrained`。
- 结果：五个目标目录均生成 `model_0.pt` 及 `transfer.json`。全部检查点 `iter=0`，优化器重新初始化，课程重新初始化；元数据的源哈希及任务 ID 均正确，Actor/Critic 首层分别为45/325输入、动作输出12，并与源模型逐张量一致。
- 验证：已执行 CPU 检查点加载与元数据核对，确认五项目标文件存在、轮次为0、网络维度45/325/12、Actor/Critic逐张量等同源策略；未启动物理仿真、Viser或任何 Elevation 训练，不能视作台阶或坡面能力验收。

## 2026-11-01 上下坡/楼梯合并与基础奖励重建

- 用户要求：把上下坡合并为一个任务、上下楼梯合并为一个任务，精简合并任务中的复杂和重复奖励，保留掉落与侧漂扣分，参考HighStep基础结构。
- 实现：新增 `PointLeg-Stairs-v0`、`PointLeg-Slope-v0` 及独立日志目录；共享 `config/elevation_basic.py` 的14项显式白名单。`walking_terrain_names`使重置、固定评估与课程指标识别双向组；上/下坡课程独立、下行从顶部出生，接口45/325/12、固定台阶、随机平台、物理布局及PPO不改。原混合和四个单方向入口及HighStep/Jump配置保留。
- 奖励：独立速度+2、完成+20、失败−50；侧漂−3/0.010米每秒死区，偏航速率−0.5；临边−1、真实接触滑移−1、姿态−0.5、机身角速度−0.1、8N至少两足承重−2、关节越限−0.2、力矩/动作各−0.02、非足端/足侧接触−0.2。移除复合循环/后扫/指定时序/足高目标/腿姿态比较/新踏面事件/重复推进等，以及依赖已删除状态的旧足高/质量指标。新任务元数据版本8。
- 模型：同一HighStep `tran_4_2/model_14100.pt` → 两个新实验目录下 `highstep_tran_4_2_14100_pretrained/model_0.pt`，iter/公共计数/优化器/课程归零；全部网络张量与源相同，严格resume通过，源SHA256不变。证据 `artifacts/elevation_paired_transfer_validation.json`。
- 检查：新9项和原专项/逻辑/行走/迁移44项聚合53项通过（5.049秒）。实际CPU迁移与加载、两条200更新CLI解析/目标路径、AST检查通过；新测试首次play/训练动作延迟对比错误已修正，不涉及产品动作修改。没有仿真、Viser或训练，不宣称移除目标后足高仍保持15厘米。
- 交付：`docs/ELEVATION_PAIRED_TRAINING.md`含奖励、范围、路径和两条可直接resume的200更新指令，项目相关记忆页同步。

## 2026-11-01 移除四个单方向高程任务

- 用户要求：删除原四个单方向入口，仅保留上下台阶、上下坡两个合并任务及四类地形混合Elevation。
- 实施：`TASK_SPECS`移除四个旧ID与配置import；`ELEVATION_TASK_TERRAINS`缩减为三个入口，使旧ID不再参与train/play/evaluate/transfer。删除 `config/stairs_up.py`、`stairs_down.py`、`slope_up.py`、`slope_down.py`，将旧专项测试改为现有入口和拒绝旧ID的检查。原四种地形内部定义、现有奖励/课程/策略接口均保留。
- 数据：所有历史日志、旧四专项model_0、两个新合并model_0和混合model_0均未删除或覆盖。旧ID已下线，旧模型参数仍可通过文件路径迁移到现有任务。
- 文档：旧专项训练页改为三个有效入口索引，合并训练页移除四专项兼容入口描述；更新项目索引、注册状态、训练方式与当前状态。
- 验证：定向配置/张量/检查点40项通过（1.639秒），涵盖两组采样/奖励/恢复及旧混合逻辑，旧ID在load/evaluate/transfer均被拒绝。未启动仿真、Viser或训练。

## 2026-11-01 Slope延长周期、加速与髋纠偏

用户报告碎步、左前髋内收、单足上坡后姿态分化和回合太短，要求增加周期、加强偏髋惩罚、速度提升至0.25～0.50米/秒并延长回合。仅在`config/slope.py`中替换冻结的`ElevationSettings`并同步select/reset事件与指令范围，速度由0.15～0.25→0.25～0.50米/秒，训练和回放时限由60→120秒。髋外展零位惩罚−3.6→−7.2，零死区/0.25弧度尺度平方和保持；通过`mdp/slope.py:slope_hip_abduction_zero`记录四腿有符号角度，不锁关节或修改动作、PD。

旧Slope17项里没有步态周期目标。新增`step_timing`−0.6，使用真实8N承重，落地确认0.06秒、离地确认0.04秒、启动宽限0.80秒。逐足稳定落地到下一次稳定落地目标0.70±0.10秒，短承重目标至少0.30秒；短周期/短承重事件按0.10秒等效代价除dt后积分，超过0.80秒未换步连续扣分，悬空超过0.55秒另罚。较长周期配合最短支撑给足端更多推进时间，不对每次换步发正奖金或强制腿序，不宣称实际行程已延长。记录逐足cycle/stance及landing_stride_x，最后一项是两落地位置世界X之差，不等于真实滑移或法向足高。

新增`mean_leg_posture`−0.6，复用`ElevationMeanPosturePenalty`，髋俯仰/膝角相对默认值EMA0.8秒、宽限1.2秒，死区6/10度、尺度12/18度。真实承重足高差>3厘米时前后容许×2.5、整个慢姿态代价×0.25，允许前后足处于不同坡段的必要屈曲，不强制瞬时四腿同角。总奖励17→19，契约revision9→10，加入120秒时限元数据。同任务旧坡面checkpoint仍可完整resume并保留网络/优化器/课程/轮次，奖励缓存重新开始；新进程才采用变化。已有1秒宽限+3秒窗口的停滞失败仍可提前触发，总时限延长不等于禁止提前失败重置。Stairs、混合Elevation、HighStep、Jump、50Hz、45/325/12和物理地图保持。

实际验证：`test_slope`+`test_elevation_paired`共21项通过（1.937秒），五文件py_compile通过。包含0.30/0.70秒周期代价排序、10/20毫秒积分、过长承重/悬空、防短促点地、缓存/局部reset/失败/非有限、单腿偏髋左右等价、慢姿态/共同屈曲/承重高差、随机速度/训练及play范围和其他任务隔离。CPU临时合成revision9坡面检查点完整resume后保留Actor张量、iter123和课程，并确认运行配置仍为新速度和120秒；未对用户真实checkpoint进行加载/覆盖，未启动仿真、Viser或训练。同步索引、任务页、验证页及`docs/ELEVATION_PAIRED_TRAINING.md`，实际步态效果待后续训练观察。

## 2026-11-01 Slope四足出口承重即成功与任务范围核对

用户指出观察到楼梯、要求仍按上下坡一起训练，并要求四腿支撑上平台即成功、重置且给奖励。源码`walking_terrain_names("slope")`和`select_elevation_terrain`一直只分配语义2/3即slope_up/down；核对保存的`logs/rsl_rl/pointleg_slope/2026-11-01_20-17-27_train_2/params/env.yaml`，select/reset两处均为terrain_kind=slope，该旧运行仍使用.15～.25米/秒/60秒。物理4×4地图保留六类几何，其中楼梯可见但不属于Slope重置采样；未修改地图或冒称已经重现用户看到的动态台阶过程。读取进程命令行未发现匹配的运行进程，不由此推断历史回放任务身份。

原Slope成功要求整机后缘越过exit_x+.10且继续满足两足支撑、姿态/速度稳定.5秒。现仅Slope改为上坡四足在顶部出口平台、下坡四足在底部出口平台实际承重，当控制步直接complete、现有task_completed结算+20，框架先算终止与奖励再auto_reset。新增`mdp/elevation.py:slope_exit_supported_feet`：每足至少一个真实、有限接触点，在本地exit_x至地块末尾之间、通道内、出口高度误差≤8毫米，地面朝向法向且法向承重≥8N。四足同时满足且初始化有效、未发生失败才完成；无额外.5秒等待，不要求全机后缘再越界10厘米。失败优先，三足、残足仍在坡上、悬空/弱承重/侧接触、下坡出生顶部均不得成功。Stairs、混合Elevation与Jump不采用新成功分支。

新增Metrics/slope_up_fraction、slope_down_fraction、slope_exit_loaded_feet，核对实际分配和终点承重足数；Slope契约revision10→11，记录completion=four_loaded_feet_on_exit_platform、8N、额外保持0秒，奖励仍19项/.25～.50/120秒。旧同任务checkpoint继续完整resume并提示契约变化，不覆盖生产模型，不自动重启训练。

验证命令`.venv/Scripts/python.exe -c "import sys, unittest; sys.path.insert(0, 'src/pointleg/tests'); suite=unittest.defaultTestLoader.loadTestsFromNames(['test_slope', 'test_elevation_paired', 'test_elevation.ElevationLogicTests', 'test_elevation.ElevationWalkingTests']); result=unittest.TextTestRunner(verbosity=1, buffer=True).run(suite); sys.exit(not result.wasSuccessful())"`，49项通过（2.590秒），六文件py_compile通过。首次平移夹具的根位置Y未随地块平移导致wrong_route断言失败，修正夹具后通过。覆盖终止/奖励管理器的+20、局部reset、失败优先、真假接触、256环境50/50及既有任务回归；未执行物理仿真、Viser、PPO、正式训练或动态回放验收。同步MEMORY索引、任务/验证页和合并训练说明。

## 2026-11-02 Slope目标速度降至0.15～0.30米/秒

用户认为目标行走速度过快，明确要求0.15～0.30米/秒。仅将`config/slope.py:slope_env_cfg`中的替换设置由`speed_range=(.25,.50)`改为`(.15,.30)`；既有配置工厂会把同一设置写入select/reset事件，并同步twist命令范围，训练及play均采用。初始速度随机化、19项奖励、步态周期、120秒时限、成功规则、上下坡课程、其他任务与45/325/12未改。保持revision11，walking契约的速度字段会记录新值；原Slope检查点可同任务完整resume，正在运行的进程不会自动热更新。本轮未启动、停止或重启任何训练/回放，未覆盖用户checkpoint。

更新现有测试的速度范围和旧checkpoint恢复后配置断言，未增加新的测试文件。实际执行三个定向用例：`test_elevation_paired.BasicElevationRewardTests.test_whitelist_and_interfaces_exclude_old_compound_terms`、`test_slope.SlopeTests.test_reset_samples_requested_commands_and_play_uses_extended_deadline`、`test_slope.SlopeTests.test_previous_slope_checkpoint_resumes_with_new_rewards_and_deadline`，全部通过（0.245秒）。覆盖训练/play范围一致、select/reset共用设置、随机采样在新范围内、120秒及Stairs隔离、CPU临时合成旧checkpoint保留网络/轮次/课程而使用新速度配置。未执行物理仿真、Viser或训练；参数检查不代表滑移或上坡效果已验收。同步MEMORY索引、任务页和合并训练说明，固定评估中间速度示例由0.375改为0.225米/秒。

# 2026-11-04 Stairs 卡滞足恢复引导

用户希望仅在足端卡住时引导足端后撤、抬升、越边和落脚。`config/stairs.py`添加权重+0.5的`stuck_foot_recovery`，实现在`mdp/stairs.py`。原先无该项；现对上行前三级各足检测非承重且靠近立面持续0.35秒，另两足至少8N、机身低速和指令前进；足底目标为台阶表面加2厘米。后撤/抬升/越边只给历史最大进度增量，落脚以踏面内部有效8N接触结算，每足每级每回合一次，1.5秒超时。奖励契约22→23，终止/课程/其他奖励权重和观测动作未改；旧同几何检查点可resume并提示奖励变更，新训练进程才加载。65项Stairs/Slope/Elevation单元测试通过，无物理仿真、可视化或训练验收。实际收益、恢复率及悬空惩罚冲突待训练验证。

2026-11-05 用户反馈右前腿摆动过快，Stairs由无摆动角速度代价改为新增`swing_joint_speed`（−0.2），四足相同、足端无8N承重时检测髋俯仰4 rad/s和膝6 rad/s阈值，超出按4 rad/s尺度平方、每关节封顶1后四足均值。新增逐腿摆动髋膝速度日志；属于软惩罚，阈值需训练数据标定，不保证足端速度或冲击下降。训练契约revision23→24，其他奖励、成功失败条件及地形不变；同布局旧checkpoint可resume并警告奖励变化，须重启进程。66项定向单元测试与py_compile通过；未运行仿真/训练。

2026-11-05 用户反馈下台阶时小腿靠台阶沿支撑。Stairs新增`down_shin_contact`权重−1，依据现有膝部刚体与地形接触最大力计算，2N死区、20N饱和；仅下行生效，逐腿力和接触足数有日志。保留−0.2通用碰撞项，无新增失败条件，控制频率采样可能漏短碰。奖励契约24→25，地形和动作未改；同布局旧checkpoint可resume，重启生效。67项定向单元检查通过，未做物理仿真或训练。

2026-11-05 用户反馈两前足同时悬空且前腿过度前伸。Stairs新增`front_support_loss`权重−1、双前足未承重0.06秒宽限、0.12秒渐增，新增`front_overreach`权重−0.5、上行接近台阶时机身坐标足端相对髋前伸超0.22米按0.12米尺度平方，单足封顶。原角度项用绝对值不约束前后方向。两项均为软约束，已有高度/停滞/成功条件不变；可能与跨阶进度冲突，需训练校准。契约25→26，同布局checkpoint可resume并提示奖励变化，需重启。69项单元测试通过，未运行仿真或训练。

2026-11-05 用户反馈后腿大腿高过机身后向轴。新增Stairs奖励`rear_thigh_above_body`−1，对后腿髋膝向量机身坐标正Z角以20°尺度平方、单腿封顶1，正Z为惩罚对象；记录有符号后腿角度。原无向角姿态奖励和其他任务保持，未添加硬终止，注意后腿越阶能力。契约26→27，同布局旧checkpoint可resume并警告奖励变化，需重启；70项定向单元测试通过，未物理仿真或训练。

2026-11-05 用户要求双前腿同时腾空立即惩罚：Stairs `front_support_loss`宽限0.06→0秒，权重−1和0.12秒惩罚上升时间保持，首次20ms周期为1/6满额，一足恢复8N支撑清零；以无8N支撑为近似，非物理接触严格检测。契约27→28，同布局旧checkpoint可resume并提示奖励变化，需新进程。70项定向单元测试通过，未仿真/训练，实际对步态和成功率的影响未知。

2026-11-05 用户要求减少前腿同时腾空并加强下行小腿碰撞惩罚。Stairs增加前脚对侧须连续8N支撑0.10秒的起脚事件惩罚−0.5；前脚同时失撑权重−1→−2，首步惩罚0.5基础加线性递增；下行小腿−1→−2.5；步态周期目标0.70→0.80秒，步态惩罚−0.2→−0.4。首两项可能与停滞惩罚、越阶动作竞争，接触未防抖；不设新失败条件。奖励契约28→29，同布局旧checkpoint可resume并提示奖励变化，需新进程。71项定向检查通过，未仿真/训练，不宣称已解决。

2026-11-06 用户反馈后腿短暂越过机身轴线且前腿仍同时腾空，要求有效承重0.20秒。Stairs `front_premature_takeoff`对侧连续承重0.10→0.20秒；`rear_thigh_above_body`权重−1→−2，角度尺度20°→10°，软惩罚不保证无越线。奖励契约29→30，旧同布局模型可resume并警告奖励变化，需重启。72项定向单元检查通过，未仿真/训练；留意停滞、抬脚能力和成功率。

2026-11-06 用户反馈右前腿在平地及上台阶时过度前伸。Stairs `front_overreach`权重−0.5→−1，移除上行入口/出口位置门槛，覆盖整个上行回合；相对髋的机身x方向22厘米阈值与12厘米尺度不变，双前足对等，下行不变。原来跨阶附近项已生效但为软惩罚，不能保证形态；权重变化需观察落脚和成功率。奖励契约30→31，同布局旧checkpoint可resume并警告变化，需重启。72项定向单元测试通过，未仿真/训练。

2026-11-06 用户认可前足缩短前伸并指出同一级可以多次迈步，Stairs `front_overreach`配置阈值22→18厘米，权重−1、上行全程、尺度12厘米和地形完成条件保持；这是软约束而非18厘米硬限位，现有逐踏面首次奖励不限制重复接触。契约31→32，旧同布局checkpoint可resume并警告奖励变化，需重启。72项定向测试通过，未物理仿真/训练。

2026-11-06 用户要求增大前足前伸惩罚，Stairs `front_overreach`权重−1→−1.5；18厘米软阈值、范围、地形及成功条件保持。奖励契约32→33，旧同布局checkpoint可resume并提示奖励变化，需重启。72项定向单元测试通过，未仿真/训练，需关注成功率和停滞。

2026-11-07 用户明确要求右前髋向前不超垂直20°、膝相对共线屈曲不低30°，且严重惩罚。新增`fr_hip_forward_limit`和`fr_knee_min_flexion`各−5，使用机身坐标真实几何有向角、10°尺度平方、单项封顶4；右前腿上下阶全程有效，不做四足均值稀释。此处垂直解释为机身向下方向，膝正常屈曲对应模型负关节方向。原其他奖励保留，不设硬限位/失败，软惩罚不能保证绝不越界。奖励契约33→34，同布局旧checkpoint可resume并提示奖励变化，需新进程。74项定向单元检查通过，未物理仿真/训练。

2026-11-07 用户反馈爬楼梯偏快，`PointLeg-Stairs-v0`前向速度指令0.15～0.30→0.10～0.20米/秒，上下台阶共用。入口`config/stairs.py:stairs_env_cfg`，重置事件与twist范围同步；奖励、终止、地形及训练接口沿用原配置。执行`test_stairs`、`test_slope`、`test_elevation_paired`、`test_elevation_fixed_stairs`和`test_elevation.ElevationLogicTests`共74项通过，另用64个合成环境验证重置采样。严格加载真实`2026-11-07_01-06-13_train_12/model_11000.pt`（SHA256 `52e25cad7b599afacc1f52847e743de7e1dd0a54430e7713ad185dd0da95c3ef`）通过，保留轮次11000和课程上限[4,4]，采用新速度并发出预期配置变化警告。需启动新训练进程，未执行物理仿真或训练。


## 2026-11-08 台阶12000迁移至梅花桩

按用户指定，将 `logs/rsl_rl/pointleg_stairs/2026-11-07_17-13-11_train_13_slow/model_12000.pt` 通过现有 `prepare_transfer("PointLeg-SteppingStones-v0", source, "stairs_train_13_slow_12000_pretrained")` 迁移至 `logs/rsl_rl/pointleg_stepping_stones/stairs_train_13_slow_12000_pretrained/model_0.pt`，同目录保存 `transfer.json`。训练时使用 `--agent.resume True --agent.load-run stairs_train_13_slow_12000_pretrained --agent.load-checkpoint model_0.pt`。优化器、轮次和公共步数从0开始。

迁移前核对了原观测列顺序、缩放、动作执行器与动作缩放。CPU逐张量核对原权重、Actor新增60列与Critic新增56列均置零、105/381/12接口及新检查点严格加载。源 SHA256 为 `28e87009fe6f19ff7c76361443f5130fbb4a6273d50f638cd7bd6f1269f4740b`，生成后再次核对一致。目标 SHA256 为 `554314e5c5ebbe1f1591648cbd5b941f8b62a2c43e74da8705362e64142a42b3`。验证范围限于配置、权重及CPU加载。

## 2026-11-08 梅花桩登台过渡及站立惩罚

用户确认登台时偶有四足悬空和较大倾斜，要求此阶段取消两项重置，并惩罚悬空与四足长时间触地不动。仅调整PointLeg-SteppingStones-v0。入口过渡为出生至四足登上左平台及之后0.5秒，出口过渡为已覆盖全部桩且机身到右平台前沿0.5米内、四足尚未登台，以及四足首次登台后0.5秒。过渡期间不因四足无接触超过0.02秒或机身倾角超过1弧度而结束，桩阵内恢复原阈值。四足登上入口后下方地面接触仍立即失败，底盘接触、关节限位及其他终止保持。

新增stepping_flight权重-1，按每个控制周期四足无接触的物理子步时间比例计算。既有stepping_support权重-0.5保持，其在四足均无接触时原始值为4，故两项完整悬空每秒合计-3。新增stepping_four_foot_standstill权重-5，参考StairsFourFootStandstill，四足均有接触且前进指令大于0.08米/秒时，记录机身前向2厘米进度，1秒宽限后0.5秒线性至满额。任一足离地、前进2厘米或进入停稳阶段清零。梅花桩现有传感器只有found和pos，此项使用接触标志，不具备Stairs的8N/法向承重过滤。首次登台0.5秒宽限、出口0.5米判断、惩罚权重属于本次训练初值。

代码位于src/pointleg/mdp/stepping_stones.py与src/pointleg/config/stepping_stones.py，输入输出仍105/381/12。实际执行 .venv/Scripts/python.exe -m unittest pointleg.tests.test_stepping_stones -q，30项通过，覆盖登台与出口过渡、宽限到期恢复、桩阵终止、严格接地优先、悬空子步积分、停滞惩罚及局部重置。不运行物理仿真、Viser或训练，新进程才使用本次配置。与旧梅花桩检查点接口一致，可在同任务继续训练后评估实际效果。

## 2026-11-08 梅花桩顶面缩小

用户要求将梅花桩边长由200×200 mm改为150×150 mm。`src/pointleg/terrains/stepping_stones.py`将`STONE_SIZE`由0.20改为0.15，作为桩顶X/Y边长；新增独立的`STONE_HEIGHT=0.20`，使桩顶仍和200 mm高平台齐平，不改变上台或下台高度。`STONE_SIZE`已被桩顶接触区域、边缘余量、首次覆盖落点奖励和相对桩观测共用，因此这些判定自动同步至新150 mm边界；桩中心位置、桩数量课程、奖励权重、终止条件及105/381/12接口未改。更新了几何单元断言和边缘落点夹具。执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，30项通过；`git diff --check`未发现本次文件空白错误，工作树已有无关的`sim2sim/pointleg_sim2sim.py:35`尾随空格。未启动仿真、Viser或训练。旧梅花桩检查点可加载，但因目标几何收紧，需新进程续训并重新评估。

## 2026-11-08 梅花桩训练速度提高

用户更正此前速度输入小数点遗漏：0.75～1.20 m/s应为0.075～0.12 m/s。`src/pointleg/config/stepping_stones.py`的`twist`前向范围改为0.075～0.12；`src/pointleg/mdp/stepping_stones.py:reset_stepping_stones`的每回合`episode_speed`采样使用相同范围，初始默认速度改为中点0.0975。横移、转向、奖励、终止、课程、地形和105/381/12接口未改。新增重置采样范围断言已同步；执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，34项通过，`git diff --check`未发现本次文件空白错误。此前按错误速度运行的`2026-11-08_16-39-13_train_1`不代表正确速度配置，必须以新进程重新训练；未停止或启动训练。

## 2026-11-08 梅花桩下右平台后刹停

用户要求终点不在右侧平台刹停，而是下台后在地面刹停。`collect_stepping_stones_substep`现在要求全桩覆盖后先四足登上右平台，以此锁存`exit_mounted`；之后只允许在右平台前沿后的800 mm通道内地面接触。四足均在该地面区域接触后才激活`brake_time`、`stepping_settle_speed`和`stepping_settle_angular`，刹停0.5秒且稳定0.5秒后成功。任务引导目标在登上右平台后改为右平台后方0.4 m的地面中心，驱动下台；未完成右平台四足登台、从侧面或其他位置接地仍立即失败。下台阶段纳入平台过渡，悬空和姿态终止继续豁免，完成下台后恢复。新增终点阶段单元测试；执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，32项通过，`git diff --check`未发现本次文件空白错误。未启动仿真、Viser或训练。观测、动作、地形尺寸及105/381/12接口不变，已有训练进程需重启。

## 2026-11-08 梅花桩对角姿态对称

用户要求增加梅花桩对角步态姿态基本对称惩罚，参考HighStep和Slope的相位门控思路。新增`SteppingDiagonalPosturePenalty`与奖励项`stepping_diagonal_posture`，权重−0.3。当前接触传感器仅有found/pos，不具备HighStep/Slope的法向力信息，因此以同为触地或同为悬空作为可比较相位；分别比较FL↔RR、FR↔RL髋俯仰/膝角。角度以环绕差计算，20°内不罚、超出按30°尺度平方，单关节封顶1，两个有效对角组平均。只在`stage==2`且不在平台过渡、失败或成功时生效，避免约束登台、出口下台和最终刹停。新增专项张量测试；执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，33项通过，`git diff --check`未发现本次文件空白错误。观测、动作和105/381/12接口不变。已按用户请求启动可见PowerShell，执行`uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.logger tensorboard --agent.upload-model False --agent.resume True --agent.max-iterations 10000 --agent.load-run stairs_12000_pretrained --agent.load-checkpoint model_0.pt --agent.run-name train_1`。启动调用无报错，但未读取训练窗口后续输出，因此不把实际加载、迭代进度或训练结果当作已验证。

## 2026-11-08 梅花桩髋外展零位惩罚

用户要求参考Stairs和Slope加入髋外展关节固定0度左右的惩罚。新增`stepping_hip_abduction_zero`，权重−7.2，复用`hip_abduction_excess`：四个`*_hip_abduction_joint`相对默认零位没有死区，绝对偏差按0.25 rad尺度平方后求和；髋俯仰和膝关节不参与。新增逐腿`Metrics/stepping_hip_abduction_{fl,fr,rl,rr}_deg`。执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，34项通过，`git diff --check`未发现本次文件空白错误。未启动仿真或新的训练。已启动的`train_1`进程不会读取此变更，必须重启训练才生效。

## 2026-11-08 梅花桩上下平台台阶约束

用户要求参考Stairs任务改善上、下平台。`PointLeg-SteppingStones-v0`新增三项，仅在既有`platform_transition`范围内生效，即出生登左平台、从桩阵登右平台及从右平台下至地面。`stepping_platform_lift`权重−2，复用现有`clearance_penalty`的足底0.24 m净空不足与边沿门控，但只在上台阶段额外施加；不会引入第二套足高事件状态。`stepping_platform_front_support`权重−2，复刻Stairs前足双失撑的0秒宽限和0.12秒递增，但以现有足端found接触近似8N承重。`stepping_platform_front_sequence`权重−0.5，前足离地时要求另一前足连续检测接触0.20秒，违例按0.10秒事件量结算。已有四足停滞、髋外展零位、对角姿态与通用净空项不改。未迁入`down_shin_contact`、`stuck_foot_recovery`、`tread_foot_placement`等依赖台阶专用膝部力传感器或踏面状态的项，避免将缺少数据的条件伪造为等价实现。执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，36项通过，`git diff --check`未发现本次文件空白错误；未启动仿真或训练。观测、动作和105/381/12接口不变，需新进程训练才采用。

## 2026-11-08 梅花桩静止惩罚加强

用户反馈策略反复异化为静止，要求增大静止惩罚。日志已显示原`stepping_four_foot_standstill`为0，旧条件要求四足同时found，策略可通过一足轻微离地或接触漏检规避。将该奖励权重−5改为−10，并将累计条件由四足同时found改为至少两足found，仍要求正向命令、非终点刹停、机身前进不足2厘米；支撑少于两足或向前推进2厘米才清零。1秒宽限与后续0.5秒线性升满保持。低速0.075～0.12 m/s正常前进约0.17～0.27秒达到2厘米，因此不应触及1秒静止宽限。更新专项测试以覆盖三足、两足持续计罚及一足支撑清零；执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，36项通过，`git diff --check`未发现本次文件空白错误。未启动仿真或训练，需重启进程采用。
## 2026-11-08 梅花桩左平台整理阶段与姿态约束

用户要求解决上左平台后机身过前、前足接近平台边缘、足端未对齐两列梅花桩及小腿前伸/膝部过低问题。仅修改`PointLeg-SteppingStones-v0`，未改桩几何、课程、速度采样、终止条件或105/381/12接口。

`collect_stepping_stones_substep`新增逐环境`setup_ready`和`setup_time`。首次四足顶面登上左平台后不再直接进入桩阵阶段，而是保持阶段1、将前向命令置零；当四足持续在左平台顶面、机身`x=0.35±0.05 m`、足端前缘不超过`x=0.65 m`、左/右足分别对齐`y=+0.20/-0.20 m`且误差不超过4 cm、双前足相对髋部机身前伸不超过18 cm、四腿髋—膝—足端几何屈曲角均不少于30°时，连续0.5秒锁存完成，再进入阶段2并恢复每回合0.075～0.12 m/s速度。整理状态复用既有四类阶段编码和严格接地观测，因此没有增加输入列。

新增整理期奖励：`stepping_setup_body_front`−3，机身超过`x=0.40 m`；`stepping_setup_front_edge`−5，足端超过`x=0.65 m`；`stepping_setup_lateral_alignment`−2，横向超出4 cm死区；`stepping_setup_front_reach`−2，双前足超过18 cm前伸；`stepping_setup_knee_flexion`−2，四腿小于30°屈曲。前伸与膝部项使用Stairs同类机身坐标的髋—膝—足端向量计算。整理期间目标速度、路线推进、对角姿态和静止惩罚均关闭；完成后恢复。记录整理进度、完成标志、机身X、前缘越界足数、横向误差、前足前伸量和逐腿膝屈曲角到TensorBoard。

验证：执行`.venv\\Scripts\\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，39项通过，覆盖半秒整理门槛、整理期速度为零、恢复采样速度、静止惩罚关闭及位置/横向/腿形奖励阈值。未运行物理仿真、Viser、PPO调试或正式训练，旧运行不会热更新；同接口检查点可加载，但奖励状态机已经变化，必须新进程续训并重新评估实际登台姿态与通过率。

## 2026-11-09 梅花桩改为平台到平台及 train_2 续训

用户反馈旧任务训练后出现原地停滞，要求将上台交给后续跳跃任务。仅更新 `PointLeg-SteppingStones-v0`。在 `src/pointleg/mdp/stepping_stones.py` 和 `src/pointleg/config/stepping_stones.py` 中，地面随机出生改为左平台默认站姿出生，机身局部位置(0.28, 0, 0.62) m。删除入口登台奖励和五项整理惩罚及零速门槛，开始即给0.075～0.12 m/s前进命令。静止惩罚触发从大于0.08 m/s改为所有正向指令，终点停稳仍排除。接地从第一个子步起失败，出生初始接触宽限最多0.15秒且只豁免悬空终止。

出口由右平台下到地面停稳改为踩齐有效桩后直接在右平台停稳。四足接触右平台顶面时前向命令归零，按物理子步计连续稳定时间，要求四足接触、机身三维线速度低于0.05 m/s、角速度低于0.15 rad/s、倾斜小于10°，满1秒即成功并自动重置。任一条件中断清零。桩数课程和105/381/12接口保留。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，43项CPU检查通过。另用静态运动学核对默认站姿足端位置，未额外启动物理验收。按指定命令从 `2026-11-08_22-52-46_train_1/model_999.pt` 续训512环境、追加1000轮，源SHA256为 `d6373fdacc99d35bc05fcbb2a64a1f2a9764e31aa631a842eb01b7f8ee686661`。首次在沙箱内因Warp缓存重命名权限退出，正式运行使用项目内 `artifacts/warp_cache`。已观察到GPU训练第1013/1999轮并生成TensorBoard日志和 `model_1000.pt`，目录为 `logs/rsl_rl/pointleg_stepping_stones/2026-11-09_12-02-01_train_2`。尚未完成1000轮或验收通关表现。

2026-11-09用户更正起始检查点为 `stairs_12000_pretrained/model_0.pt`，随后取消重启并要求终止当前任务。已向原 `train_2` 会话发送 Ctrl+C，随后按进程命令行复查，未发现剩余 PointLeg 训练进程。更正后的训练命令未执行。

## 2026-11-09 梅花桩增强停滞惩罚

用户反馈回放中左前足碰到第一桩后退回平台停住。检查 `2026-11-09_12-22-42_train_2` 的训练参数和 TensorBoard。该运行从 `stairs_12000_pretrained/model_0.pt` 初始化，仍使用−10静止权重，迭代999的静止项为−0.2569、有效桩覆盖率为0.50、成功率为0。训练标量是并行回合汇总，不能定位单次回放的接触序列。

仅调整 `PointLeg-SteppingStones-v0` 的 `SteppingFourFootStandstill` 与配置。权重−10→−30，参考Stairs的 `forward_stall` 将宽限1.0→0.60秒、线性递增0.5→0.40秒。有正向指令且未进入终点停稳时，仅按机身相对上一次进度锚点再前进2厘米清除计时，去除至少两足接触的计时门槛。退回后重新到达旧位置、接触丢失或触桩事件均不更新进度锚点。新增 `Metrics/stepping_standstill_s`，原奖励键保留。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，38项通过，覆盖最低命令速度正常行走、接触掉线、退回后回到旧位置、真实前进、权重与时间窗以及终点停稳免罚。未启动仿真、回放或训练，新训练进程使用本配置。

## 2026-11-09 梅花桩右前髋后摆软约束

用户观察到跨桩时右前腿大腿向后收得过多。静态核对 `mjcf/pointleg.xml` 及 MuJoCo 运动学，`fr_hip_pitch_joint` 正角使右前膝相对髋向机身后方移动。仅 `PointLeg-SteppingStones-v0` 在 `src/pointleg/mdp/stepping_stones.py` 和 `src/pointleg/config/stepping_stones.py` 增加 `stepping_fr_hip_backward_limit`。行进阶段以关节模型零位为参考，超过70°的部分按10°尺度平方计罚，原始代价封顶4，权重−5。记录 `Metrics/stepping_fr_hip_pitch_deg`，终点停稳时停用。此项为软惩罚，训练后需评估越界率。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，39项通过。定向测试覆盖前摆、70°边界、80°计罚、封顶、其他腿、失败与终点停稳门控。未运行物理步进或训练。

## 2026-11-09 梅花桩四腿后摆与对角对称

用户反馈跨桩时前后对角腿姿态差异使足端拉得过开。仅调整 `PointLeg-SteppingStones-v0`。用 MuJoCo 静态运动学分别核对四髋在−80°、0°、+80°时的方向，正角均为后摆。将右前髋后摆项扩为 `stepping_hip_backward_limit`，四腿独立按70°阈值、10°尺度平方计罚，单腿原始代价封顶4再相加，权重−5。保留右前角度指标并加入其余三腿。

原 `stepping_diagonal_posture` 的−0.3惩罚替换为 `stepping_diagonal_posture_symmetry` 的+0.6奖励。FL↔RR和FR↔RL仅比较同接触相位的髋俯仰和膝角，用20°尺度的指数分数，两对角组平均。至少两足检测接触且机身沿赛道前进超过0.02 m/s才计分，速度系数线性升至前进指令时的满额。保留右平台过渡与终点停稳门控。入口为 `src/pointleg/mdp/stepping_stones.py` 和 `src/pointleg/config/stepping_stones.py`。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，39项通过。覆盖四腿独立计罚和叠加、对角两组髋膝角差、接触相位与原地停滞门控。未启动物理步进或训练，跨桩平衡仍需训练后验证。

## 2026-11-09 梅花桩髋外展改为左右镜像

用户反馈髋外展零位惩罚导致站距太窄，限制调整姿态。仅 `PointLeg-SteppingStones-v0` 删除任务的 `stepping_hip_abduction_zero`（−7.2）和继承自Flat的 `hip_abduction_excess`（−1），同时移除继承的 `static_abduction_symmetry`（−0.35）。使用现有 `joint_mirror` 实现 `stepping_hip_abduction_symmetry`（−7.2），对左前与右前、左后与右后两组有符号髋外展角之和的平方取平均，沿用直立系数，无速度指令门控。左右可同时外展或内收，等幅反号时零代价，不规定站距。保留模型关节限位、逐腿外展角日志、70°髋俯仰后摆和对角姿态奖励。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，39项通过。覆盖三项旧奖励从任务配置移除、非零镜像站姿零代价、左右同向和幅度不同受罚。未启动仿真或训练，新训练进程生效。

## 2026-11-10 梅花桩四腿大小腿防共线

用户要求四条腿的大小腿轴线较小夹角至少40°。仅对 `PointLeg-SteppingStones-v0` 在 `src/pointleg/mdp/stepping_stones.py` 和 `src/pointleg/config/stepping_stones.py` 增加 `stepping_leg_axis_angle_min`，权重−5。由髋俯仰连杆与膝连杆的局部Z轴求较小夹角，避免把髋轴至膝轴的横向结构偏移计入腿杆角度。每腿取 `max(0, (40°−夹角)/40°)^2` 后相加，有效回合内支撑和摆动均计，记录 `Metrics/stepping_{leg}_leg_axis_angle_deg`。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones -q`，40项通过，覆盖逐腿计罚、40°边界、反向共线及钝角换算。真实模型静态运动学核对，膝关节位置0°时较小夹角0～0.79°，−40°时约40°，默认站姿膝角约−64.46°时较小夹角约64.46°，−150°时较小夹角约30°。未执行物理步进或训练，新进程才使用该软惩罚。

## 2026-11-10 梅花桩出生对齐、首步跟进与新训练

用户反馈首桩附近静止、出生调整时足端前移与右前抢步，以及左前跨第一桩时右后拖住。仅调整 `PointLeg-SteppingStones-v0`，代码位于 `config/stepping_stones.py` 和 `mdp/stepping_stones.py`。采用Crawl的左右镜像外展方式，按静态运动学从0 rad改为左+0.135、右−0.135 rad，机身出生高度从台面上方0.420 m降至0.408 m，保持原髋俯仰0.550 rad与膝−1.125 rad。四足出生在±0.20 m桩列中心1 cm内，前后位置保持，仍直接进入行进阶段。Crawl整套深蹲角不适配现有70°髋后摆和40°防共线奖励。

首桩只有左前足接触顶面才计覆盖，此前不给其他桩覆盖奖励。右前在左前明确起步前抢先抬脚约2 cm计连续罚−3。左前开始后，右后真实起摆、抬升≥2 cm、从起脚点前移≥4 cm并在合法面落脚，计一次+3；未跟进0.20秒后，迟滞罚−4在0.30秒内线性升满。前进时连续4秒无新增2 cm机身前向进度按失败重置，和原静止惩罚共用按控制步去重的计时器，终点停稳豁免。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，50项通过。验证包含实际模型出生静态运动学、首桩顺序、右后原地抬落不计跟步、有效跟步一次性奖励、停滞终止与接口兼容。随后按用户指定从 `stairs_12000_pretrained/model_0.pt`（SHA256 `554314e5c5ebbe1f1591648cbd5b941f8b62a2c43e74da8705362e64142a42b3`）启动512环境、4001轮。确认GPU加载该检查点并运行至日志显示第6轮。输出目录为 `logs/rsl_rl/pointleg_stepping_stones/2026-11-10_13-20-55_new_train_1`。长训和实际通关效果待验证。

## 2026-11-10 梅花桩第二步增加首组换步门槛

用户反馈前腿迈出第二步时后腿还留在平台上。原逻辑在左前触顶第一桩后立即结算后续桩，右后跟步仅为独立奖励。仅调整 `PointLeg-SteppingStones-v0`，将右后有效跟步改为第二桩及后续桩的计分前置条件。右后落脚必须发生在左前开始首步之后，且左前实际踩着第一桩、右后合法支撑时才锁存放行。未放行时接触后续桩不给覆盖、落点或抬腿事件奖励，提前碰桩的足要重新完成有效摆腿落脚才能解除该限制。

右前抢步罚延伸至放行前，提前踩桩未重迈时也继续计罚，权重仍−3。复用四类阶段观测中的阶段1表示左前已起步而首组换步未完成，内部行进/停稳逻辑仍为2/3。新回合指标 `Episode_Metrics/stepping_first_transfer_ready` 记录是否通过门槛。入口为 `src/pointleg/mdp/stepping_stones.py` 和 `src/pointleg/config/stepping_stones.py`。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，51项通过，覆盖早于左前的右后步不算跟进、后续桩不追溯计分、短触点抖动无效、有效重迈、阶段和重置。未运行物理步进或重启训练，现有进程需重启后采用。

## 2026-11-10 梅花桩机身速度跟踪与三拍开场

用户反馈足端先跨、机身留在原处，并确认左前先踩原有第一桩，右前和左后双摆，最后右后单摆。仅调整 `PointLeg-SteppingStones-v0`。新增 `stepping_forward_velocity`（+5），以赛道实际X向机身速度跟踪每回合正向命令。跟踪高斯尺度为命令速度的0.45倍，并乘0～1的正向速度比。停稳、静止、后退和失败不计分。保留原目标方向速度与路线进度项。

用三拍开场替换之前的右后先跟步门槛及其三个奖励项。左前踩第一桩且其余三足合法支撑后进入双摆。右前和左后离地有重叠，且分别抬高≥2 cm、前移≥4 cm并合法落脚，双摆才完成。右后按相同阈值单摆落脚，其余三足必须持续支撑。双摆和右后单摆完成各计一次 `stepping_opening_step` +3，指定支撑足失撑时 `stepping_opening_support` −3。双摆完成前不发后续桩覆盖分，提前碰桩后须有效重迈。三拍完成后恢复自由行进，出口停稳也要求三拍完成。桩位与105/381/12维度保持，复用4维阶段观测区分开场各拍，增加 `Episode_Metrics/stepping_opening_phase`。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，52项通过，验证开场相位、错误支撑不放行、双摆和单摆落脚、机身速度以及检查点接口。未启动仿真或训练。

## 2026-11-10 梅花桩开场抢抬、双摆反馈及低速防滑

用户反馈右前先抬、落脚后向后滑，随后停在首桩。此前 `train_new_3` 到886轮，近100轮的开场阶段均值1.00263、覆盖率0.49855、成功率0。仅调整 `PointLeg-SteppingStones-v0`。第一拍右前在已有合法支撑后持续离地20 ms时扣一次−1，原支撑惩罚保留。第二拍增加一次性分段信号，右前和左后分别抬过台面1 cm各+0.5，共同离地持续10 ms +1，各自有效向前落脚各+1。各里程碑每回合只计一次，原双摆完成+3仍在。

双摆判定由两足同帧均高出顶面2 cm改为两足共同离地连续10 ms，峰值抬高≥2 cm、前移≥4 cm、有效落脚和指定支撑足门槛仍逐足检查。用 `stepping_foot_slip`（−2）替换继承的 `feet_slip`（−0.5），取消0.1 m/s指令门槛。只对行进中合法顶面支撑足的水平世界速度计罚，0.02 m/s内免罚，按0.12 m/s尺度平方并逐足封顶。代码在 `src/pointleg/mdp/stepping_stones.py` 和 `src/pointleg/config/stepping_stones.py`。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，54项通过。覆盖抢抬离地去抖及单次结算、两足抬脚峰值错开、双摆奖励分段和不重复、低速指令下支撑足滑动。未运行物理步进、未启动或重启训练。

## 2026-11-10 梅花桩前进速度提高

用户要求将梅花桩行进速度提高。仅调整 `PointLeg-SteppingStones-v0`，前进指令采样范围0.075～0.12→0.20～0.40 m/s，初始化中值0.0975→0.30 m/s。配置的 `twist` 范围及每回合重置均匀采样共用 `STEPPING_FORWARD_SPEED_RANGE`。行进方向、终点停稳置零逻辑及奖励权重保留。执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，54项通过。未运行物理步进、未启动或重启训练。

## 2026-11-10 梅花桩开场四足停稳及机身水平

用户反馈开场右前抢抬、机身倾斜和首桩前伸过远。仅调整 `PointLeg-SteppingStones-v0` 的 `config/stepping_stones.py` 与 `mdp/stepping_stones.py`。重置进入零速准备阶段，四足在左平台顶面且机身三维线速度小于0.05 m/s连续0.5秒才进入行进，4秒未完成则失败。准备期沿用支撑和右前抢抬惩罚，使用停稳速度和角速度惩罚，冻结开场换步和桩覆盖计分。原水平项+1换为 `stepping_body_level` +3，依据机身与重力反方向夹角按12°尺度高斯衰减。记录准备完成比例和机身倾角。

出生髋摆0.55→0.68 rad、膝角−1.125→−1.40 rad、镜像外展±0.135→±0.15 rad，机身高出台面0.408→0.364 m、距左平台后沿0.28→0.33 m。MuJoCo静态运动学验证四足均在平台内且保持桩列对齐，前足距首桩中心的前向距离约31→25 cm。阶段编码和停稳进度复用原观测字段，接口仍为105/381/12。旧同任务检查点可加载，需新训练进程学习新状态。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，56项通过。未运行物理步进或训练，实际抢抬频率及过桩效果待训练验证。

## 2026-11-10 梅花桩出生后倒调整

仅调整 `PointLeg-SteppingStones-v0` 的 `config/stepping_stones.py`。旧姿态的静态重心投影位于四足支撑范围内，四足足底离台面2.4～5.5 mm。后腿髋摆0.68→0.80 rad，右前膝−1.40→−1.385 rad，机身高出台面0.364→0.362 m。出生位置仍为平台后沿起0.33 m，四足开场停稳判定、奖励和动作/观测维度沿用既有设置。默认关节角也作为策略动作偏置，旧检查点可加载，新进程才采用新姿态。

用MuJoCo静态运动学和质量分布核对，后足后移约4.9 cm，纵向重心至后支撑边余量19.5→24.0 cm，四足初始离台面0.1～0.9 mm且在平台内，前足至首桩中心前向距离约25 cm。执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，56项通过。未运行物理步进或训练，动态后倒是否消除待验证。

## 2026-11-10 梅花桩改为对角双摆起步

仅调整 `PointLeg-SteppingStones-v0` 的开场规则，入口 `mdp/stepping_stones.py:_track_opening_gait`。静态运动学中出生重心投影距右前与左后的支撑连线约1.1 cm，投影在线段内。由左前单摆、右前和左后双摆、右后单摆，改为左前和右后双摆、左后单摆、右前单摆。首桩仍仅左前触顶计分，首拍另要求两腿有效向前落脚且左前此时仍支撑首桩。右前提前离地惩罚延续到第二拍结束。分段奖励移至首拍，各拍完成均发放+3。第二桩及后续计分从第三拍完成的落脚开始，提前触桩须重新摆动。开场后继续自由选步，观测和动作维度仍为105/381/12，已有检查点可加载，但需新训练进程学习新阶段含义。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer -q`，56项通过，覆盖新三拍顺序、指定支撑失效、右前抢抬和提前触桩不计分。未运行物理步进或启动训练，对角起步的动态稳定性待验证。

## 2026-11-11 梅花桩整理阶段固定出生关节目标

用户反馈机器人抬起一足停在左平台直至重置。读取 `2026-11-11_07-44-16_new_train_5_1` TensorBoard，最后100轮为5243～5342，平均回合200控制步即4.0秒，`stepping_setup_ready` 与首桩覆盖率均为0，记录到整理超时。行进停滞和60秒总超时均为0。此前零速度指令仍由策略直接控制关节。

仅为 `PointLeg-SteppingStones-v0` 配置 `SteppingStonesJointPositionAction`。整理阶段向现有关节动作处理输入零增量，按出生关节角控制，并阻止整理阶段的策略输出进入动作延迟缓存。四足在左平台顶面且机身线速度小于0.05 m/s连续0.5秒后，由策略通过现有动作延迟及低通滤波接管。保留4秒整理超时，三拍及终点判定沿用原规则。观测和动作接口为105/381/12，旧检查点可由新进程加载。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，75项通过。新增定向检查覆盖并行环境独立放行、整理动作不入延迟缓存及部分重置。未运行物理步进或训练，站稳通过率待实际回放验证。

同日用户确认保留上述整理与策略交接方案。恢复中途撤下的动作配置和回归测试，交接继续使用当时的关节观测，旧训练日志仅代表当次训练。再次执行同一组定向测试，75项通过。动态交接待新进程回放验证。

## 2026-11-11 梅花桩抑制右前腿持续失撑

用户反馈站稳接管后右前抬起并维持悬空。仅在 `PointLeg-SteppingStones-v0` 中增加 `stepping_opening_right_support`，站稳后的前两拍生效，权重−8，连续失去合法顶面支撑20 ms后开始计罚，接下来100 ms线性升到满额。复用右前抢抬计时，按合法顶面接触清零，原一次性扣分沿用。进入第三拍右前单摆后清零，记录 `Metrics/stepping_opening_right_unsupported_s`。已有策略可加载，需要新训练进程采用新奖励。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，76项通过。覆盖计罚阈值、侧面接触、恢复合法支撑和第三拍释放。未执行物理步进或训练。最新可读日志 `2026-11-11_07-44-16_new_train_5_1` 最后100轮的整理完成率仍为0，不能用作站稳后回放的验证。

## 2026-11-11 梅花桩开场后髋过高惩罚

用户反馈开场跨首桩时后髋偏高。仅为 `PointLeg-SteppingStones-v0` 增加 `stepping_opening_rear_hip_height`。使用四个髋外展轴的机身安装点世界坐标，两后髋平均高度减去两前髋平均高度，超过3 cm后按超出量除以5 cm的平方计罚，原始代价封顶4，权重−4。只在策略接管后的开场三拍生效，记录 `Metrics/stepping_opening_rear_minus_front_hip_cm`。旧检查点可以加载，新进程续训后采用新奖励。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，77项通过，覆盖高度方向、余量、封顶、阶段门控和出生姿态。未运行物理步进或训练，姿态改善待回放验证。

## 2026-11-11 梅花桩允许换腿时短暂腾空

用户观察到首桩后的换腿有短时四足腾空。仅为 `PointLeg-SteppingStones-v0` 调整失撑判定。原站稳后连续无接触超过20 ms判失败，改为连续达到120 ms判失败，使用整数物理子步计时。接触仍每5 ms检测，任一足重新接触即清零连续计时。`stepping_flight` 权重−1→−4，继续按每个控制周期内四足无接触的子步占比计罚。整理期原失撑阈值、右平台过渡豁免及地面接触失败沿用原规则。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，78项通过。覆盖首桩后40 ms腾空、恢复接触清零、120 ms终止边界、出口过渡豁免和地面接触失败。未运行物理步进或训练，实际换腿通过率待新进程回放验证。

## 2026-11-11 提高梅花桩行进停滞惩罚

用户反馈首步向右横移后静止。仅将 `PointLeg-SteppingStones-v0` 的 `stepping_four_foot_standstill` 权重由−30调至−45，使用原行进阶段停滞计时和4秒终止。机身沿赛道X向新增2 cm进度时清零，横移期间仍累计。停滞0.60秒开始计罚，1秒达到满额。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，78项通过，定向测试验证横移10 cm后停滞仍持续累计及新权重。未运行物理步进或训练，策略效果待续训回放。

## 2026-11-11 梅花桩延长桩顶支撑

用户反馈桩顶支撑偏短，要求机身再向前推进一些再换腿。仅为 `PointLeg-SteppingStones-v0` 增加 `stepping_stone_early_takeoff`，权重−3，单足单次原始代价0～1。逐足记录在同一桩顶的接触时长和机身赛道X向净位移，前足目标0.18秒和4 cm，后足0.24秒和5 cm。两项相对缺口取较大值，离开时一次结算。离开原桩满15 ms或检测到转移到另一根桩确认本次离地。短暂丢失接触后回到同一桩保留原支撑记录，丢失期间不计时，离地后机身运动不算入支撑推进。观测与动作维度为105/381/12，原检查点可在新进程续训。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，80项通过，覆盖前后足目标差异、平台足豁免、短暂接触丢失、离地后位移和事件去重。未执行物理步进或训练，实际支撑时长待回放验证。

## 2026-11-11 梅花桩出生前足收回到髋摆轴下方

用户反馈出生前足偏前。仅调整 `PointLeg-SteppingStones-v0` 的出生前腿关节角。左前髋摆0.68→0.751 rad、膝−1.40→−1.404 rad，右前髋摆0.68→0.741 rad、膝−1.385→−1.388 rad。以各自髋摆轴的X坐标为前后目标，沿用0.33 m出生X和0.362 m台面相对高度。

MuJoCo静态运动学确认左前足后收2.8 cm，右前足后收2.4 cm，前后方向对齐误差小于0.2 mm，前足距首桩中心约27.8 cm，四足离台面均不足1 mm。执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，80项通过。出生角也用作动作偏置，新进程采用。未执行物理步进或训练，实际首步效果待回放验证。

## 2026-11-11 梅花桩机身朝向与赛道对齐

用户要求机身轴线对准梅花桩的行进方向，减少前进中的横移。仅为 `PointLeg-SteppingStones-v0` 增加 `stepping_route_alignment`，权重+2。偏航以赛道正X轴为基准，横向速度取世界Y分量。两者的高斯尺度分别为12°和0.08 m/s，得分再乘实际正向速度与指令速度之比并将该比值限制在0～1。该项在有效行进期间生效。

执行 `.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，81项通过，覆盖双向偏航和横移、角度跨周、静止、倒退和阶段门控。旧检查点可从新进程续训。未执行物理步进或训练，实际横移幅度待回放验证。


## 2026-11-11 梅花桩开场退化诊断与续训

读取 `2026-11-11_14-03-54_new_train_5_4` TensorBoard 第8700～13250轮，训练成功率全程0，任一迭代的覆盖均值最高0.50（最简单两桩赛道），未见超过第一拍的回合终止均值。第9000～9099轮开场完成拍数均值0.931，关节越限每个重置批次均值0.022。第13151～13250轮开场完成拍数降至0.209，覆盖率0.466，关节越限均值2.068，成为主要终止项。整理完成率均为1。指标来自训练汇总，未回放定位具体越限关节。选 `model_9100.pt` 保留退化前的首拍能力，源SHA256 `eb3304c53d2440e9c5043ac22e3aa26be9ef7d18548ce279416ce28225e82d58`。

仅调整 `PointLeg-SteppingStones-v0` 奖励。`src/pointleg/mdp/stepping_stones.py` 将三拍一次性奖励从+3/+3/+3改为+3/+8/+12，合法触地判据和首桩覆盖奖沿用。`src/pointleg/config/stepping_stones.py` 增加行进期 `stepping_joint_limit_approach`，权重−12，使用实测和40 ms速度外推得到的最小硬限位余量，低于0.10 rad时按缺口平方逐步扣分，原始代价封顶1，取最危险关节。硬限位终止余量为0.02 rad，Actor/Critic/动作接口为105/381/12。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，82项通过。检查点CPU只读加载确认迭代、网络维度和优化器字段。使用 `.venv\Scripts\pointleg.exe train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.logger tensorboard --agent.upload-model False --agent.resume True --agent.max-iterations 2000 --agent.save-interval 100 --agent.load-run 2026-11-11_14-03-54_new_train_5_4 --agent.load-checkpoint model_9100.pt --agent.run-name new_train_5_4_recovery_9100` 在CUDA 0启动。确认正确载入、保存新的model_9100.pt并跑至至少9104，新惩罚项已记录。独立目录 `logs/rsl_rl/pointleg_stepping_stones/2026-11-11_17-40-41_new_train_5_4_recovery_9100`，训练PID20380，旧训练继续运行。计划末次更新编号11099，当前未完成2000轮，策略效果未验收。

## 2026-11-11 梅花桩取消起步前整理

用户要求从左平台出生后直接前进。仅为 `PointLeg-SteppingStones-v0` 移除出生整理状态、0.5秒站稳门槛、关节目标屏蔽和4秒整理超时。重置时进入行进状态和开场第一拍，并立即下发0.20～0.40 m/s前进指令。出生关节角继续用作策略动作偏置，应用基础动作延迟和低通滤波。终点连续1秒停稳仍是成功条件，原停稳进度观测用于终点，Actor/Critic/动作维度为105/381/12。普通续训或回放从当前任务注册配置读取环境，已有检查点可加载，新进程采用本次改动。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，81项通过。覆盖出生立即发指令、首个控制步接收动作、终点指令归零及终点停稳进度。未执行物理步进或启动训练。

## 2026-11-11 梅花桩首拍抑制外扩与右后跟步

用户反馈直接起步时四足外扩并腾空，左前前迈时右后滞后。静态运动学测得出生足端与对应桩列横向误差最大6.9 mm，四足底距台面最大0.6 mm。读取 `2026-11-11_19-36-09_train_1` 第900～999轮，覆盖均值0.498，开场完成拍数均值0，成功率0。数据只反映回合统计。

仅为 `PointLeg-SteppingStones-v0` 增加首拍 `stepping_opening_foot_row` 和 `stepping_opening_right_rear_advance`。前者使用几何共用桩列位置±0.20 m，容差2.5 cm、超差5 cm尺度平方、逐足封顶1后平均，权重−6。后者要求右后有效抬脚且右前与左后支撑，并且左前也在有效摆动或已接触首桩。用右后相对起脚位置的前送增量刷新本回合历史最佳幅度，4 cm满额，权重+2。首拍仍需左右对角共同离地及有效落脚才判完成，起步沿用直接下发指令，旧检查点可用新进程续训。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，83项通过。覆盖对称外扩、局部赛道坐标、有效右后前送和重复动作无额外奖励。未运行物理步进或训练，实际起步效果待续训回放。

## 2026-11-11 梅花桩首桩按有支撑的摆动计分

用户从回放发现四腿外扩伴随跳上首桩。原先左前足触到首桩顶面即可记覆盖。仅在 `PointLeg-SteppingStones-v0` 将首桩计分改为左前足最近合法支撑点位于左平台前后范围内，摆动期间右前和左后保持合法支撑，离地至少20 ms后落到首桩顶面。首桩覆盖、落点和抬腿奖励共用判定，首拍推进也要求首桩已有覆盖或当次有效落脚。连续腾空120 ms终止与地面接触终止沿用原规则。

MuJoCo静态逆运动学检查在机身前移4 cm、其余三足保持出生落点时可将左前落到首桩中心，左前大小腿轴线较小夹角约55°。该检查是单个静态位形验证。执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，84项通过，覆盖短跳触桩、在桩上重踩、返回左平台再迈和正常双摆。未运行物理步进或训练，动态跨桩效果待续训回放。

## 2026-11-11 梅花桩首拍奖励改为两腿共同前送

用户反馈左前和右后先微动后收回，随后仅右后缓慢前移。读取 `2026-11-11_21-11-42_train_2` 最后100轮，右后独立前送奖平均约1.999分，首拍完成拍数均值为0，两桩平均覆盖率约0.042%。将 `PointLeg-SteppingStones-v0` 的独立右后前送奖+2替换为 `stepping_opening_pair_forward_progress`，满额+4。已有合法双摆重叠后，左前从起脚点向首桩中心前送超过3 cm开始计分，右后前送4 cm为满额目标，取两腿较小进度，只对本回合最佳进度的增量发奖。有效落脚可以先后发生，首桩计分及首拍完成判据沿用当前规则。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，84项通过，覆盖单腿前送、轻微摆动、重复往返及右后先落脚。旧检查点可从新进程续训。未执行物理步进或训练，动作改善待回放验证。

## 2026-11-11 梅花桩前进指令提至0.30～0.40 m/s

用户反馈机身前进跟不上足端周期。仅在 `PointLeg-SteppingStones-v0` 将 `STEPPING_FORWARD_SPEED_RANGE` 从0.20～0.40调整为0.30～0.40 m/s。训练配置与重置均匀采样共用该常量，初始缓冲中值0.30→0.35 m/s。新进程采用本次修改，旧检查点可续训。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，84项通过，覆盖配置范围和重置采样。未执行物理步进或训练。

## 2026-11-11 梅花桩右后步幅和第二拍左后单摆反馈

用户反馈右后步幅短，并确认停滞出现在首拍结束后右前提前摆动。核对腿序和阶段掩码，第二拍本来指定左后，右前提前触第二桩不能推进阶段。`2026-11-11_22-37-20_new_train_3_2` 最后100轮平均完成0.97拍、两桩覆盖率47.2%、成功率0。

仅调整 `PointLeg-SteppingStones-v0`。首拍右后有效步幅由4 cm调至6 cm，过程奖满额目标由4 cm调至8 cm。保留已有合法双摆且左前踩稳首桩的进度，允许右后补一步。第二拍新增 `stepping_opening_left_rear_forward_progress`，在其余三足合法支撑时按左后有效摆动的前移进度发放，4 cm累计满额+3。第二拍右前失撑从首个子步开始计罚，60 ms满额−12。现有阶段观测编码和动作维度继续可用于同任务检查点。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，86项通过。覆盖右后短步后补迈、第二拍右前抢步不推进阶段、左后过程计分及正常三拍。未运行物理步进或训练，实际动作待续训回放。

## 2026-11-11 梅花桩整机停滞惩罚翻倍

用户反馈策略频繁静止。仅将 `PointLeg-SteppingStones-v0` 的 `stepping_four_foot_standstill` 权重由−45调至−90。整机沿赛道X向每新增2 cm进度时清零，停滞0.60秒开始扣分、1秒满额，行进时连续4秒停滞触发终止。首拍和第二拍共用行进停滞计时。新训练进程采用该权重。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，86项通过，覆盖首拍切换到第二拍时的累计停滞和新权重。未运行物理步进或训练，静止是否减少待续训回放。

## 2026-11-12 梅花桩机身随步推进奖励

用户反馈足端迈8 cm时机身只前移约1 cm。仅为 `PointLeg-SteppingStones-v0` 增加 `stepping_body_step_tracking`，权重+3。采用赛道X向位移，目标等于本次有效摆动足前移距离之和的一半，单摆等于该足步幅一半，双摆等于两足步幅平均。双摆按离地时间重叠归组，机身位移从首足离地量到末足落脚。合法落脚且各足前移至少2 cm才结算，开场继续校验规定腿序和首桩接触。得分按位移误差和目标步幅缩放，零正向机身位移时得分为零。观测和动作维度为105/381/12，旧检查点可在新进程续训。

执行 `.venv\Scripts\python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q`，87项通过。覆盖单足8 cm对应机身4 cm、两足6 cm及10 cm对应机身8 cm、先后落脚、机身只移动1 cm及开场错误摆动腿。未执行物理步进或训练，实际机身跟进幅度待续训回放。

## 2026-11-12 梅花桩训练重建为自由迈步基线

2026-11-12用户指定 `sim2sim/policy/optical_policy/model_1999.pt` 迁移到梅花桩任务。执行 `.venv/Scripts/python.exe -B -m pointleg.run transfer PointLeg-SteppingStones-v0 --checkpoint-file sim2sim/policy/optical_policy/model_1999.pt --output-run optical_1999_pretrained`，输出 `logs/rsl_rl/pointleg_stepping_stones/optical_1999_pretrained/model_0.pt`。校验原有网络参数逐项相同，45/325扩展为105/381的新增列为零，优化器状态清空、迭代为0。源SHA256 `2e35db3a3fa5369fdde4f8242fbb88e7e1f92b68b0418f97f4edb2b76a5e02f6`，输出SHA256 `9a52d25040b85eb01b306f7855ef5232283e048083f3c5511e0fa6c2def991e2`。未执行仿真或训练。

针对起步静止、异常高抬腿和多项反馈相互牵制，将 `PointLeg-SteppingStones-v0` 由48项精简至10项。覆盖 +30→+20，成功 +100→+40，失败 −150→−12，前向速度 +5→+2.5，赛道对齐 +2→+0.5，机身水平 +3→+0.5 且按前向速度门控，腾空 −4→−2，支撑足打滑 −2→−0.5，关节软限位 −10→−5，动作平滑保留 −0.01。移除前三拍、抬脚、跟步、对称、换脚时机与静止惩罚等38项，保留4秒停滞终止。取消120 ms全足腾空终止。

代码入口为 `src/pointleg/config/stepping_stones.py` 和 `src/pointleg/mdp/stepping_stones.py`。子步按接触位置和法向判定顶面，任意足连续接触同一桩20 ms后记覆盖。全部覆盖且右平台四足稳定1秒算成功。课程2–8桩、出生姿态、速度0.30–0.40 m/s与60秒时限沿用原值。观测和动作维度105/381/12，阶段从三拍编码改为行进和右台站稳。TensorBoard加入 `stepping_forward_speed_mps` 和 `stepping_peak_foot_clearance_m`，保留覆盖、成功及终止原因。旧检查点可按结构加载，已运行的进程需重启才使用本版逻辑。

执行 `.venv/Scripts/python.exe -B -m unittest pointleg.tests.test_stepping_stones pointleg.tests.test_stepping_stones_transfer pointleg.tests.test_pointleg -q -b`，54项CPU测试通过。转移使用测试生成的临时检查点，未使用或修改现有训练模型。未执行物理步进、回放或训练，实际步态和收敛需新训练评估。

## 2026-11-18 梅花桩课程与固定腿序重建

用户要求最终级采用比赛场景，最低三个桩，并重建固定步态与奖励。用户确认第二步为左后加右前抬腿，循环中误写的后腿为右后。几何参照本仓库 `sim2sim/terrain/scene_terrain.xml`，没有从外部比赛文件推断尺寸。

默认地形由 7 行、两种 15/20 cm 尺寸、2～8 桩及 5～10 cm 投影间隙，改为 4 行单一比赛尺寸，3/4/5/6 桩、20 cm 方桩、高 20 cm、固定 10 cm 投影间隙、30 cm 前进投影中心距、60 cm 同侧中心距。最终级起台 x=0～0.8 m、六桩中心 x=1.0～2.5 m、出口台 x=2.7～3.5 m，逐几何参数与 XML 一致。旧 `--competition-six-curriculum` 预设转为相同桩数课程，`--competition-six` 固定最终级。默认解锁窗口 100→256，当前最高级成功率 ≥80% 解锁，失败且率 <40% 可降自身等级。

`mdp/fixed_stone_gait.py` 实现训练与原生回放共用的批量 Torch 顺序状态。前六步依次为 FL→1+RR 台上短步、FR→2+RL 台上短步、RR 台上短步、FL→3、RR→2、RL→1+FR→4。然后按 FL、RR、RL+FR 三拍沿同侧前进，超出本级末桩的目标改为出口，全部四足到出口后结束步序。短步初值 6/8/8 cm，实际足底中心距起台边沿至少 5.5 cm。资格要求摆动离面 ≥40 ms、净空峰值 ≥2 cm、双摆连续同时净空 ≥2 cm 至少 20 ms、指定面 3 cm 落点容差与 80 ms 连续有效承重。其余足保持原面且漂移 ≤3 cm，违规清除摆动资格。有效承重沿用足端 ≥8 N、同腿小腿向上受力 <8 N。

当前奖励撤下独立前迈、右后跟步/等待、左前支撑/违规/前伸、抢交接、起步推进/抢迈、正向速度、机身水平/高度及路线正收益。加入接近 +2/步、完成 +4/步、违规 −2/足/次、失撑 −8/s、错落 −3/s、起台台沿 −5/s、净空不足 −1.5/s、超过 2 s 后渐增至 −2/s。覆盖整条合计 +20→+12，中心 +2→+0.5，失败 −12→−25，膝最小屈角 −8→−2，超高 −5→−2。成功 +40、出口质量合计最多 +4 及其他几何/姿态代价见奖励文档当前表格。单步接近只支付最好值增量，完成只结算一次。六桩有 14 步，完成上限 56，接近上限 28，覆盖上限 12。停滞终止 4→8 s，60 s 回合，出口安定 0.70 s、最多整理 5 s。

Actor/Critic/动作由 105/381/12 改为 123/399/12，原观测末尾追加 18 维足目标、摆动掩码、步序和耗时。`transfer.py` 接受 45/325、105/381、123/399 来源，仅扩展首层输入，新列置零，其他参数严格加载，使用新优化器、轮次和课程。新检查点带 `pointleg_fixed_stone_gait=1`。旧 105 维模型先迁移再续训，原生 Sim2Sim 和 ONNX 导出仍自动选择 105/123 宽度和接触槽语义。未对用户真实检查点执行迁移，既有权重文件未作为本轮测试输入。

代码入口包括 `src/pointleg/config/stepping_stones.py`、`terrains/stepping_stones.py`、`mdp/stepping_stones.py`、`mdp/fixed_stone_gait.py`、`transfer.py`、`rsl_rl/runners/stepping_stones_runner.py`、`sim2sim/stepping_stones_replay.py`、`sim2sim/pointleg_sim2sim.py`、`sim2sim/export_onnx.py`。`tools/check_stepping_stones.py` 同步四级和 123/399，但未运行该物理工具。文档为 `docs/STEPPING_STONES_REWARDS.md` 当前日期段落，历史段落保留并标记。

实际执行 136 项检查，15.571 s 全部通过。覆盖全部步序与出口、双摆同步、擦地拒绝、弱承重、小腿借力、实际台沿余量、部分重置、非有限输入、事件奖励经理积分、失败优先、观测追加列、旧网络扩展和新旧 ONNX 接口。静态原生检查只编译场景和执行前向数据计算，没有调用 `mj_step`。首次系统 Temp 中的 ONNX 原子替换遭沙箱权限错误，将 Python 临时目录指向工作区后重跑通过，导出发布逻辑没有因此调整。测试输出保留在本轮工具结果中，无独立报告文件。复现选择如下。

```powershell
$checks = @(
    'pointleg.tests.test_fixed_stone_gait',
    'pointleg.tests.test_stepping_stones',
    'pointleg.tests.test_stepping_stones_transfer',
    'pointleg.tests.test_pointleg',
    'sim2sim.tests.test_pointleg_export',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_fixed_gait_policy_selects_shared_replay_targets_and_observations',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_stepping_support_progress_is_selected_by_checkpoint',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_switch_to_stepping_stones_preserves_state_and_changes_observations',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_stepping_entry_spawn_and_policy_selection_have_the_same_start_in_either_order',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_stepping_switch_requires_course_but_preserves_flat_task',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_stepping_contact_tracks_real_stone_top_and_resets',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_right_front_trace_records_rearward_offset_once_at_confirmed_landing',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_replay_blocks_left_front_stone_coverage_and_reports_the_violation',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_stepping_replay_keeps_policy_active_until_stable_exit_or_timeout',
    'sim2sim.tests.test_pointleg_sim2sim.PointLegSim2SimTests.test_stepping_replay_skipped_left_front_target_does_not_count_coverage'
)
.venv/Scripts/python.exe -c "import tempfile,unittest; tempfile.tempdir=r'E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab'; unittest.main(module=None)" @checks -q -b
```

15 个相关 Python 文件 AST 解析通过，没有行尾空白。本轮没有启动物理步进、Viser、PPO 或正式训练，没有证明实际步幅、平衡或最终级成功率。当前进程是否采用新代码未经检查，后续使用新进程和当前接口模型。以上入口短步、时间阈值和奖励权重为本轮设计初值，后续动态训练效果待验证。

## 2026-11-18 修复只抬右后后停住的起步门槛

用户反馈新固定步态只抬一次右后腿后保持不动。读取 `logs/rsl_rl/pointleg_stepping_stones/2026-11-18_01-26-20_fixed_gait_30000` 的事件日志至第 8924 轮，`stepping_gait_completed_steps` 和 `stepping_gait_phase` 全程为 0。最近 200 条记录平均步序计时 0.000143 s、覆盖率 0、平均回合长度 581.283 控制步，步态进度与等待奖励接近 0。末段 mean_reward 约 −26.892078。源码中 `ready` 必须等待四足在入口台同时有效承重 80 ms，未激活时目标掩码、计时和等待代价全为 0。这一门槛为单腿悬空静止留下规避步态要求的路径。

纯 Torch 输入保持右后足离面、高出台面 5 cm，其他三足在入口台承重，连续更新 200×5 ms。修复前结果为 ready=False、age=0、delay=0、摆动掩码=[0,0,0,0]。修复后为 ready=True、age=0.9999992 s、delay=0.2762497 s、掩码=[1,0,0,1]、完整步数 0。该输入没有进行物理仿真。

`FixedStoneGait.initialize` 改为使用首帧有限足位直接建立首步目标。训练 `fixed_gait_observation` 与原生 `refresh_observations` 在首次策略输入时初始化，`update` 也可直接初始化。首步计时随物理子步推进，初始化与实际整步成功各自按对应条件执行。规定支撑面初始化为入口台，后续完整腿序继续由指定面、足端 ≥8 N、同腿小腿 <8 N、40 ms 离面、2 cm 峰值净空、双足 20 ms 重叠、3 cm 落点误差和 80 ms 稳定落脚检查推进。

步态进度 +2 改为逐足累积 25% 抬脚高度和 75% 合格摆动后的归一化落点接近量。抬脚在 2 cm 达到该部分满额，历史最好值增量控制总预算。进度反馈可在另一只规定摆动足尚未抬起时获得，整步完成 +4 继续要求双摆同步。原等待项 −2/s 增加缺失摆动代价，0.30～0.60 s 按未取得摆动资格的规定腿比例线性升到该比例满额，未同步双摆至少按一半计费，与原整步 2～4 s 超时取较大值，封顶 −2/s。只完成右后、左前未动时 0.60 s 后至少 −1/s。增加 `stepping_gait_ready`、`stepping_gait_lifted_fraction`、`stepping_gait_pair_ready`，用于区分未激活、缺少抬脚和缺少双摆重叠。

修改入口为 `src/pointleg/mdp/fixed_stone_gait.py`、`mdp/stepping_stones.py`、`config/stepping_stones.py`、`sim2sim/stepping_stones_replay.py` 及两份针对性测试。奖励文档当前规则已同步。Actor/Critic/动作为 123/399/12，现有迁移种子可直接加载。本轮只读取真实权重元数据，种子 `flat_8700_fixed_gait_pretrained/model_0.pt` 为 iter=0、SHA256 `a0f481a6e0bc213c427c0f62e56a712d57ef10c92172b0f28e156ab8f41d72a9`。退化运行 `model_8900.pt` 为 iter=8900、SHA256 `bd1aefaf2441d68af67a3b4035bb34e7838cfc8c41fb48432af3a94fbd8ded03`。没有修改或迁移这两份文件。

复用上一条调整的 `$checks` 组合和 Python 临时目录设置，执行 `pointleg.tests.test_fixed_stone_gait`、`test_stepping_stones`、`test_stepping_stones_transfer`、`test_pointleg`、`sim2sim.tests.test_pointleg_export` 与十个明确无物理步进的原生用例，142 项在 18.940 s 全部通过。增加六项回归，覆盖单只后足始终不承重时的激活与计时、独立抬脚反馈封顶、只迈后足的持续等待代价、失败/非有限启动拒绝、无接触载荷的首次观测、诊断指标。原生用例同步检查首次观测和重置后的状态。6 个修改 Python 文件 AST 解析通过，没有行尾空白。首次组合运行中的原生测试仍假设 reset 后 ready=False，已将断言更新为首次 observe 后已初始化，并检查 age/完成步数为 0。

建议结束旧进程后，从已有迁移种子以 512 环境重训 30000 轮，运行名 `fixed_gait_start_fix`，完整命令写入 `memory/04_training_workflow.md`。本轮没有停止或启动训练进程，没有执行物理步进、Viser 或 PPO，没有验证实际策略已恢复迈步。进程需重启后采用修正逻辑，后续重点读取新增激活与同步指标及完整步数。

## 2026-11-18 固定步态二次漏洞审查

用户要求继续审查类似漏洞，范围为固定步态初始化、摆动资格、步序推进、奖励结算、重置、同桩支撑和课程续训。检查训练/原生共享状态、接触收集器、成功与失败优先级、事件清零以及 runner 保存恢复。没有启动物理仿真或训练。

确认四种摆动资格复用错误，并先添加回归验证旧代码，四项全部失败，两个合成环境均从 phase=0 错误推进到 1。情形分别为完整摆动后落回原位，再保持接触移到目标；20 ms 高抬后着地，下一次低于 2 cm 的擦地摆动持续 40 ms；一次合格双摆后再次单足起落；应支撑足失撑期间悬空 40 ms，恢复后只悬空 20 ms 即落地。后续修复分别清除错误落脚资格、在新飞行起点重置峰值和本足资格、在任何规定足重新起飞时清除双摆确认、只累计支撑与足间距有效期间的连续摆动时长。落点、承重、峰值和时长现在属于同次有效摆动。`swing_attempted` 记录本步曾获得合格摆动的事实，专门保留持续错误落脚代价的原有语义，不能用于完成判定。最好值进度与事件预算仍各自有界。

共享桩还存在所有足目标重合的问题。RR 到第 2 桩时 FR 尚支撑在第 2 桩，后续 RL/RR 也会共用奇数桩。旧目标都是桩心，MJCF 的机器人自身碰撞筛选和原仅前后腿距离代价不能阻止全部重叠计分情形。现在在目标已有保持足支撑且桩心间距不足 5 cm 时，从桩心与 x/y 各 ±4 cm 的四个候选中，选择距既有保持足最远的位置。实际所有足球心间距至少 5 cm 才可获得有效进度、摆动资格和合格落脚。5 cm 对应当前球足 2.5 cm 半径的直径。足与异腿足/小腿的距离代价从前后腿组合扩展到所有异腿组合，权重 −4 和 5 cm 阈值沿用。地形尺寸与用户指定腿序保持对应，落点在同根桩面上避让。

课程解锁等级和统计窗口此前没有写入检查点，只有公共步数被基础 runner 保存。`SteppingStoneCurriculum` 现在通过 `state_dict/load_state_dict` 保存和校验等级、窗口，实例挂在环境上供专用 runner 访问。`SteppingStonesOnPolicyRunner.save` 写入 `stepping_stone_curriculum`，完整恢复读取这部分状态，Actor-only 加载保留当前课程、轮次及公共步数。此前没有课程元数据的检查点使用当前默认课程，已有历史窗口不能从该文件恢复。本次未改变真实检查点文件。

代码入口为 `src/pointleg/mdp/fixed_stone_gait.py`、`src/pointleg/mdp/stepping_stones.py`、`src/pointleg/rsl_rl/runners/stepping_stones_runner.py`，以及两份定向测试。当前模型接口 123/399/12，可继续加载现有对应模型。共享状态同步作用于 123 维原生回放。新进程采用修正后才会保存课程元数据，未核实用户当前进程是否已重启。

复用前条 `$checks` 组合和工作区 Python 临时目录设置，运行固定步态、梅花桩、迁移、普通点足、ONNX 模块与十项明确无物理步进的原生用例，151 项在 21.490 s 全部通过。比前轮新增九项，分别覆盖上述四种资格复用、共享桩落点避让、足重叠拒绝、两后足距离代价、完整续训恢复课程、Actor-only 保留课程和计数。已有合法完整腿序、3/6 桩完整通过、单足启动、事件积分、失败优先、部分重置、旧接口与 ONNX 测试通过。五个修改 Python 文件 AST 解析和行尾空白检查通过。本轮测试使用临时合成模型，未对用户真实模型进行恢复训练，未生成独立报告文件。

151 项检查证明这些配置与状态路径符合回归预期，实际动作、训练收敛和最终级通行率仍待后续训练验收。项目记忆与奖励文档已同步当前规则，旧记录保留在后文。

## 2026-11-18 修复向后摆腿的奖励偏差

用户提供两张回放截图，反馈开始向后迈腿。核对最新运行 `2026-11-18_12-04-36_train_1`，其 `agent.yaml` 为从 `flat_8700_fixed_gait_pretrained/model_0.pt` 启动 1000 次训练，保存至 `model_999.pt`。最后 100 条 TensorBoard 记录平均赛道前向速度 −0.230413 m/s，完整步数和覆盖均为 0，激活均值 1，双摆确认均值 0.972994，合格抬脚比例 0.986435，回合 31.7939 控制步、约 0.635878 s。mean_reward 从前 100 条均值 −54.06152 改善到末 100 条 −27.849496，同时行为退化为快速向后结束回合。

使用 MJCF 静态前向运动学核对当前出生姿态，没有推进物理时间。四足 X 坐标约为 FL 0.722067、FR 0.722289、RL 0.219165、RR 0.218624 m。首步 LF 目标向前 0.277933 m，RR 目标向前 0.060 m。源码目标方向正确，原 25% 抬脚进度独立于移动方向，且未对机身或规定摆动足后退单独计罚。纯 Torch 双足向后 5 cm、抬高 5 cm、保持 0.6 s，旧规则可取得进度奖励 +0.5，等待积分为 0。固定失败 −25 还允许短回合避开后续持续代价。这些机制与实际日志中的快速后退失败行为一致。

`FixedStoneGait` 的抬脚部分现在乘以朝起点到目标方向的投影位移比例，0～2 cm 线性增加到满额，原地或反向抬脚的这部分收益为 0。新增 `backward` 子步代价，配置 `stepping_gait_backward` 权重 −4/s。仅规定摆动足计入，退到起点与目标中较后者之后超过 1 cm 才开始，5 cm 尺度平方、单足封顶后平均。朝后方目标进行必要纠偏和支撑足均豁免。整步进度 +2 与完成 +4 的预算继续使用当前配置。同一后摆张量输入修正后进度收益 0，后退项约 −1.535999 分。

新增 `stepping_backward_velocity` −3/s，使用赛道 X 向机身速度，负向死区 0.02 m/s、0.10 m/s 尺度平方封顶，行进阶段生效，出口整理豁免。新增 `stepping_early_failure` −4，按失败时距起步 8 s 窗口末尾的剩余秒数一次性计费，较短回合以其时限截短窗口。0.60 s 失败额外 −29.6，加原失败 −25 合计 −54.6；8 s 后额外项为 0，成功豁免。8 s 与当前无进度终止时间对应，目的是减弱过早失败规避持续代价的激励，实际策略是否因此改善需训练验证。

修改入口为 `src/pointleg/mdp/fixed_stone_gait.py`、`mdp/stepping_stones.py`、`config/stepping_stones.py` 及两份针对性测试。历史 handoff 配置夹具显式移除本轮新增方向/提前失败项，继续验证其原接口。当前 Actor/Critic/动作为 123/399/12，可加载已有迁移种子。没有改变任何真实权重文件，未执行迁移或启动训练。

复用前条 `$checks` 组合及工作区临时目录设置，159 项在 23.299 s 全部通过，比前轮新增八项。覆盖后摆无收益、竖直抬脚无收益、朝后方目标纠偏可用、支撑足豁免、机身后退与出口阶段、提前失败事件计费、成功豁免、控制周期积分一致性。原固定腿序、同桩避让、资格防复用、课程续训、旧接口、ONNX 和静态原生用例通过。五个修改 Python 文件 AST 与行尾空白检查通过。本轮没有执行物理步进、Viser、PPO 或训练，未验证新策略动作及通过率。

真实 `train_1/model_999.pt` 已核对 iter=999、输入 123/399、课程 cap=0，SHA256 为 `b152d3ff21bbe1fde0cb458505091a9ce92d1a5b9e41d4af3d65b7317a146c32`。原迁移种子 `flat_8700_fixed_gait_pretrained/model_0.pt` SHA256 仍为 `a0f481a6e0bc213c427c0f62e56a712d57ef10c92172b0f28e156ab8f41d72a9`。下一轮建议从该种子启动新进程采用当前规则，用户的实际运行轮数以其新命令为准。本轮未停止或启动训练进程。

## 2026-11-18 总账复查及用户授权 2000 轮训练启动

用户继续询问奖励漏洞，复查确认两处。其一为8s之后额外早退项归零，构造8/9s失败的额外项均为0，而等待1s会继续承担2分满额等待代价，导致后半段仍存在提前结束省罚的路径。窗口现由8扩到60s，按实际回合时限截短，每剩余秒额外−4。0.6s、8s、9s失败总分分别−262.6、−233、−229。该回报尺度明显变化，应以完成步数、覆盖和成功等行为指标评价新训练。

其二为成功标记在子步中锁存后，同控制周期又触发物理失败，旧函数仍能发40分。人工状态使用success=True、success_pulse=1、重力方向对应90°倾斜，旧逻辑显示bad_orientation=True、stepping_succeeded=True、success_reward=40。成功终止现移动到所有终止项末尾，读取已计算的姿态、机身接触、关节限位、NaN等物理失败，物理失败优先。有效成功恰在回合时限边界时保留。直接调用且无TerminationManager时复核姿态。

增加五项总账/冲突回归，复用既有组合选择，164项在36.001s通过。检查后半段计价、同周期物理失败、直接调用姿态、重复未完成摆动不重复支付、重复部分落脚不重复发覆盖与中心。初次执行仅因float32的−262.6积分误差3.05e−5略高于3e−5容差失败，测试绝对容差改1e−4后通过。三个修改Python文件AST与行尾空白检查通过。奖励说明及相关项目记忆已同步。

用户随后明确要求“改完开始给我训练自己训练2000轮”。据此实际执行GPU训练。发现 `2026-11-18_14-14-44_train_2` 仍使用旧8s窗口、10000轮参数，核对命令行后结束Python PID20740及启动器34192，保留其已有事件日志和model_0.pt。读取进程身份与停止该目标进程使用了已授权的提权执行，自动审批通过。

2026-11-18本地14:25:04启动 `logs/rsl_rl/pointleg_stepping_stones/2026-11-18_14-25-04_train_3_audit_2000`，命令为 `.venv/Scripts/python.exe -u -m pointleg.run train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run flat_8700_fixed_gait_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 2000 --agent.save-interval 100 --agent.run-name train_3_audit_2000 --agent.logger tensorboard --agent.upload-model False`。GPU为RTX5060Laptop8GiB，cuda:0。源种子SHA256为 `a0f481a6e0bc213c427c0f62e56a712d57ef10c92172b0f28e156ab8f41d72a9`。工具执行会话78799，GPU进程15928、启动器37588。

已通过保存的env.yaml核对512环境、失败窗口60.0、终止末项stepping_succeeded，通过agent.yaml确认2000次和正确来源。首份model_0.pt可读取，iter=0、Actor/Critic123/399，并保存课程cap0和窗口。14:29本地已至第47轮，平均赛道速度+0.3581m/s，完整步数/覆盖/成功仍0。训练正在进行，预期末次model_1999.pt，需要继续监测到完成并核验模型、哈希及末段指标。启动不构成策略通过验收。
## 2026-11-19 梅花桩训练命令生成器

按用户要求将页面的 MuJoCo 单环境和 Viser 单环境回放命令移至前两项，其他命令保持原有相对顺序；仅调整页面排列，不改变训练或回放配置。

按用户要求将页面训练环境数从每次增减 1 的数字输入改为 1 至 65536 的 2 的幂次下拉选择，默认仍为 512。属于页面输入调整，不影响训练配置，需重新选择并复制命令才会使用新值。

再次按用户要求支持不同任务的实验目录。根据 `src/pointleg/__init__.py` 与 `config/` 中当前任务及实验名称配置，页面自动识别唯一对应的任务并生成对应回放和续训命令；共享 `pointleg_rough` 的两个任务需手动选择。续训以模型路径中的实验目录上级为 `--log-root`，任务与实验目录不一致时禁用续训，避免误载同名运行。迁移目标单独选择，仍默认梅花桩，可从其他任务模型迁移。Node 模拟页面检查 Flat、Stairs、Jump、自定义根目录、Rough 歧义、目录不匹配及迁移目标通过；未在浏览器实测、未执行迁移、回放或训练。

后续按用户要求将继续训练移至第二项，末尾加入 `uv run pointleg transfer PointLeg-SteppingStones-v0 --checkpoint-file ... --output-run ...`，源路径沿用输入模型，输出名称默认为 `lift5cm_pretrained` 并可更改；目标输出目录已存在时迁移程序会拒绝覆盖。Node 模拟检查七项顺序、默认和自定义输出、非标准源目录及复制回退通过；未在浏览器实测，未执行迁移、回放或训练。

为减少手动替换模型路径，增加独立页面 `tools/训练命令生成器.html`。输入 Windows 检查点路径后生成六条 PowerShell 命令，续训默认 512 环境、1000 轮、train_1；目录名和文件名作为完整正则精确匹配，非标准梅花桩日志目录禁用续训命令。需要在 rc_mjlab 工作目录运行，页面不能验证模型存在性、来源和兼容性。Node 模拟页面检查六条命令、单/四环境参数、正则转义、路径及数字校验和复制回退通过；未在浏览器实测复制，未启动训练或仿真。用户示例路径在当前工作区不存在。

## 2026-11-20 Colab Notebook 启动入口

按用户要求新增 `notebooks/colab_pointleg.ipynb`，为 PointLeg 提供 Google Colab 启动入口。Notebook 依次检查 Python 版本、克隆或快进更新 `aoioi222/RL_pointleg`、安装 `uv`、运行 `uv sync --locked --extra cu128`、显示 `nvidia-smi` 并执行 `pointleg list`。训练配置默认 `PointLeg-Flat-v0`、64 环境、1000 次更新和运行名 `colab_flat`，训练命令通过 `--log-root` 写入 Google Drive 的 `MyDrive/pointleg_logs`。

训练开关 `START_TRAINING` 默认 `False`，因此本次仅新增入口，不会在用户打开 Notebook 时自动启动物理步进、Viser、PPO 或正式训练。未改动任务、奖励、地形、观测、动作、模型接口或检查点；无新检查点兼容性变化。仅执行了 Notebook JSON 结构与训练命令参数的静态核对，未在 Colab 安装依赖、挂载 Drive、启动物理仿真或训练。模型权重、`pretrained/` 与历史日志仍不在仓库中，续训或回放须由用户提供相应检查点。

## 2026-11-21 梅花桩更新同步

从开发源目录同步当前梅花桩训练、地形和文档更新至作品集仓库。同步内容包括固定步态、奖励、课程与地形配置，以及对应的定向测试文件。同步后的当前规则包含：足端触地失败终止扣分为 −50，其他失败终止保持 −25；右后首步目标前移 3 cm；左后从桩或入口台跨越到后续目标时，低于顶面 5 cm 的下坠按最高 −8/s 计罚。

本轮仅同步已有源目录内容并恢复 Colab 文档入口，未手动调整任务、奖励、地形、观测、动作、模型接口或训练参数。未运行单元测试、物理仿真或训练；仅进行 Git 差异与空白字符静态核对。源目录记录的既有定向检查结果不代表本轮重新执行的结果。

