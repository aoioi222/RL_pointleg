# 固定触发平台跳跃

2026-10-27：`PointLeg-Jump-v0` 已切换到本文的平台任务。旧三墙任务、330 mm 目标墙和 `--warmup` 命令属于历史实现，不再用于当前训练入口。Elevation 的楼梯、坡面训练保留原配置。

## 地形和课程

一个 Actor 同时训练两类场景，各占一半环境。每块地图 3.0 × 1.6 m，有效平台宽 0.8 m；起跳平台位于 x=0.08～1.10 m，落地平台延伸到 x=2.92 m。起跳和落地平台顶面等高。固定出生基座 x=0.64 m、y=0.80 m、朝向 +X、速度为零，关节使用默认站姿；通过当前 XML 的正运动学校正球足高度，避免出生时从空中跌落。

| 项目 | 无墙间隙 | 带墙间隙 |
|---|---:|---:|
| 最终平台高 | 200 mm | 200 mm |
| 最终平台净间距 | 360 mm | 360 mm |
| 墙顶高出平台 | 无 | 300 mm，最终离地 500 mm |
| 墙厚 | 无 | 始终 80 mm |
| 起始平台高 | 30 mm | 30 mm |
| 起始间距 | 54 mm | 80 mm |
| 起始墙顶高出平台 | 无 | 45 mm |

高度轴与间距轴分别有五档比例：15%、36.25%、57.5%、78.75%、100%。高度轴同时缩放平台高度与墙的相对高度；两平台始终等高。只改变等高平台的离地高度，本身不会增加必须跳起的高度；带墙任务的墙顶升高才增加竖直净空要求。间距轴缩放 360 mm；带墙起步按用户确认保底 80 mm，墙居中放置。

五档高度与五档间距交叉，共 25 个组合，乘两种地形共 50 块物理地图，布局为 10×5。每个组合均有真实几何，不把多个课程档映射到同一低难度地块。地图总地块面积约 240 平方米，原 16 块 12×8 m 地形为 1536 平方米。

两种地形各自维护两个课程上限。只用该轴当前上限的已结束有效回合统计，最近 100 回合成功率达到 80% 才提升该轴一档。重置时只从已解锁的组合采样，偏重上限。失败不增加上限，数值无效回合不用于升级。跳距始终固定为 400～500 mm，不随早期小间距缩短。

## 触发、跳距和判定

每次环境 reset 对应部署端的一次技能触发，不安排接近或助跑过程。控制频率保持 50 Hz、物理频率 200 Hz。触发后第一帧就输出跳跃策略动作；允许物理下蹲和蹬地，要求 0.6 s 内形成有效腾空。该时间、初始漂移容许和落稳阈值是可调整的训练初值，不是已验证的实机响应性能。

有效起跳先要求四足曾在起跳平台承重，然后整机无地形接触至少 0.02 s，且前向、向上速度均大于 0.05 m/s。起跳前支撑前足接触点位移最多 40 mm、基座前后漂移最多 80 mm；摆动足收腿不被误判为助跑。起跳后返回起跳平台、未腾空就跨步到对面平台、落地后再跳均失败。

每只前脚最后一次起跳平台接触点作为起跳参考，每只后脚第一次落地平台承重接触点作为落地参考。沿 +X 计算所有前后脚配对的跨度：最小跨度必须大于 0.400 m，最大跨度必须小于 0.500 m。数值边界额外保留 1 微米浮点容差，等于 400 或 500 mm 均不通过。初次后足落地距离超限立即失败，后续挪脚不能挽回；落地后继续滑动到区间外也失败。

成功需要四足同时在目标平台顶面连续承重 0.5 s，滚转和俯仰绝对值小于 12°、基座线速度模小于 0.10 m/s、角速度模小于 0.50 rad/s。任意物理子步失去一足支撑都会清零稳定计时。

足端只有起跳平台和落地平台顶面属于合法接触。墙、地面、平台侧面、边缘外、其他地形接触均失败；机身或腿杆接触地形也失败。非法接触按 `found` 检查，不用力阈值忽略轻微擦碰；四足承重另用 1 N 阈值，顶面位置容差 8 mm，法向必须接近竖直。失败在物理子步锁存，在当前控制步结束重置。每回合最长 3 s，超时按失败，不按成功或时间截断。

## 奖励

保留有符号奖励，事件除以控制 dt，交奖励管理器再乘 dt，单次事件总额不因控制周期改变。路径势函数折扣与 PPO gamma=0.995 一致，终止势设为零。

| 项 | 回报或权重 | 目的 |
|---|---:|---|
| 势函数进度 | 5 | 引导后足整体前移，后退扣回进度 |
| 首次有效起跳 | +1 | 单次事件，重复离地不重发 |
| 四足最低净空峰值增量 | 总额最多 +2 | 引导同时收足，超过目标不再给分 |
| 首次目标平台落地 | +3 | 仅有效腾空后触地一次 |
| 四足稳定成功 | +30 | 核心目标 |
| 失败 | -50 | 非法接触、错距、迟跳、助跑、重复跳、超时等 |
| 时间、居中、侧漂、航向、姿态 | -0.2、-0.5、-0.5、-0.3、-0.2 | 减少拖延和偏移，腾空阶段允许正常俯仰 |
| 力矩、动作变化 | -0.02、-0.01 | 约束控制代价 |
| 落地线速度、角速度、支撑滑移 | -1、-0.2、-0.2 | 引导落稳 |

不使用持续向前速度跟踪、站立不动、支撑不足、普通步态腾空时长等会阻碍起跳或鼓励落地继续走的奖励。失败当步屏蔽正的阶段事件，早期奖励总额有上限，不能反复蹲跳刷分。当前权重通过逻辑和训练链路检查，收敛性、真实跳距与实机通过率仍需正式训练及固定种子评估。

## 策略和部署接口

Actor/Critic/动作仍是 45/325/12 维，但 Actor 第 6～8 项（零基索引，原速度指令槽）改为 `(1, elapsed_seconds / 3, 0)`。第一个值表示本次技能活动，第二个值是自本次触发起的归一化单调时间；其余本体观测、关节顺序和动作滤波沿用基础工程。地形类型、间距和墙高只影响仿真反馈和 Critic，不输入 Actor，训练同一个跳跃策略。

`src/pointleg/jump_trigger.py:JumpTrigger` 提供部署端相同的命令时钟。触发上升沿即开始，重复请求不重置正在执行的时间；稳定落地或中止后调用 `finish()`。最迟 3 s 结束；在非活动期由外部控制器接管，不能继续运行此 Actor 或以零动作代表已验证的安全接管。应同时按实际部署控制链处理上一动作和低通滤波状态；本次没有接入 ROS/实机控制器，也未验证切换瞬态。

旧三墙 Jump 的 45 维数值尺寸相同，但命令语义不同，禁止直接续训或部署。可以使用 `transfer` 仅迁移兼容网络权重，然后重新训练；优化器、迭代与课程归零。新检查点保存 `jump_contract` 与双轴课程，完整续训检查契约一致。ONNX 导出附带触发时钟契约。旧 CPU Sim2Sim 速度命令回放入口拒绝当前 Jump，使用训练环境的 `play` 和 `evaluate`；不要用手动 `J` 槽位直接运行新模型。

## 启动与验收

在 `E:/OneDrive/Desktop/RC_WheelLeg-main/05_software/train/rc_mjlab` 的 PowerShell 内复用现有环境：

```powershell
uv run --locked --no-sync pointleg train PointLeg-Jump-v0 --env.scene.num-envs 256 --agent.run-name fixed_platform_jump --agent.logger tensorboard --agent.upload-model False

uv run --locked --no-sync pointleg transfer PointLeg-Jump-v0 --checkpoint-file "源模型路径/model_xxx.pt" --output-run platform_pretrained
uv run --locked --no-sync pointleg train PointLeg-Jump-v0 --env.scene.num-envs 256 --agent.resume True --agent.load-run platform_pretrained --agent.load-checkpoint model_0.pt --agent.run-name fixed_platform_jump

uv run --locked --no-sync pointleg evaluate PointLeg-Jump-v0 --checkpoint-file "新模型路径/model_xxx.pt" --profile nominal --episodes-per-case 100 --output artifacts/jump_platform_nominal.json
uv run --locked --no-sync pointleg evaluate PointLeg-Jump-v0 --checkpoint-file "新模型路径/model_xxx.pt" --profile robust --episodes-per-case 100 --output artifacts/jump_platform_robust.json
```

从零训练和迁移训练选择一种。`--final-only` 只测两种满难度地形；默认覆盖全部 50 个组合，五组种子分配每个组合的 100 个回合。nominal/robust 各用每用例 90%/80% 通过门槛。`--warmup` 已停用，课程初始档承担前置学习。

日志仍位于默认 `logs/rsl_rl/pointleg_jump/<运行目录>/`。TensorBoard 中 `Episode_Metrics/jump_*` 记录成功、失败、起跳延迟、跳距最小/最大/均值、距离有效标志、最低足端峰值净空、稳定时长、落地速度、非法接触与各类失败、台高/间距/墙相对高度和实际双轴等级。`Curriculum/jump_levels/platform_gap_success_rate` 与 `platform_wall_success_rate` 为分类型最近 100 有效回合成功率，另有两类地形的 height/gap 上限。

未测得后足落地的回合距离填 0，需结合 `jump_distance_valid` 和 `jump_landed` 解读；未起跳的延迟按 0.6 s 删失值记录，结合 `jump_airborne` 和 `jump_late_launch` 解读。不能把混合回合的平均距离或训练总回报当作几何通过率。

## 本次实际验证

- 264 项点足测试、16 项独立 Sim2Sim/导出测试通过；Jump 专项包含全部 50 个几何射线、非法接触锁存、严格跳距边界、落地后滑移、四足连续稳定、球足真实最低点出生校准、双轴课程和检查点恢复。
- 真实 GPU 场景 4 环境、160 控制步通过，45/325/12 观测动作与奖励有限。
- 球足校准后，8 环境、每轮 64 步、2 次 PPO 更新完成，运行目录 `logs/rsl_rl/pointleg_jump/2026-10-27_09-04-57_jump_platform_final_check`。`model_1.pt` SHA256 为 `f18edac041c67c2dd6810a9e143cbef04cbb2a2d72a7be5d5534a8baaf4370fa`。已读取 TensorBoard 文件确认67种标量，包括关键跳距和分类型成功率；成功率为0，课程仍为起始档。更早调试运行还实际完成了一轮完整续训。
- 调试模型只用于链路诊断；更早的50组合每个一次评估为 `artifacts/jump_platform_smoke_evaluation.json`，0/50成功，不是正式多种子验收。更早的ONNX诊断导出为 `artifacts/jump_platform_smoke.onnx`，与原Actor最大绝对误差约 `8.20e-8`。该较早模型的SHA256为 `7e75c650699e10c5e8426575a5a83ce9644ff22eeeb834886226f0510ae83bed`，与球足校准后的模型区分使用。
- 校准后的模型另完成当前代码的50用例诊断，报告 `artifacts/jump_platform_final_evaluation.json`，0/50成功，全部因起跳前漂移超限失败。当前代码从该检查点完整续训一轮成功，运行目录 `logs/rsl_rl/pointleg_jump/2026-10-27_09-06-49_jump_platform_final_resume`。这些结果证明入口与恢复链路可运行，不能证明已学会跳跃或漂移阈值已经标定。
- 没有启动正式长训练、部署到实机或产出已验收跳跃模型。
