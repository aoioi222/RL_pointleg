# 梅花桩奖励调整

日期：2026-10-27。适用任务：`PointLeg-SteppingStones-v0`。

本次在现有平地派生的梅花桩任务中落实目标方向推进和首次有效落足结算。场地、2～8 桩课程、随机出生净距、单足接地失败与全覆盖后出口停稳规则保留。Actor/Critic/动作维度仍为 105/381/12。

## 参考来源与取舍

参考 [legged_robot_competition](https://github.com/jindadu00/legged_robot_competition) 的 README：目标方向速度、边缘风险、分阶段完成；参考其 `go2_robot.py` 的首次接触结算方式。核对时默认 master 的 GO2 代码没有接入 README 所述的 `tracking_goal_vel`、`reach_all_goal`、`feet_edge`，本次是针对本项目重新实现，不声称直接复现了对方完整策略。

不引入长腾空正奖励、主动跳跃、总奖励非负裁剪或仅按机身到终点判成功。已有边缘代价与通关事件继续使用，不重复添加相同目的的奖励。独立登台预训练仍是后续可选实验，本次没有另建任务、改变出生分布或放宽主任务。

## 目标方向推进

`mdp/stepping_stones.py:goal_velocity_reward` 替换原 `forward_velocity_reward`，配置名从 `stepping_forward_velocity` 改为 `stepping_goal_velocity`，权重仍为 1。旧函数保留供历史调用，但当前梅花桩配置不再使用它。

目标完全由现有平台位置、桩位、有效掩码和覆盖掩码推导，不增加策略输入：

- 登台前：入口平台顶面中心的水平位置。
- 桩阵内：最前一组尚未全部覆盖的两桩，其纵向中点作为机身引导位置，横向固定在场地中心线。单个桩先覆盖不会切换到另一排桩中心；奇数尾桩只使用其纵向位置，仍保持横向居中。
- 全覆盖后：出口平台中心。覆盖顺序与用哪只足均不是硬判定条件，提前踩到后续桩仍正常计覆盖。

用世界水平速度在目标方向上的投影除以回合指令速度，正向收益上限为 1，登台阶段再乘 0.3。距离目标不足 5 cm、停稳、成功、失败时不发推进奖励。负向投影保留负值，不在 -1 处裁剪。

每个目标记录本回合的历史最小距离，只有创造新的接近记录才发正速度收益；退后再走回旧位置、位置不变但速度观测为正，都不会重复领取。目标切换的首步只建立基准，不产生切换奖励；同一控制步重复读取返回同一结果。回合重置清空相应环境的历史。

原 `route_progress_reward` 的全程固定势函数保留，权重 10、gamma=0.9995；不随临时目标切换重定义势函数。真实仿真另发现并修复旧实现将 `progress_ready` 逐环境布尔张量替换成 Python `True` 的错误，避免第二次奖励计算与局部重置报错；首步不发进度奖励的语义保留。

## 分开计算落点和抬腿质量

原实现先取抬腿质量和落点质量的最大值，再同时发放两项奖励，可能使居中落足替代足底净空、或高抬腿替代安全落点。本次拆为独立的每桩最佳质量：

- `landing_pulse`：只使用首次覆盖时、由离地转为桩顶接触的足的落点余量，余量达到 2 cm 后饱和。全桩落点奖励总额仍不超过 10。
- `lift_pulse`：只使用可验证摆腿后落足的足底峰值净空；足底从 0.20 m 至 0.24 m 对应质量 0～1，达到 0.24 m 后不再因抬得更高或悬得更久增加收益。四足首次登台总额不超过 2，全桩首次覆盖总额不超过 2。

抬腿结算要求先有合法支撑，再离地至少 0.020 s，摆腿期间持续保留至少两只其他合法支撑足，最后落到合法顶面且回合未失败。侧面接触不作为合法支撑；未完成登台时地面支撑正常允许。初始化悬空、短接触抖动和失支撑的摆腿不拿抬腿分，但不额外改变真实顶面接触的覆盖规则。

当前模型球足半径 25 mm；`_foot_bottom` 使用足端 site 姿态修正旋转后球足的真实最低点，并供摆腿峰值、净空不足及超高代价共同使用。不是在所有姿态下把 site 的 Z 当作足底高度。

同一桩多足同时首次接触，分别取落点质量与净空质量中的最佳值，各只结算一次；它们不再互相替代。同一足再次登上入口平台、已覆盖桩反复抬落都不给重复正事件奖励。

任一子步发生任务失败，清除本控制周期已经积累的登台、覆盖、落点、抬腿和成功正事件。奖励读取时也屏蔽框架其他失败终止；单次失败 -150 保留，未引入总奖励截零。

## 边缘、指标与未完成项

已有 `stepping_edge` 权重 -0.3、安全余量 2 cm 保留，安全区内不额外强迫脚踩几何中心。抬腿、落点、覆盖和终点权重分别仍为 1、10、30、100；没有通过盲目加大所有正奖励来迁入参考思路。

新增回合指标：

| TensorBoard 名称 | 含义 |
|---|---|
| `Episode_Metrics/stepping_goal_distance` | 控制步机身到当前引导目标水平距离的回合平均，米 |
| `Episode_Metrics/stepping_low_clearance_fraction` | 有效完成摆腿中足底峰值低于 0.24 m 的比例 |
| `Episode_Metrics/stepping_swing_landing_count` | 有效完成摆腿样本数；无样本时不足比例为 0，必须结合此计数解释 |
| `Episode_Metrics/stepping_edge_contact_fraction` | 控制步桩顶接触中安全余量不足 2 cm 的比例的回合平均 |

2 cm、4 cm、20 ms、两足支撑及速度权重都是仿真初值，需要正式训练验证。本次抬腿质量使用峰值，不宣称已实现设计文档中的全路径扫掠最小净空；接触归属仍采用当前 found/pos 通道和几何区域判定，没有新增接触法向或多接触槽。专用正式评估器、课程窗口持久化和 105 维部署接口仍需后续适配。

## 验证与使用

在项目目录使用已有环境，不安装依赖：

```powershell
.\.venv\Scripts\python.exe -m unittest pointleg.tests.test_stepping_stones -v
.\.venv\Scripts\python.exe -m unittest discover -s src/pointleg/tests -p 'test_*.py' -q
.\.venv\Scripts\python.exe tools/check_stepping_stones.py --steps 64
```

专项共 23 项，点足全量 282 项测试已通过。`tools/check_stepping_stones.py --steps 64` 已通过真实 GPU 仿真：七环境分别对应2～8桩，观测/奖励有限，检测到真实支撑，接口105/381/12。报告为 `artifacts/stepping_stones_reward_check.json`；零动作短仿真不是策略通过率验证。本次没有训练策略、覆盖检查点或停止已有训练进程。

新奖励与新曲线需要重启训练进程才会采用。已有 105/381/12 梅花桩检查点尺寸不变，但奖励目标和 critic 回报分布已变，续训后仍须重新评估；45 维平地模型仍不能原样严格加载为当前梅花桩策略。

## 2026-10-31 平地8700迁移与512环境训练

已按用户指定执行迁移，源文件为`logs/rsl_rl/pointleg_flat/2026-10-22_09-27-04/model_8700.pt`，目标为`logs/rsl_rl/pointleg_stepping_stones/flat_8700_pretrained/model_0.pt`，并保存同目录`transfer.json`。源文件未修改，SHA256为`c7af6d864a0cfde0f91b382ca0854016a806879392d9327224e10d036e8e52aa`；目标SHA256为`c6e0b9b647d7477e0f2e72095fc6e83ba3e65e01bad6513982ecdb83e7d87ade`。

`pointleg transfer`现已支持SteppingStones：保留Actor原45列、Critic原325列输入权重及全部其他网络参数，新增60/56列置零，严格拒绝未知输入维度或其他网络结构不匹配。当前源保存的观测顺序及scale已核对；Critic原四足接触槽仍在相同位置，由梅花桩专用接触项替换。优化器、迭代和公共步数重置，主任务仍从2桩开始。不是继续平地的8700轮，也没有把8700重命名成0而不处理输入。

迁移操作已执行，不要再次迁移到同名目录。直接在工程目录执行以下训练命令：

```powershell
Set-Location "E:\OneDrive\Desktop\RC_WheelLeg-main\05_software\train\rc_mjlab"
uv run --locked --no-sync pointleg train PointLeg-SteppingStones-v0 --env.scene.num-envs 512 --agent.resume True --agent.load-run flat_8700_pretrained --agent.load-checkpoint model_0.pt --agent.max-iterations 10000 --agent.run-name flat8700_stepping --agent.logger tensorboard --agent.upload-model False
```

该指令已用tyro解析，并核对会加载上述目标文件；从迁移迭代0追加10000次更新，框架编号0～9999。日志保存于默认`logs/rsl_rl/pointleg_stepping_stones/<时间戳>_flat8700_stepping/`，不会覆盖源平地运行或迁移种子。

本轮17项定向CPU网络/迁移检查通过，包括6项新梅花桩迁移检查和11项原Elevation/Jump恢复检查。实际model_0已严格resume加载，逐参数核对原权重保持且新增列全零，32组随机网络输入对照最大绝对误差Actor约1.91e-6、Critic约5.25e-6；报告`artifacts/stepping_stones_flat8700_transfer_validation.json`。按用户当前验证约束，本轮未运行物理仿真、Viser或PPO，也未启动上述10000轮训练。这些检查不代表梅花桩策略已验收。
