# B 类修复与验证边界

## 已处理的代码问题

| 项目 | 当前处理 |
|---|---|
| B02 | Flat 的高度奖励与出生高度共用 `STANDING_BASE_HEIGHT=0.42`，移除无依据的 0.36 m 配置差异；不是实机姿态标定。 |
| B03 | Crawl 奖励与终止共用 `root_ground_clearance()` 和 `base_height_sensor`，测根位置到正下方局部地面的高度。无命中、背面、非有限数据不给高度奖励并终止；全身净空仍属 C07。 |
| B08 | 时间窗口内按结束回合数累计表现，到期后的首次 reset 触发评估，不再要求全局步数整除回合长度。初始化零长度回合不计入，提前失败仍按完整回合时长归一化。 |
| B09 | 默认 nominal 评估移除物理随机化、随机初态、推力、观测噪声和随机动作延迟。robust 配置单独保留训练随机化；两者均冻结课程。 |
| B10 | `dr.pseudo_inertia` 按同一个密度倍率同步缩放质量与惯量，同时采样质心偏移，避免独立事件覆盖。密度倍率为 0.85–1.45，质心各轴偏移 ±0.05 m；属于鲁棒性训练假设，不是实测标定。 |
| B11 | `upward` 按真实公式说明为正权重的直立奖励：直立、水平、倒置分别为 4、1、0。未改变公式和权重，并修正匍匐只靠姿态判负等过时说明。 |
| B12 | 增加 16 环境、8 步 rollout、1 个 minibatch、1 次学习 epoch、默认 2 次迭代的调试配置；不改变网络结构或正式配置。 |
| B13 | 增加并运行真实 PPO 更新、checkpoint 策略／优化器／全局步数恢复、继续训练、ONNX 数值校验和 CPU 回放检查。 |

## 调试与检查

在项目根目录执行：

```powershell
uv run --locked --no-sync pointleg train PointLeg-Flat-v0 --debug
uv run --locked --no-sync pointleg train PointLeg-Rough-v0 --debug
uv run --locked --no-sync pointleg train PointLeg-Crawl-v0 --debug

# 原生 CLI 参数可覆盖调试默认值
uv run --locked --no-sync pointleg train PointLeg-Flat-v0 --debug --env.scene.num-envs 4 --agent.max-iterations 1

# 仅进行 CPU 平地零动作姿态诊断，不训练
uv run --locked --no-sync python -m sim2sim.check_training --task PointLeg-Crawl-v0 --pose-only

# 真实 PPO 更新、保存恢复、导出及回放，默认 cuda:0
uv run --locked --no-sync python -m sim2sim.check_training --task PointLeg-Flat-v0

uv run --locked --no-sync python -m unittest discover -v
```

`--debug` 仅使用本地 TensorBoard 日志，关闭模型上传；默认日志目录为 `logs/rsl_rl/`。闭环检查在 `artifacts/pointleg_checks/` 中保留独立的配置快照、日志、checkpoint、ONNX 和 `report.json`，拒绝覆盖已有目录；名称中的 `Z` 表示 UTC 时间。

每个任务执行两次 PPO 更新，保存后主动清零全局步数并恢复，核对策略与优化器状态，再继续训练一次。随后用真实更新后的权重导出 ONNX，比较 PyTorch／ONNX 动作并运行 1 秒 CPU 平地回放。检查也核对 GPU 中质量与惯量的相同比例缩放，并记录默认姿态的 5 秒零动作表现。

调试配置保留训练随机化，只缩小采样规模与地形数量。Rough 地形释放仍按控制步数定义；改变 rollout 长度会改变对应的迭代轮数，不代表课程已经标定。

## 评估模式

默认 `pointleg play` 使用 nominal。Python 中可显式创建随机鲁棒性评估配置：

```python
from mjlab.envs import ManagerBasedRlEnv
from pointleg.config.evaluation import evaluation_env_cfg

cfg = evaluation_env_cfg("PointLeg-Rough-v0", mode="robust")
env = ManagerBasedRlEnv(cfg, device="cuda:0")
```

nominal 固定默认种子以便复现地形和指令采样，但不会将速度命令设成恒定零，也不承诺 CPU／GPU 轨迹逐位一致。CPU Sim2Sim 仍为无随机化的平地基准，不执行 GPU 地形生成与事件；显式 `--action-delay` 仍能用于固定延迟测试，不等同于 robust 模式。比较轨迹时仍需对齐地形、初态和命令。

## 仍待完成

- B01：默认姿态的静态平衡未解决。5 秒零动作诊断中 Flat/Rough 根高度约 0.248 m、倾斜约 21.27°；Crawl 高度约 0.052 m、倾斜约 0.94°。不能根据训练可运行就确认可稳定站立。
- B04、B05：持续扭矩、48 V 扭矩—转速包络、惯量、摩擦和 PD 等仍待标定。
- B03 的全身净空：根高度不是机身最高点高度，也不能证明四肢和整个机身已通过横杆或通道；比赛几何与规则仍需 C07 实现。
- B12：各任务 PPO 超参数、2048 环境显存及完整课程尚未完成长期资源与效果验收。
- B13：短程检查不代表收敛、稳定步态、越障成功率或真机可部署。恢复沿用当前框架范围，不包含完整物理场景、随机数流及所有课程内部统计，不是任意中断位置的逐步精确续跑。
