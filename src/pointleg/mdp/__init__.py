# =============================================================================
# mdp/__init__.py — MDP 组件库（刻意留空）
#
# 【为什么不在这里统一 re-export】
#   本目录下的模块都是 mjlab manager 的"插件"，由 env_cfgs.py 显式按需导入
#   （如 from ..mdp.rewards import track_linear_velocity）。如果在这里统一 import：
#     会加载未使用的组件及其依赖，增加任务注册的耦合。
#   因此保持本文件为空，导入关系全部显式写在 env_cfgs.py 里。
#
# 【目录内容索引】
#   rewards.py                奖励函数库（速度跟踪、姿态、接触、诊断指标）
#   commands.py               速度指令生成器（死区过滤 + 地形自适应 + 单轴采样）
#   curriculums.py            课程函数（指令范围课程 + 地形等级课程）
#   lowpass_actions.py        动作项：低通滤波 + 随机延迟（sim2real）
#   disturbances.py           连续低频外力扰动事件（备用）
#   mode_command.py           离散运动模式指令 walk/climb/crouch（备用）
#   only_positive_rewards.py  通过启动事件配置单个环境的奖励下限
# =============================================================================
