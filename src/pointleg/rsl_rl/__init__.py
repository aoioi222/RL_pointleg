# =============================================================================
# rsl_rl/__init__.py — HIM 训练链（自实现的 RSL-RL 子集）
#
# 【这套代码是什么】
#   HIMLoco 思路的本地实现：非对称 actor-critic + 观测历史估计器 + PPO，
#   是 mjlab 内置 rsl_rl runner 的**替代训练链**。
#   当前 robot/__init__.py 注册的三个任务走的仍是 mjlab 内置 runner，
#   本包既没有被 env_cfgs.py 引用，也没有在这里导出 runner/wrapper，
#   需要另写入口脚本把 HimMjlabVecEnvWrapper + HIMOnPolicyRunner 接起来才能用。
#
# 【模块职责】
#   modules/him_estimator.py        从观测历史估计线速度与隐变量（对比学习）
#   modules/him_actor_critic.py     非对称 actor-critic 网络
#   algorithms/him_ppo.py           PPO 主算法（多一路估计器更新）
#   storage/him_rollout_storage.py  经验缓冲区，额外存 mu/sigma/下一步特权观测
#   env/vec_env.py                  环境抽象接口
#   wrappers/…                      把 mjlab 环境适配成上面的接口（未导出）
#   runners/…                       训练主循环（未导出）
#
# 【未导出 wrappers / runners 的原因】它们依赖 him_mjlab 包（不在本仓库，
#   pyproject 里也没有依赖声明），在这里导入会让整个 robot 包 import 失败。
# =============================================================================

from .modules.him_estimator import HIMEstimator
from .modules.him_actor_critic import HIMActorCritic
from .algorithms.him_ppo import HIMPPO
from .storage.him_rollout_storage import HIMRolloutStorage
from .env.vec_env import VecEnv