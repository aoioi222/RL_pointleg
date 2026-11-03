# =============================================================================
# rl_cfg.py — PPO 训练超参数配置
#
# 【模块定位】
#   各任务的算法层配置，由点足包入口在注册启用任务时调用。
#   注意这里只描述"怎么训练"，不涉及环境本身（那是 env_cfgs.py 的事）。
#
# 【继承关系】rough_ppo_runner_cfg 是唯一的完整定义，另两个都调它再改字段：
#       rough_ppo_runner_cfg  (完整模板)
#              ├── flat_ppo_runner_cfg   改：实验名 + 最大迭代 10000
#              └── crawl_ppo_runner_cfg  改：仅实验名
#   所以改公共超参只需要动 rough 那个函数，另两个自动生效。
#
# 【关键超参说明】
#   entropy_coef=0.003  熵奖励系数，控制探索强度。训练后期策略已经定型时
#                       如果还保持高熵，动作会一直抖，常见做法是二/三轮训练时
#                       手动降到 0.001 → 0.0008（注释里也是这么建议的）。
#   schedule="adaptive" 按 KL 散度自适应调学习率：KL 太大就降 lr 防止策略崩，
#                       KL 太小就升 lr 加快收敛，比固定 lr 稳得多。
#   num_mini_batches    缓冲区越大，切得越细。注释给出了 2048→4、4096→8 的对照，
#                       改 num_envs 时这两个数要一起调，否则单批样本太少、梯度噪声大。
# =============================================================================

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """
    平地环境PPO训练配置
    基于rough_ppo_runner_cfg粗糙地形配置继承，仅修改实验名与最大训练迭代次数
    返回：PPO训练运行器配置实例，用于平地步态训练

    注意 max_iterations 这里从 20000 降到 10000：平地任务简单、收敛快，
    练太久反而会因为熵系数没降而停留在抖动状态。
    """
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_flat"  # 实验名称，日志/模型保存文件夹名
    cfg.max_iterations = 10_000         # 平地最大训练迭代轮次
    return cfg


def high_step_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Keep the walking networks and isolate the high-step experiment logs."""
    cfg = flat_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_high_step"
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg


def rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """
    粗糙地形PPO基础配置（公共基础模板）
    Actor、Critic网络结构、PPO算法超参数全部在此定义
    flat、crawl模式均复用此配置，仅少量字段做覆盖修改
    返回：完整PPO on‑policy训练运行器配置
    """
    return RslRlOnPolicyRunnerCfg(
        # Actor策略网络配置
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),  # Actor隐藏层维度：3层MLP
            activation="elu",              # 激活函数ELU
            obs_normalization=False,       # 是否开启观测归一化，此处关闭
                                           # 关闭的原因：观测已在 env_cfgs 里按项做了 scale
                                           # （如角速度 ×0.25、关节速度 ×0.05），
                                           # 再叠一层在线归一化会让量纲含义变得难以追踪
            distribution_cfg={
                "class_name": "GaussianDistribution",  # 动作分布：高斯分布（连续动作空间）
                "init_std": 1.0,                       # 初始动作标准差
                                                       # 1.0 偏大，意味着开局动作幅度接近满量程，
                                                       # 配合 entropy_coef 一起构成早期探索强度
                "std_type": "log",                     # std参数存储形式：log域，网络输出log_std
                                                       # log 域保证 std 恒正，且梯度尺度更均匀
            },
        ),
        # Critic价值网络配置
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),  # Critic隐藏层维度，与Actor结构保持一致
            activation="elu",              # 激活函数ELU
            obs_normalization=False,       # 观测归一化关闭，与Actor对齐
        ),
        # PPO算法核心超参数
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,               # value loss损失权重系数
            use_clipped_value_loss=True,       # 使用裁剪版value loss，防止价值函数更新震荡
            clip_param=0.2,                    # PPO重要性采样裁剪系数 ε
            entropy_coef=0.003,                # 熵奖励系数，鼓励探索；第一轮0.003，第二轮0.001，第三轮0.0008
            num_learning_epochs=5,             # 每批数据重复训练轮次；缓冲区2048设5，4096设3
            num_mini_batches=4,                # minibatch数量；缓冲区2048设4，4096设8
            learning_rate=8.0e-4,              # 优化器学习率
            schedule="adaptive",               # KL自适应学习率调度策略
            gamma=0.99,                        # 回报折扣因子
            lam=0.95,                          # GAE‑Lambda广义优势估计系数
            desired_kl=0.01,                   # 目标KL散度，adaptive调度以此调整lr
            max_grad_norm=1.0,                 # 梯度裁剪最大范数，防止梯度爆炸
        ),
        experiment_name="pointleg_rough",  # 粗糙地形实验名称
                                        # 决定日志与模型保存目录：logs/<experiment_name>/<时间戳>/
        save_interval=100,              # 每多少迭代保存一次模型checkpoint
        num_steps_per_env=24,           # 每个环境每次采集的时间步数量
                                        # rollout 总样本数 = num_steps_per_env × num_envs
                                        # 默认 24×2048 = 49152，缓冲区很大，
                                        # 显存吃紧时优先降 num_envs（这也是之前 8G 显卡 OOM 的原因）
        max_iterations=20_000,          # 粗糙地形最大训练迭代轮次
    )

def crawl_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """
    匍匐姿态训练PPO配置
    直接继承rough_ppo_runner_cfg全部超参，仅修改实验名称
    返回：匍匐模式训练运行器配置
    """
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_crawl"  # 实验名称，日志/模型保存文件夹名
    return cfg


def elevation_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """高程越障的同结构迁移配置和较长折扣时间尺度。"""
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_elevation"
    cfg.algorithm.gamma = 0.995
    cfg.algorithm.learning_rate = 3.0e-4
    return cfg


def bridge_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """单木桥 PPO 初始配置，独立保存训练日志与模型。"""
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_bridge"
    return cfg


def stepping_stones_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """梅花桩 PPO 初始配置，独立保存训练日志与模型。"""
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_stepping_stones"
    cfg.algorithm.gamma = 0.9995
    cfg.algorithm.learning_rate = 3.0e-4
    return cfg


def jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """平台间隙及中墙共用一个固定跳跃策略。"""
    cfg = elevation_ppo_runner_cfg()
    cfg.experiment_name = "pointleg_jump"
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg
