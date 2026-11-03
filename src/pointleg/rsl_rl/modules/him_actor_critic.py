# =============================================================================
# him_actor_critic.py — 非对称 Actor-Critic 网络（HIM 版）
#
# 【三个输入通道，维度各不相同，别搞混】
#   num_actor_obs      —— actor 看到的"历史拼接观测"（(history+1) × 单步维度）
#   num_one_step_obs   —— 单步观测维度，用来把历史切回"当前这一步"
#   num_critic_obs     —— critic 看到的观测（= 单步 actor 观测 + 特权信息）
#
# 【actor 的实际输入是拼出来的 4 段】
#       actor_input = [ 当前单步观测 | vel(3) | latent(16) ]
#   其中 vel 和 latent 都来自 HIMEstimator。所以 mlp_input_dim_a 是
#   num_one_step_obs + 3 + 16 —— 这里的 3 和 16 是硬编码的，
#   如果改估计器的 enc_hidden_dims 或把速度维度从 3 改掉，这两个常数必须同步改。
#
# 【为什么 actor 不直接用历史】
#   历史只喂给估计器（在 update_distribution 里 obs_history 只用于 self.estimator），
#   actor 本体是纯 MLP，输入是"当前观测 + 估计出来的隐式信息"。
#   这样做的好处是部署时 actor 是静态图，没有时序状态需要维护。
# =============================================================================

import torch
import torch.nn as nn
from torch.distributions import Normal
from ..modules.him_estimator import HIMEstimator


class RunningMeanStd:
    """在线均值/方差估计（Welford 增量式），用于观测归一化。

    n 初值给 1e-4 而不是 0，是为了避免第一步除零；同时让初始均值/方差
    （0 和 1）在早期几乎不起作用，样本一多就迅速被真实统计取代。
    """

    def __init__(self, shape, device):
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        """增量更新：M2 是平方偏差累积量，避免存全部历史数据。"""
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count


class Normalization:
    """对张量做 (x − mean) / (std + eps) 的在线归一化。"""

    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)
        return x


class HIMActorCritic(nn.Module):
    """非对称 actor-critic：actor 走估计器路径，critic 直接吃特权观测。

    is_recurrent=False 是给 rsl_rl 的 runner 看的标记：表示这不是 RNN 策略，
    runner 不需要维护 hidden state。本仓库用的是自带 HIMOnPolicyRunner，
    该字段实际只起到"声明"作用。
    """
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_one_step_obs,
                 num_actions,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(HIMActorCritic, self).__init__()

        activation = get_activation(activation)

        # history_size 是从维度反推出来的：总观测长度 ÷ 单步长度 = 历史段数。
        # 注意这个除法假设"历史段每段长度都等于单步观测长度"，如果以后
        # 往 obs 里加了非拼接类的项（比如整段高度图），这个推断就会失效。
        self.history_size = int(num_actor_obs / num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs

        mlp_input_dim_a = num_one_step_obs + 3 + 16   # 当前观测 + 估计速度(3) + 隐变量(16)
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=num_one_step_obs,
        )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')

        # Action noise：std 是可学习参数（不是固定常数），会随训练自适应下调
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # 关掉 Normal 的参数校验：该校验每次 sample 都做 Python 层范围检查，
        # 在数万个并行环境上开销明显，而这里的取值范围完全可控，不需要校验。
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        """按层依次做正交初始化，gain 由 scales 逐层给定。

        正交初始化的作用：让每一层的输出方差在深层网络里保持稳定，
        避免 ELU 网络前向信号逐层衰减或爆炸。注意这里只初始化 Linear 层，
        激活函数层会被自动跳过，所以 scales 的长度应等于 Linear 层数。
        """
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
         for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        """非循环网络不需要跨 episode 保存状态，故为空实现。

        HIMPPO.process_env_step 每个仿真步都会调用它；保留这个接口是为了
        与 rsl_rl 的 runner 协议兼容（换成 RNN 策略时才会真正用到）。
        """
        pass

    def forward(self):
        """本网络不支持直接 forward，必须走 act / evaluate 两条路径。

        原因：actor 需要 obs_history（估计器输入），critic 需要 critic_obs，
        两者的输入不能共用一套 forward 签名，所以显式禁止误用。
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        """由观测历史构造动作分布 N(mean, std)，训练时调用（带梯度）。

        三步：估计器出 (vel, latent) → 与当前单步观测拼接 → 过 actor MLP 得到均值。
        标准差不是网络输出，而是独立的可学习参数 self.std，对所有状态、所有动作维共享，
        这是 rsl_rl 系列策略的惯例（比让网络输出 std 更稳定）。

        两处 NaN/Inf 检查是调试用的"熔断器"：一旦出现就立刻抛异常并打印各段极值，
        否则 NaN 会静默地传遍整个 rollout，最后表现为 loss 变 nan 却查不出源头。
        """
        with torch.no_grad():
            # 估计器推理不建图：估计器由自己的 update() 单独训练，
            # 这里只把它当作"特征提取器"使用。
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat(
            (obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1
        )

        if torch.isnan(actor_input).any() or torch.isinf(actor_input).any():
            print(f"[ERROR] NaN/Inf detected in actor_input before normalization!")
            print(f"  - obs_history: min={obs_history.min():.4f}, max={obs_history.max():.4f}")
            print(f"  - vel: min={vel.min():.4f}, max={vel.max():.4f}")
            print(f"  - latent: min={latent.min():.4f}, max={latent.max():.4f}")
            raise ValueError("NaN/Inf in actor_input before normalization")

        mean = self.actor(actor_input)

        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"[ERROR] NaN/Inf in actor output (mean)!")
            for name, param in self.actor.named_parameters():
                if torch.isnan(param).any():
                    print(f"  - NaN in actor weight: {name}")
                if torch.isinf(param).any():
                    print(f"  - Inf in actor weight: {name}")
            raise ValueError("NaN or Inf detected in actor network output!")

        self.distribution = Normal(mean, mean * 0. + self.std)
        # 上面的 mean * 0. + self.std 是"造一个与 mean 同形状、且与计算图解耦的 std 张量"，
        # 等价于 self.std.expand_as(mean)，但写法更短。不能直接传 self.std，
        # 否则广播时形状不匹配。

    def act(self, obs_history=None, **kwargs):
        """训练时采样动作：从分布里随机取，因此带探索噪声。"""
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """计算动作的对数概率（PPO 重要性比率的分子/分母）。按动作维求和。"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        """部署时用：直接返回分布均值（确定性动作），不采样。"""
        vel, latent = self.estimator(obs_history)
        actions_mean = self.actor(
            torch.cat(
                (obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1
            )
        )
        return actions_mean

    def test_inference(self, obs_history, observations=None):
        """调试用：除了动作均值，额外返回 (vel, latent)，便于观察估计器学到了什么。

        observations 参数是为了兼容 rsl_rl 的推理接口签名而保留，本实现不使用。
        """
        vel, latent = self.estimator(obs_history)
        actions_mean = self.actor(
            torch.cat(
                (obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1
            )
        )
        estimator_output = torch.cat((vel, latent), dim=-1)
        return actions_mean, estimator_output

    def evaluate(self, critic_observations, **kwargs):
        """critic 前向：只吃特权观测，输出状态价值 V(s)。"""
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    """按名字查激活函数。注意 crelu 被映射成 ReLU，sigmoid 未被上面的估计器启用。"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None