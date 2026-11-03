# =============================================================================
# him_estimator.py — HIM 估计器（隐式特权信息提取 / 对比学习模块）
#
# 【要解决的问题】
#   策略部署时拿不到特权信息（机身真实线速度、地形高度图等），但训练时 critic 有。
#   如果只在训练时用特权信息、部署时直接扔掉，策略网络就永远学不会"从可观测的历史
#   里推断出这些信息"。HIM（Hybrid Implicit-explicit Estimator）的做法是：
#   训练一个估计器，从**一段观测历史**中回归出线速度 vel 和隐变量 latent，
#   再把 (当前单步观测, vel, latent) 拼起来喂给 actor。
#   于是 actor 在训练时就有了"近似特权信息"的输入，部署时估计器照常工作，
#   输入维度完全不变——这是 sim2real 可落地的关键。
#
# 【三部分结构】
#   encoder  : 输入整段历史 (temporal_steps × num_one_step_obs)
#              → 输出 (vel(3维), z(latent))，是部署时真正用的推理路径
#   target   : 输入**下一步的单步观测** → 输出 z_target
#              只在训练时用，提供对比学习的"正样本参考"
#   proto    : 32 个可学习原型向量，配合 Sinkhorn 做聚类式的对比学习
#
# 【两个损失】
#   estimation_loss = MSE(预测 vel, 真实 vel)       —— 显式部分，直接监督
#   swap_loss       = 交换预测的对比损失             —— 隐式部分，让 z 学到
#                      "能区分不同运动状态"的表示（而不是无意义的数值回归）
#
# 【温度与 Sinkhorn】
#   score = z · proto^T 得到 [batch, 32] 的相似度矩阵，Sinkhorn 把它归一化成
#   一个"双随机"概率分配矩阵（每行每列和都相等），相当于软聚类分配 q。
#   swap_loss 要求 encoder 的分配能预测 target 的分配、反之亦然，
#   使 z 在"运动模式"这一语义维度上保持一致，比直接回归 z_t 稳定得多。
# =============================================================================

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class HIMEstimator(nn.Module):
    """从观测历史中估计线速度与运动隐变量，供 actor 作为额外输入使用。"""

    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        # enc_hidden_dims 最后一项 16 是隐变量维度，实际输出维度是 16 + 3 = 19
        # （前 3 维给 vel，后 16 维给 latent z），这是 HIM 原论文的标准划分。
        if kwargs:
            print("Estimator.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Encoder：输入 = 整段历史展平（temporal_steps × num_one_step_obs）
        # 用展平而不是 RNN，是为了避开循环网络不能并行、训练慢的问题。
        # 代价是历史长度必须固定，靠 obs 分块拼接来构造这条长向量。
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        # 输出维度 = 隐变量维度 + 3（3 维从头部切出来当线速度预测）
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target：输入单步观测，输出与 encoder 的 latent 同维度的目标表示
        # 它不看历史，只提供"当前这一刻应该对应什么语义"的参考，防止 encoder
        # 靠历史里的冗余信息走捷径。
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype：32 个隐空间锚点，对比学习的"聚类中心"
        # 用 nn.Embedding 只是借它的权重表，不做索引查表。
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer：估计器有独立的优化器和学习率，不与 PPO 主优化器共享
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        """供外部取 (速度, 隐变量) 且不建计算图，用于推理/日志。"""
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        """部署/推理路径：只跑 encoder，且整体不建计算图。

        注意 obs_history.detach()：估计器的输入永远不回传到环境或 PPO 主网络，
        这是刻意的——否则对比学习的梯度会污染策略网络。
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)     # latent 单位化，保证相似度只反映方向
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        """与 forward 相同但不 detach，专门给 update() 训练时用（需要梯度）。"""
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        """用一批 rollout 数据更新估计器，返回 (显式速度回归损失, 对比学习损失)。

        next_critic_obs 的布局约定（与 env_cfgs.py 的 critic 观测顺序耦合，改动需同步）：
            [0 : num_one_step_obs]                     下一步的 Actor 单帧观测
            [num_one_step_obs : +3]                    特权线速度（真实标签，只在训练时有）
        目标编码器只读取 Actor 观测，不混入其后的特权信息。
        """
        if lr is not None:
            # 估计器跟随 PPO 的自适应学习率一起调，避免两套 lr 打架
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        if next_critic_obs.ndim != 2 or next_critic_obs.shape[1] < self.num_one_step_obs + 3:
            raise ValueError("HIM critic observations must contain the actor prefix followed by three velocity values")
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs + 3].detach()
        next_obs = next_critic_obs[:, :self.num_one_step_obs].detach()

        z_s = self.encoder(obs_history)
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        # 原型向量每步重新单位化（写回权重），保证做点积时比较的是方向而非模长，
        # 否则网络可以通过放大模长来刷高相似度分数。
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        # Sinkhorn 分配矩阵不参与梯度（当作伪标签），是 SwAV 风格的自监督做法
        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        # swap：用 encoder 的分配去解释 target 的分布，反之亦然（互为正样本）
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    """把相似度矩阵 softmax 后做行/列交替归一化，得到双随机分配矩阵。

    eps=0.05 是 softmax 温度，越小分配越"硬"（接近 one-hot）；
    iters=3 只迭代 3 轮，是精度与速度的折中（这里不需要严格收敛）。
    注意 Q 先转置再运算，最后乘 B 再转置回来，是为了让"原型维在前"便于按行归一。
    """
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
