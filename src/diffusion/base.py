"""
扩散模型抽象接口。

变异策略已从「截断扩散」升级为「精英插值变异」（Elite Interpolation Mutation）：
  - 废弃：对单个精英加噪后截断扩散（局部搜索，探索范围受限）
  - 新增：在两个精英之间做适应度加权插值 + 小扰动（全局交叉，直接利用两个高适应度点）

插值公式：
    R_mut = λ * R_a + (1 - λ) * R_b + ε_small
    λ = ρ_a / (ρ_a + ρ_b)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class BaseDiffusionModel(ABC):
    """扩散模型抽象基类。

    子类必须实现：
      - random_init_rewards       随机初始化奖励种群
      - generate_rewards_given_tau 按 Tau 条件生成奖励
      - train_on_population       按种群训练扩散模型

    子类可选覆盖：
      - interpolate_mutate        精英插值变异（已提供默认实现，可按需覆盖）

    已废弃（请勿再实现）：
      - truncated_diffusion_mutate  截断扩散变异
    """

    # ------------------------------------------------------------------
    # 必须实现的抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def random_init_rewards(self, population_size: int) -> List[torch.Tensor]:
        """随机初始化 population_size 个奖励张量，返回列表。"""

    @abstractmethod
    def generate_rewards_given_tau(
        self, tau: float, n_samples: int = 1
    ) -> List[torch.Tensor]:
        """给定条件 τ，用扩散模型采样 n_samples 个奖励张量。"""

    @abstractmethod
    def train_on_population(
        self,
        population: List[Any],
    ) -> None:
        """用当前种群（包含 tau, rewards, rho）训练/微调扩散模型。"""

    # ------------------------------------------------------------------
    # 插值变异（默认实现，子类可覆盖以定制扰动分布）
    # ------------------------------------------------------------------

    def interpolate_mutate(
        self,
        elite_pool: List[Tuple[float, torch.Tensor, float]],
        n_offspring: int,
        noise_std: float = 0.01,
        seed: int | None = None,
    ) -> List[torch.Tensor]:
        """精英插值变异。

        从精英集合中随机抽取两个个体 (τ_a, R_a, ρ_a) 和 (τ_b, R_b, ρ_b)，
        在奖励空间中做适应度加权插值并添加小扰动：

            λ      = ρ_a / (ρ_a + ρ_b)
            R_mut  = λ * R_a + (1 - λ) * R_b + ε_small

        参数
        ----
        elite_pool   : 精英列表，每个元素为 (tau, reward_tensor, fitness)。
                       至少需要 2 个元素。
        n_offspring  : 需要生成的子代数量。
        noise_std    : 小扰动标准差 ε_small ~ N(0, noise_std²)；默认 0.01。
        seed         : 随机种子，None 表示不固定。

        返回
        ----
        List[torch.Tensor]: 长度为 n_offspring 的变异奖励列表。

        异常
        ----
        ValueError: elite_pool 少于 2 个元素时抛出。
        """
        if len(elite_pool) < 2:
            raise ValueError(
                f"interpolate_mutate 至少需要 2 个精英个体，当前只有 {len(elite_pool)} 个。"
            )

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        offspring: List[torch.Tensor] = []

        for _ in range(n_offspring):
            # 无放回抽取两个不同精英
            idx_a, idx_b = _sample_two_indices(len(elite_pool), rng)
            tau_a, R_a, rho_a = elite_pool[idx_a]
            tau_b, R_b, rho_b = elite_pool[idx_b]

            # 适应度加权插值系数
            denom = rho_a + rho_b
            if denom <= 0:
                # 两者适应度均为零时退化为均匀插值
                lam = 0.5
            else:
                lam = rho_a / denom  # 适应度高的亲本贡献更大

            # 奖励插值 + 小扰动（防止种群塌缩）
            R_mut = lam * R_a + (1.0 - lam) * R_b
            epsilon = torch.zeros_like(R_mut).normal_(mean=0.0, std=noise_std, generator=rng)
            R_mut = R_mut + epsilon

            offspring.append(R_mut)

        return offspring

    # ------------------------------------------------------------------
    # 废弃方法（保留签名以便给出明确错误提示）
    # ------------------------------------------------------------------

    def truncated_diffusion_mutate(self, *args, **kwargs):  # noqa: ANN001
        raise NotImplementedError(
            "truncated_diffusion_mutate 已废弃。\n"
            "请改用 interpolate_mutate(elite_pool, n_offspring, noise_std)。\n"
            "详见 src/diffusion/base.py 文档。"
        )


# 向后兼容别名
DiffusionModel = BaseDiffusionModel


# ------------------------------------------------------------------
# 内部工具函数
# ------------------------------------------------------------------

def _sample_two_indices(pool_size: int, rng: torch.Generator) -> Tuple[int, int]:
    """从 [0, pool_size) 中无放回采样两个不同下标。"""
    perm = torch.randperm(pool_size, generator=rng)
    return int(perm[0].item()), int(perm[1].item())
