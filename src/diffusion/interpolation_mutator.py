"""
精英插值变异器（Elite Interpolation Mutator）

可作为独立组件嵌入 pipeline，也可被 BaseDiffusionModel 子类委托调用。

设计动机
--------
截断扩散变异的局限：
  · 只在单个精英的邻域内随机游走（局部搜索）
  · 探索范围受限于单个个体的噪声球
  · 不利用种群中其他精英携带的结构信息

精英插值变异的优势：
  · 直接利用两个高适应度点之间尚未被访问的奖励空间
  · 适应度加权插值 → 子代天然偏向更好的亲本（软选择压力）
  · 凸组合保持奖励结构的合理性（两个好结构的加权平均大概率仍是好结构）
  · 小扰动 ε_small 防止早熟收敛（种群塌缩）
  · 与遗传算法中 BLX-α 交叉算子同源，理论保证更强

插值公式
--------
    λ      = ρ_a / (ρ_a + ρ_b)          # 适应度归一化权重
    R_mut  = λ · R_a + (1-λ) · R_b + ε  # 加权插值 + 小扰动
    ε      ~ N(0, σ²)                    # 防塌缩扰动，σ 默认 0.01
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class EliteIndividual:
    """精英个体的结构化表示。"""
    tau: float                   # 条件参数 τ（用于扩散生成）
    reward: torch.Tensor         # 奖励张量 R，形状由具体任务决定
    fitness: float               # 适应度 ρ（越大越好）

    def __post_init__(self) -> None:
        if self.fitness < 0:
            raise ValueError(
                f"适应度 fitness 必须 ≥ 0，当前值为 {self.fitness}。"
                "若你的适应度指标可为负，请先做平移（如 fitness - min_fitness）。"
            )


@dataclass
class InterpolationConfig:
    """插值变异的超参数。"""
    noise_std: float = 0.01       # 小扰动标准差（防种群塌缩）
    fitness_eps: float = 1e-8     # 防除零的极小量
    allow_self_crossover: bool = False  # 是否允许同一个体与自身交叉（pool_size=1 时退化用）
    seed: Optional[int] = None    # 随机种子，None 表示不固定


# --------------------------------------------------------------------------
# 核心变异器
# --------------------------------------------------------------------------

class EliteInterpolationMutator:
    """精英插值变异器。

    用法示例
    --------
    >>> mutator = EliteInterpolationMutator(InterpolationConfig(noise_std=0.02))
    >>> offspring = mutator.mutate(elite_pool, n_offspring=10)

    参数
    ----
    config : InterpolationConfig
        超参数配置，含噪声强度、随机种子等。
    """

    def __init__(self, config: Optional[InterpolationConfig] = None) -> None:
        self.config = config or InterpolationConfig()
        self._rng = torch.Generator()
        if self.config.seed is not None:
            self._rng.manual_seed(self.config.seed)

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def mutate(
        self,
        elite_pool: Sequence[EliteIndividual],
        n_offspring: int,
    ) -> List[torch.Tensor]:
        """从精英集合生成 n_offspring 个变异子代。

        参数
        ----
        elite_pool   : 精英个体序列，至少 2 个（allow_self_crossover=True 时允许 1 个）。
        n_offspring  : 要生成的子代数量。

        返回
        ----
        List[torch.Tensor]: 变异后的奖励张量列表，顺序与生成顺序一致。
        """
        self._validate_pool(elite_pool)
        offspring: List[torch.Tensor] = []

        for i in range(n_offspring):
            ind_a, ind_b = self._select_parent_pair(elite_pool)
            R_mut = self._interpolate(ind_a, ind_b)
            offspring.append(R_mut)

        return offspring

    def mutate_with_info(
        self,
        elite_pool: Sequence[EliteIndividual],
        n_offspring: int,
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """同 mutate，但额外返回每对亲本的适应度权重，便于调试/记录。

        返回
        ----
        List of (R_mut, lam, noise_scale)：
          · R_mut      : 变异奖励
          · lam        : 插值权重（偏向 ind_a 的程度）
          · noise_scale: 实际使用的扰动标准差
        """
        self._validate_pool(elite_pool)
        results = []

        for _ in range(n_offspring):
            ind_a, ind_b = self._select_parent_pair(elite_pool)
            lam = self._compute_lambda(ind_a.fitness, ind_b.fitness)
            R_mut_base = lam * ind_a.reward + (1.0 - lam) * ind_b.reward
            epsilon = torch.zeros_like(R_mut_base).normal_(
                mean=0.0, std=self.config.noise_std, generator=self._rng
            )
            results.append((R_mut_base + epsilon, lam, self.config.noise_std))

        return results

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _validate_pool(self, elite_pool: Sequence[EliteIndividual]) -> None:
        min_size = 1 if self.config.allow_self_crossover else 2
        if len(elite_pool) < min_size:
            raise ValueError(
                f"elite_pool 至少需要 {min_size} 个精英个体，"
                f"当前只有 {len(elite_pool)} 个。"
            )

    def _select_parent_pair(
        self, elite_pool: Sequence[EliteIndividual]
    ) -> Tuple[EliteIndividual, EliteIndividual]:
        """无放回随机抽取两个亲本（pool_size=1 时退化为自交叉）。"""
        if len(elite_pool) == 1:
            return elite_pool[0], elite_pool[0]

        perm = torch.randperm(len(elite_pool), generator=self._rng)
        return elite_pool[perm[0].item()], elite_pool[perm[1].item()]

    def _compute_lambda(self, rho_a: float, rho_b: float) -> float:
        """计算适应度归一化插值权重 λ = ρ_a / (ρ_a + ρ_b)。"""
        denom = rho_a + rho_b + self.config.fitness_eps
        return rho_a / denom

    def _interpolate(
        self, ind_a: EliteIndividual, ind_b: EliteIndividual
    ) -> torch.Tensor:
        """执行单次插值变异。"""
        lam = self._compute_lambda(ind_a.fitness, ind_b.fitness)
        R_mut = lam * ind_a.reward + (1.0 - lam) * ind_b.reward
        epsilon = torch.zeros_like(R_mut).normal_(
            mean=0.0, std=self.config.noise_std, generator=self._rng
        )
        return R_mut + epsilon
