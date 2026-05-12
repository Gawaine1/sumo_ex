from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DiffusionModel(ABC):
    """
    扩散模型抽象接口（用户自定义实现）。

    主流程要求：
    - 随机初始化扩散模型（2）
    - 以 Tau 为条件生成奖励 R（4.2 / 5.3.5）
    - 用当前种群训练扩散模型（5.1）
    - 截断扩散生成 R（对精英个体做变异）（5.3.1）
    """

    @abstractmethod
    def init_random(self) -> None:
        """随机初始化扩散模型参数（无返回值，内部就地初始化）。"""

    @abstractmethod
    def train_on_population(self, population: list[Any]) -> None:
        """
        用当前种群训练扩散模型（5.1）。

        population 中每个元素建议包含 (Tau, R, ρ)（框架内为 Individual，但也允许用户自定义解析）。
        """

    @abstractmethod
    def generate_reward(self, tau: Any) -> Any:
        """
        把 Tau 作为条件生成奖励 R（4.2 / 5.3.5）。

        返回值用途：
        - 与经验库合并构建 DQN 训练数据（4.3 / 5.3.6）
        """

    @abstractmethod
    def generate_reward_truncated(
        self,
        tau: Any,
        base_rewards: Any,
        *,
        add_noise_steps: int,
        denoise_steps: int,
    ) -> Any:
        """
        截断扩散生成奖励 R（精英个体奖励的变异）（5.3.1）。

        注意：为保证“围绕精英个体原有 R 做变异”的语义，
        该接口需要同时接收 base_rewards（精英个体的原奖励/权重矩阵）。

        返回值用途：
        - 与经验库合并构建 DQN 训练数据（5.3.2）
        """

