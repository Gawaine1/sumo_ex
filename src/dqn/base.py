from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DqnModule(ABC):
    """
    DQN 模块抽象接口（用户自定义实现）。

    主流程要求：
    - 随机初始化策略（数量 M*N）
    - 用经验库 + 奖励 R 形成训练数据（把 r 填进去）
    - 为每个智能体训练一个 DQN，得到可执行策略
    """

    @abstractmethod
    def init_random_policies(self, num_agents: int) -> list[Any]:
        """
        随机初始化 num_agents 个策略。

        返回值用途：
        - 作为环境仿真输入，收集经验库与 Tau（4.1）
        """

    @abstractmethod
    def build_training_data(self, experience_buffers: list[list[Any]], rewards: Any) -> Any:
        """
        将经验库和奖励 R 合并形成 DQN 训练数据（用 rewards 填充经验中的 r）。

        返回值用途：
        - 输入到 train_per_agent()，训练每个智能体的 DQN（4.4 / 5.3.3 / 5.3.7）
        """

    @abstractmethod
    def train_per_agent(self, training_data: Any) -> list[Any]:
        """
        为每个智能体训练一个 DQN，返回训练好的策略列表（长度 N）。

        返回值用途：
        - 用训练好的策略进行仿真（4.5 / 5.3.4 / 5.3.8 等）
        """

