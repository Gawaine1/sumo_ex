from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EliteSelector(ABC):
    """
    精英选择策略抽象接口（用户自定义实现）。

    对应主流程第 5.2 步：
    - 根据种群个体的 ρ 选择精英种群（选多少、如何选由实现与 yaml 配置决定）
    """

    @abstractmethod
    def select_elites(self, population: list[Any], *, elite_count: int) -> list[Any]:
        """
        从 population 中选择 elite_count 个精英个体。

        返回值用途：
        - 进入 5.3 循环，对精英个体做截断扩散变异与再训练评估
        """

