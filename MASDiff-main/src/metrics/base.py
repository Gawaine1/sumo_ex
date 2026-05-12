from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """
    ρ 评价指标抽象接口（用户自定义实现）。

    主流程中 ρ 越大越好，可理解为适应度。
    例如：与 Q 的 MSE 经过某种变换（需要相应处理）。
    """

    @abstractmethod
    def compute_rho(self, q: Any, simulation_data: Any) -> float:
        """
        计算 ρ。

        返回值用途：
        - 用于精英选择（5.2）
        - 用于保留 ρ 最大的 M 个个体形成新种群（5.5）
        """

