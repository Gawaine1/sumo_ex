from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from src.environments.base import Environment


class QProvider(ABC):
    """
    Q（目标/标签/参考轨迹等）的读取或创建接口。

    对应主流程第 1 步：
    - 读取或创建 Q（如果没有就创建，创建函数用户自定义）
    """

    @abstractmethod
    def load_or_create_q(self, environment: Environment) -> Any:
        """
        返回 Q（具体结构由用户自定义）。

        返回值用途：
        - 用于后续评价指标 ρ 的计算（例如与仿真数据的 MSE 等）
        """

