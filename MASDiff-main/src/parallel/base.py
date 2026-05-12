from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ParallelExecutor(ABC):
    """
    并行/串行执行的抽象接口。

    主流程中两处明确“留出并行接口”：
    - 4. for i=1..M 形成初始种群
    - 5.3 遍历精英种群做变异
    """

    @abstractmethod
    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """对 items 应用 fn，返回结果列表（保持顺序）。"""

    def close(self) -> None:
        """可选：释放线程池/进程池等资源。"""
        return None

