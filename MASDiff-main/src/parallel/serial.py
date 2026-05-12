from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from src.parallel.base import ParallelExecutor

T = TypeVar("T")
R = TypeVar("R")


class SerialExecutor(ParallelExecutor):
    """默认串行执行器（满足“并行接口”但不并行）。"""

    def __init__(
        self,
        *,
        show_progress: bool = True,
        desc: str = "SerialExecutor",
        max_updates: int = 20,
    ) -> None:
        """
        参数：
        - show_progress: 是否输出进度
        - desc: 输出前缀
        - max_updates: 最多输出多少次进度（避免刷屏）
        """
        self.show_progress = bool(show_progress)
        self.desc = str(desc)
        self.max_updates = int(max_updates)

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        # 尽量获取长度以便输出进度；主流程里传入的通常是 list(range(...)) 或 list(elites)
        if isinstance(items, (list, tuple)):
            seq = list(items)
        else:
            seq = list(items)

        total = len(seq)
        if total == 0:
            return []

        results: list[R] = []
        if not self.show_progress:
            for x in seq:
                results.append(fn(x))
            return results

        # 控制输出频率：最多 max_updates 次（含最后一次）
        max_updates = max(1, self.max_updates)
        step = max(1, total // max_updates)

        print(f"{self.desc}: 0/{total}")
        for i, x in enumerate(seq, start=1):
            results.append(fn(x))
            if i == total or (i % step == 0):
                print(f"{self.desc}: {i}/{total}")
        return results

