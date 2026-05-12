from __future__ import annotations

import importlib
from typing import Any

try:
    torch = importlib.import_module("torch")
    F = importlib.import_module("torch.nn.functional")
except Exception as e:  # pragma: no cover
    raise ImportError(
        "无法导入 torch。`src/evolution/temperature_selection.py` 需要 PyTorch 来做 softmax/multinomial。"
    ) from e

from src.evolution.base import EliteSelector


class TemperatureEliteSelector(EliteSelector):
    """
    温度采样的精英选择策略（参考 CrowdNav_Re/run_dqn_es.py:950-954）。

    思路：
    - 用个体的 ρ 作为 score（ρ 越大越好）
    - probs = softmax(temperature * scores)
    - 用 multinomial 从 probs 采样 elite_count 个索引（可选是否放回）
    """

    def __init__(self, *, temperature: float = 1.0, replacement: bool = True) -> None:
        self.temperature = float(temperature)
        self.replacement = bool(replacement)

    def select_elites(self, population: list[Any], *, elite_count: int) -> list[Any]:
        if elite_count <= 0:
            raise ValueError("elite_count 必须 > 0")
        if len(population) == 0:
            raise ValueError("population 不能为空")

        # scores 直接用 rho（越大越好）
        scores = torch.tensor([float(getattr(ind, "rho")) for ind in population], dtype=torch.float32)

        # temperature * scores -> softmax
        probs = F.softmax(self.temperature * scores, dim=0)

        # multinomial 采样
        k = int(elite_count)
        replacement = self.replacement
        if (not replacement) and k > len(population):
            # 不放回时最多只能采样 len(population) 个
            k = len(population)

        selected_indices = torch.multinomial(probs, k, replacement=replacement).tolist()
        return [population[i] for i in selected_indices]

