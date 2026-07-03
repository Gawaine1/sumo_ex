from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    """
    种群中的一个个体。

    按 `主流程.txt`：
    - (Tau, 对应的 R, ρ) 是一个个体
    - 同时需要把经验库与训练出来的 DQN（策略）也保存一下
    """

    tau: Any
    rewards: Any
    rho: float

    # 额外保存（便于后续变异/再训练/复现）
    experience_buffers: list[list[Any]] = field(default_factory=list)
    policies: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


Population = list[Individual]

