from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import Any, Optional

try:
    torch = importlib.import_module("torch")
except Exception as e:  # pragma: no cover
    raise ImportError(
        "无法导入 torch。`src/q/sumo_ryl_q.py` 需要 PyTorch 来保存/加载 Q（张量）。"
    ) from e

from src.environments.base import Environment
from src.q.base import QProvider


class SumoRylQProvider(QProvider):
    """
    sumo_ryl 的 QProvider 实现。

    你的要求：
    - “仿真不用 dqn”：这里生成 Q 时，不提供任何可用策略，环境会走“只用 A*”的路径规划。
    - “q 的结构跟 simulation_data 一样”：因此直接用 environment.simulate_evaluate(...) 的输出作为 Q。

    返回的 Q（默认）：
    - torch.Tensor，shape = [end_tick/sample_interval, num_road]，例如 [100,114]
    """

    def __init__(
        self,
        *,
        num_car: int,
        cache_path: Optional[str] = "outputs/q_sumo_ryl.pt",
        force_recompute: bool = False,
    ) -> None:
        self.num_car = int(num_car)
        if self.num_car <= 0:
            raise ValueError("num_car 必须 > 0")
        self.cache_path = str(cache_path) if cache_path else None
        self.force_recompute = bool(force_recompute)

    def load_or_create_q(self, environment: Environment) -> Any:
        # 1) 若有缓存且不强制重算，则直接加载
        if self.cache_path and (not self.force_recompute):
            pkl = Path(self.cache_path)
            if pkl.exists() and pkl.stat().st_size > 0:
                try:
                    with pkl.open("rb") as f:
                        data = pickle.load(f)

                    if torch.is_tensor(data):
                        q = data.detach().cpu()
                    else:
                        q = torch.as_tensor(data, dtype=torch.float32)

                    print(f"加载q:{q}")
                    return q
                except Exception as e:
                    print(f"警告: 通过 pickle 加载缓存的 Q 失败，路径: {pkl}，将重算。错误信息: {e}")
                    # 缓存不可用时回退到重算
                    pass

        # 2) 生成 Q：仿真不用 dqn（只用 A*）
        #    关键点：传入 policies=[None]*num_car，使环境在规划时走“无 policy”分支
        policies = [None] * self.num_car
        q = environment.simulate_evaluate(policies)

        # 3) 可选写缓存
        if self.cache_path:
            pkl = Path(self.cache_path)
            if pkl.parent != Path("."):
                pkl.parent.mkdir(parents=True, exist_ok=True)

            q_to_dump = q.detach().cpu() if torch.is_tensor(q) else q
            with pkl.open("wb") as f:
                pickle.dump(q_to_dump, f)

        return q

