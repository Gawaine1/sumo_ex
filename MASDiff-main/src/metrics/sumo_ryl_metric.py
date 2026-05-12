from __future__ import annotations

import importlib
from typing import Any, Optional

try:
    torch = importlib.import_module("torch")
except Exception as e:  # pragma: no cover
    raise ImportError(
        "无法导入 torch。`src/metrics/sumo_ryl_metric.py` 需要 PyTorch 来计算张量 MSE。"
    ) from e

from src.metrics.base import Metric


class SumoRylMetric(Metric):
    """
    sumo_ryl 的 ρ 计算：
    1) 对 q 和 simulation_data 做 MSE（越小越好）
    2) 通过单调变换得到 ρ（越大越好）

    默认变换：rho = 1 / (1 + mse)  （范围 (0, 1]，数值更稳定）

    说明：
    - 期望 q 与 simulation_data 形状一致，例如：
      - q: [100, 114]
      - simulation_data: [100, 114]
    """

    def __init__(
        self,
        *,
        transform: str = "inv1p",
        eps: float = 1e-8,
        temperature: float = 1.0,
        rho_scale: float = 1.0,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        self.transform = str(transform).lower()
        self.eps = float(eps)
        self.temperature = float(temperature)
        self.rho_scale = float(rho_scale)
        self.clip_min = clip_min if clip_min is None else float(clip_min)
        self.clip_max = clip_max if clip_max is None else float(clip_max)

    def compute_rho(self, q: Any, simulation_data: Any) -> float:
        q_t = self._to_tensor(q, name="q")
        s_t = self._to_tensor(simulation_data, name="simulation_data")
        q_t, s_t = self._align_shapes(q_t, s_t)

        if q_t.shape != s_t.shape:
            raise ValueError(f"q 与 simulation_data 形状必须一致，但得到 q={tuple(q_t.shape)}，sim={tuple(s_t.shape)}")

        mse = torch.mean((q_t - s_t) ** 2)

        # MSE(越小越好) -> rho(越大越好)
        if self.transform == "neg":
            rho = -mse
        elif self.transform == "inv":
            rho = 1.0 / (mse + max(self.eps, 1e-12))
        elif self.transform == "inv1p":
            rho = 1.0 / (1.0 + mse)
        elif self.transform == "exp":
            temp = max(self.temperature, 1e-12)
            rho = torch.exp(-mse / temp)
        else:
            raise ValueError(f"未知 transform={self.transform}(支持:neg/inv/inv1p/exp)")

        rho = rho * self.rho_scale
        if self.clip_min is not None or self.clip_max is not None:
            rho = torch.clamp(
                rho,
                min=self.clip_min if self.clip_min is not None else -float("inf"),
                max=self.clip_max if self.clip_max is not None else float("inf"),
            )

        return float(rho.detach().cpu().item())

    def _to_tensor(self, x: Any, *, name: str) -> "torch.Tensor":
        if isinstance(x, torch.Tensor):
            t = x.detach().float().cpu()
        else:
            # 支持 list / numpy 等
            t = torch.tensor(x, dtype=torch.float32)
        if t.numel() == 0:
            raise ValueError(f"{name} 不能为空")
        return t

    def _align_shapes(self, q_t: "torch.Tensor", s_t: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        兼容只差 singleton 维度的情况，例如:
        - q: [1651]
        - simulation_data: [1, 1651]
        """
        if q_t.shape == s_t.shape:
            return q_t, s_t

        q_squeezed = q_t.squeeze()
        s_squeezed = s_t.squeeze()
        if q_squeezed.shape == s_squeezed.shape:
            return q_squeezed, s_squeezed

        return q_t, s_t

