from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.import_utils import ModuleSpec


@dataclass(frozen=True)
class AlgorithmConfig:
    """主流程超参数。"""

    M: int  # 种群规模
    N: int  # 智能体数量
    K: int  # 进化迭代次数


@dataclass(frozen=True)
class EliteConfig:
    """精英选择相关配置（选择策略由用户模块实现）。"""

    elite_count: int


@dataclass(frozen=True)
class TruncatedDiffusionConfig:
    """截断扩散变异的参数（由扩散模型实现解释/使用）。"""

    add_noise_steps: int
    denoise_steps: int


@dataclass(frozen=True)
class LoggingConfig:
    """日志/记录相关配置。"""

    best_rho_csv_path: str  # 每次进化迭代记录“最优ρ”的 CSV 路径
    initial_population_path: str | None  # 初始种群缓存文件路径


@dataclass(frozen=True)
class MasDiffConfig:
    """
    一套完整模块配置（一个 YAML 对应一套）。

    注意：这里的每个 ModuleSpec 都指向“用户自定义实现”的具体类；
    框架只提供 base 接口与主流程调度。
    """

    seed: int
    algorithm: AlgorithmConfig

    # 可自定义模块（用户提供 class_path + kwargs）
    q_provider: ModuleSpec
    environment: ModuleSpec
    dqn_module: ModuleSpec
    diffusion_model: ModuleSpec
    elite_selector: ModuleSpec
    metric: ModuleSpec
    parallel_executor: ModuleSpec

    # 进化/变异相关配置（由主流程使用；具体行为由模块实现）
    elite: EliteConfig
    truncated_diffusion: TruncatedDiffusionConfig

    # 记录/日志相关配置
    logging: LoggingConfig

    # 额外保留字段（给用户扩展/透传用）
    extra: dict[str, Any]

