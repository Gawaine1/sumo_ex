from __future__ import annotations

import random
import time
from typing import Any

from src.pipeline.types import Individual
from src.pipeline.steps import (
    step_3_init_random_rewards,
    step_4_evaluate_only_astar_reward,
    step_4_build_only_astar_runtime_individual,
    step_5_3_2_simulate_collect,
    step_5_3_3_generate_reward,
    step_5_3_4_simulate_and_compute_rho,
)
from src.utils.import_utils import ModuleSpec, instantiate


def _maybe_cleanup_cuda_cache() -> None:
    """Best-effort cleanup of worker-side Python and CUDA cached memory."""
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def extract_diffusion_state(diffusion_model: Any) -> dict[str, Any]:
    """
    从扩散模型中提取“可广播到 Ray worker”的轻量状态。

    目前主要支持：
    - SumoRylDiffusionModel: 保存 diffusion_model.net.state_dict()
    - 其它实现：若提供 state_dict() 也会尝试使用
    """
    # 优先：有 net（SumoRylDiffusionModel）
    net = getattr(diffusion_model, "net", None)
    if net is not None and hasattr(net, "state_dict") and callable(getattr(net, "state_dict")):
        return {"kind": "net_state_dict", "net": net.state_dict()}

    # 退化：直接 state_dict（若用户实现提供）
    if hasattr(diffusion_model, "state_dict") and callable(getattr(diffusion_model, "state_dict")):
        return {"kind": "state_dict", "state": diffusion_model.state_dict()}

    raise TypeError("扩散模型不支持提取 state_dict（无法在 Ray worker 中复现同一模型参数）。")


def load_diffusion_state(diffusion_model: Any, state: dict[str, Any]) -> None:
    kind = state.get("kind")
    if kind == "net_state_dict":
        net = getattr(diffusion_model, "net", None)
        if net is None or (not hasattr(net, "load_state_dict")):
            raise TypeError("目标扩散模型没有 net.load_state_dict，无法加载广播参数。")
        net.load_state_dict(state["net"])
        return None
    if kind == "state_dict":
        if not hasattr(diffusion_model, "load_state_dict"):
            raise TypeError("目标扩散模型没有 load_state_dict，无法加载广播参数。")
        diffusion_model.load_state_dict(state["state"])
        return None
    raise ValueError(f"未知 diffusion state kind={kind}")


def _maybe_seed(seed: int | None) -> None:
    if seed is None:
        return None
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
    return None


def _resolve_ray_value(value: Any) -> Any:
    """Return the payload itself when Ray already auto-resolved the ObjectRef."""
    try:
        import ray  # type: ignore
    except Exception:
        return value

    object_ref_type = getattr(ray, "ObjectRef", None)
    if object_ref_type is not None and isinstance(value, object_ref_type):
        return ray.get(value)
    return value


def _prepare_only_astar_context(task: dict[str, Any]) -> dict[str, Any]:
    # 支持串行执行：优先使用 q，如果没有则尝试 q_ref
    q_value = task.get("q")
    if q_value is None:
        q_value = _resolve_ray_value(task.get("q_ref"))
    return {
        "environment": instantiate(task["environment_spec"]),
        "metric": instantiate(task["metric_spec"]),
        "q": q_value,
    }


def build_initial_individual(task: dict[str, Any]) -> Individual:
    """
    构建初始种群的一个个体（支持 Ray 和串行执行）。

    task 字段（由 runner 组装）：
    - i: 个体索引（仅用于 metadata）
    - N: 智能体数量
    - seed: 可选，任务随机种子
    - q: Q 值（串行执行时直接传递）
    - q_ref: ray ObjectRef（Ray 执行时使用）
    - environment_spec/metric_spec: ModuleSpec（用于 worker 内实例化）
    """
    _maybe_seed(task.get("seed"))

    # 不再强制要求 ray 模块，支持串行执行
    i = int(task["i"])
    N = int(task["N"])

    result: Individual | None = None
    try:
        ctx = _prepare_only_astar_context(task)
        q = ctx["q"]
        env = ctx["environment"]
        metric = ctx["metric"]

        initial_rewards = step_3_init_random_rewards(
            q=q,
            num_agents=N,
            reward_min=float(task.get("reward_min", 0.0)),
            reward_max=float(task.get("reward_max", 5.0)),
        )

        result = step_4_evaluate_only_astar_reward(env, initial_rewards, q=q, metric=metric)
        return result
    finally:
        _maybe_cleanup_cuda_cache()


def evaluate_reward_candidate(task: dict[str, Any]) -> Individual:
    """评估一个候选奖励矩阵 R，返回 only-astar 运行时个体（支持 Ray 和串行执行）。"""
    _maybe_seed(task.get("seed"))

    # 不再强制要求 ray 模块，支持串行执行
    result: Individual | None = None
    try:
        ctx = _prepare_only_astar_context(task)
        q = ctx["q"]
        env = ctx["environment"]
        metric = ctx["metric"]
        rewards = task["rewards"]
        result = step_4_evaluate_only_astar_reward(env, rewards, q=q, metric=metric)
        return result
    finally:
        _maybe_cleanup_cuda_cache()


def evaluate_reward_batch(task: dict[str, Any]) -> list[Individual]:
    """批量评估候选奖励矩阵（支持 Ray 和串行执行）。"""
    _maybe_seed(task.get("seed"))

    # 不再强制要求 ray 模块，支持串行执行
    results: list[Individual] = []
    try:
        ctx = _prepare_only_astar_context(task)
        q = ctx["q"]
        env = ctx["environment"]
        metric = ctx["metric"]
        rewards_batch_payload = task.get("rewards_batch_ref", task.get("rewards_batch"))
        rewards_batch = list(_resolve_ray_value(rewards_batch_payload) or [])

        for rewards in rewards_batch:
            ind = step_4_evaluate_only_astar_reward(env, rewards, q=q, metric=metric)
            results.append(ind)
        return results
    finally:
        _maybe_cleanup_cuda_cache()


def train_diffusion_on_population(task: dict[str, Any]) -> dict[str, Any]:
    """
    在远程侧直接消费初始种群，执行 5.1 扩散训练（支持 Ray 和串行执行）。

    返回：
    - diffusion_state: 训练后的扩散模型参数 state_dict
    """
    _maybe_seed(task.get("seed"))

    # 不再强制要求 ray 模块，支持串行执行
    diffusion_state = _resolve_ray_value(task["diffusion_state_ref"])
    population_refs = list(task["population_refs"])

    diffusion = instantiate(task["diffusion_model_spec"])
    load_diffusion_state(diffusion, diffusion_state)

    # 支持串行执行：如果是串行执行，population_refs 可能已经是 population
    if population_refs and isinstance(population_refs[0], dict):
        population = population_refs
    else:
        # Ray 执行：需要 ray.get 解析
        try:
            import ray
            population = list(ray.get(population_refs))
        except Exception:
            # 回退：假设已经是 population
            population = population_refs

    diffusion.train_on_population(population)

    return {
        "diffusion_state": extract_diffusion_state(diffusion),
        "population_count": len(population),
    }


class OnlyAstarMutateActor:
    """Persistent Ray actor for only-astar mutation work."""

    def __init__(
        self,
        *,
        q_ref: Any,
        environment_spec: ModuleSpec,
        diffusion_model_spec: ModuleSpec,
        metric_spec: ModuleSpec,
        actor_index: int = 0,
    ) -> None:
        self.actor_index = int(actor_index)
        self.q = _resolve_ray_value(q_ref)
        self.env = instantiate(environment_spec)
        self.diffusion = instantiate(diffusion_model_spec)
        self.metric = instantiate(metric_spec)

    def set_diffusion_state(self, state: dict[str, Any]) -> None:
        load_diffusion_state(self.diffusion, state)

    def mutate(self, task: dict[str, Any]) -> Individual:
        _maybe_seed(task.get("seed"))

        elite_pool: list[Individual] = task["elite_pool"]
        k = int(task["iteration_k"])
        interpolation_cfg = dict(task.get("interpolation") or {})
        noise_std = float(interpolation_cfg.get("noise_std", 0.01))

        timings: dict[str, float] = {}
        t_mut_total = time.perf_counter()

        t = time.perf_counter()
        # 使用精英插值变异替代截断扩散
        mutated_rewards = self._interpolate_mutate_rewards(elite_pool, noise_std=noise_std)
        timings["5.3.1"] = time.perf_counter() - t

        t = time.perf_counter()
        _, tau_2 = step_5_3_2_simulate_collect(self.env, mutated_rewards)
        timings["5.3.2"] = time.perf_counter() - t

        t = time.perf_counter()
        rewards_2 = step_5_3_3_generate_reward(self.diffusion, tau_2)
        timings["5.3.3"] = time.perf_counter() - t

        t = time.perf_counter()
        simulation_data_2, rho_2 = step_5_3_4_simulate_and_compute_rho(
            self.env,
            rewards_2,
            q=self.q,
            metric=self.metric,
        )
        timings["5.3.4"] = time.perf_counter() - t

        mutant = step_4_build_only_astar_runtime_individual(
            tau=tau_2,
            rewards=rewards_2,
            rho=float(rho_2),
            simulation_data=simulation_data_2,
        )
        mutant.metadata["iteration_k"] = k
        mutant.metadata["parent_rho"] = float(max(ind.rho for ind in elite_pool))
        mutant.metadata["actor_index"] = self.actor_index
        mutant.metadata["timing_mutation"] = {"total": time.perf_counter() - t_mut_total, **timings}
        return mutant

    def _interpolate_mutate_rewards(self, elite_pool: list[Individual], noise_std: float = 0.01) -> Any:
        """精英插值变异：从精英池中随机抽取两个个体做适应度加权插值。"""
        if len(elite_pool) < 2:
            raise ValueError(f"精英插值变异需要至少 2 个精英个体，当前只有 {len(elite_pool)} 个")

        # 随机抽取两个精英
        idx_a, idx_b = random.sample(range(len(elite_pool)), 2)
        elite_a = elite_pool[idx_a]
        elite_b = elite_pool[idx_b]

        # 适应度加权插值系数
        rho_a = float(elite_a.rho)
        rho_b = float(elite_b.rho)
        denom = rho_a + rho_b
        if denom <= 0:
            lam = 0.5
        else:
            lam = rho_a / denom

        # 奖励插值
        try:
            import torch
            R_a = torch.tensor(elite_a.rewards, dtype=torch.float32)
            R_b = torch.tensor(elite_b.rewards, dtype=torch.float32)
            R_mut = lam * R_a + (1.0 - lam) * R_b
            # 添加小扰动防止种群塌缩
            epsilon = torch.randn_like(R_mut) * noise_std
            R_mut = R_mut + epsilon
            return R_mut
        except Exception:
            # 降级到 numpy 或纯 Python
            import numpy as np
            R_a = np.array(elite_a.rewards, dtype=np.float32)
            R_b = np.array(elite_b.rewards, dtype=np.float32)
            R_mut = lam * R_a + (1.0 - lam) * R_b
            epsilon = np.random.randn(*R_mut.shape) * noise_std
            R_mut = R_mut + epsilon
            return R_mut

    def close(self) -> None:
        self.q = None
        self.env = None
        self.diffusion = None
        self.metric = None
        _maybe_cleanup_cuda_cache()


def mutate_one(task: dict[str, Any]) -> Individual:
    """
    对精英池做插值变异（支持 Ray 和串行执行）。

    task 字段：
    - elite_pool: list[Individual]（精英个体池，至少 2 个）
    - iteration_k: 当前迭代 k
    - seed: 可选
    - q: Q 值（串行执行时直接传递）
    - q_ref: ray ObjectRef（Ray 执行时使用）
    - diffusion_state_ref: ray ObjectRef（当前迭代训练后的扩散参数）
    - interpolation: dict(noise_std)
    - environment_spec/diffusion_model_spec/metric_spec: ModuleSpec
    """
    _maybe_seed(task.get("seed"))

    # 不再强制要求 ray 模块，支持串行执行
    elite_pool: list[Individual] = task["elite_pool"]
    k = int(task["iteration_k"])
    interpolation_cfg = dict(task.get("interpolation") or {})
    noise_std = float(interpolation_cfg.get("noise_std", 0.01))

    result: Individual | None = None
    try:
        q = task.get("q")
        if q is None:
            q = _resolve_ray_value(task["q_ref"])
        diffusion_state = _resolve_ray_value(task["diffusion_state_ref"])

        env = instantiate(task["environment_spec"])
        diffusion = instantiate(task["diffusion_model_spec"])
        metric = instantiate(task["metric_spec"])
        load_diffusion_state(diffusion, diffusion_state)

        # =========================
        # 5.3 精英插值变异（对应 runner.py 的 mutate_one）
        # =========================
        # 5.3.1) 从精英池中随机抽取两个个体做适应度加权插值
        mutated_rewards = _interpolate_mutate_rewards(elite_pool, noise_std=noise_std)

        # 5.3.2) 用变异奖励 R' + 纯 A* 仿真，重新收集经验库与 Tau
        _, tau_2 = step_5_3_2_simulate_collect(env, mutated_rewards)

        # 5.3.3) 把新 Tau 作为条件重新生成奖励 R
        rewards_2 = step_5_3_3_generate_reward(diffusion, tau_2)

        # 5.3.4) 用新奖励 R + 纯 A* 仿真并计算 ρ
        simulation_data_2, rho_2 = step_5_3_4_simulate_and_compute_rho(env, rewards_2, q=q, metric=metric)

        # 变异后的运行时对象同样不再携带 experience_buffers / policies。
        result = step_4_build_only_astar_runtime_individual(
            tau=tau_2,
            rewards=rewards_2,
            rho=float(rho_2),
            simulation_data=simulation_data_2,
        )
        return result
    finally:
        _maybe_cleanup_cuda_cache()


def _interpolate_mutate_rewards(elite_pool: list[Individual], noise_std: float = 0.01) -> Any:
    """精英插值变异：从精英池中随机抽取两个个体做适应度加权插值。"""
    if len(elite_pool) < 2:
        raise ValueError(f"精英插值变异需要至少 2 个精英个体，当前只有 {len(elite_pool)} 个")

    # 随机抽取两个精英
    idx_a, idx_b = random.sample(range(len(elite_pool)), 2)
    elite_a = elite_pool[idx_a]
    elite_b = elite_pool[idx_b]

    # 适应度加权插值系数
    rho_a = float(elite_a.rho)
    rho_b = float(elite_b.rho)
    denom = rho_a + rho_b
    if denom <= 0:
        lam = 0.5
    else:
        lam = rho_a / denom

    # 奖励插值
    try:
        import torch
        R_a = torch.tensor(elite_a.rewards, dtype=torch.float32)
        R_b = torch.tensor(elite_b.rewards, dtype=torch.float32)
        R_mut = lam * R_a + (1.0 - lam) * R_b
        # 添加小扰动防止种群塌缩
        epsilon = torch.randn_like(R_mut) * noise_std
        R_mut = R_mut + epsilon
        return R_mut
    except Exception:
        # 降级到 numpy 或纯 Python
        import numpy as np
        R_a = np.array(elite_a.rewards, dtype=np.float32)
        R_b = np.array(elite_b.rewards, dtype=np.float32)
        R_mut = lam * R_a + (1.0 - lam) * R_b
        epsilon = np.random.randn(*R_mut.shape) * noise_std
        R_mut = R_mut + epsilon
        return R_mut

