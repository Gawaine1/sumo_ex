from __future__ import annotations

import random
from typing import Any

from src.pipeline.types import Individual
from src.pipeline.steps import (
    step_3_init_random_policies,
    step_4_1_simulate_collect,
    step_4_2_generate_reward,
    step_4_3_build_dqn_training_data,
    step_4_4_train_dqn_per_agent,
    step_4_5_simulate_and_compute_rho,
    step_4_build_individual,
    step_5_3_1_truncated_diffusion_mutate_reward,
    step_5_3_2_build_dqn_training_data,
    step_5_3_3_train_dqn_per_agent,
    step_5_3_4_simulate_collect,
    step_5_3_5_generate_reward,
    step_5_3_6_build_dqn_training_data,
    step_5_3_7_train_dqn_per_agent,
    step_5_3_8_simulate_and_compute_rho,
)
from src.utils.import_utils import ModuleSpec, instantiate


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


def build_initial_individual(task: dict[str, Any]) -> Individual:
    """
    Ray task：构建初始种群的一个个体（对应 runner.py 中 build_initial_individual 的语义）。

    task 字段（由 runner 组装）：
    - i: 个体索引（仅用于 metadata）
    - N: 智能体数量
    - seed: 可选，任务随机种子
    - q_ref: ray ObjectRef（Q 可能很大，用 object store 传）
    - diffusion_state_ref: ray ObjectRef（扩散模型参数 state_dict）
    - environment_spec/dqn_module_spec/diffusion_model_spec/metric_spec: ModuleSpec（用于 worker 内实例化）
    """
    _maybe_seed(task.get("seed"))

    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ray worker 中无法导入 ray。") from e

    i = int(task["i"])
    N = int(task["N"])

    q = ray.get(task["q_ref"])
    diffusion_state = ray.get(task["diffusion_state_ref"])

    # 注意：这里刻意复用 steps.py 的函数，以确保 Ray 并行路径与 runner.py 串行路径语义一致。
    env = instantiate(task["environment_spec"])
    dqn = instantiate(task["dqn_module_spec"])
    diffusion = instantiate(task["diffusion_model_spec"])
    metric = instantiate(task["metric_spec"])

    load_diffusion_state(diffusion, diffusion_state)

    # =========================
    # 3-4. 构建初始个体（对应 runner.py 的 build_initial_individual）
    # =========================
    # 3) 随机初始化策略（DQN），为该个体初始化 N 个智能体策略
    init_policies = step_3_init_random_policies(dqn, num_agents=N)

    # 4.1) 用随机策略仿真一次，收集经验库与 Tau（奖励留空）
    experience_buffers, tau = step_4_1_simulate_collect(env, init_policies)

    # 4.2) 把 Tau 作为扩散模型条件生成奖励 R
    rewards = step_4_2_generate_reward(diffusion, tau)

    # 4.3) 将经验库与 R 合并形成 DQN 训练数据（用 rewards 填充经验中的 r）
    training_data = step_4_3_build_dqn_training_data(dqn, experience_buffers, rewards)

    # 4.4) 为每个智能体训练一个 DQN（得到策略列表）
    trained_policies = step_4_4_train_dqn_per_agent(dqn, training_data)

    # 4.5) 用训练好的策略再仿真一次，得到 simulation_data 并计算 ρ
    simulation_data, rho = step_4_5_simulate_and_compute_rho(env, trained_policies, q=q, metric=metric)

    # 关键：不要把 policies 放进 Individual（巨大且后续未使用，会导致 Ray/内存爆炸）
    ind = step_4_build_individual(
        tau=tau,
        rewards=rewards,
        rho=float(rho),
        experience_buffers=experience_buffers,
        policies=[],
    )
    ind.metadata["initial_index"] = i
    ind.metadata["simulation_data"] = simulation_data
    return ind


def mutate_one(task: dict[str, Any]) -> Individual:
    """
    Ray task：对一个精英个体做变异（对应 runner.py 中 mutate_one 的语义）。

    task 字段：
    - elite: Individual（注意：runner 已确保 elite.policies 为空，避免巨大传输）
    - iteration_k: 当前迭代 k
    - seed: 可选
    - q_ref: ray ObjectRef
    - diffusion_state_ref: ray ObjectRef（当前迭代训练后的扩散参数）
    - truncated_diffusion: dict(add_noise_steps, denoise_steps)
    - environment_spec/dqn_module_spec/diffusion_model_spec/metric_spec: ModuleSpec
    """
    _maybe_seed(task.get("seed"))

    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Ray worker 中无法导入 ray。") from e

    elite: Individual = task["elite"]
    k = int(task["iteration_k"])
    trunc = dict(task.get("truncated_diffusion") or {})
    add_noise_steps = int(trunc.get("add_noise_steps", 1))
    denoise_steps = int(trunc.get("denoise_steps", 1))

    q = ray.get(task["q_ref"])
    diffusion_state = ray.get(task["diffusion_state_ref"])

    env = instantiate(task["environment_spec"])
    dqn = instantiate(task["dqn_module_spec"])
    diffusion = instantiate(task["diffusion_model_spec"])
    metric = instantiate(task["metric_spec"])
    load_diffusion_state(diffusion, diffusion_state)

    # =========================
    # 5.3 精英个体变异（对应 runner.py 的 mutate_one）
    # =========================
    # 5.3.1) Tau 条件下做“截断扩散”生成变异奖励 R'
    mutated_rewards = step_5_3_1_truncated_diffusion_mutate_reward(
        diffusion,
        elite.tau,
        elite.rewards,
        add_noise_steps=add_noise_steps,
        denoise_steps=denoise_steps,
    )

    # 5.3.2) 用“精英个体保存的经验库”与变异奖励合并形成训练数据
    training_data_1 = step_5_3_2_build_dqn_training_data(dqn, elite.experience_buffers, mutated_rewards)

    # 5.3.3) 为每个智能体训练一个 DQN（得到一版策略）
    trained_policies_1 = step_5_3_3_train_dqn_per_agent(dqn, training_data_1)

    # 5.3.4) 用该策略仿真，重新收集经验库（奖励留空）与新的 Tau
    experience_buffers_2, tau_2 = step_5_3_4_simulate_collect(env, trained_policies_1)

    # 5.3.5) 把新 Tau 作为条件生成奖励 R
    rewards_2 = step_5_3_5_generate_reward(diffusion, tau_2)

    # 5.3.6) 将新经验库与新奖励合并形成训练数据
    training_data_2 = step_5_3_6_build_dqn_training_data(dqn, experience_buffers_2, rewards_2)

    # 5.3.7) 再训练每个智能体的 DQN（得到最终策略）
    trained_policies_2 = step_5_3_7_train_dqn_per_agent(dqn, training_data_2)

    # 5.3.8) 用最终策略仿真并计算 ρ
    simulation_data_2, rho_2 = step_5_3_8_simulate_and_compute_rho(env, trained_policies_2, q=q, metric=metric)

    # 汇总形成变异个体（同样不保存 policies，避免巨大对象在 Ray 中传输/落盘）
    mutant = step_4_build_individual(
        tau=tau_2,
        rewards=rewards_2,
        rho=float(rho_2),
        experience_buffers=experience_buffers_2,
        policies=[],
    )
    mutant.metadata["iteration_k"] = k
    mutant.metadata["parent_rho"] = float(elite.rho)
    mutant.metadata["simulation_data"] = simulation_data_2
    return mutant

