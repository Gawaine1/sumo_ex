from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.dqn.base import DqnModule
from src.diffusion.base import DiffusionModel
from src.environments.base import Environment
from src.evolution.base import EliteSelector
from src.metrics.base import Metric
from src.pipeline.types import Individual, Population
from src.q.base import QProvider


# =========================
# 1. 读取或创建 Q
# =========================
def step_1_load_or_create_q(q_provider: QProvider, environment: Environment) -> Any:
    """
    主流程第 1 步：
    1.读取或创建 Q（这是目标，如果没有就创建，创建函数用户自定义）
    """
    q = q_provider.load_or_create_q(environment)
    return q


# =========================
# 2. 随机初始化扩散模型
# =========================
def step_2_init_diffusion_model(diffusion: DiffusionModel) -> DiffusionModel:
    """
    主流程第 2 步：
    2.随机初始化扩散模型
    """
    diffusion.init_random()
    return diffusion


# =========================
# 3. 随机初始化策略（DQN）
# =========================
def step_3_init_random_policies(dqn: DqnModule, *, num_agents: int) -> list[Any]:
    """
    主流程第 3 步：
    3.随机初始化策略（dqn），数量：M*N（M是种群规模，N是智能体数量）
    """
    policies = dqn.init_random_policies(num_agents)
    return policies


# =========================
# 4. 形成初始种群：单个个体的构建步骤
# =========================
def step_4_1_simulate_collect(env: Environment, policies: list[Any]) -> tuple[list[list[Any]], Any]:
    """
    主流程第 4.1 步：
    4.1 用随机初始化策略（dqn）仿真一次，收集每个智能体的经验库（奖励留空）和 Tau
    """
    experience_buffers, tau = env.simulate_collect(policies)
    return experience_buffers, tau


def step_4_2_generate_reward(diffusion: DiffusionModel, tau: Any) -> Any:
    """
    主流程第 4.2 步：
    4.2 把 Tau 作为扩散模型的条件生成奖励 R
    """
    rewards = diffusion.generate_reward(tau)
    return rewards


def step_4_3_build_dqn_training_data(dqn: DqnModule, experience_buffers: list[list[Any]], rewards: Any) -> Any:
    """
    主流程第 4.3 步：
    4.3 将经验库和 R 合并形成 dqn 训练数据（用奖励 R 填充经验中的 r）
    """
    training_data = dqn.build_training_data(experience_buffers, rewards)
    return training_data


def step_4_4_train_dqn_per_agent(dqn: DqnModule, training_data: Any) -> list[Any]:
    """
    主流程第 4.4 步：
    4.4 为每个智能体训练一个 dqn
    """
    trained_policies = dqn.train_per_agent(training_data)
    return trained_policies


def step_4_5_simulate_and_compute_rho(
    env: Environment,
    policies: list[Any],
    *,
    q: Any,
    metric: Metric,
) -> tuple[Any, float]:
    """
    主流程第 4.5 步：
    4.5 再用训练好的 dqn 仿真一次，得到仿真数据，计算 ρ（越大越好）
    """
    simulation_data = env.simulate_evaluate(policies)
    rho = metric.compute_rho(q, simulation_data)
    return simulation_data, float(rho)


def step_4_build_individual(
    *,
    tau: Any,
    rewards: Any,
    rho: float,
    experience_buffers: list[list[Any]],
    policies: list[Any],
) -> Individual:
    """
    汇总第 4 步结果形成一个个体：
    (Tau, R, ρ) + 保存经验库与 DQN（策略）。
    """
    return Individual(
        tau=tau,
        rewards=rewards,
        rho=rho,
        experience_buffers=experience_buffers,
        policies=policies,
        metadata={},
    )


# =========================
# 5. 进化迭代：K 次
# =========================
def step_5_1_train_diffusion_with_population(diffusion: DiffusionModel, population: Population) -> DiffusionModel:
    """
    主流程第 5.1 步：
    5.1 用当前的种群训练扩散模型
    """
    diffusion.train_on_population(population)
    return diffusion


def step_5_2_select_elite_population(selector: EliteSelector, population: Population, *, elite_count: int) -> Population:
    """
    主流程第 5.2 步：
    5.2 根据种群个体的 ρ 选择精英种群
    """
    elites = selector.select_elites(population, elite_count=elite_count)
    return list(elites)


def step_5_3_1_truncated_diffusion_mutate_reward(
    diffusion: DiffusionModel,
    tau: Any,
    base_rewards: Any,
    *,
    add_noise_steps: int,
    denoise_steps: int,
) -> Any:
    """
    主流程第 5.3.1 步：
    5.3.1 把 Tau 作为条件，然后使用截断扩散生成 R（奖励变异）

    说明：
    - base_rewards 来自精英个体原有的 R（用于“围绕精英 R 做变异”）
    """
    mutated_rewards = diffusion.generate_reward_truncated(
        tau,
        base_rewards,
        add_noise_steps=add_noise_steps,
        denoise_steps=denoise_steps,
    )
    return mutated_rewards


def step_5_3_2_build_dqn_training_data(dqn: DqnModule, experience_buffers: list[list[Any]], rewards: Any) -> Any:
    """
    主流程第 5.3.2 步：
    5.3.2 将经验库和 R 合并形成 dqn 训练数据（用奖励 R 填充经验中的 r）
    """
    return step_4_3_build_dqn_training_data(dqn, experience_buffers, rewards)


def step_5_3_3_train_dqn_per_agent(dqn: DqnModule, training_data: Any) -> list[Any]:
    """
    主流程第 5.3.3 步：
    5.3.3 为每个智能体训练一个 dqn
    """
    return step_4_4_train_dqn_per_agent(dqn, training_data)


def step_5_3_4_simulate_collect(env: Environment, policies: list[Any]) -> tuple[list[list[Any]], Any]:
    """
    主流程第 5.3.4 步：
    5.3.4 用训练的 dqn 仿真，收集经验库（奖励留空）和 Tau
    """
    return step_4_1_simulate_collect(env, policies)


def step_5_3_5_generate_reward(diffusion: DiffusionModel, tau: Any) -> Any:
    """
    主流程第 5.3.5 步：
    5.3.5 把 Tau 作为扩散模型的条件生成奖励 R
    """
    return step_4_2_generate_reward(diffusion, tau)


def step_5_3_6_build_dqn_training_data(dqn: DqnModule, experience_buffers: list[list[Any]], rewards: Any) -> Any:
    """
    主流程第 5.3.6 步：
    5.3.6 将经验库和 R 合并形成 dqn 训练数据（用奖励 R 填充经验中的 r）
    """
    return step_4_3_build_dqn_training_data(dqn, experience_buffers, rewards)


def step_5_3_7_train_dqn_per_agent(dqn: DqnModule, training_data: Any) -> list[Any]:
    """
    主流程第 5.3.7 步：
    5.3.7 为每个智能体训练一个 dqn
    """
    return step_4_4_train_dqn_per_agent(dqn, training_data)


def step_5_3_8_simulate_and_compute_rho(
    env: Environment,
    policies: list[Any],
    *,
    q: Any,
    metric: Metric,
) -> tuple[Any, float]:
    """
    主流程第 5.3.8 步：
    5.3.8 再用训练好的 dqn 仿真一次，得到仿真数据，计算 ρ
    """
    return step_4_5_simulate_and_compute_rho(env, policies, q=q, metric=metric)


def step_5_4_add_mutants_to_population(population: Population, mutants: Population) -> Population:
    """
    主流程第 5.4 步：
    5.4 将变异后的精英种群加入原有种群
    """
    return list(population) + list(mutants)


def step_5_5_keep_top_m(population: Population, *, M: int) -> Population:
    """
    主流程第 5.5 步：
    5.5 保留 ρ 最大的 M 个个体，形成新种群
    """
    # ρ 越大越好
    sorted_pop = sorted(population, key=lambda ind: float(ind.rho), reverse=True)
    return sorted_pop[:M]


def step_5_6_record_best_rho(population: Population, *, iteration_k: int, csv_path: str) -> tuple[Population, float]:
    """
    主流程第 5.6 步：
    5.6 记录种群最好的 ρ 值，并把每次进化迭代种群最好的 ρ 值追加写入 CSV 文件。

    返回值：
    - population: 原样返回（用于继续主流程的 population 变量）
    - best_rho: 本次迭代新种群中的最优 ρ（越大越好），用于写入 CSV，也可用于外部监控
    """
    if len(population) == 0:
        raise ValueError("population 不能为空，无法记录 best_rho")

    best_rho = max(float(ind.rho) for ind in population)

    p = Path(csv_path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)

    write_header = (not p.exists()) or (p.stat().st_size == 0)
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["iteration_k", "best_rho"])
        writer.writerow([iteration_k, best_rho])

    return population, best_rho


def step_clone_elite_for_mutation(elite: Individual) -> Individual:
    """
    可选辅助：克隆精英个体（便于实现“从精英个体出发做变异”的语义）。
    这里做浅拷贝；具体深拷贝策略由用户数据结构决定。
    """
    return replace(elite)

