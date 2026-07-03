from __future__ import annotations

from typing import Any

from src.config.schema import MasDiffConfig
from src.dqn.base import DqnModule
from src.diffusion.base import DiffusionModel
from src.environments.base import Environment
from src.evolution.base import EliteSelector
from src.metrics.base import Metric
from src.parallel.base import ParallelExecutor
from src.pipeline.steps import (
    step_1_load_or_create_q,
    step_2_init_diffusion_model,
    step_3_init_random_policies,
    step_4_1_simulate_collect,
    step_4_2_generate_reward,
    step_4_3_build_dqn_training_data,
    step_4_4_train_dqn_per_agent,
    step_4_5_simulate_and_compute_rho,
    step_4_build_individual,
    step_5_1_train_diffusion_with_population,
    step_5_2_select_elite_population,
    step_5_3_1_truncated_diffusion_mutate_reward,
    step_5_3_2_build_dqn_training_data,
    step_5_3_3_train_dqn_per_agent,
    step_5_3_4_simulate_collect,
    step_5_3_5_generate_reward,
    step_5_3_6_build_dqn_training_data,
    step_5_3_7_train_dqn_per_agent,
    step_5_3_8_simulate_and_compute_rho,
    step_5_4_add_mutants_to_population,
    step_5_5_keep_top_m,
    step_5_6_record_best_rho,
)
from src.pipeline.types import Individual, Population
from src.q.base import QProvider
from src.utils.import_utils import instantiate


def _ensure_instance(obj: Any, expected_type: type, *, name: str) -> None:
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"模块 `{name}` 需要是 {expected_type.__name__} 的实例（或其子类），"
            f"但得到 {type(obj).__name__}"
        )


def run_masdiff(cfg: MasDiffConfig) -> Population:
    """
    执行 MASDiff 主流程。

    说明：
    - 框架只调度主流程；所有具体实现由 cfg 中的 class_path 指向的用户模块提供。
    - 主流程每一步都封装为一个函数，runner 负责串联返回值。
    """

    # ---------- 按配置加载用户自定义模块 ----------
    print("0. 加载配置并实例化各模块")
    q_provider = instantiate(cfg.q_provider)
    environment = instantiate(cfg.environment)
    dqn_module = instantiate(cfg.dqn_module)
    diffusion_model = instantiate(cfg.diffusion_model)
    elite_selector = instantiate(cfg.elite_selector)
    metric = instantiate(cfg.metric)
    executor = instantiate(cfg.parallel_executor)

    _ensure_instance(q_provider, QProvider, name="q_provider")
    _ensure_instance(environment, Environment, name="environment")
    _ensure_instance(dqn_module, DqnModule, name="dqn_module")
    _ensure_instance(diffusion_model, DiffusionModel, name="diffusion_model")
    _ensure_instance(elite_selector, EliteSelector, name="elite_selector")
    _ensure_instance(metric, Metric, name="metric")
    _ensure_instance(executor, ParallelExecutor, name="parallel_executor")

    M = cfg.algorithm.M
    N = cfg.algorithm.N
    K = cfg.algorithm.K

    try:
        # =========================
        # 1. 读取或创建 Q
        # =========================
        print("1. 读取或创建 Q")
        q = step_1_load_or_create_q(q_provider, environment)

        # =========================
        # 2. 随机初始化扩散模型
        # =========================
        print("2. 随机初始化扩散模型")
        diffusion_model = step_2_init_diffusion_model(diffusion_model)

        # =========================
        # 3~4. 形成初始种群（默认串行，留出并行接口）
        # =========================
        print(f"3-4. 构建初始种群（种群规模 M={M}，智能体数量 N={N}）")
        def build_initial_individual(i: int) -> Individual:
            # 3. 随机初始化策略（dqn），数量：M*N（这里为第 i 个个体初始化 N 个智能体策略）
            init_policies = step_3_init_random_policies(dqn_module, num_agents=N)

            # 4.1 用随机初始化策略仿真一次，收集经验库与 Tau（奖励留空）
            experience_buffers, tau = step_4_1_simulate_collect(environment, init_policies)

            # 4.2 把 Tau 作为扩散模型的条件生成奖励 R
            rewards = step_4_2_generate_reward(diffusion_model, tau)

            # 4.3 将经验库和 R 合并形成 dqn 训练数据（用 rewards 填充经验中的 r）
            training_data = step_4_3_build_dqn_training_data(dqn_module, experience_buffers, rewards)

            # 4.4 为每个智能体训练一个 dqn
            trained_policies = step_4_4_train_dqn_per_agent(dqn_module, training_data)

            # 4.5 再用训练好的 dqn 仿真一次，得到仿真数据，计算 ρ
            simulation_data, rho = step_4_5_simulate_and_compute_rho(
                environment,
                trained_policies,
                q=q,
                metric=metric,
            )

            # （Tau，对应的 R，ρ）就是种群的一个个体；并保存经验库和 dqn（策略）
            ind = step_4_build_individual(
                tau=tau,
                rewards=rewards,
                rho=rho,
                experience_buffers=experience_buffers,
                policies=trained_policies,
            )
            ind.metadata["initial_index"] = i
            ind.metadata["simulation_data"] = simulation_data
            return ind

        population: Population = executor.map(build_initial_individual, list(range(1, M + 1)))
        print("4. 初始种群构建完成")

        # =========================
        # 5. for k = 1..K 进化迭代
        # =========================
        for k in range(1, K + 1):
            print(f"5. 进化迭代（k={k}/{K}）")
            # 5.1 用当前的种群训练扩散模型
            print("5.1 用当前的种群训练扩散模型")
            diffusion_model = step_5_1_train_diffusion_with_population(diffusion_model, population)

            # 5.2 根据种群个体的 ρ 选择精英种群
            print("5.2 根据种群个体的 ρ 选择精英种群")
            elites = step_5_2_select_elite_population(elite_selector, population, elite_count=cfg.elite.elite_count)

            # 5.3 遍历精英种群（留出并行处理接口）
            # 注意：该步骤会对每个精英个体执行多步操作（5.3.1~5.3.8），为避免刷屏只在外层打印一次
            print("5.3 遍历精英种群并生成变异个体（5.3.1~5.3.8）")
            def mutate_one(elite: Individual) -> Individual:
                # 5.3.1 Tau 条件下截断扩散生成奖励 R（对精英奖励做变异）
                mutated_rewards = step_5_3_1_truncated_diffusion_mutate_reward(
                    diffusion_model,
                    elite.tau,
                    elite.rewards,
                    add_noise_steps=cfg.truncated_diffusion.add_noise_steps,
                    denoise_steps=cfg.truncated_diffusion.denoise_steps,
                )

                # 5.3.2 将“精英个体保存的经验库”与变异奖励合并形成训练数据
                training_data_1 = step_5_3_2_build_dqn_training_data(
                    dqn_module,
                    elite.experience_buffers,
                    mutated_rewards,
                )

                # 5.3.3 为每个智能体训练一个 dqn（得到一版训练策略）
                trained_policies_1 = step_5_3_3_train_dqn_per_agent(dqn_module, training_data_1)

                # 5.3.4 用训练的 dqn 仿真，重新收集经验库（奖励留空）和 Tau
                experience_buffers_2, tau_2 = step_5_3_4_simulate_collect(environment, trained_policies_1)

                # 5.3.5 把新 Tau 作为扩散条件生成奖励 R
                rewards_2 = step_5_3_5_generate_reward(diffusion_model, tau_2)

                # 5.3.6 将新经验库和新奖励合并形成训练数据
                training_data_2 = step_5_3_6_build_dqn_training_data(dqn_module, experience_buffers_2, rewards_2)

                # 5.3.7 再训练每个智能体的 dqn（得到最终策略）
                trained_policies_2 = step_5_3_7_train_dqn_per_agent(dqn_module, training_data_2)

                # 5.3.8 用最终策略仿真并计算 ρ
                simulation_data_2, rho_2 = step_5_3_8_simulate_and_compute_rho(
                    environment,
                    trained_policies_2,
                    q=q,
                    metric=metric,
                )

                # 形成变异后的精英个体（Tau，对应的 R，ρ）并保存经验库与策略
                mutant = step_4_build_individual(
                    tau=tau_2,
                    rewards=rewards_2,
                    rho=rho_2,
                    experience_buffers=experience_buffers_2,
                    policies=trained_policies_2,
                )
                mutant.metadata["iteration_k"] = k
                mutant.metadata["parent_rho"] = elite.rho
                mutant.metadata["simulation_data"] = simulation_data_2
                return mutant

            mutants: Population = executor.map(mutate_one, elites)

            # 5.4 将变异后的精英种群加入原有种群
            print("5.4 将变异后的精英种群加入原有种群")
            merged_population = step_5_4_add_mutants_to_population(population, mutants)

            # 5.5 保留 ρ 最大的 M 个个体，形成新种群
            print("5.5 保留 ρ 最大的 M 个个体，形成新种群")
            population = step_5_5_keep_top_m(merged_population, M=M)

            # 5.6 记录本次进化迭代新种群中最好的 ρ（写入 CSV，路径由配置决定）
            print("5.6 记录本次进化迭代种群最优 ρ 到 CSV")
            population, best_rho_k = step_5_6_record_best_rho(
                population,
                iteration_k=k,
                csv_path=cfg.logging.best_rho_csv_path,
            )
            print(f"    本次迭代最优 ρ = {best_rho_k}")

        return population
    finally:
        executor.close()

