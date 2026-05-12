from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

    # ---------- 计时统计（只修改 runner.py，不改其他文件） ----------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_rows: list[dict[str, Any]] = []

    def _add_timing_row(
        *,
        scope: str,
        part: str,
        name: str,
        duration_s: float,
        iteration_k: Optional[int] = None,
        count: Optional[int] = None,
    ) -> None:
        timing_rows.append(
            {
                "run_id": run_id,
                "scope": scope,  # global / iter / iter_mutate_mean / iter_mutate_sum
                "iteration_k": iteration_k,
                "part": part,
                "name": name,
                "duration_s": duration_s,
                # percent 在结束时统一补齐（需要总时长）
                "percent": None,
                "count": count,
            }
        )

    def _finalize_percents() -> None:
        # global: 相对 run_total
        run_total = None
        for r in timing_rows:
            if r.get("scope") == "global" and r.get("part") == "run_total":
                run_total = r.get("duration_s")
                break
        if isinstance(run_total, (int, float)) and run_total > 0:
            for r in timing_rows:
                if r.get("scope") == "global" and r.get("part") != "run_total":
                    r["percent"] = (float(r["duration_s"]) / float(run_total)) * 100.0

        # iter: 相对 5.iter_total（按 iteration_k 分组）
        iter_total_by_k: dict[int, float] = {}
        for r in timing_rows:
            if r.get("scope") == "iter" and r.get("part") == "5.iter_total" and isinstance(r.get("iteration_k"), int):
                iter_total_by_k[int(r["iteration_k"])] = float(r["duration_s"])
        for r in timing_rows:
            if r.get("scope") == "iter" and r.get("part") != "5.iter_total" and isinstance(r.get("iteration_k"), int):
                denom = iter_total_by_k.get(int(r["iteration_k"]))
                if denom and denom > 0:
                    r["percent"] = (float(r["duration_s"]) / denom) * 100.0

        # mutate 聚合：相对 mutate_total_mean / mutate_total_sum（按 iteration_k 分组）
        mutate_total_mean_by_k: dict[int, float] = {}
        mutate_total_sum_by_k: dict[int, float] = {}
        for r in timing_rows:
            if not isinstance(r.get("iteration_k"), int):
                continue
            k = int(r["iteration_k"])
            if r.get("scope") == "iter_mutate_mean" and r.get("part") == "5.3.mutate_total_mean":
                mutate_total_mean_by_k[k] = float(r["duration_s"])
            if r.get("scope") == "iter_mutate_sum" and r.get("part") == "5.3.mutate_total_sum":
                mutate_total_sum_by_k[k] = float(r["duration_s"])
        for r in timing_rows:
            if not isinstance(r.get("iteration_k"), int):
                continue
            k = int(r["iteration_k"])
            if r.get("scope") == "iter_mutate_mean" and r.get("part") != "5.3.mutate_total_mean":
                denom = mutate_total_mean_by_k.get(k)
                if denom and denom > 0:
                    r["percent"] = (float(r["duration_s"]) / denom) * 100.0
            if r.get("scope") == "iter_mutate_sum" and r.get("part") != "5.3.mutate_total_sum":
                denom = mutate_total_sum_by_k.get(k)
                if denom and denom > 0:
                    r["percent"] = (float(r["duration_s"]) / denom) * 100.0

    def _write_timing_csv() -> None:
        if not timing_rows:
            return
        try:
            best_rho_path = Path(cfg.logging.best_rho_csv_path)
            timing_csv_path = best_rho_path.with_name(f"{best_rho_path.stem}_timings.csv")
            if timing_csv_path.parent:
                os.makedirs(timing_csv_path.parent, exist_ok=True)

            fieldnames = ["run_id", "scope", "iteration_k", "part", "name", "duration_s", "percent", "count"]
            write_header = not timing_csv_path.exists() or timing_csv_path.stat().st_size == 0
            with open(timing_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                for r in timing_rows:
                    w.writerow(r)
            print(f"计时统计 CSV 已写入：{timing_csv_path}")
        except Exception as e:
            # 计时落盘失败不应影响主流程
            print(f"[WARN] 写入计时 CSV 失败：{e}")

    # ---------- 按配置加载用户自定义模块 ----------
    print("0. 加载配置并实例化各模块")
    _t_run_start = time.perf_counter()
    _t0 = time.perf_counter()
    q_provider = instantiate(cfg.q_provider)
    environment = instantiate(cfg.environment)
    dqn_module = instantiate(cfg.dqn_module)
    diffusion_model = instantiate(cfg.diffusion_model)
    elite_selector = instantiate(cfg.elite_selector)
    metric = instantiate(cfg.metric)
    executor = instantiate(cfg.parallel_executor)
    _add_timing_row(scope="global", part="0", name="加载配置并实例化各模块", duration_s=time.perf_counter() - _t0)

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
        _t1 = time.perf_counter()
        q = step_1_load_or_create_q(q_provider, environment)
        _add_timing_row(scope="global", part="1", name="读取或创建 Q", duration_s=time.perf_counter() - _t1)

        # =========================
        # 2. 随机初始化扩散模型
        # =========================
        print("2. 随机初始化扩散模型")
        _t2 = time.perf_counter()
        diffusion_model = step_2_init_diffusion_model(diffusion_model)
        _add_timing_row(scope="global", part="2", name="随机初始化扩散模型", duration_s=time.perf_counter() - _t2)

        # =========================
        # 3~4. 形成初始种群（默认串行，留出并行接口）
        # =========================
        print(f"3-4. 构建初始种群（种群规模 M={M}，智能体数量 N={N}）")
        _t34 = time.perf_counter()
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
        _add_timing_row(scope="global", part="3-4", name="构建初始种群（3-4）", duration_s=time.perf_counter() - _t34)
        print("4. 初始种群构建完成")

        # =========================
        # 5. for k = 1..K 进化迭代
        # =========================
        _t5_total = time.perf_counter()
        for k in range(1, K + 1):
            print(f"5. 进化迭代（k={k}/{K}）")
            _t_iter = time.perf_counter()
            # 5.1 用当前的种群训练扩散模型
            print("5.1 用当前的种群训练扩散模型")
            _t51 = time.perf_counter()
            diffusion_model = step_5_1_train_diffusion_with_population(diffusion_model, population)
            _add_timing_row(scope="iter", iteration_k=k, part="5.1", name="用当前种群训练扩散模型", duration_s=time.perf_counter() - _t51)

            # 5.2 根据种群个体的 ρ 选择精英种群
            print("5.2 根据种群个体的 ρ 选择精英种群")
            _t52 = time.perf_counter()
            elites = step_5_2_select_elite_population(elite_selector, population, elite_count=cfg.elite.elite_count)
            _add_timing_row(scope="iter", iteration_k=k, part="5.2", name="选择精英种群", duration_s=time.perf_counter() - _t52)

            # 5.3 遍历精英种群（留出并行处理接口）
            # 注意：该步骤会对每个精英个体执行多步操作（5.3.1~5.3.8），为避免刷屏只在外层打印一次
            print("5.3 遍历精英种群并生成变异个体（5.3.1~5.3.8）")
            def mutate_one(elite: Individual) -> Individual:
                _tim: dict[str, float] = {}
                _t_mut_total = time.perf_counter()
                # 5.3.1 Tau 条件下截断扩散生成奖励 R（对精英奖励做变异）
                _t = time.perf_counter()
                mutated_rewards = step_5_3_1_truncated_diffusion_mutate_reward(
                    diffusion_model,
                    elite.tau,
                    elite.rewards,
                    add_noise_steps=cfg.truncated_diffusion.add_noise_steps,
                    denoise_steps=cfg.truncated_diffusion.denoise_steps,
                )
                _tim["5.3.1"] = time.perf_counter() - _t

                # 5.3.2 将“精英个体保存的经验库”与变异奖励合并形成训练数据
                _t = time.perf_counter()
                training_data_1 = step_5_3_2_build_dqn_training_data(
                    dqn_module,
                    elite.experience_buffers,
                    mutated_rewards,
                )
                _tim["5.3.2"] = time.perf_counter() - _t

                # 5.3.3 为每个智能体训练一个 dqn（得到一版训练策略）
                _t = time.perf_counter()
                trained_policies_1 = step_5_3_3_train_dqn_per_agent(dqn_module, training_data_1)
                _tim["5.3.3"] = time.perf_counter() - _t

                # 5.3.4 用训练的 dqn 仿真，重新收集经验库（奖励留空）和 Tau
                _t = time.perf_counter()
                experience_buffers_2, tau_2 = step_5_3_4_simulate_collect(environment, trained_policies_1)
                _tim["5.3.4"] = time.perf_counter() - _t

                # 5.3.5 把新 Tau 作为扩散条件生成奖励 R
                _t = time.perf_counter()
                rewards_2 = step_5_3_5_generate_reward(diffusion_model, tau_2)
                _tim["5.3.5"] = time.perf_counter() - _t

                # 5.3.6 将新经验库和新奖励合并形成训练数据
                _t = time.perf_counter()
                training_data_2 = step_5_3_6_build_dqn_training_data(dqn_module, experience_buffers_2, rewards_2)
                _tim["5.3.6"] = time.perf_counter() - _t

                # 5.3.7 再训练每个智能体的 dqn（得到最终策略）
                _t = time.perf_counter()
                trained_policies_2 = step_5_3_7_train_dqn_per_agent(dqn_module, training_data_2)
                _tim["5.3.7"] = time.perf_counter() - _t

                # 5.3.8 用最终策略仿真并计算 ρ
                _t = time.perf_counter()
                simulation_data_2, rho_2 = step_5_3_8_simulate_and_compute_rho(
                    environment,
                    trained_policies_2,
                    q=q,
                    metric=metric,
                )
                _tim["5.3.8"] = time.perf_counter() - _t

                # 形成变异后的精英个体（Tau，对应的 R，ρ）并保存经验库与策略
                _mut_total = time.perf_counter() - _t_mut_total
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
                mutant.metadata["timing_mutation"] = {"total": _mut_total, **_tim}
                return mutant

            _t53 = time.perf_counter()
            mutants: Population = executor.map(mutate_one, elites)
            _add_timing_row(scope="iter", iteration_k=k, part="5.3", name="生成变异个体（wall time）", duration_s=time.perf_counter() - _t53)

            # 聚合 5.3.* 的“每个精英个体内部”耗时（用于看子步骤占比；sum 不等于 wall time）
            mut_timings: list[dict[str, Any]] = []
            for m in mutants:
                meta = getattr(m, "metadata", None)
                if isinstance(meta, dict):
                    tm = meta.get("timing_mutation")
                    if isinstance(tm, dict) and isinstance(tm.get("total"), (int, float)):
                        mut_timings.append(tm)
            if mut_timings:
                cnt = len(mut_timings)
                total_sum = sum(float(t["total"]) for t in mut_timings)
                total_mean = total_sum / cnt if cnt else 0.0
                _add_timing_row(
                    scope="iter_mutate_sum",
                    iteration_k=k,
                    part="5.3.mutate_total_sum",
                    name="变异内部总耗时（sum，非 wall time）",
                    duration_s=total_sum,
                    count=cnt,
                )
                _add_timing_row(
                    scope="iter_mutate_mean",
                    iteration_k=k,
                    part="5.3.mutate_total_mean",
                    name="变异内部总耗时（mean / 每个精英个体）",
                    duration_s=total_mean,
                    count=cnt,
                )
                for sub in ["5.3.1", "5.3.2", "5.3.3", "5.3.4", "5.3.5", "5.3.6", "5.3.7", "5.3.8"]:
                    sub_sum = sum(float(t.get(sub, 0.0)) for t in mut_timings)
                    sub_mean = sub_sum / cnt if cnt else 0.0
                    _add_timing_row(
                        scope="iter_mutate_sum",
                        iteration_k=k,
                        part=sub,
                        name=f"变异内部步骤 {sub}（sum）",
                        duration_s=sub_sum,
                        count=cnt,
                    )
                    _add_timing_row(
                        scope="iter_mutate_mean",
                        iteration_k=k,
                        part=sub,
                        name=f"变异内部步骤 {sub}（mean）",
                        duration_s=sub_mean,
                        count=cnt,
                    )

            # 5.4 将变异后的精英种群加入原有种群
            print("5.4 将变异后的精英种群加入原有种群")
            _t54 = time.perf_counter()
            merged_population = step_5_4_add_mutants_to_population(population, mutants)
            _add_timing_row(scope="iter", iteration_k=k, part="5.4", name="合并变异个体到种群", duration_s=time.perf_counter() - _t54)

            # 5.5 保留 ρ 最大的 M 个个体，形成新种群
            print("5.5 保留 ρ 最大的 M 个个体，形成新种群")
            _t55 = time.perf_counter()
            population = step_5_5_keep_top_m(merged_population, M=M)
            _add_timing_row(scope="iter", iteration_k=k, part="5.5", name="保留 top-M 形成新种群", duration_s=time.perf_counter() - _t55)

            # 5.6 记录本次进化迭代新种群中最好的 ρ（写入 CSV，路径由配置决定）
            print("5.6 记录本次进化迭代种群最优 ρ 到 CSV")
            _t56 = time.perf_counter()
            population, best_rho_k = step_5_6_record_best_rho(
                population,
                iteration_k=k,
                csv_path=cfg.logging.best_rho_csv_path,
            )
            _add_timing_row(scope="iter", iteration_k=k, part="5.6", name="记录最优 ρ 到 CSV", duration_s=time.perf_counter() - _t56)
            print(f"    本次迭代最优 ρ = {best_rho_k}")

            _add_timing_row(
                scope="iter",
                iteration_k=k,
                part="5.iter_total",
                name="单次进化迭代总耗时（wall time）",
                duration_s=time.perf_counter() - _t_iter,
            )

        _add_timing_row(scope="global", part="5", name="进化迭代总耗时（5）", duration_s=time.perf_counter() - _t5_total)
        return population
    finally:
        _add_timing_row(scope="global", part="run_total", name="主流程总耗时（wall time）", duration_s=time.perf_counter() - _t_run_start)
        _finalize_percents()
        _write_timing_csv()
        executor.close()

