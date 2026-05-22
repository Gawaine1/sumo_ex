# 星际争霸（SMAC）场景文档

## 问题定义

在星际争霸多智能体战斗环境（SMAC）中，智能体（战斗单位）仅能获得稀疏的全局奖励（例如胜/负）。这导致独立学习的 DQN 策略难以收敛。本框架解决的核心问题是：

> **如何利用条件扩散模型，根据每个智能体的局部观测自动生成稠密奖励矩阵，从而有效训练每个单位的 DQN 策略，并通过种群进化持续优化奖励函数，最终提升战斗胜率？**

**形式化描述：**

- 环境包含 $N$ 个友方单位，每个单位有动作空间 $A$（移动、攻击等）。
- 从仿真中收集条件信息 $\tau \in \mathbb{R}^{N \times d}$，$\tau_i$ 为单位 $i$ 的观测（血量、位置、敌我距离等）。
- 扩散模型 $p_\theta(R|\tau)$ 生成奖励矩阵 $R \in \mathbb{R}^{N \times A}$。
- 使用 $R$ 作为即时奖励训练每个单位的 DQN，然后以胜率相关指标 $\rho$ 评估策略。
- 通过精英选择 + 截断扩散变异迭代改进扩散模型。

---

## 框架流程图

```{mermaid}
flowchart TD
    Start([开始]) --> Init["0. 初始化模块
    SMAC环境 / DQN / 扩散模型 / 胜率指标"]
    Init --> Q["1. 加载或创建基准 Q
    （内置脚本战斗一次，记录胜率）"]
    Q --> RandDiff["2. 随机初始化扩散模型"]

    RandDiff --> PopLoop{"对每个个体 i=1..M"}
    PopLoop --> InitPolicy["3. 为每个战斗单位初始化随机 DQN"]
    InitPolicy --> SimCollect["4.1 执行一次 SMAC 战斗
    收集经验 (obs, action) 与 tau
    （tau = 各单位最终观测）"]
    SimCollect --> GenReward["4.2 扩散模型生成奖励矩阵 R (N×A)"]
    GenReward --> BuildData["4.3 用 R 替换原始奖励，构建训练数据"]
    BuildData --> TrainDQN["4.4 训练每个单位的 DQN"]
    TrainDQN --> SimEval["4.5 再次战斗，计算胜率指标 ρ"]
    SimEval --> StoreInd["存储个体 (tau, R, ρ, 经验库)"]
    StoreInd --> PopLoop

    PopLoop --> |"M 个个体完成"| Population["初始种群"]

    Population --> EvoLoop{"进化轮次 k=1..K"}
    EvoLoop --> TrainDiff["5.1 用种群中所有 (tau,R) 训练扩散模型"]
    TrainDiff --> SelectElite["5.2 基于 ρ 的温度采样选择精英"]
    SelectElite --> MutateLoop{"对每个精英"}
    MutateLoop --> TruncMut["5.3.1 截断扩散：
    精英奖励加噪→部分去噪，产生变异奖励"]
    TruncMut --> Rebuild["5.3.2 用原经验+变异奖励重建训练数据"]
    Rebuild --> Retrain1["5.3.3 在原 DQN 上训练一次"]
    Retrain1 --> Resim["5.3.4 新 DQN 再次战斗，收集新 tau 和经验"]
    Resim --> Regenerate["5.3.5 根据新 tau 生成新奖励"]
    Regenerate --> Retrain2["5.3.6 再次训练 DQN"]
    Retrain2 --> Reeval["5.3.7 最终战斗，计算新 ρ"]
    Reeval --> StoreMut["存储变异个体"]
    StoreMut --> MutateLoop

    MutateLoop --> |"所有精英处理完"| Merge["5.4 合并原种群与变异种群"]
    Merge --> TopM["5.5 保留 ρ 最高的 M 个个体"]
    TopM --> Record["5.6 记录本轮最优 ρ 到 CSV"]
    Record --> EvoLoop

    EvoLoop --> |"K 轮结束"| End([结束])
```

---

## 分步骤说明

### 步骤 0 – 模块初始化

根据 YAML 配置动态加载以下模块（需符合抽象基类）：

| 模块 | 实现 | 说明 |
|------|------|------|
| **环境** | `SC2Environment` | 封装 SMAC，提供 `simulate_collect` 和 `simulate_evaluate` |
| **DQN 模块** | `SC2DqnModule` | 管理每个单位的 DQN 网络，用生成的奖励训练 |
| **扩散模型** | `SC2DiffusionModel` | 条件噪声预测网络，输入 $\tau$，输出奖励矩阵 |
| **Q 提供者** | `SC2QProvider` | 加载或运行内置脚本得到基准胜率 $Q$ |
| **指标** | `SC2Metric` | 比较当前策略与 $Q$ 的胜率，计算 $\rho$ |
| **精英选择器** | `TemperatureEliteSelector` | 按 $\rho$ 进行温度采样 |
| **并行执行器** | `SerialExecutor` 或 `RayExecutor` | 控制个体并行度 |

---

### 步骤 1 – 读取或创建基准 Q

调用 `SC2QProvider.load_or_create_q(environment)`。若缓存不存在，则使用全为 `None` 的策略列表（即 SMAC 内置脚本）运行一次完整战斗，将返回的 `battles_won / battles_game` 等统计作为基准 $Q$ 并缓存。

---

### 步骤 2 – 随机初始化扩散模型

`SC2DiffusionModel.init_random()` 创建条件 UNet 和噪声调度器（DDPM/DDIM），模型参数随机。

---

### 步骤 3～4 – 构建初始种群（重复 M 次）

#### 3. 初始化随机策略

`SC2DqnModule.init_random_policies(num_agents=N)` 为每个战斗单位生成一个随机初始化的 DQN 网络（输入观测维度，输出动作价值）。

#### 4.1 仿真并收集经验库与 tau

`SC2Environment.simulate_collect(policies)` 运行一次 SMAC 对局：

- 记录每个时间步的观测 `obs` 和选择的动作 `action`，存入 `experience_buffers`（奖励暂时填 0）。
- 对局结束后，将最后一个时间步的所有单位观测堆叠成 `tau`，形状 $(N, d_\text{obs})$。

#### 4.2 根据 tau 生成奖励矩阵 R

`SC2DiffusionModel.generate_reward(tau)` 使用 DDIM 采样生成奖励矩阵 $R \in \mathbb{R}^{N \times A}$，$R[i,a]$ 表示单位 $i$ 执行动作 $a$ 时应获得的即时奖励。

#### 4.3 构建 DQN 训练数据

`SC2DqnModule.build_training_data(experience_buffers, R)` 将每个经验中的奖励占位符替换为 $R[i, \text{action}]$，形成 $(s, a, Q_\text{target})$ 三元组。

#### 4.4 训练每个智能体的 DQN

`SC2DqnModule.train_per_agent(training_data)` 对每个单位的 DQN 进行 MSE 回归，使 $Q(s, a)$ 逼近生成的 $Q_\text{target}$。

#### 4.5 再次仿真并计算 ρ

`SC2Environment.simulate_evaluate(trained_policies)` 使用训练后的策略重新战斗，返回仿真数据（胜率、总奖励等）。

`SC2Metric.compute_rho(q, simulation_data)` 计算适应度 $\rho$，例如：

$$\rho = \frac{1}{1 + |\text{winrate}_Q - \text{winrate}_\text{cur}|}$$

$\rho$ 越大表示越接近基准性能。

---

### 步骤 5 – 进化迭代 K 轮

**5.1 训练扩散模型**：提取所有 $(\tau, R)$ 对，训练条件扩散模型。

**5.2 精英选择**：温度采样选出精英个体。

**5.3 变异**：对每个精英执行截断扩散变异 → 重建数据 → 训练 DQN → 再次仿真 → 再次生成奖励 → 再次训练 → 最终评估。

**5.4～5.6**：合并种群 → 保留 Top-M → 记录最优 $\rho$。

---

## 与交通仿真场景的对比

| 维度 | SMAC 场景 | SUMO 场景 |
|------|----------|---------|
| **智能体** | 战斗单位（离散动作：移动/攻击）| 车辆（连续路径选择）|
| **宏观目标** | 胜率最大化 | LCC / AQL 与真实数据对齐 |
| **奖励结构** | $R \in \mathbb{R}^{N \times A}$（动作奖励矩阵）| $R \in \mathbb{R}^{T \times N}$（时序奖励序列）|
| **策略训练** | 独立 DQN | Hybrid A\* + Q-learning |
| **反事实推理** | 不涉及 | 支持（场景 2：禁止右转）|
