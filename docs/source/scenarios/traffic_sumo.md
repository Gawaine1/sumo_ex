# 交通仿真（SUMO）场景文档

## 问题定义

在基于 SUMO（城市移动仿真）的大规模交通网络中，每辆车（智能体）仅能感知局部环境（当前道路、速度、周边车辆），而全局目标（如最大连通高速路段 LCC、关键路口平均排队长度 AQL）却只能以宏观聚合指标的形式观测。这导致传统 MARL 方法无法同时学到可迁移的个体奖励函数与满足宏观目标的集体策略。本框架解决的核心问题是：

> **如何在无专家示范的条件下，利用条件扩散模型自动生成每辆车的个体奖励序列，通过离线强化学习训练分布式导航策略，并借助进化搜索持续改进轨迹质量，最终使仿真交通流的宏观状态逼近真实观测数据，并支持反事实推理（如"若某路口禁止右转，拥堵将如何变化"）？**

**形式化描述：**

- 环境包含 $N = 1557$ 辆车，每辆车有路径选择动作空间（Hybrid A\* + Q-learning）。
- 从仿真中收集联合轨迹 $\tau = \{\tau_i\}_{i=1}^N$，$\tau_i = \{(s_i, a_i)_t\}_{t=1}^T$ 为车辆 $i$ 的状态-动作序列。
- 扩散模型 $p_\theta(R|\tau)$ 根据轨迹生成奖励序列 $R \in \mathbb{R}^{T \times N}$，$R[t,i]$ 为车辆 $i$ 在时步 $t$ 的即时奖励。
- 使用 $R$ 训练每辆车的导航策略 $\pi_i$，以宏观差距 $D(\pi) = \sum_t d(X_t, X_t^*)$ 评估策略质量。
- 通过精英选择 + 截断扩散变异迭代优化轨迹-奖励种群，逐步缩小宏观差距。

---

## 框架流程图

```{mermaid}
flowchart TD
    Start([开始]) --> Init["0. 初始化模块
    SUMO环境 / 离线RL / 扩散模型 / 宏观差距指标"]
    Init --> Q["1. 提取目标宏观状态序列 X*
    （从真实摄像头数据提取 LCC 或 AQL）"]
    Q --> RandDiff["2. 随机初始化条件扩散模型 pθ"]

    RandDiff --> PopLoop{"对每个个体 j=1..M"}
    PopLoop --> InitPolicy["3. 为每辆车初始化随机导航策略 πj"]
    InitPolicy --> SimCollect["4.1 执行一次 SUMO 仿真
    收集联合轨迹 τj = {(s,a)t}"]
    SimCollect --> GenReward["4.2 扩散模型采样奖励序列
    Rj ~ pθ(R | τj)"]
    GenReward --> TrainRL["4.3 用 (τj, Rj) 离线训练每辆车的策略 πj"]
    TrainRL --> SimEval["4.4 再次仿真，聚合宏观状态 {Xt}
    计算宏观差距 Dj = Σ d(Xt, Xt*)"]
    SimEval --> StoreInd["存储个体 (τj, Rj, ρj = −Dj)"]
    StoreInd --> PopLoop

    PopLoop --> |"M 个个体完成"| Population["初始种群 P"]

    Population --> EvoLoop{"进化轮次 k=1..K"}
    EvoLoop --> TrainDiff["5.1 用种群加权损失更新扩散模型
    权重 w ∝ 1/D，强调高质量样本"]
    TrainDiff --> SelectElite["5.2 轮盘赌选择精英子集 E ⊂ P"]
    SelectElite --> MutateLoop{"对每个精英 (τ, R, ρ)"}
    MutateLoop --> TruncMut["5.3.1 截断扩散变异：
    R 加噪后部分去噪 → Rmut"]
    TruncMut --> Retrain1["5.3.2 用 (τ, Rmut) 重新训练策略 πmut"]
    Retrain1 --> Resim["5.3.3 执行 πmut，收集新轨迹 τnew"]
    Resim --> Regenerate["5.3.4 采样新奖励 Rnew ~ pθ(R | τnew)"]
    Regenerate --> Retrain2["5.3.5 用 (τnew, Rnew) 再次训练 πnew"]
    Retrain2 --> Reeval["5.3.6 仿真评估，计算新宏观差距 Dnew
    ρnew = −Dnew"]
    Reeval --> StoreMut["存储变异个体 (τnew, Rnew, ρnew)"]
    StoreMut --> MutateLoop

    MutateLoop --> |"所有精英处理完"| Merge["5.4 合并原种群与变异子代"]
    Merge --> TopM["5.5 按 ρ 降序保留前 M 个个体"]
    TopM --> Record["5.6 记录本轮最优宏观差距到 CSV"]
    Record --> EvoLoop

    EvoLoop --> |"K 轮结束"| CF{"需要反事实推理？"}
    CF --> |"是"| CFInfer["6. 将策略迁移至新场景
    （如禁止右转）执行仿真，预测 AQL"]
    CF --> |"否"| End([结束])
    CFInfer --> End
```

---

## 分步骤说明

### 步骤 0 – 模块初始化

根据 YAML 配置动态加载以下模块：

| 模块 | 实现说明 |
|------|---------|
| **环境** | `SUMOEnvironment` — 封装 SUMO，提供 `simulate_collect`（收集轨迹）和 `simulate_evaluate`（计算宏观状态）接口 |
| **离线 RL 模块** | `SUMOOfflineRLModule` — 管理每辆车的 Hybrid A\* + Q-learning 策略，接受轨迹-奖励对进行离线训练 |
| **扩散模型** | `SUMODiffusionModel` — 多头注意力条件降噪网络，输入轨迹 $\tau$，输出奖励序列 $R \in \mathbb{R}^{T \times N}$ |
| **宏观指标** | `SUMOMetric` — 单目标（LCC 绝对差）或多目标（AQL Tchebycheff 距离）宏观差距计算 |
| **精英选择器** | `RouletteWheelSelector` — 基于归一化适应度 $\rho$ 的轮盘赌采样 |
| **并行执行器** | `RayExecutor`（大规模）或 `SerialExecutor`（调试） |

---

### 步骤 1 – 提取目标宏观状态序列 $\{X_t^*\}$

从真实城市监控摄像头数据提取目标宏观状态：

**单目标（LCC）**

每分钟统计平均车速较高的最大连通道路集合大小 $\text{LCC}_t^*$：

$$d(X_t, X_t^*) = |\text{LCC}_t - \text{LCC}_t^*|$$

**多目标（AQL）**

统计 $K$ 条关键道路的平均排队长度向量，使用等权重 Tchebycheff 距离：

$$d(X_t, X_t^*) = \max_{1 \le k \le K} |AQL_{t,k} - AQL_{t,k}^*|$$

---

### 步骤 2 – 随机初始化扩散模型

`SUMODiffusionModel.init_random()` 创建条件降噪网络，采用 1000 步线性噪声调度（$\beta$：$0.0001 \to 0.02$），推理时使用 5 步截断过程。

---

### 步骤 3～4 – 构建初始种群（重复 M=500 次）

#### 3. 初始化随机策略

`SUMOOfflineRLModule.init_random_policies(num_agents=N)` 为每辆车生成随机初始化的 Q-learning 参数，输入为车辆局部观测（当前道路 ID、速度、目的地距离等），输出各候选路径的动作价值。

#### 4.1 仿真并收集联合轨迹

`SUMOEnvironment.simulate_collect(policies)` 运行一次完整 SUMO 仿真：

- 每辆车按当前策略选择路径。
- 记录所有时间步的状态-动作对，构成轨迹 $\tau_j$，奖励暂置为 0。

#### 4.2 根据轨迹采样奖励序列

`SUMODiffusionModel.generate_reward(τ_j)` 使用截断 DDPM 从纯噪声出发，5 步迭代去噪生成奖励序列 $R_j \in \mathbb{R}^{T \times N}$。

#### 4.3 离线训练每辆车的策略

`SUMOOfflineRLModule.train_offline(τ_j, R_j)` 将轨迹中的奖励替换为 $R_j[t, i]$，构建 $(s, a, r, s')$ 四元组，对每辆车的 Q-network 进行离线 MSE 回归更新。

#### 4.4 仿真评估并计算适应度

`SUMOEnvironment.simulate_evaluate(trained_policies)` 用更新后策略重新运行仿真，收集每分钟宏观状态 $\{X_t\}$。

`SUMOMetric.compute_rho(X, X*)` 计算适应度：$\rho_j = -D_j$，$\rho$ 越大表示集体行为越接近真实观测。

---

### 步骤 5 – 进化迭代 K 轮

#### 5.1 加权训练扩散模型

提取种群中所有 $(\tau_j, R_j)$ 对，以宏观差距的倒数为样本权重训练扩散模型，使模型更多地从接近真实数据的轨迹中学习。

#### 5.2 轮盘赌精英选择

按归一化适应度权重 $w_j = \gamma \cdot \frac{\rho_j - \min \rho}{\max \rho - \min \rho}$ 软选择精英子集 $E$。

#### 5.3 截断扩散变异

对每个精英 $(\tau, R, \rho) \in E$ 执行策略改进循环：

| 子步骤 | 操作 | 含义 |
|--------|------|------|
| **5.3.1** | $R \xrightarrow{\text{加噪}} R^\text{truncated} \xrightarrow{\text{去噪}} R^\text{mut}$ | 在原奖励邻域内探索新激励结构 |
| **5.3.2** | 用 $(\tau, R^\text{mut})$ 训练 $\pi^\text{mut}$ | 车辆在新激励下调整导航偏好 |
| **5.3.3** | 执行 $\pi^\text{mut}$ 得到 $\tau^\text{new}$ | 新行为产生新的路网流量模式 |
| **5.3.4** | $R^\text{new} \sim p_\theta(R \| \tau^\text{new})$ | 扩散模型为新轨迹匹配奖励 |
| **5.3.5** | 用 $(\tau^\text{new}, R^\text{new})$ 训练 $\pi^\text{new}$ | 充分利用新探索轨迹 |
| **5.3.6** | 仿真评估，计算 $\rho^\text{new}$ | 判断变异是否改善宏观差距 |

#### 5.4～5.6 种群更新与记录

合并原种群与变异子代，按 $\rho$ 降序保留前 $M=500$ 个个体，将最优宏观差距追加写入 CSV。

---

### 步骤 6 – 反事实推理

训练完成后，学到的奖励函数编码了每辆车的个体偏好（如偏好拥堵少的路、偏好宽干道等）。

**实验设计（场景 2：禁止右转）：**

1. 仅用场景 1（正常）的宏观数据 $\{AQL_t^*\}_{s1}$ 训练奖励模型和策略。
2. 将学到的策略原封不动地迁移至场景 2（禁止在某路口右转）。
3. 在场景 2 运行仿真，预测新的 $\{AQL_t\}_{s2}$。
4. 与合成真值 $\{AQL_t^*\}_{s2}$ 对比，量化反事实预测误差。

**结果（5 条关键道路的 AQL 预测误差）：**

| 方法 | road76 | road16 | road64 | road2 | road79 |
|------|--------|--------|--------|-------|--------|
| **MASDiff** | **2** | **−3** | **1** | **2** | **−1** |
| CGAN | 4 | 7 | −3 | −8 | 1 |
| LDM_IMIT | −3 | 1 | 11 | 6 | 7 |
| LDM_CFG | 19 | 2 | 4 | 7 | 5 |
| LDM_PCFG | −5 | −1 | 4 | −10 | 6 |

MASDiff 在两个场景、大多数道路上均取得最小绝对误差。

---

## 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| 智能体数量 $N$ | 1557 辆车 | 重庆某区真实路网 |
| 路段数 / 交叉口数 | 114 / 45 | SUMO 路网规模 |
| 种群大小 $M$ | 500 | 进化种群容量 |
| 扩散模型隐维度 | 128 | 多头注意力层宽度 |
| 噪声步数（训练）| 1000 步线性 | $\beta$: 0.0001 → 0.02 |
| 噪声步数（推理）| 5 步截断 | 加噪 + 去噪各 5 步 |
| 硬件 | AMD EPYC 9554 + 4× RTX 4090 + 256GB RAM | 实验服务器 |
