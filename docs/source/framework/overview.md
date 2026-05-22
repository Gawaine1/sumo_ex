# 框架总览

## 问题形式化

MASDiff 将多智能体学习建模为 **去中心化马尔可夫决策过程（Dec-MDP）**，元组定义为 $(S, A, R, \pi)$：

- $S = \{s_i\}_{i=1}^N$：各智能体状态集合
- $A = \{a_i\}_{i=1}^N$：各智能体动作集合
- $R = \{r_i\}_{i=1}^N$：各智能体奖励序列
- $\pi = \{\pi_i\}_{i=1}^N$：各智能体策略集合

**宏观聚合**：将所有智能体的微观状态通过聚合函数映射为宏观状态序列 $\{X_t\}_{t=0}^T$。

**优化目标**：

$$\{\pi_k\}_{k=1}^N,\ \{R_k\}_{k=1}^N = \arg\min \sum_{t=1}^T \|X_t - X_t^*\|$$

$$\text{s.t.}\quad \pi_k = \arg\max_{\pi_k} \mathbb{E}\!\left[\sum_{t=1}^T \gamma^t r_{k,t}\right],\quad \forall k$$

## 宏观差距函数

性能通过宏观差距函数 $D: \Pi \to \mathbb{R}^+$ 量化：

$$D(\pi) = \sum_{t=1}^T d(X_t,\ X_t^*)$$

距离函数 $d$ 根据目标类型选择：

| 目标类型 | $X_t$ | 距离 $d$ |
|---------|-------|---------|
| 单目标（如 LCC）| 标量 | 绝对差 $\|X_t - X_t^*\|$ |
| 多目标（如 AQL）| 向量 $\in \mathbb{R}^K$ | Tchebycheff：$\max_k \omega_k \|X_{t,k} - X_{t,k}^*\|$ |

## 理想训练框架

MASDiff 的核心建立在两个理想化假设之上：

```{admonition} 假设 1：理想奖励生成器 O（最优奖励选择）
:class: note

存在理想奖励生成器 $\mathcal{O}$。给定轨迹 $\tau$，它能找到使宏观差距最小的奖励函数 $R^*$：

$$\mathcal{O}(\tau) = \arg\min_R D(\pi(\tau, R))$$
```

```{admonition} 假设 2：理想轨迹生成器 G（渐进轨迹改进）
:class: note

存在理想轨迹生成器 $\mathcal{G}$，结合 $\mathcal{O}$ 使用时，每次迭代宏观差距单调递减：

$$D\!\left(\pi(\tau^{(k+1)},\, \mathcal{O}(\tau^{(k+1)}))\right) < D\!\left(\pi(\tau^{(k)},\, \mathcal{O}(\tau^{(k)}))\right)$$
```

在这两个假设下，框架保证宏观差距单调下降。实际中，MASDiff 用**条件扩散模型**近似 $\mathcal{O}$，用**进化策略**近似 $\mathcal{G}$。

## 两阶段近似

```{mermaid}
graph TD
    O["理想奖励生成器 O"] -->|近似为| DM["条件扩散模型 p_θ(R|τ)"]
    G["理想轨迹生成器 G"] -->|近似为| ES["进化策略\n（精英选择 + 截断扩散变异）"]
    DM & ES --> Cycle["自举循环\n更好的轨迹 → 更好的策略\n更好的策略 → 更好的轨迹"]
```

详细说明见：
- [条件扩散模型](diffusion_model.md)
- [进化策略](evolutionary_strategy.md)
- [离线强化学习](offline_rl.md)
