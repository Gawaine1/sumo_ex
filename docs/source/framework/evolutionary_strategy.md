# 进化策略

进化策略近似理想轨迹生成器 $\mathcal{G}$，通过种群进化在轨迹空间中高效搜索，使宏观差距持续减小。

## 完整算法（Algorithm 2）

```{mermaid}
flowchart TD
    Init["初始化 pθ 和空种群 P"] --> RandPol["为 M 个体生成随机策略 {πj}"]
    RandPol --> PopBuild["种群构建循环 j=1..M\nrollout τj → 采样 Rj ~ pθ\n离线训练 πj → 评估 ρj\n存入 P"]
    PopBuild --> EvoLoop["进化循环 k=1..K"]
    EvoLoop --> UpdateDiff["5.1 用加权损失更新 pθ"]
    UpdateDiff --> SelectE["5.2 轮盘赌选择精英 E ⊂ P"]
    SelectE --> MutLoop["对每个精英 (τ, R, ρ) ∈ E"]
    MutLoop --> Mut["变异 Rj → Rmut（截断扩散）\n训练 πmut → rollout τnew\n采样 Rnew → 训练 πnew\n评估 ρnew"]
    Mut --> Merge["合并 P ∪ 变异子代"]
    Merge --> TopM["保留 Top-M（按 ρ）"]
    TopM --> EvoLoop
    TopM --> Return["返回最优策略 π* 和 pθ"]
```

## 种群初始化

初始种群 $P = \{(\tau_j, R_j, \rho_j)\}_{j=1}^M$，每个个体的产生流程：

1. 随机初始化策略 $\pi_j$
2. 执行 rollout 得到轨迹 $\tau_j$
3. 从扩散模型采样奖励：$R_j \sim p_\theta(R|\tau_j)$
4. 用 $(\tau_j, R_j)$ 离线训练策略 $\pi_j$
5. 仿真评估，计算适应度：$\rho_j = -D(\pi_j)$

## 适应度评估

适应度定义为宏观差距的负值：

$$\rho_j = -D(\pi(\tau_j, R_j)) = -\sum_{t=1}^T d(X_t, X_t^*)$$

$\rho$ 越大（即 $D$ 越小），个体越优秀。

## 轮盘赌精英选择

归一化适应度权重：

$$w_j = \gamma \cdot \frac{\rho_j - \min_k \rho_k}{\max_k \rho_k - \min_k \rho_k}$$

其中 $\gamma$ 为缩放系数。按 $w_j$ 进行 softmax 抽样，选出精英子集 $E \subset P$。

```{admonition} 与 top-k 选择的区别
:class: tip
轮盘赌（软选择）保留种群多样性，避免早熟收敛。纯 top-k 硬选择容易陷入局部最优。
```

## 变异：策略改进循环

对每个精英 $(\tau, R, \rho) \in E$ 执行以下循环，生成新个体：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 截断扩散变异 $R \to R^\text{mut}$ | 在原奖励邻域内探索新激励结构 |
| 2 | 用 $(\tau, R^\text{mut})$ 训练 $\pi^\text{mut}$ | 让策略适应新的奖励信号 |
| 3 | 执行 $\pi^\text{mut}$ 得到 $\tau^\text{new}$ | 新策略产生新的轨迹 |
| 4 | 采样 $R^\text{new} \sim p_\theta(R|\tau^\text{new})$ | 扩散模型为新轨迹匹配奖励 |
| 5 | 用 $(\tau^\text{new}, R^\text{new})$ 训练 $\pi^\text{new}$ | 充分利用新轨迹信息 |
| 6 | 评估 $\rho^\text{new} = -D(\pi^\text{new})$ | 判断变异是否有效 |

## 种群更新

```
P_new = top_M(P ∪ {所有变异子代}, key=ρ)
```

合并原种群与变异子代后，按适应度降序保留前 $M$ 个个体，实现精英保留（elitism）。

## 超参数建议

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| 种群大小 $M$ | 500 | 平衡多样性与计算开销 |
| 精英数量 | $M/10$ | 约 50 个 |
| 温度系数 $\gamma$ | 1.0 | 控制选择压力 |
| 变异加噪步数 $t$ | 50～200 | 较大 $t$ 探索范围更广 |
| 进化轮次 $K$ | 至收敛 | 通常 50～200 轮 |
