# 条件扩散模型

条件扩散模型 $p_\theta(R|\tau)$ 是 MASDiff 中近似理想奖励生成器 $\mathcal{O}$ 的核心组件。它根据智能体的联合轨迹 $\tau$ 生成对应的奖励序列 $R$。

## 模型架构

网络由四个模块顺序组成：

```{mermaid}
graph LR
    R["奖励序列 R ∈ R^{T×N}"] --> IP["① 自适应输入投影\n线性变换 + 时序池化\n→ Z_R ∈ R^{N×d_h}"]
    TAU["轨迹条件 τ"] --> CE["② 动态条件嵌入\n线性映射 + 全局平均池化\n→ Z_τ ∈ R^{N×d_h}"]
    IP & CE --> ATT["③ 多头注意力\nC = Concat(Z_τ, Z_R, t)\n捕获全局依赖"]
    ATT --> OUT["④ 输出层\n卷积精炼 + 自适应反投影\n→ 预测噪声 ε_θ"]
```

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| 自适应输入投影 | $R \in \mathbb{R}^{T \times N}$ | $Z_R \in \mathbb{R}^{N \times d_h}$ | 将奖励序列映射至隐空间 |
| 动态条件嵌入 | $\tau$ | $Z_\tau \in \mathbb{R}^{N \times d_h}$ | 将轨迹编码为条件向量 |
| 多头注意力 | $C = [Z_\tau; Z_R; t]$ | attention output | 建模智能体间全局依赖 |
| 输出层 | attention output | $\epsilon_\theta \in \mathbb{R}^{T \times N}$ | 预测添加的噪声 |

隐维度 $d_h = 128$。

## 前向扩散过程

逐步向干净奖励 $R^0$ 添加高斯噪声：

$$q(R^t | R^{t-1}, \tau) = \mathcal{N}\!\left(R^t;\ \sqrt{1-\beta_t}\, R^{t-1},\ \beta_t I\right)$$

任意步 $t$ 可由 $R^0$ 直接计算：

$$R^t = \sqrt{\bar\alpha_t}\, R^0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)$$

$$\bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)$$

**噪声调度**：线性调度，$\beta$ 从 $0.0001$ 线性增至 $0.02$，共 $L=1000$ 步。

## 加权训练

为了让模型更多学习高质量样本，引入基于宏观差距的样本权重：

$$\min_\theta\; \mathbb{E}_{R^t, \tau, \epsilon, t}\!\left[\, w(\tau, R) \cdot \|\epsilon - \epsilon_\theta(R^t, \tau, t)\|_2^2 \,\right]$$

权重 $w(\tau, R) \propto 1/D(\pi(\tau, R))$，宏观差距越小，对应样本权重越大。

## 推理：从噪声到奖励

给定新轨迹 $\tau$，从纯噪声 $R^L \sim \mathcal{N}(0, I)$ 出发，迭代应用反向过程：

$$R^{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(R^t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(R^t, \tau, t)\right) + \sigma_t z,\quad z \sim \mathcal{N}(0, I)$$

**截断推理**：实际推理时仅执行 **5 步**（而非完整 1000 步），大幅加速生成速度，性能损失可忽略。

## 截断扩散变异

进化阶段用于对精英奖励做邻域变异：

$$\text{加噪：}\quad R^{\text{truncated}} = \sqrt{\bar\alpha_t}\, R + \sqrt{1-\bar\alpha_t}\, \epsilon$$

$$\text{去噪：}\quad R^{\text{mut}} \sim p_\theta(R^{\text{truncated}} | \tau)$$

控制加噪步数 $t$ 即可控制变异幅度：$t$ 越大，探索范围越广；$t$ 越小，越贴近原始奖励。
