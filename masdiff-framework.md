# MASDiff 详细项目说明文档

## 1. 项目简介

MASDiff 是一个围绕“**多智能体仿真 + DQN 策略训练 + 条件扩散奖励生成 + 进化迭代优化**”组织起来的实验框架。

这个项目的核心定位不是提供某一个固定场景下的完整算法产品，而是提供：

- 一套**固定主流程**；
- 一组可替换的**抽象接口**；
- 一种基于 **YAML + Python 动态导入** 的模块拼装方式；
- 一套已经落地的 **SUMO 路网仿真示例实现（sumo_ryl）**。

换句话说，MASDiff 负责把算法流程串起来，而环境、Q 的构造、DQN 训练、扩散模型、精英选择、评价指标、并行执行器都可以由用户自行实现并接入。

---

## 2. 项目要解决的问题

从代码结构和主流程定义来看，MASDiff 关注的问题可以概括为：

1. 在一个多智能体环境中先随机生成一批策略；
2. 通过环境仿真收集状态信息 `Tau` 和经验库；
3. 用扩散模型根据 `Tau` 生成奖励 `R`；
4. 用 `R` 反向构建 DQN 训练数据并训练策略；
5. 用训练后的策略再次仿真，计算每个个体相对于目标 `Q` 的指标 `ρ`；
6. 将 `(Tau, R, ρ)` 作为种群个体，持续做扩散训练、精英选择、截断扩散变异和保优迭代。

它本质上是一个“**扩散模型辅助奖励生成的种群式策略搜索框架**”。

---

## 3. 项目整体架构

## 3.1 逻辑分层

项目可分为 4 层：

1. **入口层**
   - `run.py`
   - 负责读取配置并调用主流程。

2. **配置层**
   - `src/config/schema.py`
   - `src/config/loader.py`
   - 负责 YAML 配置解析、类型化、校验。

3. **流程调度层**
   - `src/pipeline/runner.py`
   - `src/pipeline/steps.py`
   - `src/pipeline/types.py`
   - 负责把整个 MASDiff 算法主流程分解成标准步骤并执行。

4. **可插拔算法/场景模块层**
   - `src/environments/`
   - `src/dqn/`
   - `src/diffusion/`
   - `src/evolution/`
   - `src/q/`
   - `src/metrics/`
   - `src/parallel/`
   - 这些模块既包含抽象基类，也包含部分内置示例实现。

## 3.2 目录结构说明

```text
MASDiff/
├─ run.py                         # 程序入口
├─ README.md                      # 简要说明
├─ 主流程.txt                     # 原始算法主流程定义
├─ requirements.txt               # 当前列出的基础依赖
├─ configs/                       # YAML 配置
├─ maps/                          # SUMO 路网/路由/仿真配置
├─ outputs/                       # 输出目录
└─ src/
   ├─ config/                     # 配置模型与加载器
   ├─ pipeline/                   # 主流程组织、步骤拆分、数据类型
   ├─ environments/               # 环境接口与 SUMO 实现
   ├─ dqn/                        # DQN 接口与 SUMO 示例实现
   ├─ diffusion/                  # 扩散模型接口与 SUMO 示例实现
   ├─ evolution/                  # 精英选择接口与示例实现
   ├─ q/                          # Q 的读取/生成接口与示例实现
   ├─ metrics/                    # rho 指标接口与示例实现
   ├─ parallel/                   # 串行/Ray 并行执行器
   └─ utils/                      # 动态导入工具
```

---

## 4. 核心主流程

项目主流程来自根目录的 `主流程.txt`，在代码中由 `src/pipeline/runner.py` 和 `src/pipeline/steps.py` 实现。

完整流程可以概括为：

### 0. 初始化模块
根据 YAML 中的 `class_path` 动态实例化：

- `q_provider`
- `environment`
- `dqn_module`
- `diffusion_model`
- `elite_selector`
- `metric`
- `parallel_executor`

并在运行开始时做类型检查，确保它们分别继承了对应的抽象基类。

### 1. 读取或创建 Q
调用：
- `QProvider.load_or_create_q(environment)`

`Q` 可以理解为目标、参考轨迹、标签或评价基准，具体结构由用户自定义。

### 2. 随机初始化扩散模型
调用：
- `DiffusionModel.init_random()`

### 3~4. 构建初始种群
设：
- `M` = 种群规模
- `N` = 每个个体中的智能体数量

对每个个体执行：

#### 3. 初始化随机策略
调用：
- `DqnModule.init_random_policies(num_agents=N)`

#### 4.1 仿真并收集经验库与 Tau
调用：
- `Environment.simulate_collect(policies)`

返回：
- `experience_buffers`
- `tau`

#### 4.2 根据 Tau 生成奖励 R
调用：
- `DiffusionModel.generate_reward(tau)`

#### 4.3 构建 DQN 训练数据
调用：
- `DqnModule.build_training_data(experience_buffers, rewards)`

#### 4.4 训练每个智能体的 DQN
调用：
- `DqnModule.train_per_agent(training_data)`

#### 4.5 再次仿真并计算 rho
调用：
- `Environment.simulate_evaluate(trained_policies)`
- `Metric.compute_rho(q, simulation_data)`

最终形成一个个体：
- `(tau, rewards, rho)`
- 同时保存经验库、元数据等附加信息。

### 5. 进行 K 轮进化
每轮包括：

#### 5.1 用当前种群训练扩散模型
- `DiffusionModel.train_on_population(population)`

#### 5.2 选择精英个体
- `EliteSelector.select_elites(population, elite_count=...)`

#### 5.3 对每个精英个体做变异和再训练
对每个精英依次执行：

1. 使用截断扩散对原奖励做变异：
   - `generate_reward_truncated(tau, base_rewards, add_noise_steps, denoise_steps)`
2. 用原经验库 + 新奖励重建训练数据；
3. 训练一次 DQN；
4. 用该 DQN 再仿真，收集新的经验库和新的 `Tau`；
5. 再次根据新 `Tau` 生成奖励；
6. 再构建训练数据；
7. 再训练一次 DQN；
8. 最终仿真并重新计算 `ρ`。

这会得到一批“变异后的精英个体”。

#### 5.4 合并种群
- 原种群 + 变异种群

#### 5.5 保留 top-M 个体
按 `ρ` 从大到小排序，只保留前 `M` 个。

#### 5.6 记录当前轮最优 rho
追加写入 CSV 文件。

---

## 5. 核心数据结构

## 5.1 Individual
定义位置：`src/pipeline/types.py`

```python
@dataclass
class Individual:
    tau: Any
    rewards: Any
    rho: float
    experience_buffers: list[list[Any]]
    policies: list[Any]
    metadata: dict[str, Any]
```

含义：

- `tau`：环境收集到的条件信息；
- `rewards`：扩散模型生成的奖励或权重；
- `rho`：适应度/指标，越大越好；
- `experience_buffers`：每个智能体对应的经验库；
- `policies`：策略列表；
- `metadata`：补充信息，例如初始索引、父代 rho、simulation_data、计时数据等。

注意：
- 在 `runner.py` 的当前实现中，为避免内存和序列化膨胀，**通常不会把训练后的 policies 真正保存在个体里**，而是存空列表 `[]`。
- 也就是说，`Individual` 结构支持保存策略，但当前主流程为了减小内存压力主动避免保存大模型对象。

## 5.2 Population
```python
Population = list[Individual]
```

即种群就是个体列表。

## 5.3 配置对象 MasDiffConfig
定义位置：`src/config/schema.py`

关键字段：

- `seed`：随机种子；
- `algorithm.M`：种群规模；
- `algorithm.N`：智能体数量；
- `algorithm.K`：进化迭代次数；
- `elite.elite_count`：每轮选择多少精英；
- `truncated_diffusion.add_noise_steps`：截断扩散加噪步数；
- `truncated_diffusion.denoise_steps`：截断扩散去噪步数；
- `logging.best_rho_csv_path`：最优 rho 输出路径；
- 各类 `ModuleSpec`：指定模块类路径和初始化参数。

---

## 6. 配置系统设计

## 6.1 配置文件格式
配置以 YAML 组织，一份 YAML 描述一整套实验。

示例入口：
- `configs/default.yaml`
- `configs/sumo_ryl.yaml`
- `configs/sumo_ryl_4000.yaml`

## 6.2 ModuleSpec 机制
定义位置：`src/utils/import_utils.py`

```python
@dataclass(frozen=True)
class ModuleSpec:
    class_path: str
    kwargs: dict[str, Any]
```

支持两种写法：

- `pkg.mod:ClassName`
- `pkg.mod.ClassName`

启动时通过 `instantiate(spec)` 动态导入并实例化对象。

这使项目具备很强的可插拔性：
- 只要实现了约定接口，就可以直接在 YAML 中替换模块。

## 6.3 配置加载器的职责
`src/config/loader.py` 负责：

- 读取 YAML；
- 转换为 `MasDiffConfig`；
- 做轻量校验；
- 确保 `M > 0`、`N > 0`、`K >= 0`、`elite_count > 0`。

---

## 7. 各模块接口说明

## 7.1 QProvider
定义位置：`src/q/base.py`

职责：
- 读取或创建目标 `Q`。

核心方法：
- `load_or_create_q(environment) -> Any`

典型用途：
- 作为评价指标的参考基准。

## 7.2 Environment
定义位置：`src/environments/base.py`

职责：
- 仿真；
- 收集经验库；
- 产出 `Tau`；
- 产出用于评价的仿真结果。

核心方法：
- `simulate_collect(policies) -> (experience_buffers, tau)`
- `simulate_evaluate(policies) -> simulation_data`

## 7.3 DqnModule
定义位置：`src/dqn/base.py`

职责：
- 初始化随机策略；
- 用经验库和奖励构建训练数据；
- 为每个智能体训练策略。

核心方法：
- `init_random_policies(num_agents)`
- `build_training_data(experience_buffers, rewards)`
- `train_per_agent(training_data)`

## 7.4 DiffusionModel
定义位置：`src/diffusion/base.py`

职责：
- 随机初始化扩散模型；
- 根据 `Tau` 生成奖励；
- 根据种群训练扩散模型；
- 做截断扩散变异。

核心方法：
- `init_random()`
- `train_on_population(population)`
- `generate_reward(tau)`
- `generate_reward_truncated(tau, base_rewards, add_noise_steps, denoise_steps)`

## 7.5 EliteSelector
定义位置：`src/evolution/base.py`

职责：
- 根据 `ρ` 从种群中选精英。

核心方法：
- `select_elites(population, elite_count)`

## 7.6 Metric
定义位置：`src/metrics/base.py`

职责：
- 计算 `ρ`。

核心方法：
- `compute_rho(q, simulation_data) -> float`

## 7.7 ParallelExecutor
定义位置：`src/parallel/base.py`

职责：
- 对初始种群构建和精英变异提供并行接口。

核心方法：
- `map(fn, items) -> list[result]`
- `close()`

---

## 8. pipeline 层实现细节

## 8.1 steps.py 的作用
`src/pipeline/steps.py` 将主流程拆成一系列标准步骤函数，例如：

- `step_1_load_or_create_q`
- `step_2_init_diffusion_model`
- `step_4_1_simulate_collect`
- `step_4_2_generate_reward`
- `step_4_3_build_dqn_training_data`
- `step_4_4_train_dqn_per_agent`
- `step_4_5_simulate_and_compute_rho`
- `step_5_1_train_diffusion_with_population`
- `step_5_2_select_elite_population`
- `step_5_3_1_truncated_diffusion_mutate_reward`
- `step_5_6_record_best_rho`

这样做的好处是：
- 主流程可读性高；
- 便于替换某一步实现；
- 便于测试和局部扩展。

## 8.2 runner.py 的作用
`src/pipeline/runner.py` 是项目实际调度核心，负责：

- 组装各步骤；
- 处理串行/并行两条执行路径；
- 汇总个体；
- 记录 best rho；
- 输出计时统计。

---

## 9. 并行执行设计

项目中明确给两段计算预留并行能力：

1. 构建初始种群（3~4）；
2. 精英变异（5.3）。

## 9.1 SerialExecutor
定义位置：`src/parallel/serial.py`

特征：
- 默认执行器；
- 不并行，但保留统一的 `map` 接口；
- 支持简单进度打印。

适合：
- 开发调试；
- 小规模实验；
- 没有部署 Ray 的环境。

## 9.2 RayExecutor
定义位置：`src/parallel/ray_executor.py`

特征：
- 基于 Ray 实现并行任务调度；
- 可按任务名配置 `num_cpus`、`num_gpus` 等资源；
- 适合大规模仿真与训练。

支持按函数名区分资源配置：
- `build_initial_individual`
- `mutate_one`

## 9.3 Ray 专用任务封装
定义位置：`src/parallel/ray_tasks.py`

这是项目中一个很重要的工程化处理：

为避免 `runner.py` 内部闭包捕获大对象后被 Ray 序列化，导致：
- remote function 过大；
- 模型对象传输成本高；
- 内存占用剧增；

项目将并行路径拆成顶层函数：
- `build_initial_individual(task)`
- `mutate_one(task)`

并借助：
- `ray.put(q)`
- `ray.put(diffusion_state)`

把大对象放入 Ray object store，再由 worker 端恢复。

这部分说明该项目不仅有算法原型，还考虑到了真实大规模实验时的分布式执行问题。

---

## 10. 运行日志与输出

## 10.1 best_rho 历史记录
每次进化迭代结束后，`step_5_6_record_best_rho()` 会把当前最优 `ρ` 追加写入 CSV。

默认字段：
- `iteration_k`
- `best_rho`

默认路径由配置决定，例如：
- `outputs/best_rho_history.csv`
- `outputs/sumo_ryl_best_rho_history.csv`

## 10.2 计时统计 CSV
`runner.py` 内部实现了较详细的计时记录逻辑，会额外生成一个与 `best_rho_csv_path` 同名 stem 的计时 CSV，例如：

- `outputs/sumo_ryl_best_rho_history.csv`
- `outputs/sumo_ryl_best_rho_history_timings.csv`

计时统计覆盖：
- 全局阶段耗时；
- 每轮迭代耗时；
- 变异阶段 5.3 的 wall time；
- 5.3 内部各子步骤 sum/mean 聚合耗时；
- 百分比占比。

这对定位性能瓶颈很有帮助。

## 10.3 其他可能输出
根据具体模块实现，可能还会产生：
- `outputs/q_sumo_ryl.pt`：Q 的缓存文件；
- 模型参数、日志、临时结果等。

---

## 11. 内置示例：sumo_ryl 场景

除了抽象框架，本项目已经提供了一套较完整的 SUMO 示例实现，主要用于路网车辆路径规划/交通状态建模。

相关文件：

- `src/environments/sumo_ryl.py`
- `src/dqn/sumo_ryl_dqn.py`
- `src/diffusion/sumo_ryl_diffusion.py`
- `src/q/sumo_ryl_q.py`
- `src/metrics/sumo_ryl_metric.py`
- `src/evolution/temperature_selection.py`
- `configs/sumo_ryl.yaml`
- `configs/sumo_ryl_4000.yaml`

## 11.1 SumoRylEnvironment
这是基于 SUMO/TraCI 的环境实现。

### 输入
- `policies`：策略列表，长度通常等于车辆数；
- 如果某个策略不可用或为 `None`，则环境会退化为只使用 A* 路径规划。

### simulate_collect 的输出
1. `experience_buffers`
   - 长度 = `num_car`；
   - 每个元素是该车的经验序列；
   - 单条经验形如：`[s, a, r]`；
   - 其中 `r` 在收集阶段为空。

2. `tau`
   - 类型：`torch.Tensor`
   - 形状：`[num_car, num_road, 2]`
   - 两个特征分别是：
     - 当前路到终点的最短距离；
     - 当前路段排队长度。

### simulate_evaluate 的输出
- `simulation_data`
- 类型：`torch.Tensor`
- 形状：`[end_tick / sample_interval, num_road]`

它表示在多个采样时刻记录下来的全网排队长度向量。

### 路径规划策略
车辆首次出现时做路径规划：
- 有 policy：使用 “DQN 选分支 + A* 补全路径”；
- 无 policy：只使用按路段长度为代价的 A*。

这使它既能支持纯规则基线，也能支持 DQN 控制。

## 11.2 SumoRylQProvider
定义位置：`src/q/sumo_ryl_q.py`

作用：
- 构造或读取参考目标 `Q`。

当前实现逻辑：
- 若缓存存在且未要求强制重算，则直接加载；
- 否则以 `policies=[None] * num_car` 调用环境评估，得到一个纯 A* 基准下的 `Q`；
- 再可选缓存到 `.pt` 文件。

因此在 sumo_ryl 场景里，`Q` 和 `simulation_data` 的结构是一致的。

## 11.3 SumoRylMetric
定义位置：`src/metrics/sumo_ryl_metric.py`

作用：
- 计算 `q` 与 `simulation_data` 的差异；
- 再把“越小越好”的误差转换成“越大越好”的 `ρ`。

默认逻辑：
- 先计算 MSE；
- 默认变换：`rho = 1 / (1 + mse)`。

支持的变换方式：
- `neg`
- `inv`
- `inv1p`
- `exp`

## 11.4 SumoRylDqnModule
定义位置：`src/dqn/sumo_ryl_dqn.py`

作用：
- 初始化每个智能体的 DQN；
- 用扩散模型生成的奖励填充经验；
- 对每个智能体单独训练一个 DQN。

关键特点：

1. **每个 agent 对应一个 DQNModel**；
2. 状态默认使用 2 维特征：
   - 到终点距离；
   - 当前排队长度；
3. 动作为下一条道路的索引；
4. 训练逻辑是“单步回归版”：
   - 用 `(s, a, r)` 直接拟合 `Q(s,a)`；
5. 如果经验数量少于 `min_experiences`，则直接返回未训练模型。

奖励填充策略支持：
- `to`：使用 `rewards[car_idx, action_idx]`；
- `from`：尝试按起始道路索引取奖励。

## 11.5 SumoRylDiffusionModel
定义位置：`src/diffusion/sumo_ryl_diffusion.py`

作用：
- 根据 `Tau` 生成奖励矩阵 `R`；
- 根据当前种群中的 `(Tau, R)` 训练条件扩散模型；
- 对精英奖励做截断扩散变异。

### 数据形状
- 输入 `tau`：`[num_car, num_road, 2]`
- 输出 `rewards`：`[num_car, num_road]`

### 内部组成
- `TauDiffusionModel`：噪声预测网络；
- `NoiseScheduler`：DDPM 调度器；
- `DDIMScheduler`：DDIM 调度器；
- 支持 `linear/cosine` beta schedule；
- 支持 `ddpm/ddim` 采样。

### 截断扩散变异逻辑
`generate_reward_truncated()` 的语义是：
- 从精英个体原有奖励 `R_base` 出发；
- 先加少量噪声；
- 再只做少量去噪；
- 得到邻域内的新奖励样本。

这是项目实现“围绕优秀个体做局部探索”的关键。

## 11.6 TemperatureEliteSelector
定义位置：`src/evolution/temperature_selection.py`

作用：
- 用 softmax 温度采样方式选择精英。

大致逻辑：
- `scores = rho`
- `probs = softmax(temperature * scores)`
- 使用 `torch.multinomial` 采样若干精英索引。

这意味着精英不一定永远是绝对 top-k，而是带有一定探索性。

---

## 12. 典型配置文件说明

## 12.1 default.yaml
定位：
- 框架演示配置；
- 使用占位的 `your_pkg.xxx:YourClass`；
- 用于说明每个模块应该如何在 YAML 中声明。

## 12.2 sumo_ryl.yaml
定位：
- SUMO 场景示例配置；
- 使用重庆路网示例；
- 设置了较大的 `M`、`N`，更接近真实实验配置。

其中典型参数包括：
- `M: 100`
- `N: 1557`
- `K: 1`
- `elite_count: 20`
- Ray 任务资源配置；
- SUMO 路网和路由文件路径；
- DQN 和扩散模型的 GPU 设备设置。

## 12.3 sumo_ryl_4000.yaml
定位：
- 更大规模车辆数实验配置；
- `N = 4000`；
- 更长仿真时长；
- 更适合高负载测试。

---

## 13. 运行方式

## 13.1 基础运行命令

```bash
python run.py --config configs/default.yaml
```

或者：

```bash
python run.py --config configs/sumo_ryl.yaml
```

## 13.2 当前 requirements.txt 中的依赖
当前项目 `requirements.txt` 只列出了：

- `PyYAML>=6.0`
- `ray[default]`

这足以支撑：
- YAML 配置读取；
- Ray 并行执行器。

但如果要运行 `sumo_ryl` 相关示例，还需要额外安装：

- `torch`
- `SUMO`
- `traci`
- `sumolib`

也就是说，当前 `requirements.txt` 更像是“框架最小依赖”，**不是完整场景依赖清单**。

---

## 14. 扩展方式

MASDiff 的一个核心价值就是便于扩展。

## 14.1 扩展一个新环境
只需继承：
- `src/environments/base.py` 中的 `Environment`

并实现：
- `simulate_collect()`
- `simulate_evaluate()`

然后在 YAML 中写入：

```yaml
environment:
  class_path: "your_pkg.envs:YourEnvironment"
  kwargs: {}
```

## 14.2 扩展一个新 DQN 模块
继承：
- `DqnModule`

实现：
- `init_random_policies()`
- `build_training_data()`
- `train_per_agent()`

## 14.3 扩展一个新扩散模型
继承：
- `DiffusionModel`

实现：
- `init_random()`
- `train_on_population()`
- `generate_reward()`
- `generate_reward_truncated()`

## 14.4 扩展评价指标
继承：
- `Metric`

实现：
- `compute_rho(q, simulation_data)`

## 14.5 扩展精英选择策略
继承：
- `EliteSelector`

实现：
- `select_elites(population, elite_count)`

## 14.6 扩展并行执行器
继承：
- `ParallelExecutor`

实现：
- `map(fn, items)`

---

## 15. 设计亮点

## 15.1 主流程和具体算法解耦
框架层只定义：
- 顺序；
- 数据接口；
- 调度规则。

具体算法留给用户实现，灵活性很高。

## 15.2 支持从小实验到大规模并行
- 小实验：`SerialExecutor`
- 大规模：`RayExecutor`

## 15.3 对大对象传输有工程考虑
当前实现明确做了以下优化：

1. `Individual` 默认不保存训练后的大策略对象；
2. Ray 路径中使用 `ray.put()` 广播大对象；
3. 将并行任务移到顶层函数中，避免闭包序列化过大。

## 15.4 计时统计比较完善
除了算法结果，还会自动输出：
- 主流程各阶段耗时；
- 单轮迭代耗时；
- 变异阶段内部各步骤耗时。

这对于实验调优非常有价值。

---

## 16. 当前项目的适用场景

从现有代码看，MASDiff 特别适合以下场景：

- 多智能体强化学习实验框架；
- 交通路网/路径规划类仿真；
- 用生成模型辅助奖励建模的研究原型；
- 需要做种群式搜索和策略变异的研究；
- 希望在同一主流程下切换不同环境和模型实现的实验平台。

---

## 17. 使用时需要注意的问题

## 17.1 接口兼容性完全由用户负责
虽然框架会检查实例是否属于正确基类，但：
- `Q` 的结构；
- `Tau` 的结构；
- `rewards` 的形状；
- `simulation_data` 的格式；
- DQN 的输入输出约定；

这些跨模块的数据契约仍需要用户自己保证一致。

## 17.2 sumo_ryl 示例对环境依赖较重
如果要运行 SUMO 场景，必须保证：
- 本机已安装 SUMO；
- `SUMO_HOME/tools` 已可用于 Python 导入；
- `traci` / `sumolib` 可正常使用；
- 地图和路由文件路径正确。

## 17.3 配置中的规模参数很大
例如 `configs/sumo_ryl.yaml` 中：
- `M=100`
- `N=1557`

这代表运算规模非常大，对：
- CPU
- GPU
- 内存
- Ray object store

都有较高要求。

## 17.4 outputs 目录会不断追加结果
CSV 记录采用追加写入方式，如果反复运行同一配置：
- `best_rho_history.csv`
- `*_timings.csv`

会持续追加内容，不会自动清空。

---

## 18. 一句话总结

MASDiff 是一个以 `主流程.txt` 为蓝本实现的、可插拔的多智能体实验框架。它把“Q 基准构造、环境仿真、DQN 训练、条件扩散奖励生成、精英选择、种群进化、并行调度、计时统计”统一到一个清晰的流程中；同时又通过抽象接口和 YAML 动态导入，允许用户替换几乎所有具体算法模块。当前仓库除框架外，还内置了一套面向 SUMO 路网仿真的 `sumo_ryl` 示例实现，可作为进一步开发和实验的基础。
