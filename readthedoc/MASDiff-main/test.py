from src.config.loader import load_config
from src.utils.import_utils import instantiate
from src.utils.calculate_fitness_based_on_rewards import calculate_fitness_based_on_rewards

# 1. 读取配置
cfg = load_config("configs/sumo_ryl_nov.yaml")

# 2. 实例化所需模块
environment = instantiate(cfg.environment)
dqn_module = instantiate(cfg.dqn_module)
metric = instantiate(cfg.metric)
q_provider = instantiate(cfg.q_provider)

# 3. 准备一批奖励 R
# 假设种群规模 M=3，每个 R 的形状是 [N, num_road]
# 实际使用时可以换成你自己的 rewards_batch
import torch

N = cfg.algorithm.N
num_road = cfg.dqn_module.kwargs["num_road"]
M = 3

rewards_batch = [
    torch.randn(N, num_road),
    torch.randn(N, num_road),
    torch.randn(N, num_road),
]

# 4. 计算每个 R 对应的 fitness
fitness_list = calculate_fitness_based_on_rewards(
    environment=environment,
    rewards_batch=rewards_batch,
    population_size=M,
    dqn_module=dqn_module,
    metric=metric,
    q_provider=q_provider,   # 如果你已经有 q，也可以直接传 q=...
    initial_population_path="outputs/example_initial_population.pkl",
    ray_init_kwargs={
        "num_cpus": 25,
        "num_gpus": 3,
    },
    task_options_by_name={
        "build_fitness_initial_seed_data": {
            "num_cpus": 2.5,
            "num_gpus": 0.3,
        },
        "evaluate_fitness_for_reward": {
            "num_cpus": 2.5,
            "num_gpus": 0.3,
        },
    },
    seed=42,
)

print("fitness_list =", fitness_list)
