# 快速开始

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- SUMO 1.18+（交通仿真场景）
- SMAC（StarCraft II 场景）
- PettingZoo（MPE 场景）

## 安装

```bash
git clone https://github.com/your-org/masdiff.git
cd masdiff
pip install -e ".[all]"
```

## 运行交通仿真场景

```bash
python main.py --config configs/sumo_lcc.yaml
```

## 运行 SMAC 场景

```bash
python main.py --config configs/smac_3m.yaml
```

## 运行 MPE 场景

```bash
python main.py --config configs/mpe_grid_nav.yaml
```

## 配置文件说明

所有实验参数通过 YAML 配置文件控制：

```yaml
# configs/sumo_lcc.yaml 示例
environment:
  type: SUMOEnvironment
  network: chongqing_district
  num_agents: 1557
  horizon: 60            # 仿真时长（分钟）

diffusion:
  hidden_dim: 128
  train_steps: 1000
  noise_schedule: linear
  beta_start: 0.0001
  beta_end: 0.02
  inference_steps: 5     # 截断推理步数

evolution:
  population_size: 500
  elite_count: 50
  iterations: 100
  temperature: 1.0       # 轮盘赌温度系数 γ

metric:
  type: LCC              # 或 AQL（多目标）
  target_data: data/chongqing_lcc_target.csv

output:
  csv_path: results/sumo_lcc_rho.csv
```

## 结果查看

进化过程中每轮的最优宏观差距会实时写入 CSV，可用以下命令绘图：

```bash
python scripts/plot_convergence.py --csv results/sumo_lcc_rho.csv
```
