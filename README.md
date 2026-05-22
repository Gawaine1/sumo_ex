# MASDiff

**Multi-Agent Scalable Diffusion** — Collective Behavior Learning for Counterfactual Reasoning in Multi-Agent Systems

[![Documentation Status](https://readthedocs.org/projects/masdiff/badge/?version=latest)](https://masdiff.readthedocs.io/zh_CN/latest/?badge=latest)

## 文档

完整文档请访问：**[masdiff.readthedocs.io](https://masdiff.readthedocs.io)**

文档内容包括：

- 框架介绍与问题形式化
- 条件扩散模型架构详解
- 进化策略与离线 RL 说明
- 三大场景文档（SUMO 交通仿真 / SMAC / MPE）
- 实验结果（宏观对齐 / 反事实推理 / 可扩展性）

## 快速开始

```bash
git clone https://github.com/your-org/masdiff.git
cd masdiff
pip install -e ".[all]"

# 运行交通仿真场景
python main.py --config configs/sumo_lcc.yaml
```

## 本地构建文档

```bash
cd docs
pip install -r requirements.txt
make html
# 打开 build/html/index.html
```

## 引用

```bibtex
@inproceedings{masdiff2024,
  title     = {MASDiff: Collective Behavior Learning for Counterfactual Reasoning in Multi-Agent Systems},
  author    = {Anonymous},
  booktitle = {Submission \#4755},
  year      = {2024}
}
```
