MASDiff 文档
============

**Multi-Agent Scalable Diffusion (MASDiff)** 是一个面向去中心化多智能体系统的集体行为学习框架。
它通过条件扩散模型与进化搜索相结合，在无需专家示范的条件下同时学习个体奖励函数与满足宏观目标的集体策略，并支持反事实推理。

.. note::
   本文档对应论文 *MASDiff: Collective Behavior Learning for Counterfactual Reasoning in Multi-Agent Systems*（Submission #4755）。

.. toctree::
   :maxdepth: 2
   :caption: 快速入门

   introduction
   quickstart

.. toctree::
   :maxdepth: 3
   :caption: 框架详解

   framework/overview
   framework/diffusion_model
   framework/evolutionary_strategy
   framework/offline_rl

.. toctree::
   :maxdepth: 3
   :caption: 场景文档

   scenarios/traffic_sumo
   scenarios/smac
   scenarios/mpe

.. toctree::
   :maxdepth: 2
   :caption: 实验结果

   experiments/rq1_alignment
   experiments/rq2_counterfactual
   experiments/rq3_scalability

.. toctree::
   :maxdepth: 1
   :caption: 参考资料

   bibliography

索引
----

* :ref:`genindex`
* :ref:`search`
