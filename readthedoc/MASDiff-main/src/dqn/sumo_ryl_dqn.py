from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from typing import Any, Optional

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    optim = importlib.import_module("torch.optim")
except Exception as e:  # pragma: no cover
    raise ImportError(
        "无法导入 torch。`src/dqn/sumo_ryl_dqn.py` 需要 PyTorch。\n"
        "请先安装 torch（例如 pip/conda 安装与你 CUDA/CPU 匹配的版本）。"
    ) from e

from src.dqn.base import DqnModule


class DQNModel(nn.Module):
    """
    DQN 网络结构（参考 CrowdNav_Re/models/dqn_model.py）。

    输入：
    - state: [batch, input_dim]
      在本框架 sumo_ryl 中，默认使用 2 维特征： [distance_to_dest, queue_length]

    输出：
    - q_values: [batch, output_dim]（output_dim = num_road）
    """

    def __init__(self, *, input_dim: int = 2, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, max(1, self.hidden_dim // 2))
        self.fc4 = nn.Linear(max(1, self.hidden_dim // 2), self.output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

    def select_action(
        self,
        state: Any,
        *,
        candidates: Optional[list[str]] = None,
        info: Optional[dict[str, Any]] = None,
        epsilon: float = 0.0,
        device: str | None = None,
    ) -> Any:
        """
        供环境调用的动作选择接口（与 SumoRylEnvironment._select_action_next_edge 兼容）。

        支持返回：
        - 候选索引（int，指 candidates 的索引）
        - 或直接返回 edge_id（str）

        说明：
        - 如果 info 中提供了 `candidate_indices`（与 candidates 对齐的 action index 列表），会优先使用；
          否则会尝试用 edge_id 解析成 int 作为 action index（仅在 edge_id 可转 int 时有效）。
        """
        if candidates is None:
            candidates = (info or {}).get("candidates") or []
        info = info or {}

        if len(candidates) == 0:
            return None

        if random.random() < float(epsilon):
            # 随机探索：返回候选索引
            return random.randrange(len(candidates))

        # 解析 state：环境传入形如 [edge_id, [dist, queue]]
        if isinstance(state, (list, tuple)) and len(state) == 2 and isinstance(state[1], (list, tuple)):
            feats = state[1]
        else:
            feats = state

        # 取前 input_dim 维
        feats = list(feats) if isinstance(feats, (list, tuple)) else [feats]
        if len(feats) < self.input_dim:
            feats = feats + [0.0] * (self.input_dim - len(feats))
        feats = feats[: self.input_dim]

        x = torch.tensor([feats], dtype=torch.float32)
        if device:
            x = x.to(device)
            self.to(device)

        with torch.no_grad():
            q = self(x).squeeze(0)  # [output_dim]

        # 候选动作索引（在 [0, output_dim) 内）
        candidate_indices = info.get("candidate_indices")
        if candidate_indices is None:
            # 尝试把 edge_id 转成 int
            tmp = []
            for eid in candidates:
                try:
                    tmp.append(int(eid))
                except Exception:
                    tmp.append(None)
            if all(v is not None for v in tmp):
                candidate_indices = tmp
            else:
                candidate_indices = None

        if candidate_indices is None:
            # 无法映射到全局动作空间：退化为随机选择
            return random.randrange(len(candidates))

        best_i = 0
        best_q = float("-inf")
        for i, a_idx in enumerate(candidate_indices):
            if a_idx is None:
                continue
            if not (0 <= int(a_idx) < self.output_dim):
                continue
            v = float(q[int(a_idx)].item())
            if v > best_q:
                best_q = v
                best_i = i

        return int(best_i)


@dataclass(frozen=True)
class SumoRylDqnTrainConfig:
    num_road: int
    input_dim: int = 2
    hidden_dim: int = 128
    device: str = "cpu"
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    min_experiences: int = 5
    # 奖励填充策略：to=使用 rewards[car_idx, action_idx]；from=尝试用 edge_id->index
    reward_fill: str = "to"  # "to" | "from"


class SumoRylDqnModule(DqnModule):
    """
    sumo_ryl 的 DQN 模块实现：
    - build_training_data: 用扩散生成的 R 填充经验中的 r
    - train_per_agent: 为每个 car/agent 训练一个 DQNModel（单步回归版本，参考 train_dqn_model.py）
    """

    def __init__(
        self,
        *,
        num_road: int,
        input_dim: int = 2,
        hidden_dim: int = 128,
        device: str = "cpu",
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        min_experiences: int = 5,
        reward_fill: str = "to",
        epsilon: float = 0.0,
    ) -> None:
        self.cfg = SumoRylDqnTrainConfig(
            num_road=int(num_road),
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            device=str(device),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            min_experiences=int(min_experiences),
            reward_fill=str(reward_fill).lower(),
        )
        self.epsilon = float(epsilon)

        if self.cfg.num_road <= 0:
            raise ValueError("num_road 必须 > 0")

    def init_random_policies(self, num_agents: int) -> list[Any]:
        """
        随机初始化 num_agents 个策略。

        返回值用途：
        - 作为环境仿真输入，收集经验库与 Tau（4.1）
        """
        policies: list[Any] = []
        for _ in range(int(num_agents)):
            m = DQNModel(input_dim=self.cfg.input_dim, output_dim=self.cfg.num_road, hidden_dim=self.cfg.hidden_dim)
            # 给 policy 挂一个默认 epsilon（用于环境 epsilon-greedy）
            m.epsilon = self.epsilon  # type: ignore[attr-defined]
            policies.append(m)
        return policies

    def build_training_data(self, experience_buffers: list[list[Any]], rewards: Any) -> Any:
        """
        将经验库和奖励 R 合并形成 DQN 训练数据（用奖励 R 填充经验中的 r）。

        约定：
        - rewards: [num_car, num_road]（torch.Tensor 或可转 tensor）
        - experience_buffers: 长度 num_car；每条经验 [s, a, r]，其中 a 是 next_road 的 index（int）
        """
        if not isinstance(rewards, torch.Tensor):
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
        else:
            rewards_t = rewards.detach().float()
        if rewards_t.dim() != 2:
            raise ValueError(f"rewards 期望 [num_car,num_road]，但得到 {tuple(rewards_t.shape)}")

        num_car = len(experience_buffers)
        num_road = rewards_t.shape[1]
        if num_road != self.cfg.num_road:
            # 不强制报错，但提示：动作空间维度需一致
            pass

        per_agent: list[list[tuple[torch.Tensor, int, float]]] = []
        for car_idx in range(num_car):
            buf = experience_buffers[car_idx]
            car_data: list[tuple[torch.Tensor, int, float]] = []
            for item in buf:
                # item: [s, a, r]
                s = item[0]
                a = int(item[1])

                # state 特征：默认取 s = [edge_id, [dist, queue]]
                feats = s[1] if isinstance(s, (list, tuple)) and len(s) == 2 else s
                feats = list(feats) if isinstance(feats, (list, tuple)) else [feats]
                if len(feats) < self.cfg.input_dim:
                    feats = feats + [0.0] * (self.cfg.input_dim - len(feats))
                feats = feats[: self.cfg.input_dim]
                state_t = torch.tensor(feats, dtype=torch.float32)

                # reward 填充：默认使用 rewards[car_idx, action_idx]
                r_val: float
                if self.cfg.reward_fill == "from":
                    # 尝试用 edge_id -> int 作为 road index
                    try:
                        from_edge_id = s[0] if isinstance(s, (list, tuple)) and len(s) == 2 else None
                        from_idx = int(from_edge_id) if from_edge_id is not None else a
                        r_val = float(rewards_t[car_idx, from_idx].item())
                    except Exception:
                        r_val = float(rewards_t[car_idx, a].item())
                else:
                    r_val = float(rewards_t[car_idx, a].item())

                car_data.append((state_t, a, r_val))
            per_agent.append(car_data)

        return {
            "per_agent_replay": per_agent,
            "num_road": int(rewards_t.shape[1]),
        }

    def train_per_agent(self, training_data: Any) -> list[Any]:
        """
        为每个智能体训练一个 DQN，返回训练好的策略列表（长度 N）。
        训练逻辑参考 CrowdNav_Re/trainers/train_dqn_model.py 的“单步回归版本”：
        - 使用 (s,a,r) 直接回归 Q(s,a) -> r
        """
        per_agent = training_data["per_agent_replay"]
        num_road = int(training_data.get("num_road", self.cfg.num_road))

        device = torch.device(self.cfg.device)
        policies: list[Any] = []

        for car_idx, replay in enumerate(per_agent):
            model = DQNModel(input_dim=self.cfg.input_dim, output_dim=num_road, hidden_dim=self.cfg.hidden_dim).to(device)
            model.epsilon = self.epsilon  # type: ignore[attr-defined]

            if len(replay) < self.cfg.min_experiences:
                # 经验不足：返回未训练模型
                policies.append(model.to("cpu"))
                continue

            opt = optim.Adam(model.parameters(), lr=self.cfg.lr)

            for _ in range(max(1, self.cfg.epochs)):
                batch_size = min(self.cfg.batch_size, len(replay))
                batch = random.sample(replay, batch_size)
                states = torch.stack([x[0] for x in batch]).to(device)  # [B, input_dim]
                actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)  # [B]
                rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)  # [B]

                q_all = model(states)  # [B, num_road]
                q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
                loss = nn.MSELoss()(q_sa, rewards)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            policies.append(model.to("cpu"))

        return policies

