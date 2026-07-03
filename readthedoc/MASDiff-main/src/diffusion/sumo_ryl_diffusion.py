from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Optional

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    F = importlib.import_module("torch.nn.functional")
except Exception as e:  # pragma: no cover
    raise ImportError(
        "无法导入 torch。`src/diffusion/sumo_ryl_diffusion.py` 需要 PyTorch。\n"
        "请先安装 torch（例如 pip/conda 安装与你 CUDA/CPU 匹配的版本）。"
    ) from e

from src.diffusion.base import DiffusionModel


# ============================================================
# 噪声调度器（参考 CrowdNav_Re/utils/noise_scheduler.py 的逻辑）
# ============================================================
class NoiseScheduler:
    """
    DDPM 噪声调度器（支持 linear/cosine beta schedule）。

    - add_noise: x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps
    - step: 根据 pred_noise 进行一步去噪（近似 DDPM 采样）
    """

    def __init__(
        self,
        *,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "linear",
        device: Any = "cpu",
    ) -> None:
        import math

        self.num_timesteps = int(num_timesteps)
        self.device = torch.device(device)
        self.schedule_type = str(schedule_type)

        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps 必须 > 0")

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        elif schedule_type == "cosine":
            # Improved DDPM cosine schedule
            s = 0.008
            steps = torch.arange(self.num_timesteps + 1, dtype=torch.float64) / self.num_timesteps
            alpha_cumprod = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
            betas = torch.clamp(betas, min=1e-4, max=0.9999).float()
        else:
            raise NotImplementedError(f"schedule_type={schedule_type} 不支持")

        self.betas = betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(self.device)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        x0/noise: [B, num_car, num_road]
        timesteps: [B]
        """
        timesteps = timesteps.to(self.device)
        x0 = x0.to(self.device)
        noise = noise.to(self.device)

        # broadcast 到 [B,1,1]
        s1 = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return s1 * x0 + s2 * noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """
        DDPM 一步去噪（返回 x_{t-1}）。
        """
        t = int(timestep)
        model_output = model_output.to(self.device)
        sample = sample.to(self.device)

        alpha_t = self.alphas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)

        variance = 0.0 if t == 0 else self.posterior_variance[t]
        if t > 0:
            noise = torch.randn_like(sample)
            x_prev = torch.sqrt(self.alphas_cumprod_prev[t]) * pred_x0 + torch.sqrt(variance) * noise
        else:
            x_prev = pred_x0
        return x_prev


class DDIMScheduler(NoiseScheduler):
    """
    DDIM 调度器（推理时可用较少步数）。
    参考 CrowdNav_Re/utils/noise_scheduler.py 的实现要点。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eta: float = 0.0
        self.inference_timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, *, num_inference_steps: int = 50) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps 必须 > 0")
        if num_inference_steps > self.num_timesteps:
            raise ValueError("num_inference_steps 不能大于 num_timesteps")

        if num_inference_steps == 1:
            timesteps = torch.tensor([self.num_timesteps - 1], dtype=torch.long)
        else:
            timesteps = torch.linspace(
                0, self.num_timesteps - 1, num_inference_steps, dtype=torch.long
            )
            timesteps = torch.flip(timesteps, dims=[0])
        self.inference_timesteps = timesteps.to(self.device)

    def _get_prev_timestep(self, timestep: int) -> int:
        if self.inference_timesteps is None:
            return max(0, timestep - 1)

        step_indices = (self.inference_timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(step_indices) == 0:
            distances = torch.abs(self.inference_timesteps - timestep)
            step_idx = int(torch.argmin(distances).item())
        else:
            step_idx = int(step_indices[0].item())

        if step_idx < len(self.inference_timesteps) - 1:
            return int(self.inference_timesteps[step_idx + 1].item())
        return 0

    def sample_prev_timestep(self, timestep: int) -> int:
        return self._get_prev_timestep(timestep)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        prev_timestep: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        if eta is None:
            eta = self.eta
        if prev_timestep is None:
            prev_timestep = self._get_prev_timestep(timestep)

        model_output = model_output.to(self.device)
        sample = sample.to(self.device)

        alpha_prod_t = torch.clamp(self.alphas_cumprod[int(timestep)], min=1e-8, max=1.0)
        alpha_prod_t_prev = (
            torch.clamp(self.alphas_cumprod[int(prev_timestep)], min=1e-8, max=1.0)
            if prev_timestep >= 0
            else torch.tensor(1.0, device=self.device)
        )

        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1.0 - alpha_prod_t)
        pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        if eta > 0 and timestep > 0:
            variance = (eta**2) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (
                1 - alpha_prod_t / alpha_prod_t_prev
            )
            variance = torch.clamp(variance, min=0.0)
            sigma = torch.sqrt(variance)
        else:
            sigma = 0.0

        sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)
        direction_coeff = torch.sqrt(torch.clamp(1.0 - alpha_prod_t_prev - sigma**2, min=0.0))

        x_prev = sqrt_alpha_prod_t_prev * pred_x0 + direction_coeff * model_output
        if eta > 0 and timestep > 0:
            x_prev = x_prev + sigma * torch.randn_like(sample)
        return x_prev


# ============================================================
# 条件噪声预测网络：tau=[B,num_car,num_road,2] 条件下预测 eps
# ============================================================
def _timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    正弦时间嵌入（类似 transformer positional encoding）。
    timesteps: [B]，返回 [B, dim]
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=device))
        * torch.arange(0, half, device=device).float()
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
    return emb


class MultiScaleResidualBlock(nn.Module):
    """
    多尺度残差块（参考 CrowdNav_Re/models/diffusion_model.py）。
    使用 GroupNorm 以支持小 batch。
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=1)
        self.conv3x3 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=5, padding=2)

        num_groups = min(32, max(1, (out_channels * 3) // 2))
        self.gn = nn.GroupNorm(num_groups, (out_channels * 3) // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Conv1d((out_channels * 3) // 2, out_channels, kernel_size=1)
        self.output_gn = nn.GroupNorm(min(32, out_channels), out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(min(32, out_channels), out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)

        out = torch.cat([out1, out3, out5], dim=1)
        out = self.gn(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.output_proj(out)
        out = self.output_gn(out)
        out = out + residual
        out = self.relu(out)
        return out


class TauDiffusionModel(nn.Module):
    """
    tau 条件扩散噪声预测网络（结构参考 DiffusionModelPQueue，且**强制条件**无分支）：

    输入：
    - x: [B, num_car, num_road]
    - condition(tau): [B, num_car, num_road, 2]
    - timesteps: [B]
    输出：
    - pred_noise: [B, num_car, num_road]
    """

    def __init__(
        self,
        *,
        num_car: int,
        hidden_dim: int = 128,
        hidden_mult: int = 10,
        dropout: float = 0.1,
        num_attention_heads: int = 8,
        condition_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_car = int(num_car)
        if self.num_car <= 0:
            raise ValueError("num_car 必须 > 0")

        self.hidden_dim = int(hidden_dim) * int(hidden_mult)
        self.condition_embed_dim = int(condition_embed_dim)

        # 时间嵌入（正弦位置编码 + MLP）
        self.time_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # 条件嵌入：把每个 (dist, queue) -> condition_embed_dim，然后对 (car, road) 做 mean pool
        self.condition_flatten_proj = nn.Sequential(
            nn.Linear(2, self.condition_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(self.condition_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # 输入投影：将每条路上的 num_car 维向量投影到 hidden_dim
        # x: [B,num_car,num_road] -> x^T: [B,num_road,num_car] -> Linear(num_car->hidden_dim)
        self.input_proj_linear = nn.Linear(self.num_car, self.hidden_dim)

        # 残差块：拼接 [x_proj, t_emb, c_emb] => hidden_dim*3
        self.res_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(self.hidden_dim * 3, self.hidden_dim, dropout=dropout),
                MultiScaleResidualBlock(self.hidden_dim, self.hidden_dim, dropout=dropout),
                MultiScaleResidualBlock(self.hidden_dim, self.hidden_dim, dropout=dropout),
                MultiScaleResidualBlock(self.hidden_dim, self.hidden_dim, dropout=dropout),
            ]
        )

        # 注意力
        self.self_attention = nn.MultiheadAttention(
            self.hidden_dim, num_attention_heads, dropout=dropout, batch_first=True
        )

        # 输出特征层：hidden_dim -> hidden_dim//8
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, self.hidden_dim // 2), self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, self.hidden_dim // 4), self.hidden_dim // 4),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 8, kernel_size=1),
            nn.ReLU(),
        )

        # 输出投影：把每条路的 hidden_dim//8 特征投影回 num_car（预测噪声）
        self.output_proj_linear = nn.Linear(self.hidden_dim // 8, self.num_car)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def positional_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        # 输出 [B, hidden_dim]
        pe = torch.zeros(timesteps.size(0), self.hidden_dim, device=timesteps.device)
        position = timesteps.unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=timesteps.device).float()
            * (-(torch.log(torch.tensor(10000.0, device=timesteps.device)) / self.hidden_dim))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x 期望 [B,num_car,num_road]，但得到 {tuple(x.shape)}")
        if condition.dim() != 4 or condition.shape[-1] != 2:
            raise ValueError(f"condition(tau) 期望 [B,num_car,num_road,2]，但得到 {tuple(condition.shape)}")

        B, num_car_in, num_road = x.shape
        if num_car_in != self.num_car:
            raise ValueError(f"x 的 num_car={num_car_in} 与模型 num_car={self.num_car} 不一致")
        if condition.shape[:3] != (B, num_car_in, num_road):
            raise ValueError(f"condition 前三维需要与 x 匹配：期望 {(B, num_car_in, num_road)}，但得到 {tuple(condition.shape[:3])}")

        # x: [B,num_car,num_road] -> [B,num_road,num_car] -> [B,num_road,hidden] -> [B,hidden,num_road]
        x_t = x.transpose(1, 2)  # [B, num_road, num_car]
        x_proj = self.input_proj_linear(x_t).transpose(1, 2)  # [B, hidden_dim, num_road]

        seq_len = num_road
        # 时间嵌入：posenc -> MLP -> broadcast 到 [B,hidden,num_road]
        t_pe = self.positional_encoding(timesteps)  # [B, hidden_dim]
        t_emb = self.time_embedding(t_pe).unsqueeze(-1).expand(-1, -1, seq_len)

        # 条件嵌入：先把每个 (dist,queue)->embed，然后对 (car,road) mean pool
        c_embed = self.condition_flatten_proj(condition)  # [B, num_car, num_road, cond_embed]
        c_pooled = c_embed.mean(dim=[1, 2])  # [B, cond_embed]
        c_emb = self.condition_proj(c_pooled).unsqueeze(-1).expand(-1, -1, seq_len)  # [B, hidden_dim, num_road]

        x_combined = torch.cat([x_proj, t_emb, c_emb], dim=1)  # [B, hidden_dim*3, num_road]

        for block in self.res_blocks:
            x_combined = block(x_combined)

        # self-attention over road dimension
        x_attn = x_combined.transpose(1, 2)  # [B, num_road, hidden_dim]
        attn_out, _ = self.self_attention(x_attn, x_attn, x_attn)
        x_combined = (x_combined + attn_out.transpose(1, 2)) / 2.0

        # 输出特征并投影回 num_car
        feat = self.output_layer(x_combined)  # [B, hidden_dim//8, num_road]
        feat_t = feat.transpose(1, 2)  # [B, num_road, hidden_dim//8]
        out = self.output_proj_linear(feat_t).transpose(1, 2)  # [B, num_car, num_road]
        return out


# ============================================================
# MASDiff 框架对接：SumoRylDiffusionModel（实现 DiffusionModel 接口）
# ============================================================
@dataclass
class SumoRylDiffusionHyperParams:
    # 训练相关
    num_epochs: int = 1
    batch_size: int = 1
    lr: float = 1e-4
    grad_clip_norm: float = 1.0

    # 扩散相关
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule_type: str = "cosine"  # linear/cosine
    sampler: str = "ddim"  # ddpm/ddim
    inference_steps: int = 50  # DDIM 推理步数
    ddim_eta: float = 0.0

    # 数值/归一化
    reward_scale: float = 5.0  # 假设奖励（或权重）主要落在 [0, reward_scale]
    tau_dist_clip: float = 1e6
    tau_dist_scale: float = 1e4
    tau_queue_scale: float = 50.0

    # 设备
    device: str = "cpu"


class SumoRylDiffusionModel(DiffusionModel):
    """
    面向 sumo_ryl 场景的“条件扩散模型”实现：

    - 条件输入：tau.shape == [num_car, num_road, 2]
      - [:,:,0] 当前路到终点最短距离（按路段长度）
      - [:,:,1] 当前路排队长度
    - 输出奖励/权重：R.shape == [num_car, num_road]

    注意：
    - 本类实现的是 MASDiff 的 DiffusionModel 接口，用于主流程中的：
      - init_random()
      - train_on_population(population)
      - generate_reward(tau)
      - generate_reward_truncated(tau, base_rewards, add_noise_steps, denoise_steps)
    """

    def __init__(
        self,
        *,
        num_car: int,
        hidden_dim: int = 128,
        hidden_mult: int = 1,
        dropout: float = 0.1,
        # Hyper params（也可在 YAML 里逐项传入）
        num_epochs: int = 1,
        batch_size: int = 1,
        lr: float = 1e-4,
        grad_clip_norm: float = 1.0,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "cosine",
        sampler: str = "ddim",
        inference_steps: int = 50,
        ddim_eta: float = 0.0,
        reward_scale: float = 5.0,
        tau_dist_clip: float = 1e6,
        tau_dist_scale: float = 1e4,
        tau_queue_scale: float = 50.0,
        device: str = "cpu",
    ) -> None:
        self.hparams = SumoRylDiffusionHyperParams(
            num_epochs=int(num_epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            grad_clip_norm=float(grad_clip_norm),
            num_timesteps=int(num_timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            schedule_type=str(schedule_type),
            sampler=str(sampler).lower(),
            inference_steps=int(inference_steps),
            ddim_eta=float(ddim_eta),
            reward_scale=float(reward_scale),
            tau_dist_clip=float(tau_dist_clip),
            tau_dist_scale=float(tau_dist_scale),
            tau_queue_scale=float(tau_queue_scale),
            device=str(device),
        )
        self.device = torch.device(self.hparams.device)

        self._net_cfg = {
            "num_car": int(num_car),
            "hidden_dim": int(hidden_dim),
            "hidden_mult": int(hidden_mult),
            "dropout": float(dropout),
        }
        self.net = TauDiffusionModel(**self._net_cfg).to(self.device)
        self._rebuild_scheduler()

    def _rebuild_scheduler(self) -> None:
        if self.hparams.sampler == "ddim":
            self.scheduler: NoiseScheduler = DDIMScheduler(
                num_timesteps=self.hparams.num_timesteps,
                beta_start=self.hparams.beta_start,
                beta_end=self.hparams.beta_end,
                schedule_type=self.hparams.schedule_type,
                device=self.device,
            )
            assert isinstance(self.scheduler, DDIMScheduler)
            self.scheduler.eta = self.hparams.ddim_eta
            self.scheduler.set_timesteps(num_inference_steps=self.hparams.inference_steps)
        elif self.hparams.sampler == "ddpm":
            self.scheduler = NoiseScheduler(
                num_timesteps=self.hparams.num_timesteps,
                beta_start=self.hparams.beta_start,
                beta_end=self.hparams.beta_end,
                schedule_type=self.hparams.schedule_type,
                device=self.device,
            )
        else:
            raise ValueError(f"sampler={self.hparams.sampler} 不支持（仅 ddpm/ddim）")

    # -------------------------
    # DiffusionModel 接口实现
    # -------------------------
    def init_random(self) -> None:
        # 重新初始化网络权重（简单做法：重新构造模块）
        self.net = TauDiffusionModel(**self._net_cfg).to(self.device)
        self._rebuild_scheduler()

    def train_on_population(self, population: list[Any]) -> None:
        """
        用当前种群训练扩散模型（5.1）。

        训练数据来自每个个体的 (tau, rewards)：
        - tau: [num_car, num_road, 2]
        - rewards: [num_car, num_road]
        """
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ind in population:
            tau = getattr(ind, "tau", None)
            rewards = getattr(ind, "rewards", None)
            if tau is None or rewards is None:
                continue
            tau_t = self._to_tau_tensor(tau)
            r_t = self._to_reward_tensor(rewards)
            samples.append((r_t, tau_t))

        if not samples:
            return None

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        self.net.train()

        # 简化：按个体 batch 训练（batch_size 通常很小，因为单样本极大）
        n = len(samples)
        bs = max(1, self.hparams.batch_size)

        for _epoch in range(max(1, self.hparams.num_epochs)):
            # 打乱
            perm = torch.randperm(n).tolist()
            for start in range(0, n, bs):
                idxs = perm[start : start + bs]
                x0_list = []
                tau_list = []
                for j in idxs:
                    r_t, tau_t = samples[j]
                    x0_list.append(self._normalize_rewards(r_t))
                    tau_list.append(self._normalize_tau(tau_t))

                x0 = torch.stack(x0_list, dim=0).to(self.device)  # [B,C,R]
                cond = torch.stack(tau_list, dim=0).to(self.device)  # [B,C,R,2]

                B = x0.shape[0]
                timesteps = torch.randint(0, self.hparams.num_timesteps, (B,), device=self.device, dtype=torch.long)
                noise = torch.randn_like(x0)
                x_t = self.scheduler.add_noise(x0, noise, timesteps)

                pred_noise = self.net(x_t, cond, timesteps)
                loss = F.mse_loss(pred_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.hparams.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.hparams.grad_clip_norm)
                optimizer.step()

    def generate_reward(self, tau: Any) -> Any:
        """
        把 Tau 作为条件生成奖励 R（4.2 / 5.3.5）。

        返回：
        - rewards: torch.Tensor，shape=[num_car, num_road]
        """
        self.net.eval()

        tau_t = self._to_tau_tensor(tau)
        cond = self._normalize_tau(tau_t).unsqueeze(0).to(self.device)  # [1,C,R,2]
        C, R = tau_t.shape[0], tau_t.shape[1]

        # 从纯噪声开始
        x = torch.randn((1, C, R), dtype=torch.float32, device=self.device)

        if isinstance(self.scheduler, DDIMScheduler) and self.scheduler.inference_timesteps is not None:
            timesteps_seq = self.scheduler.inference_timesteps.tolist()
            for t in timesteps_seq:
                t_tensor = torch.full((1,), int(t), device=self.device, dtype=torch.long)
                pred_noise = self.net(x, cond, t_tensor)
                prev_t = self.scheduler.sample_prev_timestep(int(t))
                x = self.scheduler.step(pred_noise, int(t), x, prev_timestep=int(prev_t), eta=self.scheduler.eta)
        else:
            for t in range(self.hparams.num_timesteps - 1, -1, -1):
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                pred_noise = self.net(x, cond, t_tensor)
                x = self.scheduler.step(pred_noise, t, x)

        rewards = self._denormalize_rewards(x.squeeze(0).detach().cpu())
        return rewards

    def generate_reward_truncated(
        self,
        tau: Any,
        base_rewards: Any,
        *,
        add_noise_steps: int,
        denoise_steps: int,
    ) -> Any:
        """
        截断扩散（用于“变异”）：
        - 以 base_rewards 作为基准样本 R_base（来自精英个体的原 R）
        - 加噪 add_noise_steps（把 R_base 推到某个噪声层级）
        - 仅用 denoise_steps 做短程去噪，得到与 R_base 邻域内的“变异样本”

        返回：
        - mutated_rewards: torch.Tensor，shape=[num_car, num_road]
        """
        add_noise_steps = max(0, int(add_noise_steps))
        denoise_steps = max(1, int(denoise_steps))

        tau_t = self._to_tau_tensor(tau)
        cond = self._normalize_tau(tau_t).unsqueeze(0).to(self.device)
        base_r = self._to_reward_tensor(base_rewards)  # [C,R]
        if base_r.shape[0] != tau_t.shape[0] or base_r.shape[1] != tau_t.shape[1]:
            raise ValueError(
                f"base_rewards shape 需要与 tau 匹配，期望 [{tau_t.shape[0]},{tau_t.shape[1]}]，但得到 {tuple(base_r.shape)}"
            )
        x0 = self._normalize_rewards(base_r.to(self.device)).unsqueeze(0)  # [1,C,R]

        # 选择加噪时间步 t_add（越大噪声越多）
        t_add = min(add_noise_steps, self.hparams.num_timesteps - 1)
        t_tensor = torch.tensor([t_add], device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x = self.scheduler.add_noise(x0, noise, t_tensor)

        # 构造一个从 t_add -> 0 的“截断”时间步序列（长度 denoise_steps）
        if denoise_steps == 1:
            timesteps_seq = [t_add]
        else:
            timesteps_seq = (
                torch.linspace(0, t_add, denoise_steps, dtype=torch.long).flip(0).tolist()
            )

        # 若是 DDIM，临时按该序列推进（用 step(prev_timestep=...)）
        if isinstance(self.scheduler, DDIMScheduler):
            for i, t in enumerate(timesteps_seq):
                t = int(t)
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                pred_noise = self.net(x, cond, t_tensor)
                prev_t = int(timesteps_seq[i + 1]) if i + 1 < len(timesteps_seq) else 0
                x = self.scheduler.step(pred_noise, t, x, prev_timestep=prev_t, eta=self.scheduler.eta)
        else:
            for t in timesteps_seq:
                t = int(t)
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                pred_noise = self.net(x, cond, t_tensor)
                x = self.scheduler.step(pred_noise, t, x)

        mutated = self._denormalize_rewards(x.squeeze(0).detach().cpu())
        return mutated

    # -------------------------
    # 内部：类型与归一化
    # -------------------------
    def _to_tau_tensor(self, tau: Any) -> torch.Tensor:
        if isinstance(tau, torch.Tensor):
            t = tau.detach().float()
        else:
            t = torch.tensor(tau, dtype=torch.float32)
        if t.dim() != 3 or t.shape[-1] != 2:
            raise ValueError(f"tau 期望 shape=[num_car,num_road,2]，但得到 {tuple(t.shape)}")
        return t

    def _to_reward_tensor(self, rewards: Any) -> torch.Tensor:
        if isinstance(rewards, torch.Tensor):
            r = rewards.detach().float()
        else:
            r = torch.tensor(rewards, dtype=torch.float32)
        if r.dim() != 2:
            raise ValueError(f"rewards 期望 shape=[num_car,num_road]，但得到 {tuple(r.shape)}")
        return r

    def _normalize_tau(self, tau: torch.Tensor) -> torch.Tensor:
        # tau[...,0] 可能为 inf：先裁剪再缩放
        dist = tau[..., 0].clone()
        queue = tau[..., 1].clone()

        dist = torch.where(torch.isfinite(dist), dist, torch.tensor(self.hparams.tau_dist_clip, device=dist.device))
        dist = torch.clamp(dist, min=0.0, max=self.hparams.tau_dist_clip) / max(self.hparams.tau_dist_scale, 1e-6)

        queue = torch.clamp(queue, min=0.0) / max(self.hparams.tau_queue_scale, 1e-6)

        out = torch.stack([dist, queue], dim=-1)
        return out

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        # 假设奖励落在 [0, reward_scale]，映射到 [-1,1]
        rs = max(self.hparams.reward_scale, 1e-6)
        r = torch.clamp(rewards, 0.0, rs)
        return (r / rs) * 2.0 - 1.0

    def _denormalize_rewards(self, x: torch.Tensor) -> torch.Tensor:
        # [-1,1] -> [0, reward_scale]
        rs = max(self.hparams.reward_scale, 1e-6)
        r = (torch.clamp(x, -1.0, 1.0) + 1.0) * 0.5 * rs
        return r

