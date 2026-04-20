from dataclasses import dataclass, field

# LeRobot 配置基类和类型
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
# 导入所需的优化器和调度器
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    type: str = "openvla"
    # ============ 1. 核心模型配置 ============
    vlm_model_name: str = "C:/Users/vipuser/Desktop/openvla-7b"
    n_obs_steps: int = 1
    n_action_steps: int = 1
    chunk_size: int = 1
    action_dim: int = 6  # 默认 6 自由度机械臂

    # OpenVLA核心机制：将[-1,1]离散化为256个等级
    n_action_bins: int = 256

    # ============ 2. 显存优化：无损 bfloat16 + LoRA ============
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # ============ 3. 炼丹超参数配置（新增补充） ============
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self):
        super().__post_init__()

    # =========================================================
    # 下面这 6 个方法就是“填补官方契约”的，没有它们系统就会报错！
    # =========================================================

    def validate_features(self) -> None:
        """检查输入特征的形状是否合法（为空则跳过校验）"""
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        """告诉系统使用什么优化器（AdamW）"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """告诉系统学习率怎么衰减（余弦退火，带预热）"""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        """状态的偏移帧列表"""
        return [0]

    @property
    def action_delta_indices(self) -> list:
        """动作序列帧的长度序列"""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """强化学习的奖励偏移（这里不用，所以是None）"""
        return None