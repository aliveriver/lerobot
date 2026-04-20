import re
import torch
from torch import nn, Tensor
from transformers import AutoProcessor
from torchvision.transforms.functional import to_pil_image

from lerobot.constants import ACTION
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.normalize import Normalize, Unnormalize

try:
    # 适配最新的 transformers 5.0+ 版本
    from transformers import AutoModelForImageTextToText
except ImportError:
    # 兼容老的 transformers 4.x 版本
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText


class OpenVLAPolicy(PreTrainedPolicy):
    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(self, config: OpenVLAConfig, dataset_stats=None):
        super().__init__(config)
        self.config = config

        # 1. 归一化工具（映射物理极限到[-1,1]）
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # 2. 加载图像文字预处理器
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name, trust_remote_code=True)

        # 3. 纯血 Bfloat16 加载大模型本体
        model = AutoModelForImageTextToText.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.bfloat16,  # 显式指定用 BFloat16 加载
            trust_remote_code=True
        )

        # 4. 开启梯度检查点：极大幅度降低训练时的激活性显存占用
        model.gradient_checkpointing_enable()

        # 5. 植入纯血 LoRA 神经连接
        if config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=config.lora_dropout,
                task_type="CAUSAL_LM",
            )
            self.vla_model = get_peft_model(model, lora_config)
            self.vla_model.print_trainable_parameters()
        else:
            self.vla_model = model

    def reset(self):
        pass

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ====== 核心魔法 1：连续浮点 -> 离散 Token ======
    def bin_actions(self, actions_float: Tensor) -> Tensor:
        """把 [-1.0, 1.0] 的浮点数切分成[0, 255] 的整数格子"""
        actions_float = torch.clamp(actions_float, -1.0, 1.0)
        action_bins = ((actions_float + 1.0) / 2.0 * (self.config.n_action_bins - 1))
        return action_bins.to(torch.long)

    # ====== 核心魔法 2：离散 Token -> 连续浮点 ======
    def unbin_actions(self, action_bins: Tensor) -> Tensor:
        """把[0, 255] 的整数格子还原回[-1.0, 1.0] 的坐标"""
        actions_float = (action_bins.float() / (self.config.n_action_bins - 1)) * 2.0 - 1.0
        return torch.clamp(actions_float, -1.0, 1.0)

    # ====== 训练核心：组装 Prompt 算 Loss ======
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch)

        images = batch["observation.images.front"]
        # 防护：如果传进来的图片带有时间序列维度[B, T, C, H, W]，提取最后一帧
        if images.ndim == 5:
            images = images[:, -1, ...]

        # 转换 Tensor 为 PIL Image 列表，喂给 OpenVLA Processor
        pil_images = [to_pil_image(img.cpu()) for img in images]

        actions_float = batch[ACTION]
        tasks = batch["task"]

        action_bins = self.bin_actions(actions_float)

        # 组装 OpenVLA 的官方标准训练语料格式
        texts = []
        for i, instruction in enumerate(tasks):
            # 将多维张量打平后再取值，防止维度错误
            act_str = "".join([f"<action_{b.item()}>" for b in action_bins[i].flatten()])
            text = f"In: What action should the robot take to {instruction}?\nOut: {act_str}"
            texts.append(text)

        # 打包送给处理器
        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(images.device)

        labels = inputs["input_ids"].clone()

        # 把补齐的 Padding Token 的标签设为 -100，让 PyTorch 的 Loss 函数忽略它们
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        outputs = self.vla_model(**inputs, labels=labels)
        loss = outputs.loss

        return loss, {"l2_loss": loss.item()}

    # ====== 推理接口1：生成预测块 (必须实现) ======
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)

        images = batch["observation.images.front"]
        if images.ndim == 5:
            images = images[:, -1, ...]

        pil_images = [to_pil_image(img.cpu()) for img in images]
        tasks = batch["task"]

        # 喂给模型前半句话，让它自己“接话”
        texts = [f"In: What action should the robot take to {instruction}?\nOut:" for instruction in tasks]
        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(images.device)

        total_action_tokens = self.config.action_dim * self.config.chunk_size

        generated_ids = self.vla_model.generate(
            **inputs,
            max_new_tokens=total_action_tokens,
            use_cache=True
        )

        # 把大模型写的回复解码成人类能读懂的文字
        generated_actions_only = generated_ids[:, inputs["input_ids"].shape[1]:]
        action_strs = self.processor.batch_decode(generated_actions_only, skip_special_tokens=True)

        parsed_actions = []
        for text in action_strs:
            matches = re.findall(r"<action_(\d+)>", text)
            bins = [int(m) for m in matches[:total_action_tokens]]

            while len(bins) < total_action_tokens:
                bins.append(128)  # 静止状态补齐
            parsed_actions.append(bins)

        action_bins = torch.tensor(parsed_actions, device=images.device)

        # 还原回浮点数，并重塑张量形状
        actions_float = self.unbin_actions(action_bins)
        actions_float = actions_float.view(images.shape[0], self.config.chunk_size, self.config.action_dim)

        # 还原物理极限（逆归一化）
        actions_float = self.unnormalize_outputs({ACTION: actions_float})[ACTION]

        return actions_float

    # ====== 推理接口2：剥离第一步 (必须实现) ======
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        actions = self.predict_action_chunk(batch, **kwargs)
        return actions[:, 0, :]