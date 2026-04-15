# Changelog

## 2026-04-15 - 为 pi0 增加 LoRA 微调支持

### 做了什么

本次改动为 `pi0` policy 增加了 LoRA 微调能力，目标是支持在加载 `lerobot/pi0` base 模型后，只训练少量 LoRA adapter 和动作/状态投影头，而不是全量微调整个 PaliGemma/Gemma 主干。

改动文件：

- `lerobot/policies/pi0/configuration_pi0.py`
  - 新增 `use_lora` 开关。
  - 新增 `lora_r`、`lora_alpha`、`lora_dropout`、`lora_target_modules`、`lora_apply_to` 参数。
  - 增加 LoRA 参数校验，避免 `train_expert_only=true` 和 language LoRA 组合时出现 dropout 被 eval 模式关闭的问题。

- `lerobot/policies/pi0/paligemma_with_expert.py`
  - 新增本地 `LoRALinear` 实现。
  - 对 `q_proj`、`k_proj`、`v_proj`、`o_proj` 这类 attention linear 层进行 LoRA 注入。
  - LoRA 开启后会冻结 PaliGemma/Gemma base 权重，只训练 LoRA adapter。
  - 修复原文件错误依赖 `pytest.Cache` 的问题，改为使用 `transformers.cache_utils.Cache`，避免运行环境未安装 pytest 时导入失败。

- `lerobot/policies/pi0/modeling_pi0.py`
  - 将 LoRA 配置传入 `PaliGemmaWithExpertModel`。
  - 优化 `get_optim_params()`，只把 `requires_grad=True` 的参数交给 optimizer，避免冻结参数进入优化器。

### 为什么不用 PEFT

当前 `pi0` 的 PaliGemma/Gemma forward 不是直接走 Hugging Face 标准模型 forward，而是仓库里自定义的 attention 计算路径，会直接访问类似：

- `paligemma.language_model.layers`
- `gemma_expert.model.layers`
- `q_proj.weight.dtype`

如果直接把模型包装成 `PeftModel`，有较高概率破坏这些访问路径。因此这次使用了轻量的本地 `LoRALinear`，直接替换 attention projection 层，同时保留原始 `q_proj.weight` 等 checkpoint key，方便继续加载 `lerobot/pi0` base 权重。

### 新增参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--policy.use_lora` | `false` | 是否开启 pi0 LoRA 微调 |
| `--policy.lora_r` | `16` | LoRA rank |
| `--policy.lora_alpha` | `32` | LoRA scaling alpha |
| `--policy.lora_dropout` | `0.05` | LoRA dropout |
| `--policy.lora_target_modules` | `("q_proj", "k_proj", "v_proj", "o_proj")` | 要替换的 attention projection 层 |
| `--policy.lora_apply_to` | `all` | LoRA 注入范围，可选 `all`、`language`、`expert` |

`lora_apply_to` 含义：

- `all`：同时对 PaliGemma language model 和 Gemma expert 注入 LoRA。
- `language`：只对 PaliGemma language model 注入 LoRA。
- `expert`：只对 Gemma expert 注入 LoRA。

### 推荐用法

#### 方案 A：推荐，冻结视觉塔，对 language + expert 做 LoRA

这个方案适合租显卡训练，效果通常比只训 expert 更有潜力。

```powershell
python lerobot/scripts/train.py `
  --dataset.repo_id=task1_data/demo `
  --policy.path=lerobot/pi0 `
  --output_dir=outputs/train/pi0_so101_base_task1_lora `
  --job_name=pi0_so101_base_task1_lora `
  --policy.device=cuda `
  --policy.use_amp=true `
  --policy.freeze_vision_encoder=true `
  --policy.train_expert_only=false `
  --policy.use_lora=true `
  --policy.lora_r=16 `
  --policy.lora_alpha=32 `
  --policy.lora_dropout=0.05 `
  --policy.lora_apply_to=all `
  --batch_size=4 `
  --num_workers=4 `
  --wandb.enable=false `
  --policy.push_to_hub=false `
  --steps=100000 `
  --save_freq=20000
```

训练内容：

- 冻结视觉塔。
- 冻结 PaliGemma/Gemma base 权重。
- 训练 PaliGemma language model 的 LoRA adapter。
- 训练 Gemma expert 的 LoRA adapter。
- 训练 pi0 的 state/action projection 头。

#### 方案 B：更省显存，只对 expert 做 LoRA

如果显存压力较大，可以只在 expert 上加 LoRA：

```powershell
python lerobot/scripts/train.py `
  --dataset.repo_id=task1_data/demo `
  --policy.path=lerobot/pi0 `
  --output_dir=outputs/train/pi0_so101_base_task1_lora_expert `
  --job_name=pi0_so101_base_task1_lora_expert `
  --policy.device=cuda `
  --policy.use_amp=true `
  --policy.freeze_vision_encoder=true `
  --policy.train_expert_only=true `
  --policy.use_lora=true `
  --policy.lora_r=16 `
  --policy.lora_alpha=32 `
  --policy.lora_dropout=0.05 `
  --policy.lora_apply_to=expert `
  --batch_size=4 `
  --num_workers=4 `
  --wandb.enable=false `
  --policy.push_to_hub=false `
  --steps=100000 `
  --save_freq=20000
```

训练内容：

- 冻结视觉塔。
- 冻结 PaliGemma 主干。
- 冻结 Gemma expert base 权重。
- 只训练 Gemma expert 的 LoRA adapter。
- 训练 pi0 的 state/action projection 头。

### 注意事项

1. 如果使用 `--policy.train_expert_only=true`，则 `--policy.lora_apply_to` 必须设置为 `expert`。

   原因是 `train_expert_only=true` 会让 PaliGemma 进入 eval 模式，如果同时给 language model 注入 LoRA，LoRA dropout 也会被关闭，训练行为不符合预期。

2. 当前实现保存的是完整模型权重，而不是单独的 adapter 文件。

   也就是说，训练后仍然按 LeRobot 原来的 `pretrained_model` 目录使用：

   ```powershell
   --policy.path=outputs/train/pi0_so101_base_task1_lora/checkpoints/xxxxxx/pretrained_model
   ```

3. 如果显存不足，优先尝试：

   - 减小 `--batch_size`
   - 使用 `--policy.lora_apply_to=expert`
   - 降低 `--policy.lora_r`

4. 当前未接入 QLoRA / 4-bit 量化训练。

   这次改动只实现 LoRA adapter 微调，没有引入 bitsandbytes 量化训练。Windows 环境下 bitsandbytes 兼容性也需要额外确认。

### 已验证

已完成轻量验证：

- `configuration_pi0.py`、`paligemma_with_expert.py`、`modeling_pi0.py` 语法编译通过。
- `PI0Config(use_lora=True)` 配置解析通过。
- `LoRALinear` 前向计算通过。
- `LoRALinear` 中 base weight 会冻结，LoRA 参数可训练。

未做完整大模型训练 dry-run，因为加载 `lerobot/pi0` 可能需要下载大模型权重并占用较长时间。
