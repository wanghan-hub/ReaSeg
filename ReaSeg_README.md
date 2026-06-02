# ReaSeg: Qwen2.5-VL + MedSAM for Medical Reasoning Segmentation

ReaSeg 是一个面向医学图像推理分割任务的纯净训练框架。核心结构为：

```text
Qwen2.5-VL
  → [SEG] token hidden state
  → ReaSegProjector
  → MedSAM image encoder
  → MedSAM mask decoder
  → segmentation mask
```

本项目用于训练“图像 + 医学问题 → 推理文本 + [SEG] → 分割掩码”的医学图像推理分割模型。

---

## 1. 项目目录结构

```text
reaseg/
├── checkpoints/
│   ├── medsam_vit_b.pth
│   └── Qwen/Qwen2.5-VL-3B-Instruct/
├── configs/
│   ├── sft_bf16.yaml
│   └── deepspeed_zero2.json
├── data/
│   ├── reason_seg_dataset.py
│   └── collate.py
├── datasets/
│   └── BraTS2023_reasonseg_rgbA/
│       ├── images/
│       ├── masks/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── losses/
│   └── mask_losses.py
├── model/
│   ├── model.py
│   ├── seg_projector.py
│   ├── medsam_wrapper.py
│   └── segment_anything/
├── scripts/
│   ├── debug_dataset.sh
│   ├── run_sft.sh
│   ├── stage1_projector_warmup.sh
│   ├── stage2_reasoning_sft.sh
│   ├── stage3_partial_decoder_adaptation.sh
│   ├── stage4_hard_finetune.sh
│   └── test_final_model.sh
├── tools/
│   ├── debug_dataset.py
│   ├── inspect_modules.py
│   └── test_reaseg.py
└── train/
    ├── train_sft.py
    └── train_utils.py
```

---

## 2. 数据格式

数据集目录示例：

```text
datasets/BraTS2023_reasonseg_rgbA/
├── images/
├── masks/
├── train.json
├── val.json
└── test.json
```

单条 JSON 示例：

```json
{
  "image": "datasets/BraTS2023_reasonseg_rgbA/images/BraTS2023_xxx_rgb_a.png",
  "mask": "datasets/BraTS2023_reasonseg_rgbA/masks/BraTS2023_xxx.png",
  "input_text": "Please identify and segment the region of peritumoral FLAIR abnormality and associated edema in this axial slice.",
  "output": "The analysis ... [SEG]"
}
```

`reason_seg_dataset.py` 应支持：

```text
image / image_path
mask / mask_path / json
input_text / query / question / input
output / outputs / answer
```

训练时推荐：

```bash
DATA_PATH=./datasets/BraTS2023_reasonseg_rgbA
```

---

## 3. 数据检查

```bash
cd /data/ReaSeg/reaseg

DATA_PATH=./datasets/BraTS2023_reasonseg_rgbA \
SPLIT=train \
SAMPLE_INDEX=0 \
BATCH_SIZE=2 \
bash scripts/debug_dataset.sh
```

确认：

```text
SEG token count in input_ids >= 1
pixel_values shape 正常
image_grid_thw 正常
medsam_image: [3, 1024, 1024]
gt_masks: [N, H, W]
Dataset and collate sanity check passed
```

---

## 4. 模块检查

Frozen decoder：

```bash
python tools/inspect_modules.py \
  --model_name_or_path ./checkpoints/Qwen/Qwen2.5-VL-3B-Instruct \
  --medsam_checkpoint ./checkpoints/medsam_vit_b.pth \
  --precision bf16 \
  --use_lora \
  --lora_scope llm \
  --train_seg_projector \
  --mask_decoder_train_mode none \
  --match seg_projector
```

Partial decoder：

```bash
python tools/inspect_modules.py \
  --model_name_or_path ./checkpoints/Qwen/Qwen2.5-VL-3B-Instruct \
  --medsam_checkpoint ./checkpoints/medsam_vit_b.pth \
  --precision bf16 \
  --use_lora \
  --lora_scope llm \
  --train_seg_projector \
  --mask_decoder_train_mode partial \
  --match mask_decoder
```

---

## 5. 分阶段训练

### Stage 1: Projector Warm-up，冻结 MedSAM decoder；让 seg_projector 学会把 [SEG] hidden 映射到 MedSAM prompt 空间

训练：
- ReaSegProjector
- LLM LoRA

冻结：
- MedSAM image encoder
- MedSAM prompt encoder
- MedSAM mask decoder
- Qwen visual encoder

目标：让 `ReaSegProjector` 学会把 `[SEG] hidden state` 映射到 MedSAM prompt embedding 空间。

```bash
bash scripts/stage1_projector_warmup.sh
```

关键参数：

```text
MASK_DECODER_TRAIN_MODE=none
DETACH_SEG_HIDDEN_FOR_MASK=1
LEARNING_RATE=1e-5
SEG_PROJECTOR_LR=1e-5
```

---

### Stage 2: Reasoning SFT，训练 LLM LoRA + Projector；让医学图像 + query → 合理推理文本 + [SEG] hidden → mask prompt

训练：
- Qwen LLM LoRA
- ReaSegProjector

冻结：
- MedSAM image encoder
- MedSAM prompt encoder
- MedSAM mask decoder

目标：让医学推理文本、`[SEG]` 位置和分割提示空间进一步对齐。

```bash
bash scripts/stage2_reasoning_sft.sh
```

关键参数：

```text
MASK_DECODER_TRAIN_MODE=none
DETACH_SEG_HIDDEN_FOR_MASK=0
LEARNING_RATE=5e-6
SEG_PROJECTOR_LR=1e-5
```

`DETACH_SEG_HIDDEN_FOR_MASK=0` 表示 mask loss 可以通过 `[SEG] hidden state` 反传到 LLM LoRA。

---

### Stage 3: Partial Mask Decoder Adaptation；让 mask decoder 的输出头适配你的 BraTS RGB 合成图像和 projector prompt

训练：
- Qwen LLM LoRA
- ReaSegProjector
- MedSAM mask decoder output heads

冻结：
- MedSAM image encoder
- MedSAM prompt encoder
- mask decoder transformer 主体

目标：让 MedSAM mask decoder 的输出头适配 ReaSegProjector 生成的 prompt embedding。

```bash
bash scripts/stage3_partial_decoder_adaptation.sh
```

关键参数：

```text
MASK_DECODER_TRAIN_MODE=partial
MASK_DECODER_LR=1e-9
DETACH_SEG_HIDDEN_FOR_MASK=1
```

稳定后可以尝试：

```text
MASK_DECODER_LR=5e-9
MASK_DECODER_LR=1e-8
```

---

### Stage 4: Hard Sample / Optional Fine-tuning

目标：针对低 Dice、边界复杂、小目标、ET/TC 等困难样本进行短程微调。

```bash
bash scripts/stage4_hard_finetune.sh
```

推荐：

```text
MASK_DECODER_TRAIN_MODE=head_plus_upscaling
EPOCHS=1
LEARNING_RATE=2e-6
SEG_PROJECTOR_LR=2e-6
MASK_DECODER_LR=1e-9
DICE_LOSS_WEIGHT=2.0
BCE_LOSS_WEIGHT=1.0
CE_LOSS_WEIGHT=0.5
```

不建议一开始使用 `MASK_DECODER_TRAIN_MODE=full`。

---

## 6. 少量数据与全量数据配置

### 少量数据，例如 1300 条

```text
Stage 1: EPOCHS=10
Stage 2: EPOCHS=5
Stage 3: EPOCHS=1~3
Stage 4: EPOCHS=1
```

示例：

```bash
DATA_PATH=./datasets/BraTS2023_reasonseg_rgbA \
EPOCHS=10 \
bash scripts/stage1_projector_warmup.sh
```

### 全量数据，例如 45000 条

```text
Stage 1: EPOCHS=1~2
Stage 2: EPOCHS=1~3
Stage 3: EPOCHS=1
Stage 4: EPOCHS=1
```

LoRA：

```text
显存保守: LORA_R=8, LORA_ALPHA=8
效果优先: LORA_R=16, LORA_ALPHA=16 或 32
```

---

## 7. 多卡训练

1 卡：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/stage1_projector_warmup.sh
```

4 卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash scripts/stage1_projector_warmup.sh
```

8 卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 bash scripts/stage1_projector_warmup.sh
```

全局 batch：

```text
global_batch = NUM_GPUS × BATCH_SIZE × GRAD_ACCUM_STEPS
```

如果想保持 1 卡时的 global batch 不变：

```text
1卡: BATCH_SIZE=1, GRAD_ACCUM_STEPS=8
4卡: BATCH_SIZE=1, GRAD_ACCUM_STEPS=2
8卡: BATCH_SIZE=1, GRAD_ACCUM_STEPS=1
```

---

## 8. 测试模型

测试脚本默认执行 teacher-forced segmentation evaluation：

```text
使用 test.json 中的 output 文本和 [SEG]
→ 提取 [SEG] hidden state
→ ReaSegProjector
→ MedSAM mask decoder
→ predicted mask
→ 与 GT mask 计算 Dice / IoU / Precision / Recall / HD95
```

这不是跳过 mask decoder；预测 mask 仍然来自 mask decoder。

运行：

```bash
bash scripts/test_final_model.sh
```

如果只跑到 Stage 3：

```bash
FINAL_CKPT_DIR=./outputs/stage3_partial_decoder/final_model \
bash scripts/test_final_model.sh
```

输出：

```text
test_metrics.csv
test_summary.json
```

---

## 9. Teacher-forced eval 与 generation eval

### Teacher-forced eval

输入包含：

```text
image + input_text + output reasoning + [SEG]
```

模型直接使用已有 `[SEG]` 位置对应的 hidden state 预测 mask。

用途：评估 segmentation branch 是否学会 `[SEG] hidden → projector → mask decoder → mask`。

### Generation eval

输入只有：

```text
image + input_text
```

模型需要先生成：

```text
reasoning text + [SEG]
```

再用生成序列中的 `[SEG]` hidden state 预测 mask。

用途：评估完整端到端推理分割能力。

建议先用 teacher-forced eval 验证分割分支，再实现 generation eval。

---

## 10. DeepSpeed checkpoint 与 zero_to_fp32

训练保存的 `final_model/` 是 DeepSpeed checkpoint，不一定直接包含单个 `pytorch_model.bin`。

运行：

```bash
python zero_to_fp32.py . pytorch_model.bin
```

可能生成：

```text
pytorch_model-00001-of-00002.bin
pytorch_model-00002-of-00002.bin
pytorch_model.bin.index.json
```

这是正常的 HuggingFace 分片 checkpoint 格式，不是失败。

测试脚本需要支持：

```text
pytorch_model.bin
或
pytorch_model.bin.index.json + pytorch_model-xxxxx.bin shards
```

---

## 11. 完整流程

```bash
# 1. 数据检查
DATA_PATH=./datasets/BraTS2023_reasonseg_rgbA bash scripts/debug_dataset.sh

# 2. 模块检查
python tools/inspect_modules.py \
  --model_name_or_path ./checkpoints/Qwen/Qwen2.5-VL-3B-Instruct \
  --medsam_checkpoint ./checkpoints/medsam_vit_b.pth \
  --precision bf16 \
  --use_lora \
  --lora_scope llm \
  --train_seg_projector \
  --mask_decoder_train_mode none

# 3. Stage 1
bash scripts/stage1_projector_warmup.sh

# 4. Stage 2
bash scripts/stage2_reasoning_sft.sh

# 5. Stage 3
bash scripts/stage3_partial_decoder_adaptation.sh

# 6. Stage 4, optional
bash scripts/stage4_hard_finetune.sh

# 7. Test
bash scripts/test_final_model.sh
```

---

## 12. 常见问题

### Q1: 为什么阶段 1 和阶段 2 都训练 LLM LoRA + Projector？

Stage 1 使用：

```text
DETACH_SEG_HIDDEN_FOR_MASK=1
```

mask loss 不更新 LLM LoRA，更稳定，主要让 projector 学 prompt 空间。

Stage 2 使用：

```text
DETACH_SEG_HIDDEN_FOR_MASK=0
```

mask loss 可以更新 LLM LoRA，让 `[SEG] hidden` 更适合分割任务。

### Q2: Stage 3 显存为什么和前两阶段差不多？

Stage 3 只多训练约 693k 个 mask decoder 参数。相对于 Qwen2.5-VL 的数十亿参数和视觉 token 激活，新增 optimizer/gradient 显存很小，所以后台显存变化不明显是正常的。

### Q3: 微调后保存的是完整 Qwen 权重还是 LoRA 权重？

当前 DeepSpeed checkpoint 保存的是训练时的模型状态，包含冻结参数和可训练参数的 ZeRO 分片。运行 `zero_to_fp32.py` 后可得到完整 state dict 或分片 state dict。

工程上建议额外实现 portable save：

```text
lora_adapter/
reaseg_trainables.pt
tokenizer/
processor/
```

### Q4: 如果微调 MedSAM，是修改原始 medsam_vit_b.pth 吗？

不是。原始 `checkpoints/medsam_vit_b.pth` 作为初始化权重保留不变。训练后的 MedSAM decoder 参数保存在训练输出 checkpoint 中，例如：

```text
outputs/stage3_partial_decoder/final_model
outputs/stage4_hard_finetune/final_model
```

不要覆盖原始 `medsam_vit_b.pth`。
