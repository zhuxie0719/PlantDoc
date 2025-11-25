## ResNet50 基线实验计划

目标：在 11 月 26 日前完成 ResNet50 迁移学习的基线训练，获得可复现的无增强 vs. 基础增强对比及学习率/优化器消融结果，形成后续扩展实验的参照。

### 数据与加载
- 数据源：`data/processed/plantdoc_224`，划分由 `data/splits/plantdoc_split_seed42.json` 控制。
- 归一化：`mean=[0.485,0.456,0.406]`，`std=[0.229,0.224,0.225]`。
- 数据增强设置：
  1. **E2（无增强）**：仅 `Resize(224)` + `CenterCrop(224)` + Normalize（已在预处理完成，训练时只需 ToTensor + Normalize）。
  2. **E3（基础增强）**：随机水平翻转、随机旋转 ±20°、RandomResizedCrop(224, scale=[0.8,1.0])、ColorJitter(0.2,0.2,0.2,0.1)。

### 模型与优化
- 模型：`torchvision.models.resnet50(weights='IMAGENET1K_V2')`，分类头改为 27 类。
- 两阶段训练：
  1. **阶段 A（冻结骨干）**：只训练 fc 层 5 个 epoch，LR=1e-3，AdamW，weight_decay=1e-4。
  2. **阶段 B（全量微调）**：解冻全网，LR=1e-4（带 CosineAnnealingLR，T_max=12），batch size=16，训练 15 epoch，监控 `val_macro_f1` 早停 (patience=3)。
- 可选对比：SGD(momentum=0.9, lr=3e-3, cosine decay) 作为 E6 中的超参数消融记录。

### 监控与输出
- 指标：`val_acc`, `val_macro_f1`, `train/val loss`。
- 记录：保存 `outputs/logs/resnet50_baseline.csv`（每 epoch 指标）、`outputs/checkpoints/resnet50_{phase}_{metric}.ckpt`。
- 可视化：`outputs/figures/resnet50_lr1e-4_curves.png`，Grad-CAM 由最佳模型 (Phase B) 生成若干样例。

### 评估流程
1. 训练结束后，载入最佳 checkpoint，在验证集上计算混淆矩阵和 per-class 指标。
2. 使用相同权重在测试集推理，保存 `outputs/predictions/resnet50_seed42.csv`（包含 `image_path, pred_label`）。
3. 记录推理耗时（单张/整集）和 GPU 统计，用于报告的“训练时间/推理速度”栏。

### 负责人行动项
- [ ] 补充 `src/training/resnet50_baseline.py`（或 notebook `20_dl_resnet50.ipynb`）实现上述流程。
- [ ] 确认 wandb/TensorBoard 是否需要接入，如有需求同步 API Key。
- [ ] 与传统方法侧共享结果表模板，确保 E2/E3 指标能直接写入最终报告表格。

