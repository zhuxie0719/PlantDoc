# ResNet50 Baseline & Augmentation Experiments

| Experiment | Purpose | Key Hyperparams | Current Status | Artifacts |
| --- | --- | --- | --- | --- |
| **E2 – 无增强基线** | 迁移学习基线（指南 E2） | `augment=none`, AdamW (`lr_head=1e-3`, `lr_backbone=1e-4`), `freeze_epochs=5`, `finetune_epochs=15`, `batch_size=16` | ✅ 完成，最佳验证 Macro-F1 ≈ **0.646**（finetune epoch 6），测试集推理已输出 `resnet50_E2_test.csv` | log: `outputs/logs/resnet50_E2.csv`; ckpt: `outputs/checkpoints/E2_resnet50_no_aug_best.pt`; preds: `outputs/predictions/resnet50_E2_test.csv` |
| **E3 – 基础增强** | 对比基础增强对性能影响（指南 E3） | `augment=basic`, 其余同 E2 | ✅ 完成，最佳验证 Macro-F1 ≈ **0.660**（finetune epoch 8），测试集推理输出 `resnet50_E3_test.csv` | log: `outputs/logs/resnet50_E3.csv`; ckpt: `outputs/checkpoints/E3_resnet50_basic_best.pt`; preds: `outputs/predictions/resnet50_E3_test.csv` |
| **E3.1 – SGD 对照 v1** | AdamW → SGD 的第一轮尝试 | `augment=basic`, `optimizer=sgd`, `freeze_epochs=3`, `finetune_epochs=15`, `lr_head=3e-3`, `lr_backbone=3e-3`, `batch_size=16`, `num_workers=0` | ✅ 已完成：最佳验证 Macro-F1 ≈ **0.627**（finetune epoch 9），测试 Macro-F1 ≈ 0.627。整体略低于 E2/E3，作为负向对照保留 | log: `outputs/logs/resnet50_E3_1.csv`; ckpt: `outputs/checkpoints/E3_1_resnet50_sgd_best.pt`; preds: `outputs/predictions/resnet50_E3_1_test.csv` |
| **E3.2 – SGD 对照 v2** | 在更长微调 + 降低学习率下的第二轮尝试 | `augment=basic`, `optimizer=sgd`, `freeze_epochs=2`, `finetune_epochs=20`, `lr_head=2e-3`, `lr_backbone=8e-4`, `batch_size=32`（Colab 调整为 24~32） | ✅ 已完成：最佳验证 Macro-F1 ≈ **0.59**（早停前），最终测试 Macro-F1 ≈ **0.55**，表现仍低于 E2/E3 | log: `outputs/logs/resnet50_E3_2.csv`（Colab）、ckpt: `outputs/checkpoints/E3_2_resnet50_sgd_best.pt`, preds: `outputs/predictions/resnet50_E3_2_test.csv` |
| **E3.3 – 优化版（AdamW 回归）** | 回到 AdamW 并延长微调，希望超过 E3 | `augment=basic`, `optimizer=adamw`, `freeze_epochs=4`, `finetune_epochs=25`, `lr_head=7e-4`, `lr_backbone=7e-5`, `batch_size=24`, `weight_decay=5e-5` | ✅ 已完成：验证集 Macro-F1 最高约 **0.65**（早停前），测试 Macro-F1 ≈ **0.628**，仍略低于 E3 | log: `outputs/logs/resnet50_E3_3.csv`; ckpt: `outputs/checkpoints/E3_3_resnet50_adamw_best.pt`; preds: `outputs/predictions/resnet50_E3_3_test.csv` |
| **E3.4 – 高级正则（AdamW + Label Smoothing）** | 引入 label smoothing、降低学习率以提升泛化 | `augment=basic`, `optimizer=adamw`, `freeze_epochs=3`, `finetune_epochs=30`, `lr_head=5e-4`, `lr_backbone=3e-5`, `batch_size=24`, `weight_decay=1e-5`, `label_smoothing=0.1` | ✅ 已完成：在 finetune epoch 13 前后触发早停，最佳验证 Macro-F1 ≈ **0.646**，测试 Macro-F1 ≈ **0.647**。验证曲线仍在 0.64~0.65 区间震荡 | log: `outputs/logs/resnet50_E3_4.csv`; ckpt: `outputs/checkpoints/E3_4_resnet50_adamw_ls_best.pt`; preds: `outputs/predictions/resnet50_E3_4_test.csv` |
| **E3.5 – 强增强 + Mixup** | 叠加 RandAugment + Mixup/CutMix + Label Smoothing | `augment=basic`, `randaug_num_ops=2`, `randaug_magnitude=12`, `mixup_alpha=0.2`, `cutmix_alpha=1.0`, `label_smoothing=0.1`, AdamW, `freeze_epochs=2`, `finetune_epochs=35`, `lr_head=7e-4`, `lr_backbone=5e-5`, `batch_size=32`, `weight_decay=5e-5` | ✅ 已完成：验证 Macro-F1 **≈0.679**、测试 Macro-F1 **≈0.679**，较之前有显著提升 | log: `outputs/logs/resnet50_E3_5.csv`; ckpt: `outputs/checkpoints/E3_5_resnet50_strong_aug_best.pt`; preds: `outputs/predictions/resnet50_E3_5_test.csv` |
| **E3.6 – 强增强 + Mixup** | 延续 E3.5 成功经验，进一步提高扰动幅度，追求验证 ≥0.70 | `augment=basic`, `randaug_num_ops=3`, `randaug_magnitude=15`, `mixup_alpha=0.3`, `cutmix_alpha=1.0`, `label_smoothing=0.1`, AdamW，`freeze_epochs=2`, `finetune_epochs=35`, `lr_head=8e-4`, `lr_backbone=6e-5`, `weight_decay=3e-5`, `batch_size=32`，`device=cuda`, `num_workers=4` | ✅ 已完成：验证 Macro-F1 **≈0.706**（finetune epoch 14），测试 Macro-F1 **≈0.700**，首次突破 0.70 | log: `outputs/logs/resnet50_E3_6.csv`; ckpt: `outputs/checkpoints/E3_6_resnet50_strong_aug_warmup_best.pt`; preds: `outputs/predictions/resnet50_E3_6_test.csv` |
| **E3.7 – 强增强 + Warmup/Dropout** | 在 E3.6 基础上引入更强 mixup、warmup、fc Dropout=0.2 | `augment=basic`, `randaug_num_ops=3`, `randaug_magnitude=16`, `mixup_alpha=0.4`, `cutmix_alpha=1.0`, `label_smoothing=0.1`, AdamW，`freeze_epochs=1`, `finetune_epochs=40`, `lr_head=7e-4`, `lr_backbone=4e-5`, `weight_decay=2e-5`, `batch_size=32`, `device=cuda`, `num_workers=4` | ✅ 已完成：验证 Macro-F1 **≈0.693**、测试 Macro-F1 **≈0.671**，相比 E3.6 略有回落，说明过强的正则导致欠拟合 | log: `outputs/logs/resnet50_E3_7.csv`; ckpt: `outputs/checkpoints/E3_7_resnet50_strong_aug_warmup_drop_best.pt`; preds: `outputs/predictions/resnet50_E3_7_test.csv` |
| 

> 每次实验完成后，请把最终验证/测试指标追加到该表中（如 Macro-F1、Top-1 Accuracy），并简要记录训练曲线、Grad-CAM、混淆矩阵等关键可视化的路径。

## 调参理由与思路

- **E2（无增强基线）**：以最小化复杂度为目标，只做图像标准化，观察纯迁移学习的上限，为后续所有实验提供参照。
- **E3（基础增强）**：在 E2 成功收敛的基础上引入翻转/旋转/ColorJitter，以缓解过拟合及覆盖真实场景的光照、角度变化；实验结果确实带来 ~1.4% 的验证增益。
- **E3.1（SGD 对照 v1）**：导师要求展示调参过程，因此将优化器从 AdamW 换成 SGD，学习率提高到 3e-3，并缩短冻结阶段，目的在于对比不同优化器的收敛特性；结果略差，说明 SGD 在该设置下不如 AdamW。
- **E3.2（SGD 对照 v2）**: 针对 E3.1 过快震荡的问题，降低学习率并延长微调（20 epoch），同时减少冻结层数，希望提升稳定性；验证略有提升但仍低于 AdamW，证明 SGD 方案需更激进的正则或其它策略。
- **E3.3（优化版实测）**：实施回归 AdamW + 长周期训练，验证仍停留在 ~0.65，说明仅靠延长训练/降低学习率不足以超过 E3，需要额外正则或更强数据扰动。
- **E3.4（实测）**：加入 label smoothing 和更低学习率后，验证曲线依旧在 0.64~0.65，且早停在 epoch 13，说明当前增强/学习率组合仍不足以突破瓶颈，需尝试更强的数据扰动。
- **E3.5（实测）**：RandAugment + Mixup/CutMix + Label Smoothing 的组合显著提升了验证性能（~0.679），证明多重正则方向有效。
- **E3.6（实测）**：更强的 RandAugment/ Mixup 将验证 Macro-F1 推至 0.706、测试 0.700，标志着 0.70 目标已达成。
- **E3.7（实测）**：加入 warmup/Dropout 后验证回落到 ~0.693，说明正则过强导致欠拟合。


