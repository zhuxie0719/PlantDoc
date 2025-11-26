# 传统方法路线使用指南

本指南说明如何使用传统机器学习方法进行PlantDoc植物病害分类任务。

## 快速开始

### 1. 环境准备

确保安装以下依赖：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image opencv-python opencv-contrib-python tqdm
```

**注意**：
- `opencv-contrib-python` 用于SIFT/SURF特征（可选）
- `scikit-image` 用于HOG和LBP特征

### 2. 数据准备

确保已完成数据预处理（成员A已完成）：
- `data/splits/plantdoc_split_seed42.json` - 数据划分文件
- `data/processed/plantdoc_224/` - 预处理后的图像

### 3. 运行基线实验（E1）

打开 `notebooks/10_traditional_baseline.ipynb` 并运行所有单元格。

该notebook将：
1. 加载数据（使用统一的数据划分）
2. 提取HOG特征
3. 训练SVM分类器
4. 评估并可视化结果
5. 保存模型和结果

**预期输出**：
- `outputs/checkpoints/E1_hog_svm_model.pkl` - 训练好的模型
- `outputs/logs/E1_hog_svm_results.csv` - 实验结果
- `outputs/figures/E1_hog_svm_confusion_matrix.png` - 混淆矩阵
- `outputs/figures/hog_visualization_example.png` - HOG特征可视化

### 4. 运行消融实验

消融实验拆成了“脚本 + Notebook”双通道：

- **第一步（推荐）**：在命令行运行 `scripts/run_traditional_ablation.py`，先缓存特征再跑批量实验。例如：
  ```bash
  # 仅需执行一次，缓存 HOG/颜色直方图/LBP 特征与标签
  python scripts/run_traditional_ablation.py extract-features

  # 复用缓存做特征组合消融（默认 SVM）
  python scripts/run_traditional_ablation.py feature-ablation

  # 在最佳特征组合上对比分类器
  python scripts/run_traditional_ablation.py classifier-ablation --feature-combo "HOG + Color + LBP"
  ```
- **第二步**：打开 `notebooks/11_traditional_ablation.ipynb`，Notebook 会优先读取 `outputs/cache/traditional_features/` 中的缓存；若缺失才会自动补算，并立即写回缓存，实现断点续跑。

## 代码结构

```
src/
├── data/
│   └── traditional_loader.py      # 数据加载工具
├── models/
│   ├── traditional_features.py    # 特征提取（HOG, SIFT, 颜色直方图, LBP）
│   └── traditional_classifiers.py # 分类器（SVM, LR, MLP）
└── eval/
    └── traditional_eval.py        # 评估与可视化工具
```

## 主要功能

### 特征提取

- **HOG**: `extract_hog_features()` - 方向梯度直方图
- **颜色直方图**: `extract_color_histogram()` - RGB/HSV/LAB颜色特征
- **LBP**: `extract_lbp_features()` - 局部二值模式纹理特征
- **SIFT/SURF**: `extract_sift_features()` / `extract_surf_features()` - 关键点特征（需要opencv-contrib）

### 分类器

- **SVM**: `train_svm()` - 支持RBF/Linear/Poly核
- **Logistic Regression**: `train_logistic_regression()` - 带L2正则化
- **MLP**: `train_mlp()` - 多层感知机

### 评估与可视化

- `plot_confusion_matrix()` - 混淆矩阵
- `plot_feature_comparison()` - 特征组合对比
- `plot_classifier_comparison()` - 分类器对比
- `visualize_hog_features()` - HOG特征可视化

## 实验记录

所有实验结果保存在 `outputs/logs/` 目录下，格式为CSV。

建议记录：
- 实验编号（E1, E7等）
- 特征类型
- 分类器类型
- 验证集/测试集准确率和F1分数
- 训练时间
- 备注

## 注意事项

1. **数据一致性**：始终使用 `plantdoc_split_seed42.json` 保持与深度学习实验一致
2. **特征维度**：不同特征组合会产生不同维度的特征向量，注意内存使用
3. **计算时间**：SIFT/SURF特征提取较慢，建议先用HOG+颜色+LBP
4. **类别不平衡**：使用 `class_weight="balanced"` 处理类别不平衡
5. **特征归一化**：使用 `combine_features(normalize=True)` 进行特征归一化

## 下一步

完成基线实验后，可以：
1. 尝试不同的HOG参数（orientations, pixels_per_cell等）
2. 添加SIFT/SURF特征（需要opencv-contrib）
3. 进行PCA降维实验
4. 尝试不同的分类器超参数
5. 分析错误样例并可视化

## 问题排查

**ImportError: scikit-image / opencv-python**
- 运行 `pip install scikit-image opencv-python`

**SURF不可用**
- 需要安装 `opencv-contrib-python` 而非 `opencv-python`
- 或跳过SURF，使用SIFT

**内存不足**
- 减小图像尺寸（如使用128x128而非256x256）
- 使用PCA降维
- 分批处理特征提取

**训练时间过长**
- 使用较小的C值（SVM）
- 减少MLP隐藏层大小
- 使用Linear SVM而非RBF







