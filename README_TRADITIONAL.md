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

打开 `notebooks/plant_traditional_optimized.ipynb` 并运行所有单元格。

该notebook将：
1. 加载数据（使用统一的数据划分）
2. 提取HOG特征
3. 训练SVM分类器
4. 评估并可视化结果
5. 保存模型和结果

**预期输出**：
- `outputs/checkpoints/E1_final_optimized_model.pkl` - 优化的模型
- `outputs/logs/E1-experimental_results.csv` - 实验结果
- `outputs/figures/hog_visualization_example.png` - HOG特征可视化
- `outputs/figures/E1_hog_svm_confusion_matrix.png` - 混淆矩阵-优化前
- `outputs/figures/E1_final_confusion_matrix.png` - 混淆矩阵-优化后
- `outputs/logs/E7_ablation_results.csv` - 消融实验结果





