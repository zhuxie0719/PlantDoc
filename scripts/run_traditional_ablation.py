#!/usr/bin/env python
"""
Command-line utilities for PlantDoc传统方法实验。

提供三个主要阶段：
1. extract-features：提取 HOG / 颜色直方图 / LBP 等特征并缓存为 .npy，顺便缓存标签与类名。
2. feature-ablation：在缓存特征上跑特征组合消融实验（默认使用 SVM）。
3. classifier-ablation：在指定特征组合上对比 SVM / Logistic Regression / MLP。

借助磁盘缓存，可以实现断点续跑：耗时的特征提取只需执行一次，后续实验直接复用缓存文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# 确保可以 import src 内的模块
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.traditional_loader import load_all_splits
from models.traditional_features import (
    extract_color_histogram,
    extract_hog_features,
    extract_lbp_features,
    combine_features,
)
from models.traditional_classifiers import (
    evaluate_model,
    train_logistic_regression,
    train_mlp,
    train_svm,
)


DEFAULT_FEATURE_COMBOS = {
    "HOG only": ["hog"],
    "HOG + Color": ["hog", "color"],
    "HOG + LBP": ["hog", "lbp"],
    "HOG + Color + LBP": ["hog", "color", "lbp"],
}


def default_paths():
    data_dir = ROOT_DIR / "data"
    outputs_dir = ROOT_DIR / "outputs"
    return {
        "split_json": data_dir / "splits" / "plantdoc_split_seed42.json",
        "processed_root": data_dir / "processed" / "plantdoc_224",
        "cache_dir": outputs_dir / "cache" / "traditional_features",
        "logs_dir": outputs_dir / "logs",
        "figures_dir": outputs_dir / "figures",
    }


def cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def save_numpy(path: Path, array: np.ndarray, force: bool = False):
    if path.exists() and not force:
        print(f"[skip] {path.name} 已存在，使用 --force 可覆盖。")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    print(f"[cache] 保存 {path}")


def load_numpy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到缓存文件：{path}")
    return np.load(path)


def save_metadata(cache_dir: Path, class_names: List[str]):
    meta_path = cache_path(cache_dir, "metadata.json")
    payload = {"class_names": class_names}
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[cache] 保存 {meta_path}")


def load_metadata(cache_dir: Path) -> Dict:
    meta_path = cache_path(cache_dir, "metadata.json")
    if not meta_path.exists():
        raise FileNotFoundError("metadata.json 不存在，请先运行 extract-features 阶段。")
    return json.loads(meta_path.read_text())


def feature_extractors():
    return {
        "hog": extract_hog_features,
        "color": extract_color_histogram,
        "lbp": extract_lbp_features,
    }


def run_extract_features(args):
    paths = default_paths()
    cache_dir: Path = args.cache_dir or paths["cache_dir"]

    print("===> 加载图像数据 ...")
    splits_data, class_names = load_all_splits(
        split_json_path=paths["split_json"],
        processed_root=paths["processed_root"],
        target_size=(args.target_size, args.target_size) if args.target_size else None,
        grayscale=False,
    )

    # 缓存标签和类名
    save_metadata(cache_dir, class_names)
    for split_name, (_, labels, _) in splits_data.items():
        save_numpy(cache_path(cache_dir, f"labels_{split_name}.npy"), labels, force=args.force)

    extractor_map = feature_extractors()

    for feature_name in args.features:
        feature_name = feature_name.lower()
        if feature_name not in extractor_map:
            raise ValueError(f"暂不支持的特征类型：{feature_name}")

        extractor = extractor_map[feature_name]
        print(f"\n===> 提取特征：{feature_name.upper()}")

        for split_name, (images, _, _) in splits_data.items():
            cache_file = cache_path(cache_dir, f"{feature_name}_{split_name}.npy")
            if cache_file.exists() and not args.force:
                print(f"[skip] {cache_file.name} 已存在，跳过计算。")
                continue

            if feature_name == "color":
                feats = extractor(images, bins=args.color_bins)
            elif feature_name == "hog":
                feats = extractor(
                    images,
                    orientations=args.hog_orientations,
                    pixels_per_cell=(args.pixels_per_cell, args.pixels_per_cell),
                    cells_per_block=(args.cells_per_block, args.cells_per_block),
                )
            else:
                feats = extractor(images)

            save_numpy(cache_file, feats, force=True)

    print("\n[done] 特征提取阶段完成。")


def load_cached_features(cache_dir: Path, feature_names: Sequence[str], splits: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    store: Dict[str, Dict[str, np.ndarray]] = {}
    for feature in feature_names:
        feature = feature.lower()
        store[feature] = {}
        for split in splits:
            path = cache_path(cache_dir, f"{feature}_{split}.npy")
            store[feature][split] = load_numpy(path)
    return store


def load_labels(cache_dir: Path, splits: Sequence[str]) -> Dict[str, np.ndarray]:
    labels = {}
    for split in splits:
        path = cache_path(cache_dir, f"labels_{split}.npy")
        labels[split] = load_numpy(path)
    return labels


def combine_with_scaler(feature_list: Sequence[str], feature_store: Dict[str, Dict[str, np.ndarray]]):
    feature_list = [f.lower() for f in feature_list]
    train_dict = {name: feature_store[name]["train"] for name in feature_list}
    train_combined, scaler = combine_features(train_dict, normalize=True)

    def transform(split: str):
        stacked = np.hstack([feature_store[name][split] for name in feature_list])
        return scaler.transform(stacked) if scaler is not None else stacked

    return {
        "train": train_combined,
        "val": transform("val"),
        "test": transform("test"),
    }


def run_feature_ablation(args):
    paths = default_paths()
    cache_dir: Path = args.cache_dir or paths["cache_dir"]
    logs_dir: Path = args.logs_dir or paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(cache_dir)
    labels = load_labels(cache_dir, ["train", "val", "test"])

    combo_defs = DEFAULT_FEATURE_COMBOS.copy()
    if args.extra_combo:
        combo_defs.update(args.extra_combo)

    required_features = {feat for feats in combo_defs.values() for feat in feats}
    feature_store = load_cached_features(cache_dir, required_features, ["train", "val", "test"])

    feature_results = []
    for combo_name, feature_list in combo_defs.items():
        print(f"\n{'='*60}\n组合：{combo_name}\n包含特征：{', '.join(feature_list)}\n{'='*60}")
        matrices = combine_with_scaler(feature_list, feature_store)

        model = train_svm(
            matrices["train"],
            labels["train"],
            kernel=args.kernel,
            C=args.c_value,
            class_weight="balanced",
        )

        val_metrics = evaluate_model(
            model,
            matrices["val"],
            labels["val"],
            metadata["class_names"],
            verbose=False,
        )
        test_metrics = evaluate_model(
            model,
            matrices["test"],
            labels["test"],
            metadata["class_names"],
            verbose=False,
        )

        feature_results.append(
            {
                "feature": combo_name,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )

        print(
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['macro_f1']:.4f}"
        )

    feature_df = pd.DataFrame(feature_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = logs_dir / f"ablation_features_{timestamp}.csv"
    feature_df.to_csv(csv_path, index=False)
    print(f"\n[done] 特征组合结果已保存：{csv_path}")


def run_classifier_ablation(args):
    paths = default_paths()
    cache_dir: Path = args.cache_dir or paths["cache_dir"]
    logs_dir: Path = args.logs_dir or paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(cache_dir)
    labels = load_labels(cache_dir, ["train", "test"])

    combo_defs = DEFAULT_FEATURE_COMBOS.copy()
    if args.extra_combo:
        combo_defs.update(args.extra_combo)

    if args.feature_combo not in combo_defs:
        raise ValueError(f"未知的特征组合：{args.feature_combo}")

    feature_list = combo_defs[args.feature_combo]
    feature_store = load_cached_features(cache_dir, feature_list, ["train", "test"])
    matrices = combine_with_scaler(feature_list, feature_store)

    classifier_results = []

    if not args.disable_svm:
        print("\n===> 训练 SVM ...")
        svm_model = train_svm(
            matrices["train"],
            labels["train"],
            kernel=args.svm_kernel,
            C=args.svm_c,
            class_weight="balanced",
        )
        svm_metrics = evaluate_model(
            svm_model,
            matrices["test"],
            labels["test"],
            metadata["class_names"],
            verbose=False,
        )
        classifier_results.append(
            {
                "classifier": "SVM",
                "accuracy": svm_metrics["accuracy"],
                "macro_f1": svm_metrics["macro_f1"],
                "weighted_f1": svm_metrics["weighted_f1"],
            }
        )

    if not args.disable_lr:
        print("\n===> 训练 Logistic Regression ...")
        lr_model = train_logistic_regression(
            matrices["train"],
            labels["train"],
            C=args.lr_c,
            class_weight="balanced",
        )
        lr_metrics = evaluate_model(
            lr_model,
            matrices["test"],
            labels["test"],
            metadata["class_names"],
            verbose=False,
        )
        classifier_results.append(
            {
                "classifier": "Logistic Regression",
                "accuracy": lr_metrics["accuracy"],
                "macro_f1": lr_metrics["macro_f1"],
                "weighted_f1": lr_metrics["weighted_f1"],
            }
        )

    if not args.disable_mlp:
        print("\n===> 训练 MLP ...")
        mlp_model = train_mlp(
            matrices["train"],
            labels["train"],
            hidden_layer_sizes=tuple(args.mlp_hidden),
            max_iter=args.mlp_max_iter,
        )
        mlp_metrics = evaluate_model(
            mlp_model,
            matrices["test"],
            labels["test"],
            metadata["class_names"],
            verbose=False,
        )
        classifier_results.append(
            {
                "classifier": f"MLP{tuple(args.mlp_hidden)}",
                "accuracy": mlp_metrics["accuracy"],
                "macro_f1": mlp_metrics["macro_f1"],
                "weighted_f1": mlp_metrics["weighted_f1"],
            }
        )

    classifier_df = pd.DataFrame(classifier_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = logs_dir / f"ablation_classifiers_{timestamp}.csv"
    classifier_df.to_csv(csv_path, index=False)
    print(f"\n[done] 分类器对比结果已保存：{csv_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="PlantDoc传统方法实验脚本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract-features
    extract_p = subparsers.add_parser("extract-features", help="提取并缓存特征")
    extract_p.add_argument(
        "--features",
        nargs="+",
        default=["hog", "color", "lbp"],
        help="需要提取的特征列表",
    )
    extract_p.add_argument("--cache-dir", type=Path, default=None, help="缓存输出目录")
    extract_p.add_argument("--target-size", type=int, default=256, help="resize大小（正方形）")
    extract_p.add_argument("--hog-orientations", type=int, default=9)
    extract_p.add_argument("--pixels-per-cell", type=int, default=8)
    extract_p.add_argument("--cells-per-block", type=int, default=2)
    extract_p.add_argument("--color-bins", type=int, default=32)
    extract_p.add_argument("--force", action="store_true", help="若已存在缓存则覆盖")
    extract_p.set_defaults(func=run_extract_features)

    # feature-ablation
    feat_ablation = subparsers.add_parser("feature-ablation", help="运行特征组合消融实验")
    feat_ablation.add_argument("--cache-dir", type=Path, default=None)
    feat_ablation.add_argument("--logs-dir", type=Path, default=None)
    feat_ablation.add_argument("--kernel", type=str, default="rbf")
    feat_ablation.add_argument("--c-value", type=float, default=1.0)
    feat_ablation.add_argument(
        "--extra-combo",
        type=json.loads,
        help='额外组合，格式如 \'{"HOG + PCA":["hog"]}\'',
    )
    feat_ablation.set_defaults(func=run_feature_ablation)

    # classifier-ablation
    clf_ablation = subparsers.add_parser("classifier-ablation", help="运行分类器对比实验")
    clf_ablation.add_argument("--cache-dir", type=Path, default=None)
    clf_ablation.add_argument("--logs-dir", type=Path, default=None)
    clf_ablation.add_argument(
        "--feature-combo",
        type=str,
        default="HOG + Color + LBP",
        help="选择使用的特征组合名称",
    )
    clf_ablation.add_argument(
        "--extra-combo",
        type=json.loads,
        help='额外组合，格式如 \'{"Custom":["hog","color"]}\'',
    )
    clf_ablation.add_argument("--disable-svm", action="store_true")
    clf_ablation.add_argument("--disable-lr", action="store_true")
    clf_ablation.add_argument("--disable-mlp", action="store_true")
    clf_ablation.add_argument("--svm-kernel", type=str, default="rbf")
    clf_ablation.add_argument("--svm-c", type=float, default=1.0)
    clf_ablation.add_argument("--lr-c", type=float, default=1.0)
    clf_ablation.add_argument(
        "--mlp-hidden",
        type=int,
        nargs="+",
        default=[128, 64],
        help="MLP隐藏层维度",
    )
    clf_ablation.add_argument("--mlp-max-iter", type=int, default=500)
    clf_ablation.set_defaults(func=run_classifier_ablation)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()



