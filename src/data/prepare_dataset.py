import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps
from tqdm import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    class_name: str
    rel_path: Path  # relative to repo `data/` folder


def collect_samples(raw_root: Path) -> List[Sample]:
    """
    Traverse TRAIN/TEST folders under the PlantDoc raw directory and collect image paths.
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root not found: {raw_root}")

    samples: List[Sample] = []
    for subset in ("TRAIN", "TEST"):
        subset_dir = raw_root / subset
        if not subset_dir.exists():
            continue
        for class_dir in sorted(subset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() not in VALID_EXTS:
                    continue
                rel_to_data = Path("raw") / img_path.relative_to(raw_root.parent)
                samples.append(Sample(class_name=class_dir.name, rel_path=rel_to_data))
    if not samples:
        raise RuntimeError(f"No images found under {raw_root}")
    return samples


def build_class_stats(samples: Sequence[Sample]) -> pd.DataFrame:
    counter = Counter(sample.class_name for sample in samples)
    df = (
        pd.DataFrame(
            {"class_name": list(counter.keys()), "count": list(counter.values())}
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    total = df["count"].sum()
    df["ratio"] = df["count"] / total
    return df


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, max(6, len(df) * 0.3)))
    sns.barplot(data=df, y="class_name", x="count", palette="viridis")
    plt.title("PlantDoc Class Distribution")
    plt.xlabel("Image count")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_sample_grid(samples: Sequence[Sample], raw_data_root: Path, output_path: Path) -> None:
    rng = random.Random(42)
    chosen = rng.sample(samples, k=min(16, len(samples)))
    n_cols = 4
    n_rows = int(np.ceil(len(chosen) / n_cols))
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    for idx, sample in enumerate(chosen):
        img_path = Path("data") / sample.rel_path
        image = Image.open(img_path).convert("RGB")
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.imshow(image)
        ax.set_title(sample.class_name, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def stratified_split(
    samples: Sequence[Sample],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[Path]]:
    class_to_samples: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        class_to_samples[sample.class_name].append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_name, class_samples in class_to_samples.items():
        class_list = class_samples.copy()
        rng.shuffle(class_list)
        n = len(class_list)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        if n_train + n_val > n - 1:
            # leave at least one sample for test if possible
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        for idx, sample in enumerate(class_list):
            if idx < n_train:
                splits["train"].append(sample.rel_path)
            elif idx < n_train + n_val:
                splits["val"].append(sample.rel_path)
            else:
                splits["test"].append(sample.rel_path)
    return splits


def save_splits_json(
    splits: Dict[str, List[Path]],
    output_path: Path,
    classes: Iterable[str],
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset": "PlantDoc",
            "seed": seed,
            "created_by": "prepare_dataset.py",
        },
        "classes": sorted(classes),
        "splits": {k: [path.as_posix() for path in v] for k, v in splits.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def preprocess_images(
    splits: Dict[str, List[Path]],
    processed_root: Path,
    target_size: int = 256,
    crop_size: int = 224,
) -> None:
    processed_root = processed_root.resolve()
    processed_root.mkdir(parents=True, exist_ok=True)
    for split_name, split_paths in splits.items():
        iterator = tqdm(split_paths, desc=f"Processing {split_name}", unit="img")
        for rel_path in iterator:
            src_path = Path("data") / rel_path
            if not src_path.exists():
                raise FileNotFoundError(f"Missing file referenced in split: {src_path}")
            class_name = src_path.parent.name
            dest_dir = processed_root / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name
            with Image.open(src_path) as img:
                image = img.convert("RGB")
                image = ImageOps.fit(
                    image,
                    size=(target_size, target_size),
                    method=Image.Resampling.BICUBIC,
                )
                if crop_size and crop_size < target_size:
                    delta = (target_size - crop_size) // 2
                    image = image.crop(
                        (delta, delta, delta + crop_size, delta + crop_size)
                    )
                image.save(dest_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA + preprocessing workflow for PlantDoc.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/PlantDoc"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/plantdoc_224"))
    parser.add_argument("--splits-out", type=Path, default=Path("data/splits/plantdoc_split_seed42.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--skip-preprocess", action="store_true", help="Only compute stats and splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root
    samples = collect_samples(raw_root)

    stats_df = build_class_stats(samples)
    logs_dir = Path("outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = logs_dir / "class_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    plot_class_distribution(stats_df, Path("outputs/figures/class_distribution.png"))
    plot_sample_grid(samples, raw_root, Path("outputs/figures/sample_grid.png"))

    splits = stratified_split(samples, args.seed, args.train_ratio, args.val_ratio)
    save_splits_json(splits, args.splits_out, stats_df["class_name"].tolist(), args.seed)

    if not args.skip_preprocess:
        preprocess_images(splits, args.processed_root)

    print(f"Saved class stats to {stats_csv}")
    print(f"Saved figures under outputs/figures")
    print(f"Wrote splits to {args.splits_out}")
    if not args.skip_preprocess:
        print(f"Processed images under {args.processed_root}")


if __name__ == "__main__":
    main()

