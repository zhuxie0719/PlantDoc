import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np


class PlantDocDataset(Dataset):
    """
    Thin dataset wrapper that reads the processed images based on the canonical split JSON.
    """

    def __init__(
        self,
        split_name: str,
        split_entries: List[str],
        class_to_idx: Dict[str, int],
        processed_root: Path,
        transform=None,
    ) -> None:
        self.split_name = split_name
        self.entries: List[Tuple[str, str]] = []
        for entry in split_entries:
            rel = Path(entry)
            class_name = rel.parent.name
            self.entries.append((class_name, rel.name))
        self.class_to_idx = class_to_idx
        self.processed_root = processed_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        class_name, filename = self.entries[idx]
        img_path = self.processed_root / self.split_name / class_name / filename
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label


class SimpleTransform:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def load_splits(split_path: Path):
    payload = json.loads(split_path.read_text())
    return payload["classes"], payload["splits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick sanity check for PlantDoc dataloader.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/plantdoc_split_seed42.json"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/plantdoc_224"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-batch", action="store_true", help="Show a single batch summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes, splits = load_splits(args.split_json)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    transform = SimpleTransform()

    dataset = PlantDocDataset(
        split_name=args.split,
        split_entries=splits[args.split],
        class_to_idx=class_to_idx,
        processed_root=args.processed_root,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Split: {args.split} | samples: {len(dataset)} | classes: {len(classes)}")
    if args.preview_batch:
        batch = next(iter(dataloader))
        images, labels = batch
        print(f"Batch images tensor: {images.shape}, dtype={images.dtype}")
        print(f"Batch labels tensor: {labels.shape}, min={labels.min().item()}, max={labels.max().item()}")
        unique, counts = labels.unique(return_counts=True)
        print("Batch label histogram:")
        for cls_idx, count in zip(unique.tolist(), counts.tolist()):
            print(f"  {classes[cls_idx]}: {count}")


if __name__ == "__main__":
    main()

