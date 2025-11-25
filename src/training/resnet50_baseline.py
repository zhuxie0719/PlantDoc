"""
Training script that follows `report/resnet50_plan.md`.

Usage example (baseline without augmentation):

python -m src.training.resnet50_baseline \
    --experiment-name E2_resnet50_no_aug \
    --augment none \
    --log-csv outputs/logs/resnet50_E2.csv \
    --ckpt-dir outputs/checkpoints \
    --pred-csv outputs/predictions/resnet50_E2_test.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import models, transforms
    from torchvision.models import ResNet50_Weights
except ImportError as exc:
    raise ImportError(
        "torchvision is required for this script. Please install it via "
        "`pip install torchvision`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet50 baseline training pipeline.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/plantdoc_split_seed42.json"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/plantdoc_224"))
    parser.add_argument("--experiment-name", type=str, default="resnet50_baseline")
    parser.add_argument("--augment", choices=["none", "basic"], default="none")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-csv", type=Path, default=Path("outputs/logs/resnet50_train_log.csv"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--pred-csv", type=Path, default=Path("outputs/predictions/resnet50_test_preds.csv"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(split_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    payload = json.loads(split_path.read_text())
    return payload["classes"], payload["splits"]


class PlantDocDataset(Dataset):
    def __init__(
        self,
        split_name: str,
        entries: List[str],
        class_to_idx: Dict[str, int],
        processed_root: Path,
        transform,
    ) -> None:
        self.split_name = split_name
        self.processed_root = processed_root
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.items: List[Tuple[str, str, str]] = []
        for entry in entries:
            rel = Path(entry)
            self.items.append((rel.parent.name, rel.name, entry))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        class_name, filename, raw_rel = self.items[idx]
        path = self.processed_root / self.split_name / class_name / filename
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label, raw_rel


def build_transforms(augment: str):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment == "none":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )


def create_model(num_classes: int, device: str):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def build_optimizer(params: Iterable, args: argparse.Namespace, lr: float):
    if args.optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    momentum = 0.9
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=args.weight_decay)


def run_epoch(model, dataloader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    preds_all: List[int] = []
    labels_all: List[int] = []
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds_all.append(outputs.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())

    preds_tensor = torch.cat(preds_all)
    labels_tensor = torch.cat(labels_all)
    acc = (preds_tensor == labels_tensor).float().mean().item()
    macro_f1 = f1_score(labels_tensor.numpy(), preds_tensor.numpy(), average="macro")
    return {"loss": running_loss / total, "acc": acc, "macro_f1": macro_f1}


def save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_log(rows: List[Dict[str, float]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(model, dataloader, classes: List[str], device: str, output_csv: Path) -> None:
    import csv

    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels, raw_paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for pred, label, raw_rel in zip(preds, labels.tolist(), raw_paths):
                rows.append(
                    {
                        "image_path": raw_rel,
                        "pred_label": classes[pred],
                        "target_label": classes[label],
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pred_label", "target_label"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)

    classes, splits = load_splits(args.split_json)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    train_transform = build_transforms(args.augment)
    eval_transform = build_transforms("none")

    train_dataset = PlantDocDataset("train", splits["train"], class_to_idx, args.processed_root, train_transform)
    val_dataset = PlantDocDataset("val", splits["val"], class_to_idx, args.processed_root, eval_transform)
    test_dataset = PlantDocDataset("test", splits["test"], class_to_idx, args.processed_root, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = args.device
    model = create_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()

    log_rows: List[Dict[str, float]] = []
    best_metric = -float("inf")
    best_ckpt = args.ckpt_dir / f"{args.experiment_name}_best.pt"

    if args.freeze_epochs > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = build_optimizer(model.fc.parameters(), args, args.lr_head)
        for epoch in range(1, args.freeze_epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            row = {
                "phase": "frozen",
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            log_rows.append(row)
            print(row)
            if val_metrics["macro_f1"] > best_metric:
                best_metric = val_metrics["macro_f1"]
                save_checkpoint(model, best_ckpt)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = build_optimizer(model.parameters(), args, args.lr_backbone)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    patience = 3
    patience_counter = 0
    for epoch in range(1, args.finetune_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "phase": "finetune",
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
        }
        log_rows.append(row)
        print(row)

        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            patience_counter = 0
            save_checkpoint(model, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    save_log(log_rows, args.log_csv)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Best checkpoint evaluated on test: {test_metrics}")
    save_predictions(model, test_loader, classes, device, args.pred_csv)


if __name__ == "__main__":
    main()

