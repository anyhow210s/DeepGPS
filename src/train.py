"""Train a transfer-learning ResNet50 photo geolocation classifier.

English:
    Trains the classifier head of a pretrained ResNet50, optionally fine-tunes
    the final ResNet block, and saves the best model and learning curves.

한국어:
    사전 학습된 ResNet50의 분류기 헤드를 학습하고, 선택적으로 마지막 ResNet
    블록을 미세 조정하며, 최적 모델과 학습 곡선을 저장합니다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    HISTORY_PATH,
    PROCESSED_DATA_DIR,
    build_datasets,
    build_model,
    ensure_output_dirs,
    freeze_backbone,
    get_device,
    save_checkpoint,
    save_class_names,
    set_seed,
    unfreeze_last_block,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_class_counts(dataset: torch.utils.data.Dataset, split_name: str) -> None:
    """Print the number of images in each class."""
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, class_index in dataset.samples:
        counts[dataset.classes[class_index]] += 1
    print(f"\n{split_name} class counts:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, running_correct / total


def plot_history(history: list[dict[str, float | int | str]]) -> None:
    """Save loss and accuracy learning curves."""
    frame = pd.DataFrame(history)
    frame.to_csv(HISTORY_PATH, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Training loss")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="Training accuracy")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "training_history.png", dpi=200)
    plt.close(figure)


def train_phase(
    phase_name: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    epoch_offset: int,
    class_names: list[str],
    best_accuracy: float,
    history: list[dict[str, float | int | str]],
) -> float:
    """Train and validate one model phase."""
    for phase_epoch in range(1, epochs + 1):
        epoch = epoch_offset + phase_epoch
        start = time.time()
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)

        history.append(
            {
                "epoch": epoch,
                "phase": phase_name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d} [{phase_name}] "
            f"train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f} acc={val_accuracy:.4f} | "
            f"{time.time() - start:.1f}s"
        )

        # Save only when validation accuracy improves.
        # 검증 정확도가 향상될 때만 최적 모델을 저장합니다.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(BEST_MODEL_PATH, model, class_names, epoch, val_accuracy)
            print(f"  Saved new best model: {BEST_MODEL_PATH}")
        plot_history(history)
    return best_accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()
    device = get_device()
    print(f"Using device: {device}")

    datasets = build_datasets(args.data_dir)
    if datasets["train"].classes != datasets["val"].classes:
        raise ValueError("Train and validation class folders do not match.")
    class_names = datasets["train"].classes
    save_class_names(class_names)
    for split_name, dataset in datasets.items():
        print_class_counts(dataset, split_name)

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
        )
        for split, dataset in datasets.items()
    }
    model = build_model(len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float | int | str]] = []
    best_accuracy = -1.0

    freeze_backbone(model)
    head_optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    best_accuracy = train_phase(
        "classifier_head",
        model,
        loaders,
        criterion,
        head_optimizer,
        device,
        args.head_epochs,
        0,
        class_names,
        best_accuracy,
        history,
    )

    if args.fine_tune_epochs > 0:
        unfreeze_last_block(model)
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        fine_tune_optimizer = torch.optim.Adam(
            trainable_parameters, lr=args.fine_tune_learning_rate
        )
        train_phase(
            "fine_tune_layer4",
            model,
            loaders,
            criterion,
            fine_tune_optimizer,
            device,
            args.fine_tune_epochs,
            args.head_epochs,
            class_names,
            best_accuracy,
            history,
        )

    print(f"\nTraining complete. Best checkpoint: {BEST_MODEL_PATH}")
    print(f"History CSV: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
