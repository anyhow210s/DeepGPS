"""Evaluate the best model on the held-out test set.

English:
    Loads the best checkpoint, measures test accuracy and per-class metrics,
    and saves a classification report and confusion matrix.

한국어:
    최적 체크포인트를 불러와 테스트 정확도와 클래스별 성능 지표를 계산하고,
    분류 보고서와 혼동 행렬을 저장합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    build_datasets,
    ensure_output_dirs,
    get_device,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    device = get_device()
    model, class_names, checkpoint = load_checkpoint(args.model, device)
    test_dataset = build_datasets(args.data_dir)["test"]
    if test_dataset.classes != class_names:
        raise ValueError("Test dataset classes do not match checkpoint classes.")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_frame = pd.DataFrame(report).transpose()
    report_frame.to_csv(REPORTS_DIR / "classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    figure = plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted location")
    plt.ylabel("True location")
    plt.title("Photo Geolocation Confusion Matrix")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close(figure)

    summary = (
        f"Test accuracy: {accuracy:.4f}\n"
        f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.4f}\n"
        f"Checkpoint epoch: {checkpoint['epoch']}\n\n"
        f"{classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)}"
    )
    (REPORTS_DIR / "evaluation_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved confusion matrix: {FIGURES_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
