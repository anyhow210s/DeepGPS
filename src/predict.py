"""Predict the top location categories for one photo.

English:
    Loads a trained ResNet50 checkpoint and prints the top location predictions
    with probabilities for a user-provided image.

한국어:
    학습된 ResNet50 체크포인트를 불러와 사용자가 지정한 이미지의 위치 예측
    결과와 확률을 높은 순서대로 출력합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from config import BEST_MODEL_PATH, get_device, get_transforms, load_checkpoint, open_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def predict_image(
    model: nn.Module,
    image: Image.Image,
    class_names: list[str],
    device: torch.device,
    top_k: int = 3,
) -> tuple[list[dict[str, float | str]], torch.Tensor]:
    """Return ranked predictions and all class probabilities for one image."""
    tensor = get_transforms()["test"](image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    top_k = min(top_k, len(class_names))
    top_probabilities, top_indices = probabilities.topk(top_k)
    predictions = [
        {"location": class_names[index], "probability": probability}
        for probability, index in zip(
            top_probabilities.tolist(), top_indices.tolist()
        )
    ]
    return predictions, probabilities


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = get_device()
    model, class_names, _ = load_checkpoint(args.model, device)
    image = open_rgb_image(args.image)
    predictions, _ = predict_image(model, image, class_names, device, args.top_k)

    print(f"Image: {args.image}")
    print("Top predictions:")
    for rank, prediction in enumerate(predictions, start=1):
        print(f"  {rank}. {prediction['location']}: {prediction['probability']:.2%}")


if __name__ == "__main__":
    main()
