"""Shared configuration and utilities for the photo geolocation project.

English:
    Defines project paths, image transforms, ResNet50 construction, device
    selection, checkpoint handling, and common image-loading helpers.

한국어:
    프로젝트 경로, 이미지 변환, ResNet50 모델 생성, 실행 장치 선택,
    체크포인트 처리, 공통 이미지 로딩 기능을 정의합니다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    # HEIC support is optional at import time, but included in requirements.txt.
    # HEIC 지원은 import 시점에는 선택 사항이지만 requirements.txt에 포함되어 있습니다.
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

BEST_MODEL_PATH = MODELS_DIR / "best_resnet50.pth"
HISTORY_PATH = REPORTS_DIR / "training_history.csv"
CLASS_NAMES_PATH = REPORTS_DIR / "class_names.json"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def ensure_output_dirs() -> None:
    """Create output directories if they do not already exist."""
    for directory in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Choose CUDA, Apple Silicon MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transforms() -> dict[str, transforms.Compose]:
    """Return ImageNet-compatible training and evaluation transforms."""
    # Strong augmentation is used only for training.
    # 강한 데이터 증강은 학습 데이터에만 적용합니다.
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def build_datasets(data_dir: Path = PROCESSED_DATA_DIR) -> dict[str, datasets.ImageFolder]:
    """Build ImageFolder datasets for train, validation, and test splits."""
    split_transforms = get_transforms()
    return {
        split: datasets.ImageFolder(data_dir / split, transform=split_transforms[split])
        for split in ("train", "val", "test")
    }


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a ResNet50 model with a location classification head."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the final classifier."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_last_block(model: nn.Module) -> None:
    """Unfreeze ResNet layer4 and the final classifier for fine-tuning."""
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    epoch: int,
    val_accuracy: float,
) -> None:
    """Save model weights and metadata needed for inference."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "image_size": IMAGE_SIZE,
        "architecture": "resnet50",
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path = BEST_MODEL_PATH, device: torch.device | None = None
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    """Load a saved ResNet50 checkpoint for evaluation or inference."""
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, checkpoint


def save_class_names(class_names: list[str]) -> None:
    """Save class names as readable JSON."""
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def open_rgb_image(path: Path) -> Image.Image:
    """Open an image and return a detached RGB copy."""
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")
