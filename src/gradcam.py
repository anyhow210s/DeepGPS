"""Generate a Grad-CAM heatmap for a ResNet50 prediction.

English:
    Visualizes which image regions influenced a ResNet50 location prediction
    by generating and saving a Grad-CAM heatmap and overlay.

한국어:
    Grad-CAM 히트맵과 오버레이 이미지를 생성하여 ResNet50 위치 예측에
    영향을 준 이미지 영역을 시각화하고 저장합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    ensure_output_dirs,
    get_device,
    get_transforms,
    load_checkpoint,
    open_rgb_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


class GradCAM:
    """Capture activations and gradients from the final ResNet convolution block."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Return a normalized Grad-CAM map for one class."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[0, class_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        # Channel weights summarize how strongly each feature map affects the class.
        # 채널 가중치는 각 특징 맵이 예측 클래스에 미치는 영향을 요약합니다.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()


def create_gradcam_visuals(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    class_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a resized heatmap and RGB overlay for a prediction class."""
    input_tensor = get_transforms()["test"](image).unsqueeze(0).to(device)
    gradcam = GradCAM(model, model.layer4[-1])
    heatmap = gradcam.generate(input_tensor, class_index)
    gradcam.close()

    original_array = np.array(image)
    heatmap = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
    colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * original_array + 0.4 * colored_heatmap)
    return heatmap, overlay


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = get_device()
    model, class_names, _ = load_checkpoint(args.model, device)
    original = open_rgb_image(args.image)
    input_tensor = get_transforms()["test"](original).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
        class_index = int(probabilities.argmax().item())
    predicted_class = class_names[class_index]
    heatmap, overlay = create_gradcam_visuals(model, original, device, class_index)
    original_array = np.array(original)

    output_path = args.output or FIGURES_DIR / f"gradcam_{args.image.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_array)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {predicted_class} ({probabilities[class_index]:.2%})")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved Grad-CAM image: {output_path}")


if __name__ == "__main__":
    main()
