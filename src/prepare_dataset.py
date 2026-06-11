"""Validate raw Takeout images and create train/validation/test splits.

English:
    Reads location folders under data/raw, skips non-image and corrupted files,
    converts valid images to JPEG, and creates reproducible 70/15/15 splits.

한국어:
    data/raw 아래의 위치별 폴더를 읽고 이미지가 아닌 파일과 손상된 파일을
    제외한 뒤, 유효한 이미지를 JPEG로 변환하고 70/15/15 데이터셋을 생성합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import Counter
from pathlib import Path

from PIL import UnidentifiedImageError
from tqdm import tqdm

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SUPPORTED_EXTENSIONS, open_rgb_image, set_seed


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing processed dataset before creating a new split.",
    )
    return parser.parse_args()


def valid_image_paths(class_dir: Path) -> list[Path]:
    """Return readable images and skip JSON, videos, and corrupted files."""
    valid_paths: list[Path] = []
    for path in sorted(class_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            open_rgb_image(path)
            valid_paths.append(path)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            print(f"Skipping corrupted or unsupported image: {path} ({error})")
    return valid_paths


def split_paths(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    """Shuffle and split paths into 70/15/15 groups."""
    import random

    rng = random.Random(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def unique_jpeg_name(source: Path) -> str:
    """Create a stable output name that avoids collisions."""
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    return f"{source.stem}_{digest}.jpg"


def save_as_jpeg(source: Path, destination: Path) -> None:
    """Convert a validated source image to a standard RGB JPEG."""
    image = open_rgb_image(source)
    image.save(destination, format="JPEG", quality=95)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {args.raw_dir}")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"{args.output_dir} is not empty. Use --overwrite to recreate the split."
        )

    class_dirs = sorted(path for path in args.raw_dir.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("At least two class folders are required in data/raw/.")

    counts: Counter[str] = Counter()
    all_splits: dict[str, dict[str, list[Path]]] = {}
    for class_index, class_dir in enumerate(class_dirs):
        paths = valid_image_paths(class_dir)
        counts[class_dir.name] = len(paths)
        all_splits[class_dir.name] = split_paths(paths, args.seed + class_index)

    print("\nValidated class counts:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")
    too_small = {class_name: count for class_name, count in counts.items() if count < 7}
    if too_small:
        raise ValueError(
            "Each class needs at least 7 valid images so train, val, and test are non-empty. "
            f"Too small: {too_small}"
        )

    # Convert to JPEG so ImageFolder can read JPG, PNG, and HEIC inputs uniformly.
    # JPG, PNG, HEIC 입력을 ImageFolder가 동일하게 읽도록 JPEG로 변환합니다.
    for class_name, splits in all_splits.items():
        for split_name, paths in splits.items():
            destination_dir = args.output_dir / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            for source in tqdm(paths, desc=f"{split_name}/{class_name}"):
                destination = destination_dir / unique_jpeg_name(source)
                save_as_jpeg(source, destination)

    print(f"\nPrepared dataset at: {args.output_dir}")
    for split_name in SPLIT_RATIOS:
        split_count = sum(len(splits[split_name]) for splits in all_splits.values())
        print(f"  {split_name}: {split_count} images")


if __name__ == "__main__":
    main()
