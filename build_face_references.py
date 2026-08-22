#!/usr/bin/env python3
"""Build normalized ArcFace reference embeddings for the Android app."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import insightface
import numpy as np


IDENTITIES = {
    "you": Path("class_A/A_1"),
    "teammate": Path("class_A/A_2"),
    "professor": Path("class_B/B"),
}
MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 512-D reference embeddings with InsightFace buffalo_l."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/synthetic"),
        help="Dataset root containing class_A/A_1, class_A/A_2 and class_B/B.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference_embeddings"),
        help="Directory for the generated .npy files.",
    )
    parser.add_argument(
        "--android-assets",
        type=Path,
        help="Also copy references and required ONNX models into this assets directory.",
    )
    parser.add_argument(
        "--insightface-root",
        type=Path,
        default=Path.home() / ".insightface",
        help="InsightFace download/cache directory.",
    )
    return parser.parse_args()


def normalized_reference(analyzer: insightface.app.FaceAnalysis, folder: Path) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    images = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        faces = analyzer.get(image)
        if not faces:
            continue
        face = max(
            faces,
            key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
        )
        embeddings.append(face.embedding.astype(np.float32))

    if not embeddings:
        raise RuntimeError(f"No usable faces found in {folder}")

    reference = np.mean(embeddings, axis=0, keepdims=True, dtype=np.float32)
    reference /= max(float(np.linalg.norm(reference)), 1e-12)
    print(f"{folder}: {len(embeddings)}/{len(images)} usable images")
    return reference


def copy_android_assets(
    references: dict[str, np.ndarray], assets_dir: Path, insightface_root: Path
) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    for name, reference in references.items():
        np.save(assets_dir / f"{name}_ref.npy", reference)

    model_dir = insightface_root.expanduser() / "models" / "buffalo_l"
    for filename in MODEL_FILES:
        source = model_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"InsightFace model not found: {source}")
        shutil.copy2(source, assets_dir / filename)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    analyzer = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(args.insightface_root.expanduser()),
        providers=["CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))

    references: dict[str, np.ndarray] = {}
    for name, relative_folder in IDENTITIES.items():
        folder = dataset / relative_folder
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {folder}")
        references[name] = normalized_reference(analyzer, folder)
        np.save(output / f"{name}_ref.npy", references[name])

    if args.android_assets:
        copy_android_assets(references, args.android_assets.resolve(), args.insightface_root)

    print(f"Reference embeddings written to {output}")


if __name__ == "__main__":
    main()
