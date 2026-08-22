#!/usr/bin/env python3
"""Build normalized ArcFace reference embeddings for the Android app."""

from __future__ import annotations

import argparse
import csv
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
        "--train-manifest",
        type=Path,
        default=Path("dataset/splits/train.csv"),
        help="CSV manifest produced by prepare_dataset_splits.py.",
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


def load_manifest(manifest: Path) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {identity: [] for identity in IDENTITIES}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            identity = row["identity"]
            if identity in paths:
                paths[identity].append(Path(row["path"]).resolve())
    missing = [identity for identity, identity_paths in paths.items() if not identity_paths]
    if missing:
        raise RuntimeError(f"Training manifest contains no images for: {', '.join(missing)}")
    return paths


def normalized_reference(
    analyzer: insightface.app.FaceAnalysis, identity: str, images: list[Path]
) -> np.ndarray:
    embeddings: list[np.ndarray] = []

    for image_path in sorted(images):
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
        raise RuntimeError(f"No usable faces found for {identity}")

    reference = np.mean(embeddings, axis=0, keepdims=True, dtype=np.float32)
    reference /= max(float(np.linalg.norm(reference)), 1e-12)
    print(f"{identity}: {len(embeddings)}/{len(images)} usable training images")
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
    train_manifest = args.train_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    analyzer = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(args.insightface_root.expanduser()),
        providers=["CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))

    references: dict[str, np.ndarray] = {}
    paths_by_identity = load_manifest(train_manifest)
    for name in IDENTITIES:
        references[name] = normalized_reference(analyzer, name, paths_by_identity[name])
        np.save(output / f"{name}_ref.npy", references[name])

    if args.android_assets:
        copy_android_assets(references, args.android_assets.resolve(), args.insightface_root)

    print(f"Reference embeddings written to {output}")


if __name__ == "__main__":
    main()
