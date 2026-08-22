#!/usr/bin/env python3
"""Create deterministic, identity-stratified 80/10/10 dataset manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


IDENTITIES = {
    "you": ("class_A", Path("class_A/A_1")),
    "teammate": ("class_A", Path("class_A/A_2")),
    "professor": ("class_B", Path("class_B/B")),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--seed", type=int, default=402)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_RATIOS}

    for identity, (class_label, relative_dir) in IDENTITIES.items():
        images = sorted(
            path for path in (dataset / relative_dir).iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(images) < 10:
            raise RuntimeError(f"Need at least 10 images for {identity}, found {len(images)}")
        random.Random(f"{args.seed}:{identity}").shuffle(images)
        train_end = int(len(images) * SPLIT_RATIOS["train"])
        val_end = train_end + int(len(images) * SPLIT_RATIOS["val"])
        partitions = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }
        for split, paths in partitions.items():
            for path in paths:
                rows_by_split[split].append({
                    "path": path.relative_to(Path.cwd()).as_posix(),
                    "class_label": class_label,
                    "identity": identity,
                })

    summary: dict[str, object] = {
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: (row["identity"], row["path"]))
        manifest = output / f"{split}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("path", "class_label", "identity"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        summary["splits"][split] = {
            "total": len(rows),
            "by_class": dict(sorted(Counter(row["class_label"] for row in rows).items())),
            "by_identity": dict(sorted(Counter(row["identity"] for row in rows).items())),
        }

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
