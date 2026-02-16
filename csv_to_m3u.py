#!/usr/bin/env python3
"""Parse the CSV index and generate M3U playlists per label."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def detect_labels(fieldnames: Iterable[str]) -> List[str]:
    labels: List[str] = []
    for name in fieldnames:
        if name.endswith("_matched"):
            labels.append(name[: -len("_matched")])
    return sorted(set(labels))


def write_m3u_files(classified: Dict[str, List[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, entries in sorted(classified.items()):
        playlist_path = output_dir / f"{label}.m3u"
        content = "\n".join(entries)
        playlist_path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def transform_entry(entry: str, drive_prefix: str, base_path: str) -> str:
    if base_path:
        entry_norm = entry.replace("\\", "/")
        base_norm = base_path.replace("\\", "/").rstrip("/")
        if entry_norm.startswith(base_norm):
            remainder = entry_norm[len(base_norm):].lstrip("/")
            remainder = remainder.replace("/", "\\")
            return f"{drive_prefix}{remainder}" if remainder else drive_prefix

    normalized = entry.replace("/", "\\")
    for sep in (":\\", ":/"):
        if sep in normalized:
            normalized = normalized.split(sep, 1)[1]
            break
    normalized = normalized.lstrip("\\/")
    return f"{drive_prefix}{normalized}" if normalized else drive_prefix


def write_variant_playlists(
    classified: Dict[str, List[str]],
    output_dir: Path,
    variants: Dict[str, str],
    base_path: str,
) -> None:
    for variant_name, drive_prefix in variants.items():
        variant_dir = output_dir / variant_name
        transformed = {
            label: [transform_entry(entry, drive_prefix, base_path) for entry in entries]
            for label, entries in classified.items()
        }
        write_m3u_files(transformed, variant_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create M3U files from an audio CSV index.")
    parser.add_argument("csv_path", help="Path to the CSV produced by eval_to_csv.py")
    parser.add_argument("--output-dir", default=".", help="Directory to write M3U playlists")
    parser.add_argument(
        "--base-path",
        default="",
        help="Prefix in CSV file paths to replace with variant drive prefixes",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists() or not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header.")

        labels = detect_labels(reader.fieldnames)
        classified: Dict[str, List[str]] = {label: [] for label in labels}

        for row in reader:
            if row.get("error"):
                continue
            file_path = row.get("file_path") or ""
            if not file_path:
                continue

            matched_labels = [label for label in labels if truthy(row.get(f"{label}_matched", ""))]
            if matched_labels:
                for label in matched_labels:
                    classified.setdefault(label, []).append(file_path)
                continue

            best_label = row.get("best_label") or ""
            if best_label:
                classified.setdefault(best_label, []).append(file_path)
                continue

            best_score = -1.0
            fallback_label = ""
            for label in labels:
                score_value = row.get(f"{label}_score", "")
                try:
                    score = float(score_value)
                except (TypeError, ValueError):
                    continue
                if score > best_score:
                    best_score = score
                    fallback_label = label
            if fallback_label:
                classified.setdefault(fallback_label, []).append(file_path)

    variants = {
        "varianta": "a:\\Music\\",
        "variantA": "A:\\Music\\",
        "variantc": "c:\\Music\\",
    }
    write_variant_playlists(classified, output_dir, variants, args.base_path)


if __name__ == "__main__":
    main()
