#!/usr/bin/env python3
"""Extract audio features, evaluate rules, and write metrics + matches to CSV."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import librosa
import numpy as np


@dataclass
class Rule:
    label: str
    description: str
    check: Callable[[Dict[str, float]], bool]


def extract_features(file_path: Path) -> Dict[str, float]:
    y, sr = librosa.load(str(file_path), sr=None, mono=True)
    duration = len(y) / sr if sr else 0.0

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_count = int(len(beats))
    beat_count_per_min = (beat_count / (duration / 60.0)) if duration > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {
        "duration": float(duration),
        "sample_rate": float(sr),
        "tempo": float(tempo),
        "beat_count": float(beat_count),
        "beat_count_per_min": float(beat_count_per_min),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_centroid_min": float(np.min(spectral_centroid)),
        "spectral_centroid_max": float(np.max(spectral_centroid)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
        "onset_strength_mean": float(np.mean(onset_strength)),
        "onset_strength_std": float(np.std(onset_strength)),
        "spectral_contrast_mean": float(np.mean(spectral_contrast)),
        "chroma_mean": float(np.mean(chroma)),
        "chroma_std": float(np.std(chroma)),
        "tonnetz_mean": float(np.mean(tonnetz)),
        "tonnetz_std": float(np.std(tonnetz)),
        "mfcc_2_mean": float(np.mean(mfcc[1])) if mfcc.shape[0] > 1 else 0.0,
        "mfcc_2_std": float(np.std(mfcc[1])) if mfcc.shape[0] > 1 else 0.0,
    }

    return features


def between(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def build_rules() -> Dict[str, List[Rule]]:
    return {
        "Lounge": [
            Rule("Lounge", "tempo 70–100", lambda f: between(f["tempo"], 70, 100)),
            Rule("Lounge", "rms_mean 0.05–0.12", lambda f: between(f["rms_mean"], 0.05, 0.12)),
            Rule("Lounge", "rms_std <0.08", lambda f: f["rms_std"] < 0.08),
            Rule(
                "Lounge",
                "spectral_centroid_mean 2000–3500",
                lambda f: between(f["spectral_centroid_mean"], 2000, 3500),
            ),
            Rule("Lounge", "onset_strength_mean <1.0", lambda f: f["onset_strength_mean"] < 1.0),
            Rule("Lounge", "spectral_contrast_mean 14–20", lambda f: between(f["spectral_contrast_mean"], 14, 20)),
            Rule("Lounge", "chroma_std <0.25", lambda f: f["chroma_std"] < 0.25),
        ],
        "Happy": [
            Rule("Happy", "tempo 100–140", lambda f: between(f["tempo"], 100, 140)),
            Rule("Happy", "spectral_centroid_mean >3500", lambda f: f["spectral_centroid_mean"] > 3500),
            Rule("Happy", "rms_mean >0.12", lambda f: f["rms_mean"] > 0.12),
            Rule("Happy", "chroma_mean >0.40", lambda f: f["chroma_mean"] > 0.40),
            Rule("Happy", "onset_strength_mean >1.2", lambda f: f["onset_strength_mean"] > 1.2),
            Rule("Happy", "tonnetz_mean >0.03", lambda f: f["tonnetz_mean"] > 0.03),
        ],
        "Melancholy": [
            Rule("Melancholy", "tempo 50–80", lambda f: between(f["tempo"], 50, 80)),
            Rule("Melancholy", "spectral_centroid_mean <2800", lambda f: f["spectral_centroid_mean"] < 2800),
            Rule("Melancholy", "rms_mean 0.05–0.15", lambda f: between(f["rms_mean"], 0.05, 0.15)),
            Rule("Melancholy", "rms_std 0.06–0.12", lambda f: between(f["rms_std"], 0.06, 0.12)),
            Rule("Melancholy", "tonnetz_std >0.12", lambda f: f["tonnetz_std"] > 0.12),
            Rule("Melancholy", "mfcc_2_mean <110", lambda f: f["mfcc_2_mean"] < 110),
        ],
        "Hard": [
            Rule("Hard", "rms_mean >0.18", lambda f: f["rms_mean"] > 0.18),
            Rule("Hard", "zero_crossing_rate_mean >0.08", lambda f: f["zero_crossing_rate_mean"] > 0.08),
            Rule("Hard", "spectral_centroid_mean >4000", lambda f: f["spectral_centroid_mean"] > 4000),
            Rule("Hard", "spectral_bandwidth_mean >4000", lambda f: f["spectral_bandwidth_mean"] > 4000),
            Rule("Hard", "onset_strength_mean >1.8", lambda f: f["onset_strength_mean"] > 1.8),
            Rule("Hard", "spectral_contrast_mean >22", lambda f: f["spectral_contrast_mean"] > 22),
        ],
        "Fast": [
            Rule("Fast", "tempo >130", lambda f: f["tempo"] > 130),
            Rule("Fast", "beat_count_per_min >2.2", lambda f: f["beat_count_per_min"] > 2.2),
            Rule("Fast", "onset_strength_mean >1.5", lambda f: f["onset_strength_mean"] > 1.5),
            Rule("Fast", "onset_strength_std >1.5", lambda f: f["onset_strength_std"] > 1.5),
        ],
        "Mellow": [
            Rule("Mellow", "tempo 60–95", lambda f: between(f["tempo"], 60, 95)),
            Rule("Mellow", "rms_mean <0.10", lambda f: f["rms_mean"] < 0.10),
            Rule("Mellow", "rms_std <0.06", lambda f: f["rms_std"] < 0.06),
            Rule("Mellow", "spectral_centroid_mean <2500", lambda f: f["spectral_centroid_mean"] < 2500),
            Rule("Mellow", "onset_strength_mean <0.8", lambda f: f["onset_strength_mean"] < 0.8),
            Rule("Mellow", "zero_crossing_rate_mean <0.05", lambda f: f["zero_crossing_rate_mean"] < 0.05),
            Rule("Mellow", "spectral_rolloff_mean <5500", lambda f: f["spectral_rolloff_mean"] < 5500),
        ],
        "Energetic": [
            Rule("Energetic", "tempo >120", lambda f: f["tempo"] > 120),
            Rule("Energetic", "rms_mean >0.15", lambda f: f["rms_mean"] > 0.15),
            Rule("Energetic", "rms_std >0.10", lambda f: f["rms_std"] > 0.10),
            Rule("Energetic", "onset_strength_mean >1.5", lambda f: f["onset_strength_mean"] > 1.5),
            Rule("Energetic", "spectral_centroid_mean >3500", lambda f: f["spectral_centroid_mean"] > 3500),
            Rule("Energetic", "spectral_bandwidth_mean >3800", lambda f: f["spectral_bandwidth_mean"] > 3800),
            Rule("Energetic", "beat_count_per_min >2.0", lambda f: f["beat_count_per_min"] > 2.0),
        ],
        "Day": [
            Rule("Day", "tempo 90–130", lambda f: between(f["tempo"], 90, 130)),
            Rule("Day", "spectral_centroid_mean >3200", lambda f: f["spectral_centroid_mean"] > 3200),
            Rule("Day", "rms_mean 0.10–0.20", lambda f: between(f["rms_mean"], 0.10, 0.20)),
            Rule("Day", "spectral_rolloff_mean >6500", lambda f: f["spectral_rolloff_mean"] > 6500),
            Rule("Day", "chroma_mean >0.38", lambda f: f["chroma_mean"] > 0.38),
            Rule("Day", "onset_strength_mean 1.0–1.8", lambda f: between(f["onset_strength_mean"], 1.0, 1.8)),
        ],
        "Night": [
            Rule("Night", "tempo 70–110", lambda f: between(f["tempo"], 70, 110)),
            Rule("Night", "spectral_centroid_mean <3000", lambda f: f["spectral_centroid_mean"] < 3000),
            Rule("Night", "rms_mean 0.05–0.14", lambda f: between(f["rms_mean"], 0.05, 0.14)),
            Rule("Night", "spectral_rolloff_mean <6000", lambda f: f["spectral_rolloff_mean"] < 6000),
            Rule("Night", "spectral_contrast_mean 14–20", lambda f: between(f["spectral_contrast_mean"], 14, 20)),
            Rule("Night", "tonnetz_std 0.10–0.18", lambda f: between(f["tonnetz_std"], 0.10, 0.18)),
        ],
        "Chill": [
            Rule("Chill", "tempo 60–100", lambda f: between(f["tempo"], 60, 100)),
            Rule("Chill", "rms_mean <0.10", lambda f: f["rms_mean"] < 0.10),
            Rule("Chill", "rms_std <0.06", lambda f: f["rms_std"] < 0.06),
            Rule("Chill", "onset_strength_mean <0.9", lambda f: f["onset_strength_mean"] < 0.9),
            Rule("Chill", "spectral_centroid_mean <3000", lambda f: f["spectral_centroid_mean"] < 3000),
            Rule("Chill", "zero_crossing_rate_mean <0.05", lambda f: f["zero_crossing_rate_mean"] < 0.05),
            Rule("Chill", "chroma_std <0.28", lambda f: f["chroma_std"] < 0.28),
        ],
        "Emotional": [
            Rule("Emotional", "rms_std >0.10", lambda f: f["rms_std"] > 0.10),
            Rule("Emotional", "spectral_centroid_std >1500", lambda f: f["spectral_centroid_std"] > 1500),
            Rule("Emotional", "tonnetz_std >0.14", lambda f: f["tonnetz_std"] > 0.14),
            Rule("Emotional", "chroma_std >0.30", lambda f: f["chroma_std"] > 0.30),
            Rule("Emotional", "onset_strength_std >1.5", lambda f: f["onset_strength_std"] > 1.5),
            Rule("Emotional", "mfcc_2_std >40", lambda f: f["mfcc_2_std"] > 40),
        ],
        "Dansing": [
            Rule("Dansing", "tempo 115–140", lambda f: between(f["tempo"], 115, 140)),
            Rule("Dansing", "beat_count_per_min >1.9", lambda f: f["beat_count_per_min"] > 1.9),
            Rule("Dansing", "onset_strength_mean >1.4", lambda f: f["onset_strength_mean"] > 1.4),
            Rule("Dansing", "rms_mean >0.14", lambda f: f["rms_mean"] > 0.14),
            Rule("Dansing", "spectral_centroid_mean >2800", lambda f: f["spectral_centroid_mean"] > 2800),
            Rule("Dansing", "zero_crossing_rate_mean 0.04–0.09", lambda f: between(f["zero_crossing_rate_mean"], 0.04, 0.09)),
            Rule("Dansing", "rms_std <0.12", lambda f: f["rms_std"] < 0.12),
        ],
        "Extreme": [
            Rule("Extreme", "tempo >150 or <60", lambda f: f["tempo"] > 150 or f["tempo"] < 60),
            Rule("Extreme", "rms_mean >0.22", lambda f: f["rms_mean"] > 0.22),
            Rule("Extreme", "zero_crossing_rate_mean >0.10", lambda f: f["zero_crossing_rate_mean"] > 0.10),
            Rule("Extreme", "spectral_bandwidth_mean >4500", lambda f: f["spectral_bandwidth_mean"] > 4500),
            Rule("Extreme", "onset_strength_mean >2.2", lambda f: f["onset_strength_mean"] > 2.2),
            Rule("Extreme", "spectral_contrast_mean >24", lambda f: f["spectral_contrast_mean"] > 24),
            Rule("Extreme", "spectral_centroid_max >8000", lambda f: f["spectral_centroid_max"] > 8000),
        ],
        "Serene": [
            Rule("Serene", "tempo 40–70", lambda f: between(f["tempo"], 40, 70)),
            Rule("Serene", "rms_mean <0.06", lambda f: f["rms_mean"] < 0.06),
            Rule("Serene", "rms_std <0.04", lambda f: f["rms_std"] < 0.04),
            Rule("Serene", "onset_strength_mean <0.6", lambda f: f["onset_strength_mean"] < 0.6),
            Rule("Serene", "spectral_centroid_mean <2200", lambda f: f["spectral_centroid_mean"] < 2200),
            Rule("Serene", "zero_crossing_rate_mean <0.04", lambda f: f["zero_crossing_rate_mean"] < 0.04),
            Rule("Serene", "tonnetz_std <0.10", lambda f: f["tonnetz_std"] < 0.10),
            Rule("Serene", "chroma_std <0.20", lambda f: f["chroma_std"] < 0.20),
        ],
    }


def classify(features: Dict[str, float]) -> Tuple[str, float, List[Tuple[str, float, int, int]]]:
    rules = build_rules()
    scores: List[Tuple[str, float, int, int]] = []
    for label, checks in rules.items():
        passed = sum(1 for rule in checks if rule.check(features))
        total = len(checks)
        score = passed / total if total else 0.0
        scores.append((label, score, passed, total))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scores[0][0], scores[0][1] if scores else ("Unknown", 0.0)
    return best_label, best_score, scores


def iter_audio_files(directory: Path) -> List[Path]:
    patterns = ["*.wav", "*.flac", "*.WAV", "*.FLAC"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return sorted(set(files))


def build_csv_header(features: Iterable[str], labels: Iterable[str]) -> List[str]:
    header = [
        "file_path",
        "best_label",
        "best_score",
        "error",
    ]
    header.extend(features)
    for label in labels:
        header.extend(
            [
                f"{label}_score",
                f"{label}_passed",
                f"{label}_total",
                f"{label}_matched",
            ]
        )
    return header


def process_file(
    file_path: str,
    min_score: float,
    feature_names: List[str],
    labels: List[str],
) -> Dict[str, object]:
    path = Path(file_path)
    row: Dict[str, object] = {
        "file_path": file_path,
    }
    try:
        features = extract_features(path)
        best_label, best_score, scores = classify(features)
        row["best_label"] = best_label
        row["best_score"] = best_score
        row["error"] = ""
        for name in feature_names:
            row[name] = features.get(name, "")
        for label, score, passed, total in scores:
            row[f"{label}_score"] = score
            row[f"{label}_passed"] = passed
            row[f"{label}_total"] = total
            row[f"{label}_matched"] = 1 if score >= min_score else 0
    except Exception as exc:  # pragma: no cover - robust CLI
        row["best_label"] = ""
        row["best_score"] = ""
        row["error"] = str(exc)
        for name in feature_names:
            row[name] = ""
        for label in labels:
            row[f"{label}_score"] = ""
            row[f"{label}_passed"] = ""
            row[f"{label}_total"] = ""
            row[f"{label}_matched"] = ""

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Index audio metrics and matches into CSV.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory containing .wav/.flac files")
    parser.add_argument("--output", default="audio_index.csv", help="Output CSV path")
    parser.add_argument("--min-score", type=float, default=0.66, help="Minimum score to mark a match")
    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    files = iter_audio_files(directory)
    if not files:
        print(f"No .wav/.flac files found under: {directory}")
        return

    rules = build_rules()
    labels = list(rules.keys())

    feature_names = list(extract_features(files[0]).keys())
    header = build_csv_header(feature_names, labels)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        max_workers = os.cpu_count() or 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            rows = executor.map(
                process_file,
                [str(path) for path in files],
                [args.min_score] * len(files),
                [feature_names] * len(files),
                [labels] * len(files),
            )
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
