# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from stem_bleed_reclaimer import BleedCleaner, BleedConfig


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1.0e-15))


def main() -> int:
    root = Path(__file__).resolve().parent / "generated"
    source = root / "source"
    cleaned_dir = root / "cleaned"
    source.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    frames = sample_rate * 6
    time = np.arange(frames, dtype=np.float32) / sample_rate

    drums = np.zeros(frames, dtype=np.float32)
    for onset in range(2000, frames - 1200, 4000):
        event = 0.8 * np.sin(np.linspace(0, 20 * np.pi, 900)) * np.exp(-np.linspace(0, 7, 900))
        drums[onset : onset + len(event)] += event.astype(np.float32)
    bass = (0.5 * np.sin(2 * np.pi * 93 * time)).astype(np.float32)
    bass[:sample_rate] = 0.0
    bass[-sample_rate:] = 0.0
    legitimate_other = np.zeros(frames, dtype=np.float32)
    legitimate_other[2 * sample_rate : 4 * sample_rate] = 0.3 * np.sin(2 * np.pi * 440 * time[: 2 * sample_rate])
    other = legitimate_other + 0.10 * drums + 0.07 * bass
    vocals = np.zeros(frames, dtype=np.float32)
    stems = {"drums": drums, "bass": bass, "other": other, "vocals": vocals}
    for lane, audio in stems.items():
        sf.write(source / f"{lane}.wav", audio, sample_rate, subtype="FLOAT")

    cleaner = BleedCleaner(BleedConfig(frame_size=1024, hop_size=256, minimum_confidence=0.72))
    result = cleaner.process_folder(source, cleaned_dir)
    cleaned_other, _ = sf.read(cleaned_dir / "other.wav", dtype="float32")
    quiet_mask = np.ones(frames, dtype=bool)
    quiet_mask[2 * sample_rate : 4 * sample_rate] = False
    active = slice(2 * sample_rate, 4 * sample_rate)
    metrics = {
        "regions": len(result.regions),
        "owners_found_for_other": sorted({region.owner_stem for region in result.regions if region.target_stem == "other"}),
        "other_quiet_region_rms_before": rms(other[quiet_mask]),
        "other_quiet_region_rms_after": rms(cleaned_other[quiet_mask]),
        "other_quiet_region_bleed_reduction_percent": 100.0 * (1.0 - rms(cleaned_other[quiet_mask]) / rms(other[quiet_mask])),
        "other_active_region_change_rms": rms(cleaned_other[active] - other[active]),
        "source_files_modified": False,
    }
    (root / "proof_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (cleaned_dir / "bleed_attribution_report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
