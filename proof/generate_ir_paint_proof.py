# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from stem_bleed_reclaimer.ir_bleed_painter import IRBleedPainter


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1.0e-15))


def main() -> int:
    output = Path(__file__).resolve().parent / "generated" / "ir_paint"
    output.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    frames = sample_rate * 8
    rng = np.random.default_rng(1984)
    drums = rng.normal(0, 0.25, frames).astype(np.float32)
    bass = rng.normal(0, 0.18, frames).astype(np.float32)
    other = rng.normal(0, 0.12, frames).astype(np.float32)
    true_bleed = (
        lfilter([0.11, 0.03, -0.02], [1.0], drums)
        + lfilter([-0.05, 0.0, 0.018], [1.0], bass)
        + lfilter([0.025, -0.01], [1.0], other)
    ).astype(np.float32)
    wanted = np.zeros(frames, dtype=np.float32)
    wanted[3 * sample_rate : 6 * sample_rate] = 0.3 * np.sin(2 * np.pi * 311 * np.arange(3 * sample_rate) / sample_rate)
    stems = {
        "drums": drums[:, None],
        "bass": bass[:, None],
        "other": other[:, None],
        "vocals": (wanted + true_bleed)[:, None],
    }
    painter = IRBleedPainter(n_fft=512, hop_size=128)
    model = painter.learn(stems, sample_rate, "vocals", [(0, 3 * sample_rate), (6 * sample_rate, frames)])
    predicted = painter.paint(stems, model)[:, 0]
    cleaned = stems["vocals"][:, 0] - predicted
    middle = slice(1000, frames - 1000)
    metrics = {
        "predicted_vs_true_bleed_correlation": float(np.corrcoef(predicted[middle], true_bleed[middle])[0, 1]),
        "bleed_rms_before": rms(true_bleed[middle]),
        "bleed_error_rms_after": rms(cleaned[middle] - wanted[middle]),
        "bleed_error_remaining_percent": 100.0 * rms(cleaned[middle] - wanted[middle]) / rms(true_bleed[middle]),
        "wanted_active_rms_before": rms(wanted[3 * sample_rate : 6 * sample_rate]),
        "wanted_active_rms_after": rms(cleaned[3 * sample_rate : 6 * sample_rate]),
    }
    sf.write(output / "true_bleed.wav", true_bleed, sample_rate, subtype="FLOAT")
    sf.write(output / "predicted_bleed.wav", predicted, sample_rate, subtype="FLOAT")
    sf.write(output / "unclean_target.wav", stems["vocals"], sample_rate, subtype="FLOAT")
    sf.write(output / "cleaned_target.wav", cleaned, sample_rate, subtype="FLOAT")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
