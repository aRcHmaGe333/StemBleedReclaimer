# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from stem_bleed_reclaimer.ir_bleed_painter import IRBleedPainter
from stem_bleed_reclaimer.mix_conservation import compare_mix_conservation


STEM_FOLDER = Path(r"E:\notable\PT_Audio\inputs\stems_mirror\kevurushtija 8bit 09 Surface Terrestrial Colonization")
ORIGINAL_MIX = Path(r"E:\code\audio\ArrangeMe\mp3 exemplary\kevurushtija 8bit 09 Surface Terrestrial Colonization.flac")
OUTPUT = Path(r"E:\notable\_scratch\StemBleedReclaimer_SURFACE_4V5_20260715")
LANES = ("bass", "drums", "other", "vocals")


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1.0e-15))


def select_quiet_regions(stems: dict[str, np.ndarray], target: str, sample_rate: int) -> list[tuple[int, int]]:
    block = 2 * sample_rate
    rows = []
    refs = sum((audio for lane, audio in stems.items() if lane != target), start=np.zeros_like(stems[target]))
    for start in range(0, len(refs) - block + 1, block):
        end = start + block
        target_rms = rms(stems[target][start:end])
        reference_rms = rms(refs[start:end])
        rows.append((target_rms / max(reference_rms, 1.0e-12), -reference_rms, start, end, target_rms, reference_rms))
    eligible = [row for row in rows if -row[1] >= np.percentile([-item[1] for item in rows], 40)]
    return [(row[2], row[3]) for row in sorted(eligible)[:24]]


def best_alignment_samples(candidate: np.ndarray, reference: np.ndarray, sample_rate: int) -> tuple[int, float]:
    stride = 100
    left = np.mean(candidate, axis=1)[::stride].astype(np.float64)
    right = np.mean(reference, axis=1)[::stride].astype(np.float64)
    max_lag = round(sample_rate / stride)
    best = (0, -2.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = left[-lag:], right[: len(right) + lag]
        elif lag > 0:
            a, b = left[: len(left) - lag], right[lag:]
        else:
            a, b = left, right
        score = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-15))
        if score > best[1]:
            best = (lag * stride, score)
    return best


def anomaly_metrics(audio: np.ndarray, reference: np.ndarray, gain: float, sample_rate: int) -> dict:
    matched = audio * gain
    residual = matched - reference
    block = sample_rate
    local = np.asarray([rms(residual[start : start + block]) for start in range(0, len(residual), block)])
    derivative = np.max(np.abs(np.diff(matched, axis=0)), axis=1)
    worst = int(np.argmax(local))
    return {
        "worst_residual_second": worst,
        "worst_residual_rms": float(local[worst]),
        "median_residual_rms": float(np.median(local)),
        "worst_to_median_ratio": float(local[worst] / max(np.median(local), 1.0e-15)),
        "maximum_adjacent_sample_jump": float(np.max(derivative)),
        "jump_p999999": float(np.percentile(derivative, 99.9999)),
        "nonfinite_samples": int(np.size(matched) - np.count_nonzero(np.isfinite(matched))),
        "samples_above_reference_peak": int(np.count_nonzero(np.abs(matched) > np.max(np.abs(reference)) + 1.0e-7)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stems: dict[str, np.ndarray] = {}
    sample_rate = None
    for lane in LANES:
        audio, current_rate = sf.read(STEM_FOLDER / f"{lane}.wav", dtype="float32", always_2d=True)
        sample_rate = current_rate if sample_rate is None else sample_rate
        if current_rate != sample_rate:
            raise ValueError("Stem sample rates differ")
        stems[lane] = audio
    reference, reference_rate = sf.read(ORIGINAL_MIX, dtype="float32", always_2d=True)
    if reference_rate != sample_rate or any(audio.shape != reference.shape for audio in stems.values()):
        raise ValueError("Demucs stem timeline differs from the original full mix")

    baseline = sum(stems.values(), start=np.zeros_like(reference))
    lag, alignment_correlation = best_alignment_samples(baseline, reference, sample_rate)
    if lag != 0:
        raise ValueError(f"Demucs timeline is shifted by {lag} samples; comparison stopped before false scoring")

    # Real separated stems are strongly correlated. A conservative ridge prevents
    # an ill-conditioned inverse from turning ordinary bleed into full-scale spikes.
    painter = IRBleedPainter(ridge_ratio=0.1, paint_chunk_seconds=30.0)
    cleaned: dict[str, np.ndarray] = {}
    predicted: dict[str, np.ndarray] = {}
    region_report = {}
    target_report = {}
    for lane in LANES:
        regions = select_quiet_regions(stems, lane, sample_rate)
        model = painter.learn(stems, sample_rate, lane, regions)
        noise = painter.paint(stems, model)
        noise_peak = float(np.max(np.abs(noise)))
        target_peak = float(np.max(np.abs(stems[lane])))
        if noise_peak > target_peak:
            raise ValueError(f"Unstable {lane} transfer: predicted bleed peak {noise_peak} exceeds target peak {target_peak}")
        predicted[lane] = noise
        cleaned[lane] = stems[lane] - noise
        region_report[lane] = [[start / sample_rate, end / sample_rate] for start, end in regions]
        target_report[lane] = {
            "target_peak": target_peak,
            "predicted_bleed_peak": noise_peak,
            "target_rms": rms(stems[lane]),
            "predicted_bleed_rms": rms(noise),
        }

    result = compare_mix_conservation(stems, cleaned, predicted, reference)
    four = result.cleaned_stem_sum
    five = result.cleaned_plus_removed_mix
    four_metrics = result.cleaned_to_original_mix
    five_metrics = result.cleaned_plus_removed_to_original_mix
    assert four_metrics is not None and five_metrics is not None
    outputs = {
        "four_cleaned_peak_matched.wav": four * four_metrics.peak_match_gain,
        "fifth_removed_noise.wav": result.combined_removed_noise,
        "five_stems_peak_matched.wav": five * five_metrics.peak_match_gain,
    }
    for name, audio in outputs.items():
        sf.write(OUTPUT / name, audio, sample_rate, subtype="FLOAT")

    report = {
        "source": str(STEM_FOLDER),
        "reference": str(ORIGINAL_MIX),
        "sample_rate": sample_rate,
        "frames": len(reference),
        "duration_seconds": len(reference) / sample_rate,
        "demucs_timeline_guard": {
            "equal_frame_count": True,
            "best_lag_samples": lag,
            "best_lag_milliseconds": lag * 1000 / sample_rate,
            "alignment_correlation": alignment_correlation,
        },
        "quiet_regions_seconds": region_report,
        "targets": target_report,
        "mix_conservation": result.metrics_dict(),
        "four_cleaned_anomalies": anomaly_metrics(four, reference, four_metrics.peak_match_gain, sample_rate),
        "five_stem_anomalies": anomaly_metrics(five, reference, five_metrics.peak_match_gain, sample_rate),
        "winner_by_peak_matched_delta_rms": "four_cleaned" if four_metrics.peak_matched_delta_rms < five_metrics.peak_matched_delta_rms else "five_stems",
    }
    (OUTPUT / "four_vs_five_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
