# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .core import BleedCleaner
from .ir_bleed_painter import IRBleedPainter
from .mix_conservation import compare_mix_conservation


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1.0e-15))


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn bleed transfer from quiet regions and paint the full-length subtraction signal.")
    parser.add_argument("stem_folder", type=Path)
    parser.add_argument("quiet_regions_json", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--original-mix", type=Path)
    args = parser.parse_args()

    definition = json.loads(args.quiet_regions_json.read_text(encoding="utf-8"))
    if definition.get("unit") != "seconds" or not isinstance(definition.get("regions"), dict):
        raise ValueError('Quiet-region JSON must contain {"unit":"seconds","regions":{...}}')
    stems, sample_rate = BleedCleaner.load_stem_folder(args.stem_folder)
    painter = IRBleedPainter()
    cleaned = {lane: np.array(audio, copy=True) for lane, audio in stems.items()}
    predicted = {lane: np.zeros_like(audio) for lane, audio in stems.items()}
    report = {"sample_rate": sample_rate, "frames": len(next(iter(stems.values()))), "targets": {}}
    for target, ranges in definition["regions"].items():
        intervals = [(round(float(start) * sample_rate), round(float(end) * sample_rate)) for start, end in ranges]
        model = painter.learn(stems, sample_rate, target, intervals)
        predicted[target] = painter.paint(stems, model)
        cleaned[target] = stems[target] - predicted[target]
        model.save(args.output_folder / "models" / f"{target}_ir_model.npz")
        report["targets"][target] = {
            "quiet_regions": [[int(start), int(end)] for start, end in model.quiet_regions],
            "reference_channels": [[lane, channel] for lane, channel in model.reference_channels],
            "predicted_bleed_rms": _rms(predicted[target]),
            "target_rms_before": _rms(stems[target]),
            "target_rms_after": _rms(cleaned[target]),
        }
    for group, audio_by_lane in (("cleaned", cleaned), ("predicted_bleed", predicted)):
        folder = args.output_folder / group
        folder.mkdir(parents=True, exist_ok=True)
        for lane, audio in audio_by_lane.items():
            sf.write(folder / f"{lane}.wav", audio, sample_rate, subtype="FLOAT")
    original_mix = None
    if args.original_mix:
        original_mix, mix_rate = sf.read(args.original_mix, dtype="float32", always_2d=True)
        if int(mix_rate) != sample_rate:
            raise ValueError("Original full mix sample rate does not match the stems")
    conservation = compare_mix_conservation(stems, cleaned, predicted, original_mix)
    mix_folder = args.output_folder / "mix_comparison"
    mix_folder.mkdir(parents=True, exist_ok=True)
    for name, audio in (
        ("original_stem_sum.wav", conservation.original_stem_sum),
        ("cleaned_stem_sum.wav", conservation.cleaned_stem_sum),
        ("fifth_combined_removed_noise.wav", conservation.combined_removed_noise),
        ("cleaned_plus_fifth_mix.wav", conservation.cleaned_plus_removed_mix),
    ):
        sf.write(mix_folder / name, audio, sample_rate, subtype="FLOAT")
    report["mix_conservation"] = conservation.metrics_dict()
    report_path = args.output_folder / "ir_paint_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
