# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MixDistance:
    delta_rms: float
    level_match_gain: float
    level_match_gain_db: float
    level_matched_delta_rms: float
    correlation: float


@dataclass(frozen=True)
class MixConservationResult:
    original_stem_sum: np.ndarray
    cleaned_stem_sum: np.ndarray
    combined_removed_noise: np.ndarray
    cleaned_plus_removed_mix: np.ndarray
    baseline_to_original_mix: MixDistance | None
    cleaned_to_original_mix: MixDistance | None
    cleaned_plus_removed_to_original_mix: MixDistance | None
    conservation_error_rms: float

    def metrics_dict(self) -> dict:
        return {
            "baseline_to_original_mix": None if self.baseline_to_original_mix is None else asdict(self.baseline_to_original_mix),
            "cleaned_to_original_mix": None if self.cleaned_to_original_mix is None else asdict(self.cleaned_to_original_mix),
            "cleaned_plus_removed_to_original_mix": None if self.cleaned_plus_removed_to_original_mix is None else asdict(self.cleaned_plus_removed_to_original_mix),
            "conservation_error_rms": self.conservation_error_rms,
        }


def _sum(stems: Mapping[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(audio, dtype=np.float32) for audio in stems.values()]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("All stems must share one shape")
    return np.sum(np.stack(arrays, axis=0), axis=0, dtype=np.float32)


def _distance(candidate: np.ndarray, reference: np.ndarray) -> MixDistance:
    if candidate.shape != reference.shape:
        raise ValueError("Original full mix and stem sums must share one shape")
    delta = candidate - reference
    delta_rms = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64)) + 1.0e-15))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate, dtype=np.float64)) + 1.0e-15))
    reference_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64)) + 1.0e-15))
    level_match_gain = reference_rms / max(candidate_rms, 1.0e-15)
    level_matched = candidate * level_match_gain
    matched_delta = level_matched - reference
    level_matched_delta_rms = float(np.sqrt(np.mean(np.square(matched_delta, dtype=np.float64)) + 1.0e-15))
    level_match_gain_db = float(20.0 * np.log10(max(level_match_gain, 1.0e-15)))
    left = candidate.reshape(-1).astype(np.float64)
    right = reference.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    correlation = float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0
    return MixDistance(delta_rms, level_match_gain, level_match_gain_db, level_matched_delta_rms, correlation)


def compare_mix_conservation(
    original_stems: Mapping[str, np.ndarray],
    cleaned_stems: Mapping[str, np.ndarray],
    predicted_bleed: Mapping[str, np.ndarray],
    original_full_mix: np.ndarray | None = None,
) -> MixConservationResult:
    original_sum = _sum(original_stems)
    cleaned_sum = _sum(cleaned_stems)
    removed_sum = _sum(predicted_bleed)
    restored_sum = cleaned_sum + removed_sum
    conservation_error = float(np.sqrt(np.mean(np.square(restored_sum - original_sum, dtype=np.float64)) + 1.0e-15))
    return MixConservationResult(
        original_stem_sum=original_sum,
        cleaned_stem_sum=cleaned_sum,
        combined_removed_noise=removed_sum,
        cleaned_plus_removed_mix=restored_sum,
        baseline_to_original_mix=None if original_full_mix is None else _distance(original_sum, original_full_mix),
        cleaned_to_original_mix=None if original_full_mix is None else _distance(cleaned_sum, original_full_mix),
        cleaned_plus_removed_to_original_mix=None if original_full_mix is None else _distance(restored_sum, original_full_mix),
        conservation_error_rms=conservation_error,
    )
