# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import numpy as np

from stem_bleed_reclaimer.mix_conservation import compare_mix_conservation


def test_combined_removed_noise_is_a_fifth_stem_and_restores_the_original_stem_sum():
    rng = np.random.default_rng(12)
    original = {lane: rng.normal(0, 0.1, (4096, 2)).astype(np.float32) for lane in ("drums", "bass", "other", "vocals")}
    predicted = {lane: rng.normal(0, 0.005, (4096, 2)).astype(np.float32) for lane in original}
    cleaned = {lane: original[lane] - predicted[lane] for lane in original}
    original_mix = np.sum(np.stack(list(original.values())), axis=0)

    result = compare_mix_conservation(original, cleaned, predicted, original_mix)

    assert result.conservation_error_rms < 1.0e-7
    assert result.cleaned_to_original_mix.delta_rms > 0.005
    assert result.cleaned_plus_removed_to_original_mix.delta_rms < 1.0e-7


def test_content_comparison_volume_matches_quieter_cleaned_stems_before_scoring():
    rng = np.random.default_rng(77)
    original = {lane: rng.normal(0, 0.1, (4096, 2)).astype(np.float32) for lane in ("drums", "bass", "other", "vocals")}
    cleaned = {lane: audio * 0.5 for lane, audio in original.items()}
    predicted = {lane: audio - cleaned[lane] for lane, audio in original.items()}
    original_mix = np.sum(np.stack(list(original.values())), axis=0)

    result = compare_mix_conservation(original, cleaned, predicted, original_mix)

    assert abs(result.cleaned_to_original_mix.level_match_gain - 2.0) < 1.0e-6
    assert abs(result.cleaned_to_original_mix.level_match_gain_db - 6.0206) < 0.001
    assert result.cleaned_to_original_mix.level_matched_delta_rms < 1.0e-7
    assert result.cleaned_to_original_mix.correlation > 0.999999
