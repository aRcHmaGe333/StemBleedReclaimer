# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from stem_bleed_reclaimer.leakage_diagnostics import LeakageRelationDiagnostic


def _references(seed: int = 7, windows: int = 20, size: int = 2048):
    rng = np.random.default_rng(seed)
    refs = {name: rng.normal(0, scale, windows * size).astype(np.float32) for name, scale in (("drums", 0.4), ("bass", 0.3), ("other", 0.2))}
    refs["vocals"] = np.zeros(windows * size, dtype=np.float32)
    return refs


def test_scaled_sum_relationship_reports_gain_and_required_amplification():
    stems = _references()
    stems["vocals"] = 0.1 * (stems["drums"] + stems["bass"] + stems["other"])
    result = LeakageRelationDiagnostic(2048, 1.0, 20).analyze_target(stems, 2048, "vocals")
    assert result.relation == "approximately_scaled_sum"
    assert abs(result.scalar_gain - 0.1) < 0.002
    assert abs(result.equivalent_amplification_db - 20.0) < 0.2
    assert result.direct_sum.explained_variance > 0.999


def test_comb_filtered_bleed_is_not_misreported_as_a_plain_quiet_copy():
    stems = _references()
    summed = stems["drums"] + stems["bass"] + stems["other"]
    stems["vocals"] = lfilter([0.08, 0.0, 0.0, -0.045], [1.0], summed).astype(np.float32)
    result = LeakageRelationDiagnostic(2048, 1.0, 20).analyze_target(stems, 2048, "vocals")
    assert result.relation == "frequency_or_phase_filtered"
    assert result.frequency_filtered_sum.explained_variance > result.direct_sum.explained_variance + 0.08


def test_different_owner_leakage_gains_require_separate_reference_channels():
    stems = _references()
    stems["vocals"] = 0.14 * stems["drums"] - 0.05 * stems["bass"] + 0.025 * stems["other"]
    result = LeakageRelationDiagnostic(2048, 1.0, 20).analyze_target(stems, 2048, "vocals")
    assert result.relation == "different_transfer_per_owner_stem"
    assert result.separate_reference_filter.explained_variance > result.frequency_filtered_sum.explained_variance + 0.08


def test_pooled_relation_is_evaluated_on_tracks_excluded_from_its_training(tmp_path):
    import soundfile as sf

    folders = []
    for track_index in range(3):
        stems = _references(seed=20 + track_index)
        stems["vocals"] = 0.12 * stems["drums"] - 0.04 * stems["bass"] + 0.02 * stems["other"]
        folder = tmp_path / f"track_{track_index}"
        folder.mkdir()
        for lane, audio in stems.items():
            sf.write(folder / f"{lane}.wav", audio, 2048, subtype="FLOAT")
        folders.append(folder)
    result = LeakageRelationDiagnostic(2048, 1.0, 20).analyze_pooled(folders)["vocals"]
    assert result.tracks == 3
    assert result.relation == "different_transfer_per_owner_stem"
    assert result.separate_reference_filter.explained_variance > 0.99
