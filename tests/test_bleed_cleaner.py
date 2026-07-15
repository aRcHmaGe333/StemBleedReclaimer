# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import numpy as np

from stem_bleed_reclaimer import BleedCleaner, BleedConfig


def _fixture(sample_rate: int = 16000):
    seconds = 4
    frames = sample_rate * seconds
    time = np.arange(frames, dtype=np.float32) / sample_rate
    drums = np.zeros(frames, dtype=np.float32)
    for onset in (2000, 6000, 10000, 14000, 18000, 22000):
        length = 900
        drums[onset : onset + length] += 0.8 * np.sin(np.linspace(0, 20 * np.pi, length)) * np.exp(-np.linspace(0, 7, length))
    bass = (0.55 * np.sin(2 * np.pi * 93 * time)).astype(np.float32)
    bass[: sample_rate] = 0.0
    bass[3 * sample_rate :] = 0.0
    legitimate_other = np.zeros(frames, dtype=np.float32)
    legitimate_other[2 * sample_rate : 3 * sample_rate] = 0.35 * np.sin(2 * np.pi * 440 * time[:sample_rate])
    other = legitimate_other + drums * 0.10 + bass * 0.07
    vocals = np.zeros(frames, dtype=np.float32)
    return {name: value[:, None] for name, value in {"drums": drums, "bass": bass, "other": other, "vocals": vocals}.items()}, legitimate_other


def test_bleed_is_attributed_to_the_actual_simultaneous_owner():
    stems, _ = _fixture()
    cleaner = BleedCleaner(BleedConfig(frame_size=1024, hop_size=256, minimum_confidence=0.72))
    regions = cleaner.analyze(stems, 16000)
    other_owners = {region.owner_stem for region in regions if region.target_stem == "other"}
    assert "drums" in other_owners
    assert "bass" in other_owners


def test_cleaning_reduces_duplicate_bleed_without_erasing_legitimate_other_material():
    stems, legitimate_other = _fixture()
    cleaner = BleedCleaner(BleedConfig(frame_size=1024, hop_size=256, minimum_confidence=0.72))
    regions = cleaner.analyze(stems, 16000)
    cleaned = cleaner.clean(stems, 16000, regions)

    silent_zone = slice(0, 16000)
    before = float(np.sqrt(np.mean(stems["other"][silent_zone, 0] ** 2)))
    after = float(np.sqrt(np.mean(cleaned["other"][silent_zone, 0] ** 2)))
    assert after < before * 0.72

    active_zone = slice(32000, 48000)
    error = cleaned["other"][active_zone, 0] - stems["other"][active_zone, 0]
    assert float(np.sqrt(np.mean(error**2))) < 0.015
    assert float(np.sqrt(np.mean(legitimate_other[active_zone] ** 2))) > 0.20


def test_analysis_is_deterministic():
    stems, _ = _fixture()
    cleaner = BleedCleaner(BleedConfig(frame_size=1024, hop_size=256, minimum_confidence=0.72))
    assert cleaner.analyze(stems, 16000) == cleaner.analyze(stems, 16000)

