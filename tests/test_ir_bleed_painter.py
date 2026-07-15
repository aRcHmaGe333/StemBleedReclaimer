# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from stem_bleed_reclaimer.ir_bleed_painter import IRBleedPainter


def test_quiet_regions_learn_ir_then_paint_and_remove_full_length_bleed():
    sample_rate = 8000
    frames = sample_rate * 8
    rng = np.random.default_rng(84)
    drums = rng.normal(0, 0.25, frames).astype(np.float32)
    bass = rng.normal(0, 0.18, frames).astype(np.float32)
    other = rng.normal(0, 0.12, frames).astype(np.float32)
    bleed = (
        lfilter([0.11, 0.03, -0.02], [1.0], drums)
        + lfilter([-0.05, 0.0, 0.018], [1.0], bass)
        + lfilter([0.025, -0.01], [1.0], other)
    ).astype(np.float32)
    wanted = np.zeros(frames, dtype=np.float32)
    wanted[3 * sample_rate : 6 * sample_rate] = 0.3 * np.sin(
        2 * np.pi * 311 * np.arange(3 * sample_rate, dtype=np.float32) / sample_rate
    )
    stems = {
        "drums": drums[:, None],
        "bass": bass[:, None],
        "other": other[:, None],
        "vocals": (wanted + bleed)[:, None],
    }
    painter = IRBleedPainter(n_fft=512, hop_size=128)

    model = painter.learn(
        stems,
        sample_rate,
        "vocals",
        quiet_regions=[(0, 3 * sample_rate), (6 * sample_rate, frames)],
    )
    predicted = painter.paint(stems, model)
    cleaned = stems["vocals"] - predicted

    correlation = np.corrcoef(bleed[1000:-1000], predicted[1000:-1000, 0])[0, 1]
    residual = cleaned[1000:-1000, 0] - wanted[1000:-1000]
    assert correlation > 0.97
    assert float(np.sqrt(np.mean(residual**2))) < float(np.sqrt(np.mean(bleed[1000:-1000] ** 2))) * 0.20
    assert float(np.sqrt(np.mean(cleaned[3 * sample_rate : 6 * sample_rate, 0] ** 2))) > 0.19
