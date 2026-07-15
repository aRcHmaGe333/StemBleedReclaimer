# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly


@dataclass(frozen=True)
class RelationMetrics:
    explained_variance: float
    residual_rms: float
    correlation: float


@dataclass(frozen=True)
class LeakageRelation:
    target_stem: str
    quiet_windows: int
    window_seconds: float
    scalar_gain: float
    equivalent_amplification_db: float
    polarity: str
    best_delay_samples: int
    best_delay_ms: float
    direct_sum: RelationMetrics
    delayed_sum: RelationMetrics
    frequency_filtered_sum: RelationMetrics
    separate_reference_filter: RelationMetrics
    relation: str

    def to_dict(self) -> dict:
        result = asdict(self)
        return result


@dataclass(frozen=True)
class PooledLeakageRelation:
    target_stem: str
    tracks: int
    quiet_windows: int
    direct_sum: RelationMetrics
    delayed_sum: RelationMetrics
    frequency_filtered_sum: RelationMetrics
    separate_reference_filter: RelationMetrics
    relation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1.0e-15))


def _score(actual: list[np.ndarray], predicted: list[np.ndarray]) -> RelationMetrics:
    y = np.concatenate(actual)
    estimate = np.concatenate(predicted)
    residual = y - estimate
    energy = float(np.sum(y * y, dtype=np.float64)) + 1.0e-15
    explained = 1.0 - float(np.sum(residual * residual, dtype=np.float64)) / energy
    denominator = float(np.linalg.norm(y) * np.linalg.norm(estimate))
    correlation_value = float(np.dot(y, estimate) / denominator) if denominator > 1.0e-12 else 0.0
    return RelationMetrics(explained, _rms(residual), correlation_value)


def _shift(signal: np.ndarray, lag: int) -> np.ndarray:
    shifted = np.zeros_like(signal)
    if lag > 0:
        shifted[lag:] = signal[:-lag]
    elif lag < 0:
        shifted[:lag] = signal[-lag:]
    else:
        shifted[:] = signal
    return shifted


class LeakageRelationDiagnostic:
    """Cross-validate how target-quiet audio relates to the other three stems."""

    def __init__(self, analysis_sample_rate: int = 12000, window_seconds: float = 1.0, maximum_windows: int = 48) -> None:
        self.analysis_sample_rate = int(analysis_sample_rate)
        self.window_seconds = float(window_seconds)
        self.maximum_windows = int(maximum_windows)

    def load(self, folder: Path) -> tuple[dict[str, np.ndarray], int]:
        stems: dict[str, np.ndarray] = {}
        source_rate = None
        source_frames = None
        for lane in ("drums", "bass", "other", "vocals"):
            audio, rate = sf.read(Path(folder) / f"{lane}.wav", dtype="float32", always_2d=True)
            if source_rate is None:
                source_rate, source_frames = int(rate), len(audio)
            elif (int(rate), len(audio)) != (source_rate, source_frames):
                raise ValueError("All four stems must be sample-aligned")
            mono = np.mean(audio, axis=1, dtype=np.float32)
            if int(rate) != self.analysis_sample_rate:
                divisor = gcd(int(rate), self.analysis_sample_rate)
                mono = resample_poly(mono, self.analysis_sample_rate // divisor, int(rate) // divisor).astype(np.float32)
            stems[lane] = mono
        return stems, self.analysis_sample_rate

    def _quiet_windows(self, stems: Mapping[str, np.ndarray], target: str, sample_rate: int) -> list[tuple[np.ndarray, np.ndarray]]:
        size = max(256, int(round(self.window_seconds * sample_rate)))
        references = [lane for lane in stems if lane != target]
        candidates = []
        total = min(len(signal) for signal in stems.values())
        for start in range(0, total - size + 1, size):
            y = stems[target][start : start + size]
            x = np.stack([stems[lane][start : start + size] for lane in references], axis=1)
            target_rms = _rms(y)
            reference_rms = _rms(np.sum(x, axis=1))
            if target_rms <= 1.0e-7 or reference_rms < target_rms * 1.8:
                continue
            candidates.append((target_rms / max(reference_rms, 1.0e-12), start, y.copy(), x.copy()))
        candidates.sort(key=lambda row: (row[0], row[1]))
        return [(row[2], row[3]) for row in candidates[: self.maximum_windows]]

    @staticmethod
    def _scalar_model(train: list[tuple[np.ndarray, np.ndarray]], test: list[tuple[np.ndarray, np.ndarray]]):
        train_y = np.concatenate([row[0] for row in train])
        train_sum = np.concatenate([np.sum(row[1], axis=1) for row in train])
        gain = float(np.dot(train_sum, train_y) / (np.dot(train_sum, train_sum) + 1.0e-15))
        predictions = [gain * np.sum(row[1], axis=1) for row in test]
        return gain, predictions

    @staticmethod
    def _delay_model(train: list[tuple[np.ndarray, np.ndarray]], test: list[tuple[np.ndarray, np.ndarray]], sample_rate: int):
        maximum_lag = max(1, int(round(0.025 * sample_rate)))
        combined = np.zeros(2 * maximum_lag + 1, dtype=np.float64)
        for y, references in train:
            x = np.sum(references, axis=1)
            values = correlate(y, x, mode="full", method="fft")
            lags = correlation_lags(len(y), len(x), mode="full")
            keep = (lags >= -maximum_lag) & (lags <= maximum_lag)
            combined += values[keep]
        candidate_lags = np.arange(-maximum_lag, maximum_lag + 1)
        lag = int(candidate_lags[int(np.argmax(np.abs(combined)))])
        train_y = np.concatenate([row[0] for row in train])
        shifted_train = np.concatenate([_shift(np.sum(row[1], axis=1), lag) for row in train])
        gain = float(np.dot(shifted_train, train_y) / (np.dot(shifted_train, shifted_train) + 1.0e-15))
        return lag, gain, [gain * _shift(np.sum(row[1], axis=1), lag) for row in test]

    @staticmethod
    def _frequency_models(train: list[tuple[np.ndarray, np.ndarray]], test: list[tuple[np.ndarray, np.ndarray]]):
        train_y = np.stack([np.fft.rfft(row[0]) for row in train])
        train_x = np.stack([np.fft.rfft(row[1], axis=0) for row in train])
        bins = train_y.shape[1]
        summed = np.sum(train_x, axis=2)
        sum_filter = np.sum(np.conj(summed) * train_y, axis=0) / (np.sum(np.abs(summed) ** 2, axis=0) + 1.0e-9)
        separate_filter = np.zeros((bins, 3), dtype=np.complex128)
        for frequency in range(bins):
            matrix = train_x[:, frequency, :]
            scale = float(np.trace(np.conj(matrix.T) @ matrix).real) / 3.0
            ridge = max(1.0e-9, scale * 1.0e-4)
            separate_filter[frequency] = np.linalg.solve(
                np.conj(matrix.T) @ matrix + ridge * np.eye(3),
                np.conj(matrix.T) @ train_y[:, frequency],
            )
        sum_predictions = []
        separate_predictions = []
        for y, references in test:
            spectrum = np.fft.rfft(references, axis=0)
            sum_predictions.append(np.fft.irfft(np.sum(spectrum, axis=1) * sum_filter, n=len(y)).astype(np.float32))
            separate_predictions.append(np.fft.irfft(np.sum(spectrum * separate_filter, axis=1), n=len(y)).astype(np.float32))
        return sum_predictions, separate_predictions

    @staticmethod
    def _classify(direct: RelationMetrics, delayed: RelationMetrics, filtered: RelationMetrics, separate: RelationMetrics, lag: int = 0) -> str:
        best = max(direct.explained_variance, delayed.explained_variance, filtered.explained_variance, separate.explained_variance)
        if best < 0.30:
            return "weakly_predictable_or_target_content_present"
        if separate.explained_variance > filtered.explained_variance + 0.08:
            return "different_transfer_per_owner_stem"
        if filtered.explained_variance > max(direct.explained_variance, delayed.explained_variance) + 0.08:
            return "frequency_or_phase_filtered"
        if delayed.explained_variance > direct.explained_variance + 0.05 and lag != 0:
            return "primarily_delayed_and_scaled"
        return "approximately_scaled_sum"

    def analyze_target(self, stems: Mapping[str, np.ndarray], sample_rate: int, target: str) -> LeakageRelation:
        windows = self._quiet_windows(stems, target, sample_rate)
        if len(windows) < 8:
            raise ValueError(f"Insufficient target-quiet evidence for {target}: {len(windows)} windows")
        train = windows[::2]
        test = windows[1::2]
        actual = [row[0] for row in test]
        gain, direct_prediction = self._scalar_model(train, test)
        lag, delayed_gain, delayed_prediction = self._delay_model(train, test, sample_rate)
        filtered_prediction, separate_prediction = self._frequency_models(train, test)
        direct = _score(actual, direct_prediction)
        delayed = _score(actual, delayed_prediction)
        filtered = _score(actual, filtered_prediction)
        separate = _score(actual, separate_prediction)
        relation = self._classify(direct, delayed, filtered, separate, lag)
        amplification = float(-20.0 * np.log10(max(abs(gain), 1.0e-12)))
        return LeakageRelation(
            target_stem=target,
            quiet_windows=len(windows),
            window_seconds=self.window_seconds,
            scalar_gain=gain,
            equivalent_amplification_db=amplification,
            polarity="inverted" if gain < 0 else "normal",
            best_delay_samples=lag,
            best_delay_ms=1000.0 * lag / sample_rate,
            direct_sum=direct,
            delayed_sum=delayed,
            frequency_filtered_sum=filtered,
            separate_reference_filter=separate,
            relation=relation,
        )

    def analyze_folder(self, folder: Path) -> dict[str, LeakageRelation]:
        stems, sample_rate = self.load(folder)
        results = {}
        for target in stems:
            try:
                results[target] = self.analyze_target(stems, sample_rate, target)
            except ValueError:
                continue
        return results

    def analyze_pooled(self, folders: list[Path]) -> dict[str, PooledLeakageRelation]:
        datasets = []
        for folder in folders:
            stems, sample_rate = self.load(folder)
            datasets.append((folder.name, stems, sample_rate))
        results: dict[str, PooledLeakageRelation] = {}
        for target in ("drums", "bass", "other", "vocals"):
            by_track = []
            for name, stems, sample_rate in datasets:
                windows = self._quiet_windows(stems, target, sample_rate)
                if len(windows) >= 8:
                    by_track.append((name, windows, sample_rate))
            if len(by_track) < 2:
                continue
            actual_all: list[np.ndarray] = []
            direct_all: list[np.ndarray] = []
            delayed_all: list[np.ndarray] = []
            filtered_all: list[np.ndarray] = []
            separate_all: list[np.ndarray] = []
            lags = []
            for held_out in range(len(by_track)):
                test = by_track[held_out][1]
                train = [window for index, (_name, windows, _rate) in enumerate(by_track) if index != held_out for window in windows]
                actual_all.extend([row[0] for row in test])
                _gain, direct_prediction = self._scalar_model(train, test)
                lag, _delay_gain, delayed_prediction = self._delay_model(train, test, by_track[held_out][2])
                filtered_prediction, separate_prediction = self._frequency_models(train, test)
                lags.append(lag)
                direct_all.extend(direct_prediction)
                delayed_all.extend(delayed_prediction)
                filtered_all.extend(filtered_prediction)
                separate_all.extend(separate_prediction)
            direct = _score(actual_all, direct_all)
            delayed = _score(actual_all, delayed_all)
            filtered = _score(actual_all, filtered_all)
            separate = _score(actual_all, separate_all)
            representative_lag = int(np.median(lags))
            results[target] = PooledLeakageRelation(
                target_stem=target,
                tracks=len(by_track),
                quiet_windows=sum(len(row[1]) for row in by_track),
                direct_sum=direct,
                delayed_sum=delayed,
                frequency_filtered_sum=filtered,
                separate_reference_filter=separate,
                relation=self._classify(direct, delayed, filtered, separate, representative_lag),
            )
        return results
