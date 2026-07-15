# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
from scipy.signal import get_window


DEFAULT_LANES = ("drums", "bass", "other", "vocals")


@dataclass(frozen=True)
class BleedConfig:
    frame_size: int = 4096
    hop_size: int = 1024
    silence_dbfs: float = -62.0
    minimum_owner_ratio: float = 2.0
    minimum_confidence: float = 0.78
    maximum_projection: float = 0.95
    transition_ms: float = 12.0


@dataclass(frozen=True)
class BleedRegion:
    target_stem: str
    owner_stem: str
    start_sample: int
    end_sample: int
    confidence: float
    spectral_similarity: float
    temporal_similarity: float
    target_rms: float
    owner_rms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CleanResult:
    sample_rate: int
    frames: int
    regions: tuple[BleedRegion, ...]
    before_rms: Mapping[str, float]
    after_rms: Mapping[str, float]
    output_paths: Mapping[str, str]

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "frames": self.frames,
            "regions": [region.to_dict() for region in self.regions],
            "before_rms": dict(self.before_rms),
            "after_rms": dict(self.after_rms),
            "output_paths": dict(self.output_paths),
        }


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1.0e-15))


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1, dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def _signed_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.clip(abs(np.dot(left, right) / denominator), 0.0, 1.0))


class BleedCleaner:
    def __init__(self, config: BleedConfig | None = None) -> None:
        self.config = config or BleedConfig()

    @staticmethod
    def load_stem_folder(folder: Path, lanes: tuple[str, ...] = DEFAULT_LANES) -> tuple[dict[str, np.ndarray], int]:
        loaded: dict[str, np.ndarray] = {}
        sample_rate: int | None = None
        frames: int | None = None
        channels: int | None = None
        for lane in lanes:
            path = Path(folder) / f"{lane}.wav"
            if not path.is_file():
                raise FileNotFoundError(f"Required stem is missing: {path}")
            audio, current_rate = sf.read(path, dtype="float32", always_2d=True)
            if sample_rate is None:
                sample_rate, frames, channels = int(current_rate), len(audio), audio.shape[1]
            elif (int(current_rate), len(audio), audio.shape[1]) != (sample_rate, frames, channels):
                raise ValueError("All stems must have identical sample rate, frame count, and channel count")
            loaded[lane] = np.ascontiguousarray(audio)
        assert sample_rate is not None
        return loaded, sample_rate

    def analyze(self, stems: Mapping[str, np.ndarray], sample_rate: int) -> list[BleedRegion]:
        lanes = tuple(stems)
        if len(lanes) < 2:
            return []
        frame_size = self.config.frame_size
        hop = self.config.hop_size
        frames = min(len(stems[lane]) for lane in lanes)
        starts = list(range(0, max(1, frames - frame_size + 1), hop))
        if not starts or starts[-1] + frame_size < frames:
            starts.append(max(0, frames - frame_size))
        window = get_window("hann", frame_size, fftbins=True).astype(np.float32)

        mono = {lane: _mono(stems[lane][:frames]) for lane in lanes}
        frame_rms = {
            lane: np.asarray([_rms(mono[lane][start : start + frame_size]) for start in starts], dtype=np.float32)
            for lane in lanes
        }
        absolute_silence = 10.0 ** (self.config.silence_dbfs / 20.0)
        evidence: list[BleedRegion] = []
        for frame_index, start in enumerate(starts):
            end = min(frames, start + frame_size)
            for target in lanes:
                target_rms = float(frame_rms[target][frame_index])
                # A stem containing steady bleed can sit above its own statistical
                # quiet threshold forever.  Owner dominance plus sonic agreement,
                # rather than target level alone, establishes local inactivity.
                if target_rms <= absolute_silence * 0.25:
                    continue
                target_frame = mono[target][start:end]
                padded_target = np.pad(target_frame, (0, frame_size - len(target_frame))) * window
                target_spectrum = np.abs(np.fft.rfft(padded_target))
                best: BleedRegion | None = None
                for owner in lanes:
                    if owner == target:
                        continue
                    owner_rms = float(frame_rms[owner][frame_index])
                    if owner_rms < target_rms * self.config.minimum_owner_ratio:
                        continue
                    owner_frame = mono[owner][start:end]
                    padded_owner = np.pad(owner_frame, (0, frame_size - len(owner_frame))) * window
                    spectral = _cosine(target_spectrum, np.abs(np.fft.rfft(padded_owner)))
                    temporal = _signed_similarity(padded_target, padded_owner)
                    confidence = 0.58 * spectral + 0.42 * temporal
                    if confidence < self.config.minimum_confidence:
                        continue
                    candidate = BleedRegion(
                        target_stem=target,
                        owner_stem=owner,
                        start_sample=start,
                        end_sample=end,
                        confidence=confidence,
                        spectral_similarity=spectral,
                        temporal_similarity=temporal,
                        target_rms=target_rms,
                        owner_rms=owner_rms,
                    )
                    if best is None or candidate.confidence > best.confidence:
                        best = candidate
                if best is not None:
                    evidence.append(best)
        return self._merge_regions(evidence)

    def _merge_regions(self, regions: list[BleedRegion]) -> list[BleedRegion]:
        merged: list[BleedRegion] = []
        for region in sorted(regions, key=lambda row: (row.target_stem, row.start_sample, row.owner_stem)):
            previous = merged[-1] if merged else None
            if previous and previous.target_stem == region.target_stem and previous.owner_stem == region.owner_stem and region.start_sample <= previous.end_sample:
                weight_left = previous.end_sample - previous.start_sample
                weight_right = region.end_sample - region.start_sample
                total = max(1, weight_left + weight_right)
                merged[-1] = BleedRegion(
                    target_stem=region.target_stem,
                    owner_stem=region.owner_stem,
                    start_sample=previous.start_sample,
                    end_sample=max(previous.end_sample, region.end_sample),
                    confidence=(previous.confidence * weight_left + region.confidence * weight_right) / total,
                    spectral_similarity=(previous.spectral_similarity * weight_left + region.spectral_similarity * weight_right) / total,
                    temporal_similarity=(previous.temporal_similarity * weight_left + region.temporal_similarity * weight_right) / total,
                    target_rms=(previous.target_rms * weight_left + region.target_rms * weight_right) / total,
                    owner_rms=(previous.owner_rms * weight_left + region.owner_rms * weight_right) / total,
                )
            else:
                merged.append(region)
        return merged

    def clean(self, stems: Mapping[str, np.ndarray], sample_rate: int, regions: list[BleedRegion]) -> dict[str, np.ndarray]:
        cleaned = {lane: np.array(audio, dtype=np.float32, copy=True) for lane, audio in stems.items()}
        fade_frames = max(1, int(round(sample_rate * self.config.transition_ms / 1000.0)))
        for region in regions:
            target = cleaned[region.target_stem]
            owner = stems[region.owner_stem]
            start, end = region.start_sample, min(region.end_sample, len(target), len(owner))
            if end <= start:
                continue
            target_part = target[start:end]
            owner_part = owner[start:end]
            denominator = np.sum(owner_part * owner_part, axis=0, dtype=np.float64) + 1.0e-12
            projection = np.sum(target_part * owner_part, axis=0, dtype=np.float64) / denominator
            projection = np.clip(projection, -self.config.maximum_projection, self.config.maximum_projection).astype(np.float32)
            confidence = float(np.clip((region.confidence - self.config.minimum_confidence) / max(1.0e-6, 1.0 - self.config.minimum_confidence), 0.25, 1.0))
            mask = np.ones(end - start, dtype=np.float32) * confidence
            edge = min(fade_frames, len(mask) // 2)
            if edge:
                curve = np.sin(np.linspace(0.0, np.pi / 2.0, edge, dtype=np.float32)) ** 2
                mask[:edge] *= curve
                mask[-edge:] *= curve[::-1]
            target[start:end] = target_part - owner_part * projection[None, :] * mask[:, None]
        return cleaned

    def process_folder(self, source: Path, destination: Path) -> CleanResult:
        stems, sample_rate = self.load_stem_folder(source)
        regions = self.analyze(stems, sample_rate)
        cleaned = self.clean(stems, sample_rate, regions)
        destination.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, str] = {}
        for lane, audio in cleaned.items():
            path = destination / f"{lane}.wav"
            sf.write(path, audio, sample_rate, subtype="FLOAT")
            output_paths[lane] = str(path.resolve())
        return CleanResult(
            sample_rate=sample_rate,
            frames=min(len(audio) for audio in stems.values()),
            regions=tuple(regions),
            before_rms={lane: _rms(audio) for lane, audio in stems.items()},
            after_rms={lane: _rms(audio) for lane, audio in cleaned.items()},
            output_paths=output_paths,
        )
