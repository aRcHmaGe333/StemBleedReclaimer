# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import istft, stft


@dataclass(frozen=True)
class IRBleedModel:
    target_stem: str
    reference_stems: tuple[str, ...]
    reference_channels: tuple[tuple[str, int], ...]
    sample_rate: int
    frames: int
    target_channels: int
    n_fft: int
    hop_size: int
    transfer: np.ndarray
    quiet_regions: tuple[tuple[int, int], ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            target_stem=self.target_stem,
            reference_stems=np.asarray(self.reference_stems),
            reference_channel_stems=np.asarray([row[0] for row in self.reference_channels]),
            reference_channel_indices=np.asarray([row[1] for row in self.reference_channels], dtype=np.int32),
            sample_rate=self.sample_rate,
            frames=self.frames,
            target_channels=self.target_channels,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            transfer=self.transfer,
            quiet_regions=np.asarray(self.quiet_regions, dtype=np.int64),
        )


class IRBleedPainter:
    """Learn cross-stem transfer in quiet regions and paint it over the full timeline."""

    def __init__(self, n_fft: int = 2048, hop_size: int = 512, ridge_ratio: float = 1.0e-5) -> None:
        if n_fft <= 0 or hop_size <= 0 or hop_size > n_fft:
            raise ValueError("n_fft and hop_size must define a valid overlap-add analysis")
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.ridge_ratio = float(ridge_ratio)

    @staticmethod
    def _validated(stems: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
        prepared = {}
        frames = None
        for lane, audio in stems.items():
            values = np.asarray(audio, dtype=np.float32)
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2 or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid audio for {lane}")
            if frames is None:
                frames = len(values)
            elif len(values) != frames:
                raise ValueError("All stems must have the same frame count")
            prepared[lane] = values
        if frames is None:
            raise ValueError("No stems supplied")
        return prepared, frames

    def _transform(self, signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        frequencies, times, spectrum = stft(
            signal,
            fs=sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_size,
            boundary="zeros",
            padded=True,
        )
        del frequencies
        return times, spectrum

    def learn(
        self,
        stems: Mapping[str, np.ndarray],
        sample_rate: int,
        target_stem: str,
        quiet_regions: list[tuple[int, int]],
    ) -> IRBleedModel:
        prepared, frames = self._validated(stems)
        if target_stem not in prepared:
            raise ValueError(f"Unknown target stem: {target_stem}")
        regions = tuple(sorted((max(0, int(start)), min(frames, int(end))) for start, end in quiet_regions if int(end) > int(start)))
        if not regions:
            raise ValueError("At least one certified quiet region is required")
        reference_stems = tuple(lane for lane in prepared if lane != target_stem)
        reference_channels = tuple((lane, channel) for lane in reference_stems for channel in range(prepared[lane].shape[1]))
        reference_spectra = []
        times = None
        for lane, channel in reference_channels:
            current_times, spectrum = self._transform(prepared[lane][:, channel], sample_rate)
            times = current_times if times is None else times
            reference_spectra.append(spectrum)
        assert times is not None
        references = np.stack(reference_spectra, axis=0)
        frame_samples = np.rint(times * sample_rate).astype(np.int64)
        quiet_mask = np.zeros(len(frame_samples), dtype=bool)
        for start, end in regions:
            quiet_mask |= (frame_samples >= start) & (frame_samples < end)
        if int(np.count_nonzero(quiet_mask)) < max(8, 2 * len(reference_channels)):
            raise ValueError("Certified quiet regions do not contain enough analysis frames")

        target_channels = prepared[target_stem].shape[1]
        frequency_bins = references.shape[1]
        transfer = np.zeros((target_channels, frequency_bins, len(reference_channels)), dtype=np.complex64)
        target_spectra = [self._transform(prepared[target_stem][:, channel], sample_rate)[1] for channel in range(target_channels)]
        for frequency in range(frequency_bins):
            matrix = references[:, frequency, quiet_mask].T
            covariance = np.conj(matrix.T) @ matrix
            scale = float(np.trace(covariance).real) / max(1, len(reference_channels))
            ridge = max(1.0e-12, scale * self.ridge_ratio)
            system = covariance + ridge * np.eye(len(reference_channels), dtype=np.complex128)
            for channel in range(target_channels):
                observed = target_spectra[channel][frequency, quiet_mask]
                transfer[channel, frequency] = np.linalg.solve(system, np.conj(matrix.T) @ observed).astype(np.complex64)
        return IRBleedModel(
            target_stem=target_stem,
            reference_stems=reference_stems,
            reference_channels=reference_channels,
            sample_rate=int(sample_rate),
            frames=frames,
            target_channels=target_channels,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            transfer=transfer,
            quiet_regions=regions,
        )

    def paint(self, stems: Mapping[str, np.ndarray], model: IRBleedModel) -> np.ndarray:
        prepared, frames = self._validated(stems)
        if frames != model.frames:
            raise ValueError("Stem timeline no longer matches the learned model")
        reference_spectra = []
        for lane, channel in model.reference_channels:
            reference_spectra.append(self._transform(prepared[lane][:, channel], model.sample_rate)[1])
        references = np.stack(reference_spectra, axis=0)
        predicted_spectrum = np.einsum("cfr,rft->cft", model.transfer, references, optimize=True)
        predicted = np.zeros((frames, model.target_channels), dtype=np.float32)
        for channel in range(model.target_channels):
            _times, audio = istft(
                predicted_spectrum[channel],
                fs=model.sample_rate,
                window="hann",
                nperseg=model.n_fft,
                noverlap=model.n_fft - model.hop_size,
                input_onesided=True,
                boundary=True,
            )
            copy_frames = min(frames, len(audio))
            predicted[:copy_frames, channel] = audio[:copy_frames].astype(np.float32)
        return predicted
