# Learned Noise-Profile Denoiser

## Product extension

A region certified as containing no wanted source may train a reusable noise profile. This is separate from cross-stem bleed attribution:

- **Cross-stem bleed** has an identifiable musical owner and is handled by synchronized owner attribution.
- **Noise-floor contamination** has no musical owner and is handled by learned spectral denoising.
- **Quiet wanted tails, textures, ambience, and attacks** are neither and must be preserved.

## Professional-quality processing contract

1. Collect multiple certified quiet selections instead of trusting one accidental pause.
2. Exclude attributed drum, bass, other, vocal, transient, tonal, and phrase-tail evidence before learning.
3. Persist per-frequency noise distributions, not one scalar threshold.
4. Support fixed profiles for stationary noise and slowly adaptive profiles for changing noise.
5. Use multi-resolution analysis: shorter windows around transients and longer windows for stable tonal/broadband noise.
6. Estimate a smooth time-frequency gain with Wiener/MMSE-class suppression, preserving phase and enforcing a conservative residual floor.
7. Smooth gain in time and frequency to prevent musical-noise speckles, pumping, watery tails, and gating.
8. Protect transients, sustained harmonics, low-level tails, and stereo coherence explicitly.
9. Process from original audio for every revision; never denoise an already-denoised intermediate.
10. Produce original, noise-only residual, cleaned output, and difference-monitor audition artifacts.
11. Persist the learned profile and all settings so later files from the same separation/source condition can reuse it.
12. Promotion requires HITL comparison against the unprocessed stem and measurable artifact checks; noise reduction alone is not success.

## Integration order

1. StemBleedReclaimer attributes and removes demonstrated cross-stem bleed.
2. Regions remaining owner-free and certified quiet train the noise model.
3. The denoiser removes stationary or slowly changing residual noise.
4. Residual analysis checks that wanted musical identity was not removed.

## Reference behavior

iZotope RX Spectral De-noise learns a fixed noise profile from noise-only selections and provides adaptive profiles for changing noise. Its professional behavior also separates tonal and broadband noise, changes time/frequency resolution, and controls artifacts. StemBleedReclaimer should reproduce those principles without copying proprietary implementation.

## PO proposal: recursively overlaid quiet regions

> I'm thinking that this could be possible - all the quiet regions summed up and mixed together (half over half) then this process repeated until they're pretty much a homogenous mass of noise which is then used to remove the core of noise signal (not sure if the volume increase needs to be reverted, you probably do). This "overlaid" noise signal would be better than an isolated part of a changing one (they literally reflect the other stems AND the changes in them. It's like the total mix minus the current stem).

### DSP interpretation

The product will preserve the intent but combine **spectral power/covariance evidence**, not raw waveform samples. Raw uncorrelated noise has random phase and cancels toward zero when repeatedly averaged; it therefore understates the noise that must be removed. A balanced 50:50 waveform average does not require gain reversal, but it also does not retain the correct stochastic noise amplitude.

The correct equivalent is:

1. Divide every certified quiet region into consistent analysis windows.
2. Preserve each window's original level; never peak-normalize the examples.
3. Convert each window into multi-resolution power spectra and cross-stem covariance evidence.
4. Combine the observations with a balanced robust median/trimmed mean so no one region or recursion order receives extra weight.
5. Keep a distribution per frequency band (floor, median, upper confidence bound), not one flattened waveform.
6. Separate the persistent owner-free noise core from changing, stem-correlated residuals.
7. Model the changing residual as synchronized `total mix minus current stem` evidence and attribute it through the existing cross-stem owner model.
8. Reconstruct a smooth suppression mask at the original signal level; do not restore an averaging gain because no destructive waveform averaging is used.

This yields the requested homogeneous noise **model** while retaining the changing cross-stem component as separate evidence rather than smearing it into the stationary noise floor.
