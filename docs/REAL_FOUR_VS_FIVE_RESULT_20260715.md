# Real four-cleaned-versus-five-stem result — 2026-07-15

## Material

- Original full mix: `kevurushtija 8bit 09 Surface Terrestrial Colonization.flac`
- Separated stems: bass, drums, other, vocals
- Duration: 417.36 seconds
- Sample rate: 44,100 Hz

## Demucs timeline guard

- Original and all four stems: 18,405,576 frames
- Best stem-sum-to-original lag: 0 samples / 0.0 ms
- Alignment correlation: 0.9948513

Demucs did not remove an absolute-silent head or tail in this set. The comparison refuses to score unequal or shifted timelines, preventing a trimmed Demucs result from being mistaken for a content difference.

## Peak-normalized content comparison

Both reconstructions were independently scaled to the original full mix's peak before content-distance scoring.

| Reconstruction | Peak gain | Peak gain dB | Delta RMS | Correlation |
|---|---:|---:|---:|---:|
| Four cleaned stems | 0.9728271 | -0.2392870 | 0.0291838 | 0.9907286 |
| Four cleaned + fifth removed-noise stem | 0.9262274 | -0.6656474 | 0.0246614 | 0.9948412 |

The five-stem reconstruction is closer to the original mix by these content metrics. This does not prove that five stems are intrinsically more accurate: adding the fifth stem algebraically restores the original separated-stem sum. The result shows that the provisional four-stem cleaning removed some wanted content along with bleed.

## Glitch protection

The initial inverse was ill-conditioned by correlated stem channels and created an invalid 18.30 peak in the predicted-noise signal. Conservative regularization and an invariant forbidding predicted bleed from exceeding its target stem removed that failure. Final predicted-bleed peaks were:

- bass: 0.2174971 against target peak 0.8807983
- drums: 0.1724018 against target peak 0.9901123
- other: 0.1351326 against target peak 0.9901123
- vocals: 0.0012329 against target peak 0.3136597

Final outputs contain zero non-finite samples and zero samples above the original reference peak after peak normalization.

## Status

The four-versus-five check is complete. The fifth stem wins this provisional real-material comparison. Promotion of the four-cleaned-stem result requires better quiet-region certification or a less destructive transfer estimate because its content distance is worse than the uncleaned separated-stem baseline.
