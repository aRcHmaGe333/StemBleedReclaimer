# StemBleedReclaimer

StemBleedReclaimer is an independent stem-cleaning product that identifies audible material leaking into the wrong stem, attributes it to the most likely owner, and conservatively removes only the duplicate leakage.

It uses synchronized evidence from all four stems. A locally inactive `other` stem can therefore reveal isolated drum and bass bleed instead of generating false `other` events.

## Current product behavior

- Reads aligned `drums.wav`, `bass.wav`, `other.wav`, and `vocals.wav`.
- Detects locally inactive regions without treating every quiet signal as silence.
- Attributes bleed using temporal correlation, spectral similarity, owner/target level, and synchronized timing.
- Suppresses a demonstrated owner projection with smooth boundaries.
- Never adds another copy when the owner already contains the sound.
- Preserves uncertain and locally active material.
- Writes new stems; source files are never modified.
- Persists every attribution and measurement in `bleed_attribution_report.json`.

## Launch

Double-click `RUN_STEM_BLEED_RECLAIMER.bat`, then select the source stem folder and a new output folder.

## Python use

```python
from pathlib import Path
from stem_bleed_reclaimer import BleedCleaner

result = BleedCleaner().process_folder(Path("stems"), Path("cleaned_stems"))
```

## Safety boundary

The first release implements conservative duplicate-bleed suppression. Restoring material into an owner stem is intentionally withheld until missing-owner evidence is independently proven; blindly adding it would double an already-present signal.

## License

All rights reserved under the APC/IPClaim license in `LICENSE`.

