# Full-Length IR Bleed Painting

## PO authority

> it's straighforward - the quiet part gives us the IR, then we "paint" in the parts where it's mixed with the actual sounds of the stem by applying the IR to the other three stems - that's how we get full length noise signal to subtract from that stem.

## Implemented signal path

For each target stem independently:

1. Certified quiet regions contain bleed but no wanted target-stem signal.
2. The other three synchronized stems remain separate reference channels.
3. Complex frequency-dependent transfer functions are learned from every reference channel to every target channel. These represent gain, polarity, delay, phase and comb/filter behavior—the impulse-response relationship.
4. The learned transfer is applied to the other three stems over the complete timeline.
5. Their predicted contributions are summed into one full-length `predicted_bleed` stem.
6. `cleaned_target = original_target - predicted_bleed` for the full track, including places where wanted target material is active.

There is no threshold gate in the painting/subtraction stage and no repeated denoising. Every revision starts from the original stems.

## Persisted outputs

- Full-length cleaned stems.
- Full-length predicted-bleed stems for direct audition.
- One reusable IR model per processed target stem.
- Quiet-region and level evidence in `ir_paint_report.json`.
- A fifth full-length stem containing the sum of all predicted/removed bleed.
- Original stem sum, cleaned stem sum, and cleaned-plus-fifth comparison mixes.

When an independent original full mix is supplied, the report measures which reconstruction is closer: four cleaned stems alone, or four cleaned stems plus the fifth removed-noise stem. The authoritative content comparison normalizes each reconstruction to the original full mix's peak before scoring. RMS-matched measurements remain diagnostic only. The report records both gains and residuals, so a quieter cleaned mix is not penalized merely for being quieter. The fifth stem also preserves every removed contribution for direct audition.

Because `cleaned + removed` algebraically reconstructs the original stem sum, five stems being closer does **not** establish that five stems are more accurate. That comparison certifies conservation and wiring. The meaningful content comparison is the globally volume-matched four-cleaned-stem mix against the independent original full mix, followed by residual/identity inspection of the fifth stem.

## Quiet-region input

The command accepts a JSON document containing certified regions in seconds:

```json
{
  "unit": "seconds",
  "regions": {
    "vocals": [[0.0, 12.5], [41.0, 53.0]],
    "other": [[16.0, 24.0]]
  }
}
```

Only the learning regions are selective. Once learned, the resulting IR paints the predicted bleed continuously over the entire target stem.
