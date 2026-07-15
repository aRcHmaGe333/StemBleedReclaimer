# Real-Stem Leakage Relation Findings — 2026-07-15

## Question

Is signal found in a nominally quiet target stem a simple attenuated copy of the other three stems, a delayed copy, a comb/phase-filtered copy, or a different transfer from each owner stem?

## Method

Three synchronized full-track four-stem folders were examined. For every target stem, up to 48 one-second low-target/high-reference windows were selected. Models were fitted on alternating windows and evaluated on held-out windows:

1. one scalar gain applied to the sum of the other three stems;
2. one gain plus a delay/polarity estimate;
3. one complex frequency/phase response applied to the summed references;
4. three independent complex frequency/phase responses, one per reference stem.

A second leave-one-track-out test trained on two tracks and evaluated on the excluded third track. Synthetic controls certified that the diagnostic correctly identifies scaled copies, comb/phase-filtered copies, and different per-owner transfers.

## Real results

- The direct scaled-sum model explained approximately **0–6.2%** of held-out target energy. A simple quiet 1:1 copy is not supported.
- Surface Terrestrial vocals: scalar **6.2%**; summed frequency/phase filter **47.1%**, correlation **0.693**.
- Back To The Start vocals: scalar **2.5%**; separate per-owner filters **41.4%**, correlation **0.676**.
- Back To The Start bass: scalar approximately **0%**; separate per-owner filters **38.5%**, correlation **0.628**.
- Several other target/track combinations remained weakly predictable. Their selected windows either contain real target material, use a changing/nonlinear leakage relation, or both.
- A single fixed relation trained across tracks failed leave-one-track-out validation for all four target stems. Therefore one universal fixed impulse response is not supported by this first three-track test.

## Engineering conclusion

The impulse-response analogy is valid for the clearly predictable cases, but the useful operator is not one scalar and is not currently one universal fixed IR. The evidence supports a **target-specific, owner-specific, frequency/phase-aware transfer** learned from multiple quiet regions within the current track. Cross-track learning should predict or initialize an adaptive IR/filter bank, not impose one fixed response on every track.

Room/cabinet convolution is approximately linear and time-invariant. Source-separation bleed can be content-dependent and time-varying; it therefore requires adaptive filtering or a bank of conditioned impulse responses. A tube amplifier likewise cannot be captured perfectly by one linear IR when its nonlinear operating point changes.

## Current limitation

The automatic quiet-window selector is provisional. It selects low-target/high-reference regions but does not yet possess studio ground-truth stem activity. Weak results must not be interpreted as proof that no relation exists. HITL-certified truly empty target regions will make the next measurement substantially stronger.
