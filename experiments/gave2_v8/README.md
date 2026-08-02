# GAVE2 V8: R2-V2 plus graph reconstruction

V8 is a clean, inference-first replacement for the V5-V7 CMRRWNet experiments. It uses the released `av` and `bv` weights from the MICCAI 2025 GAVE-winning R2-V2 method, preserves multilabel artery/vein crossings, and treats graph reconstruction as an optional candidate behind a quantitative safety gate.

## Outputs

- `v8-r2v2-direct`: artery and vein probabilities from the high-sensitivity `av` model, with all-vessel support from the union of `av` and `bv`.
- `v8-r2v2-graph`: crossing-aware segment labeling selected only if it improves the observed leaderboard score proxy without excessive Dice or sensitivity loss.
- Task 3 initially reuses the leaderboard-proven V6 refined text files. This isolates the Task 1/2 experiment and protects the known Task 3 score.

Both candidates retain the full image canvas and produce native `1536 x 1024` RGB PNGs in challenge order: red artery, green all vessels, blue vein.

## Reproducibility

`assets.py` checks out R2-V2 commit `7f6a8ea7a51782b1e0f89723a9ec137ba0a29913` and downloads release `v1`. Every source revision, configuration, and checkpoint is pinned and SHA256-verified. The checkpoints are not committed to this repository.

The released source has no license file in the pinned checkout. V8 therefore imports the downloaded upstream model and preprocessing modules at runtime instead of vendoring them.

## Recommended order

1. Run released `av` and `bv` inference on validation and build the direct candidate.
2. Run the same inference on training and evaluate the direct output with the sampled-path scorer.
3. Search the conservative graph candidates. Build the graph candidate only after reading the gate report.
4. Submit the direct candidate first. Its official result is the transfer diagnostic for the public GAVE-winning weights.

The Colab notebook automates this order and can resume interrupted inference from per-case float16 stores.

