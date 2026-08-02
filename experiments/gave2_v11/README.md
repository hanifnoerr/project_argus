# GAVE2 V11: audited Task 3 calibration

V11 freezes the official-scored V8 submission for Task 1 and Task 2 and changes only four Task 3 fields. It does not retrain or alter R2-V2 segmentation.

## Release decision

- Frozen: `AVR`, `CRAE`, and `CRVE` from V8.
- Replaced: artery density, vein density, artery fractal dimension, and vein fractal dimension.
- V8 source is pinned to SHA256 `88267cc219240d17186ab45199185834c7433a83a2202e919ebde00687d732d7`.
- Final ZIP layout is `梯度不下降队/Task1`, `梯度不下降队/Task2`, and `梯度不下降队/Task3`.

## Why it passed

The calibrator uses quantized V8 probabilities, physical target-specific feature groups, nested repeated cross-validation, fold-wise median fallbacks, paired bootstrap gates, and a separate validation-domain audit. Hyperparameter selection occurs only inside each outer training fold.

Primary seed 77 accepted the four replacement targets with a 13.58% aggregate held-out error reduction. Locked-protocol seed 2026 accepted the same targets with a 15.37% reduction. AVR worsened under both seeds and is therefore frozen.

The independent release audit verifies CRC, the exact 150-member root layout (`Task1/`, `Task2/`, and `Task3/`), Task 1/2 byte identity, field-level Task 3 changes, and certification hashes. The team ID is used only in the outer ZIP filename.

## Commands

```bash
python -m experiments.gave2_v11.dataset \
  --data-root GAVE2_preliminary \
  --source v8_direct \
  --prediction-store runs/gave2_r2v2_v8/predictions/training/direct \
  --output runs/gave2_r2v2_v11/audit/training_v8_features.npz

python -m experiments.gave2_v11.calibrator \
  --cache runs/gave2_r2v2_v11/audit/training_v8_features.npz \
  --output-dir runs/gave2_r2v2_v11/audit/nested
```

Use the Colab notebook for validation extraction, domain auditing, submission construction, and independent readback.

No offline method can guarantee a leaderboard improvement. V11 is a statistically gated submission candidate, not a claimed score.
