# GAVE2 CMRRWNet V7

V7 is a full-resolution, three-fold fine-tuning stage built on the certified V6 CMRRWNet checkpoints. It leaves V6 artifacts unchanged.

## Changes From V6

- Interprets the three model logits as artery evidence, vessel probability, and vein evidence.
- Uses a two-class conditional A/V softmax so artery and vein cannot overlap at the `0.5` decision boundary.
- Trains with vessel detection, balanced A/V classification, channel Dice, and artery/vessel/vein clDice terms.
- Selects checkpoints with a connected-component path score instead of the V6 local `3x3` topology proxy.
- Cross-fits hysteresis and skeleton-branch reconstruction over the immutable three folds.
- Keeps the leaderboard-proven V6 refined Task 3 output for the first V7 submission.

## Colab

Upload `miccai_v7.zip` to `MyDrive/MICCAI2026/`, then run `GAVE2_CMRRWNet_V7_3Fold_BF16_Colab.ipynb` from top to bottom. The notebook requires CUDA with BF16 support but does not require a particular GPU model.

Expected Drive inputs:

- `runs/gave2_cmrrwnet_v6_3fold/fold_manifest.json`
- Six certified V6 `best.pt` checkpoints
- V6 base and refined submission folders

Outputs:

- `<team>_cmrrwnet_v7_control_hybrid.zip`: V6 base Task 1/2 plus refined Task 3
- `<team>_cmrrwnet_v7_main.zip`: V7 Task 1/2 plus refined Task 3

The main ZIP uses connected-path reconstruction only when its cross-fitted OOF gate passes. Otherwise it automatically promotes the raw exclusive V7 probabilities.
