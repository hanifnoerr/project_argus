# GAVE2 Four-Output Ensemble Runbook

## Production CMRRWNet V2

For the serious L4 submission, use `GAVE2_CMRRWNet_V2_L4_Colab.ipynb` with
`miccai_cmrrwnet_v2.zip`. This isolated workflow writes only to
`MyDrive/MICCAI2026/runs/gave2_cmrrwnet_v2`, requires five certified
`best.pt` checkpoints per task, selects TTA and calibration using OOF
predictions, calibrates Task 3 from Task 2 OOF masks, and creates a fresh
scientifically validated submission ZIP. Legacy commands below remain for
the earlier four-branch prototype and are not used by the v2 notebook.

This package creates four testable outputs:

- `submissions/cmrrwnet/team_id`
- `submissions/sam3/team_id`
- `submissions/yolo_native/team_id`
- `submissions/ensemble/team_id`

All segmentation outputs are full-frame RGB probability PNGs at native GAVE2 geometry: `1536 x 1024`, with `R=artery`, `G=all vessels`, `B=vein`.

## Environments

Main environment for CMRRWNet, YOLO-native, prediction, ensembling, validation:

```powershell
conda create -n gave2-main python=3.11 -y
conda activate gave2-main
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow
```

SAM3 environment, if you want official SAM3/SAM3.1 dependencies beside this native decoder branch:

```powershell
conda create -n gave2-sam3 python=3.12 -y
conda activate gave2-sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git external/sam3
cd external/sam3
pip install -e ".[train]"
```

The implemented `sam3` branch is a native full-resolution semantic decoder branch. It is SAM-style and compatible with the separate SAM3 environment, but it does not depend on prompt-only SAM masks.

## Smoke Tests

Run tiny one-epoch checks first:

```powershell
python -m experiments.gave2_ensemble.train --branch cmrrwnet --task task2 --data-root GAVE2_preliminary --epochs 1 --folds 2 --limit-cases 2 --base-channels 16 --batch-size 1 --grad-accum 1 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 5
python -m experiments.gave2_ensemble.train --branch sam3 --task task2 --data-root GAVE2_preliminary --epochs 1 --folds 2 --limit-cases 2 --base-channels 16 --batch-size 1 --grad-accum 1 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 5
python -m experiments.gave2_ensemble.train --branch yolo_native --task task2 --data-root GAVE2_preliminary --epochs 1 --folds 2 --limit-cases 2 --base-channels 16 --batch-size 1 --grad-accum 1 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 5
```

## Full Training

`--epochs 150` is a maximum cap, not a target. The trainer saves `best.pt` using validation `best_dice` and stops after `25` stale epochs by default. The recommended CMRRWNet first pass uses `--preprocess gray_clahe`, `--loss-mode official_bce3`, and `--grad-accum 1`.

Train Task 2 first:

```powershell
python -m experiments.gave2_ensemble.train --branch cmrrwnet --task task2 --data-root GAVE2_preliminary --epochs 150 --folds 5 --batch-size 1 --grad-accum 1 --base-channels 64 --num-iterations 5 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 25 --early-stopping-metric best_dice
python -m experiments.gave2_ensemble.train --branch sam3 --task task2 --data-root GAVE2_preliminary --epochs 150 --folds 5 --batch-size 1 --grad-accum 1 --base-channels 64 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 25 --early-stopping-metric best_dice
python -m experiments.gave2_ensemble.train --branch yolo_native --task task2 --data-root GAVE2_preliminary --epochs 150 --folds 5 --batch-size 1 --grad-accum 1 --base-channels 64 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 25 --early-stopping-metric best_dice
```

Then train Task 1:

```powershell
python -m experiments.gave2_ensemble.train --branch cmrrwnet --task task1 --data-root GAVE2_preliminary --epochs 150 --folds 5 --batch-size 1 --grad-accum 1 --base-channels 64 --num-iterations 5 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 25 --early-stopping-metric best_dice
python -m experiments.gave2_ensemble.train --branch sam3 --task task1 --data-root GAVE2_preliminary --epochs 150 --folds 5 --batch-size 1 --grad-accum 1 --base-channels 64 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 25 --early-stopping-metric best_dice
python -m experiments.gave2_ensemble.train --branch yolo_native --task task1 --data-root GAVE2_preliminary --epochs 150 --folds 5 --batch-size 1 --grad-accum 1 --base-channels 64 --amp bf16 --preprocess gray_clahe --loss-mode official_bce3 --early-stopping-patience 25 --early-stopping-metric best_dice
```

## Prediction

Generate validation predictions for each branch:

```powershell
python -m experiments.gave2_ensemble.predict --branch cmrrwnet --task task1 --data-root GAVE2_preliminary --tta flips --preprocess auto
python -m experiments.gave2_ensemble.predict --branch cmrrwnet --task task2 --data-root GAVE2_preliminary --tta flips --preprocess auto
python -m experiments.gave2_ensemble.predict --branch sam3 --task task1 --data-root GAVE2_preliminary --tta flips --preprocess auto
python -m experiments.gave2_ensemble.predict --branch sam3 --task task2 --data-root GAVE2_preliminary --tta flips --preprocess auto
python -m experiments.gave2_ensemble.predict --branch yolo_native --task task1 --data-root GAVE2_preliminary --tta flips --preprocess auto
python -m experiments.gave2_ensemble.predict --branch yolo_native --task task2 --data-root GAVE2_preliminary --tta flips --preprocess auto
```

## Task 3

Generate biomarker TXT files from each branch's Task 2 output:

```powershell
python -m experiments.gave2_ensemble.biomarkers --branch cmrrwnet --data-root GAVE2_preliminary
python -m experiments.gave2_ensemble.biomarkers --branch sam3 --data-root GAVE2_preliminary
python -m experiments.gave2_ensemble.biomarkers --branch yolo_native --data-root GAVE2_preliminary
```

The current Task 3 implementation is a deterministic proxy from Task 2 probabilities and ROI masks. It is good for a complete four-output test cycle, but the next competitive improvement should replace or calibrate it with optic-disc-aware Zone C measurement.

## Ensemble

Default scalar weights:

```powershell
python -m experiments.gave2_ensemble.ensemble --data-root GAVE2_preliminary
python -m experiments.gave2_ensemble.biomarkers --branch ensemble --data-root GAVE2_preliminary
```

Optional per-channel weights after generating training-split out-of-fold predictions:

```powershell
python -m experiments.gave2_ensemble.predict_oof --branch cmrrwnet --task task2 --data-root GAVE2_preliminary
python -m experiments.gave2_ensemble.predict_oof --branch sam3 --task task2 --data-root GAVE2_preliminary
python -m experiments.gave2_ensemble.predict_oof --branch yolo_native --task task2 --data-root GAVE2_preliminary
python -m experiments.gave2_ensemble.optimize_ensemble_weights --data-root GAVE2_preliminary --submission-root submissions_oof --task-name Task2 --out runs/gave2_ensemble/ensemble_weights_task2.json
python -m experiments.gave2_ensemble.ensemble --data-root GAVE2_preliminary --weights-json runs/gave2_ensemble/ensemble_weights_task2.json
python -m experiments.gave2_ensemble.biomarkers --branch ensemble --data-root GAVE2_preliminary
```

## Validate

```powershell
python -m experiments.gave2_ensemble.validate_outputs --data-root GAVE2_preliminary --height 1024 --width 1536
```

Validation must report `ok: true` for all four branches before submitting.
