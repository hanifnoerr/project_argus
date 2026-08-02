# Five-State Full-Resolution Residual Pipeline

This repository contains the source code used to produce preliminary submission
`GAVE2-S013` for the GAVE2 challenge. The submission was made by team
`梯度不下降队` on 19 July 2026.

## Authors

- Hanif Noer Rofiq, Master of Artificial Intelligence student, Monash University
- Xinhe Yang, Master of Artificial Intelligence student, Monash University

## Recorded preliminary result

| Result | Value |
|---|---:|
| Final preliminary rank | 14 |
| Overall score | 7.69256 |
| Task 1 score | 7.91771 |
| Task 2 score | 7.95995 |
| Task 3 score | 7.31260 |

## Method scope

The pipeline uses the public R2-V2 `av` and `bv` checkpoints as fixed teacher
models. R2-V2 was the winning method of the GAVE challenge at MICCAI 2025. We
did not train or modify those teacher weights.

Our GAVE2-specific stages are:

- a five-state residual U-Net for background, artery-only, vein-only, crossing,
  and uncertain-vessel states;
- native `1536 x 1024` processing without crop-based training;
- CFP-to-FFA registration with pinned MINIMA correspondences for Task 2;
- one prevalence-stratified three-fold split for training and model selection;
- R5.1 prune-only calibration with protected teacher paths; and
- a gated Task 3 vein-density correction. Other Task 3 values remain fixed.

This is an adaptation of public R2-V2 weights to the GAVE2 data and output
protocol. It is not presented as a new foundation architecture.

## Reproduction entry point

Open and run:

`submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb`

The notebook downloads pinned external source revisions and weights, verifies
their SHA-256 hashes, prepares the organizer data, trains three folds, selects
the final settings, and writes the competition ZIP. It expects a BF16-capable
CUDA GPU. The recorded run used one NVIDIA L4 GPU with 22.03 GiB of memory.

Install the Python dependencies with:

```bash
python -m pip install -r experiments/gave2_v13/requirements.txt
```

Run the source-only tests with:

```bash
python -m pytest tests/gave2_v13 tests/gave2_v12 \
  tests/gave2_v8/test_store_and_fusion.py \
  tests/gave2_v8/test_submission.py \
  --ignore=tests/gave2_v12/test_task3.py -q
```

The omitted Task 3 audit reads the organizer data. After placing the dataset at
`GAVE2_preliminary/`, remove the `--ignore` option to run the complete release
test set used by the notebook.

## Files not included

The GAVE2 dataset, trained fold checkpoints, downloaded R2-V2 weights, MINIMA
weights, and run directories are excluded. The notebook obtains the external
weights from their original releases and checks their hashes. Access to the
organizer dataset is required.

`archive_manifest.json` records the SHA-256 hash of every file in the audited
source snapshot. The corresponding source ZIP has SHA-256
`b52c81a6f1ceadaf2c9553c47766a0aa6be241f8d0b23fec9b0ba05b65ee0ef2`.
