# GAVE2-S013 Reproduction Source

This repository contains the source used for the final GAVE2 preliminary
submission from team `梯度不下降队`.

| Field | Value |
|---|---:|
| Submission ID | `GAVE2-S013` |
| Release | `v13-r51-fine-calibration` |
| Runtime build | `gave2-v13-r6-r51-fine-calibration` |
| Overall score | `7.69256` |
| Task 1 | `7.91771` |
| Task 2 | `7.95995` |
| Task 3 | `7.31260` |
| Final preliminary rank | `14` |

## Authors

- Hanif Noer Rofiq, Master of Artificial Intelligence student, Monash University
- Xinhe Yang, Master of Artificial Intelligence student, Monash University

## Included source

The final pipeline has three required stages:

1. `experiments/gave2_v8`: inference with the released R2-V2 artery/vein and
   binary-vessel teachers.
2. `experiments/gave2_v12`: CFP-to-FFA registration and generation of the
   registered-FFA teacher used by the final selector.
3. `experiments/gave2_v13`: five-state residual training, R5.1 topology-safe
   selection, Task 3 vein-density correction, submission assembly, and release
   checks.

The small modules retained under `gave2_ensemble` and `gave2_v11` are imported
by those three stages. No unrelated CMRRWNet, SAM, YOLO, V9, V10, V14, or V15
experiment is included.

## Data and external weights

The organizer dataset is not redistributed. Place the provided directory at:

```text
GAVE2_preliminary/
```

The notebooks acquire the public R2-V2 and MINIMA sources and checkpoints.
`NOTICE.md` records their source revisions and checkpoint SHA-256 values. The
acquisition code verifies these values before use.

## Reproduction sequence

The final V13 selector uses the registered-FFA teacher produced by the first
notebook. Run the notebooks in this order on the same Google Drive account:

1. `submission/GAVE2_R2V2_FFA_Residual_V12_Colab.ipynb`
2. `submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb`

Before opening Colab, build the two runtime archives from this checkout:

```bash
python scripts/audit_source_tree.py
python scripts/build_miccai_v12_archive.py --output miccai_v12.zip --force
python scripts/build_miccai_v13_archive.py --output miccai_v13.zip --force
```

Each runtime archive includes the organizer data from `GAVE2_preliminary/` but
excludes weights, checkpoints, and run directories. Upload both archives to
`MyDrive/MICCAI2026/`. The notebooks verify the archive manifest before
extracting it.

The notebooks expect a BF16-capable CUDA GPU. The recorded run used one NVIDIA
L4 GPU with 22.03 GiB of memory. V12 creates the prerequisite run under
`MyDrive/MICCAI2026/runs/gave2_v12_safe_3fold`; V13 reads that immutable output
and writes its own run under `runs/gave2_v13_channel_path_3fold`.

## Python environment and tests

Install a PyTorch build compatible with the selected CUDA runtime, then run:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/gave2_v13 tests/gave2_v12 \
  tests/gave2_v8/test_store_and_fusion.py \
  tests/gave2_v8/test_submission.py \
  --ignore=tests/gave2_v12/test_task3.py -q
```

`tests/gave2_v12/test_task3.py` reads the organizer dataset. Run it after
placing `GAVE2_preliminary/` at the repository root.

## Release boundaries

The repository does not contain the organizer dataset, external checkpoints,
trained fold checkpoints, cached predictions, or competition output images.
`archive_manifest.json` records the exact source tree. Run
`python scripts/audit_source_tree.py` after cloning to verify it.

