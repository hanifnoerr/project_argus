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

Shared utility modules provide dependencies imported by these stages.

## Data and external weights

The organizer dataset is not redistributed. Place the provided directory at:

```text
GAVE2_preliminary/
```

The notebooks acquire the public R2-V2 and MINIMA sources and checkpoints.
`NOTICE.md` records their source revisions and checkpoint SHA-256 values. The
acquisition code verifies these values before use.

## Reproduce the submitted archive

V13 requires the selected Task 1 and Task 2 probability stores produced by V12.
Run both notebooks on the same Google Drive account and keep the V12 outputs
when starting V13.

### 1. Check out the locked source

```bash
git clone https://github.com/hanifnoerr/project_argus.git
cd project_argus
git checkout gave2-s013-verification
python scripts/audit_source_tree.py
```

The tag `gave2-s013-verification` identifies the source used for this
verification release.

### 2. Add the organizer data and build the runtimes

Place the organizer directory at `project_argus/GAVE2_preliminary/`. Build both
runtime archives from the repository root:

```bash
python scripts/build_miccai_v12_archive.py --output miccai_v12.zip --force
python scripts/build_miccai_v13_archive.py --output miccai_v13.zip --force
```

The builders package `GAVE2_preliminary/` with the source needed by each
notebook. Upload the archives to these exact Google Drive paths:

```text
MyDrive/MICCAI2026/miccai_v12.zip
MyDrive/MICCAI2026/miccai_v13.zip
```

### 3. Run V12 from a clean Drive run directory

Open `submission/GAVE2_R2V2_FFA_Residual_V12_Colab.ipynb` in Colab and run all
cells in order. Use a BF16-capable CUDA GPU. The notebook downloads and verifies
the pinned R2-V2 and MINIMA assets, creates the three folds with seed 77, trains
both tasks, and writes the prerequisite stores under:

```text
MyDrive/MICCAI2026/runs/gave2_r2v2_v8/predictions/
MyDrive/MICCAI2026/runs/gave2_v12_safe_3fold/predictions/selected/
```

Keep those directories for the V13 run. V13 checks each V12
`completion_manifest.json` before model selection.

### 4. Run V13 with the V12 outputs retained

Open `submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb` and run all cells in
order. The recorded run used Python 3.12, PyTorch 2.11.0+cu128, Kornia 0.8.3,
and one NVIDIA L4 with 22.03 GiB. Its selected profile was the following for
both Task 1 and Task 2:

```json
{
  "base_channels": 24,
  "batch_size": 2,
  "activation_checkpointing": false
}
```

Check `runs/gave2_v13_channel_path_3fold/selected_profiles.json` before
training. A fallback profile changes the trained model and is not the recorded
GAVE2-S013 configuration.

### 5. Verify the final output

The notebook writes the competition archive to:

```text
MyDrive/MICCAI2026/submissions/gave2_v13_r51_topology_safe/
  v13_candidate/梯度不下降队.zip
```
the archieve can be downloaded here:
```text
https://drive.google.com/drive/folders/1smF1kTs5RxDSgEmLjBCwmrJNa2neuPbs?usp=sharing\
```

The recorded archive contains `Task1/`, `Task2/`, and `Task3/` at its root. Its
size is `69,894,748` bytes and its SHA-256 is:

```text
3e69271eb51d98bdc919345bfe0c0854ec42bdc389a29494cec686d86a0f7e03
```

The V13 notebook also checks the 100 MB limit, folder counts, threshold
preservation after seven-bit compaction, Task 3 provenance, and release gates.

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

## Source integrity

`archive_manifest.json` records the exact source tree. Run
`python scripts/audit_source_tree.py` after cloning to verify it.
