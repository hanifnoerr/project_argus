# GAVE2 CMRRWNet V6

Run `GAVE2_CMRRWNet_V6_3Fold_BF16_Colab.ipynb` from top to bottom on a CUDA GPU with BF16 support.

## Inputs

- Google Drive archive: `MyDrive/MICCAI2026/miccai_v6.zip`
- The archive must extract `GAVE2_preliminary`, `experiments`, `tests`, and the official CMRRWNet source under `knowledge_base`.
- The replacement archive contains the dataset, V6 source, tests, and official architecture source. The notebook extracts it directly without embedded source patches or GitHub authentication.
- The configured team ID is `梯度不下降队`. Change `TEAM_ID` in the first cell only if Baidu displays a different exact team name.

## Training

- Exactly three folds: validation sizes `17`, `17`, and `16`, seed `77`.
- Native `1536 x 1024` canvases; no crops or rotations.
- Task 2: maximum 100 epochs, no early stop before epoch 40.
- Task 1: maximum 50 epochs, no early stop before epoch 25.
- Early stopping: seven stale epochs, `min_delta=1e-4`.
- Task 2 is trained before Task 1.
- The hardware-neutral memory test selects batch `6`, `4`, or `2`, uses accumulation when batch 2 is selected, and retries with activation checkpointing when required.

All base and refiner folds write resumable checkpoints under:

`MyDrive/MICCAI2026/runs/gave2_cmrrwnet_v6_3fold`

## Outputs

The base ZIP is certified before optional refiner training:

`MyDrive/MICCAI2026/梯度不下降队_cmrrwnet_v6_3fold_base.zip`

The optional cross-fitted residual refiner is adopted independently for Task 1 and Task 2 only when all validation gates pass. Its candidate is:

`MyDrive/MICCAI2026/梯度不下降队_cmrrwnet_v6_3fold_refined.zip`

If a refiner fails or is rejected, the corresponding task in the refined candidate remains identical to the certified base output. Task 3 is refit from refined Task 2 probabilities only when the Task 2 refiner is accepted.

Each ZIP is reopened and CRC-tested and must contain exactly 50 Task 1 PNGs, 50 Task 2 PNGs, and 50 Task 3 TXT files before optional automatic runtime disconnection.
