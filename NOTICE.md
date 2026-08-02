# External Method and Asset Notice

This repository does not vendor external model weights.

## R2-V2

The pipeline uses the released R2-V2 `av` and `bv` checkpoints as fixed
teachers. R2-V2 is the winning method of the GAVE challenge at MICCAI 2025.

- Source: https://github.com/j-morano/R2-V2
- Commit: `7f6a8ea7a51782b1e0f89723a9ec137ba0a29913`
- `av.pth` SHA-256: `74d425afb714384cb3f4d5db9cc852c1ea6d7552e46c866e29a3777db12b9d80`
- `bv.pth` SHA-256: `db816a3867e8bc235661e76def115ef9a0a865fb34fd4f4c8259586b7f096a61`

## MINIMA

MINIMA is used to obtain CFP-to-FFA correspondences. It does not predict vessel
labels.

- Source: https://github.com/LSXI7/MINIMA
- Commit: `796e7721174f9f829b79b3702bf8c2ae9a3d447a`
- `minima_loftr.ckpt` SHA-256: `810d19773ff898ba04a68c99a3eff9c112210bf884214bd76aec885e83b0e257`

The acquisition scripts verify these revisions and hashes before use. Users
must comply with the licenses and terms of the upstream projects and the GAVE2
dataset.

