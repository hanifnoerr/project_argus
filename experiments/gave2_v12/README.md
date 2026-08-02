# GAVE2 V12: R2-V2 + Registered FFA Residual Refinement

V12 is a separate experiment. It does not modify V8-V11 or their outputs.

## Why this version exists

V8 is the strongest public submission so far (`7.49452`) and has excellent
sensitivity/topology but weak Dice. V9-V11 showed that unconstrained threshold
or morphology changes can improve Dice while losing much more topology. V12
therefore learns only a bounded residual on top of the released R2-V2 teacher.

The public leaderboard values are consistent with Task 1/2 weights of 40%
classification, 20% Dice, and 40% topology. Because the organizer's topology
implementation is unavailable, checkpoint selection uses only reproducible
classification and Dice metrics. Both tasks can only prune V8 class support,
and the positive V8 class skeleton is immutable to negative corrections.

## Data contract

- Native `1536 x 1024` canvas only; no crop and no resize during training.
- Task 1 input: CFP RGB, local green contrast, V8 A/Vessel/V probabilities,
  and field-of-view mask (8 channels).
- Task 2 adds registered FFA_A, FFA_AV, temporal difference, and two local
  perfusion cues (13 channels total).
- FFA_A and FFA_AV are registered independently to CFP. MINIMA LoFTR
  correspondences are fit with similarity, affine, then projective RANSAC. A
  phase transform is used only after geometric QA; otherwise identity is
  recorded for that phase.
- The adapter contains a narrow `np.float` compatibility shim required by the
  pinned MINIMA commit and maps its removed `kornia.utils.grid` import to
  Kornia's supported `kornia.utils.create_meshgrid` API. External matcher
  failures are printed and recorded; affected cases use identity registration
  instead of aborting the run.
- The notebook stops before training unless each split/phase has valid MINIMA
  matches for at least 90% of cases, preventing an all-identity fallback run.
- Label crossings remain artery and vein simultaneously. The vessel channel is
  always at least the A/V union.

## Safety contract

1. Every fold starts at an exact epoch-0 V8 reproduction.
2. Three-fold OOF calibration must improve the observed pixel-score component
   in every fold while preserving 100% of the teacher class skeleton.
3. The final gate checks deterministic support/skeleton invariants and treats
   the sampled path proxy as diagnostic only. V9/V10 proved that proxy does not
   reliably track the organizer's COR/INF implementation.
4. `v12_safe` falls back independently to V8 for any rejected task.
5. Task 3 remains the proven V8 payload. The V11 live result showed that the
   available optic-disc heuristic was not reliable enough to change it.
6. BF16 training uses the crossing head's raw logits with masked
   `binary_cross_entropy_with_logits`; non-memory smoke-test failures abort
   immediately rather than selecting a smaller model.
7. The notebook verifies an explicit runtime build ID before extraction, so a
   stale Drive ZIP cannot be combined with newer notebook cells.
8. Full per-case registration reports remain on Drive; notebook output prints
   compact summaries so Colab does not truncate the useful diagnostics.

## Outputs

The Colab notebook produces one independently certified candidate,
`v12_safe`, containing only gated Task 1/2 changes. The ZIP is named with the
team name and contains exactly `Task1/`, `Task2/`, and `Task3/` at its root.

The score target is an experiment objective, not a guarantee. The notebook
emits `DO_NOT_SUBMIT` unless the worst-fold projection clears `7.95`; otherwise
it names exactly one candidate ZIP for a cautious leaderboard test.
