# GAVE2 V13: Channel-Coupled Path and FFA Biomarker Model

V13 is a clean experiment built from the V8 R2-V2 teacher. It does not alter
V8-V12 artifacts.

## Why V12 stopped

V12 completed successfully and improved OOF Dice/classification, but its
`prune` correction preserved every V8 class path and froze Task 3. Its strict
projection was `7.628752`, so it correctly emitted `DO_NOT_SUBMIT`.

## V13 changes

- Full native `1536 x 1024` canvas; no crop or resize.
- Five mutually exclusive semantic states: background, artery, vein,
  crossing, and uncertain vessel.
- A/V correction is allowed only inside trusted R2-V2 vessel support.
- Loss includes classification, state coupling, Dice, soft clDice, and
  artery/vein centerline supervision.
- OOF postprocessing explicitly evaluates the sampled COR/INF path protocol
  before accepting a setting. Geodesic hysteresis is retained only for
  reproducibility of the rejected first run.
- Task 3 may replace only AVR and vein density, using registered FFA features.
  Each replacement needs repeated nested CV, bootstrap support, and a
  validation-domain audit. Other proven V8 biomarkers remain frozen.

## R5 repair after the first OOF run

The original geodesic selector was rejected correctly: it reassigned about
12.6% of the teacher's A/V positives and reduced Dice on every fold. R5 keeps
the trained V13 raw predictions but removes geodesic class growth from the
default search. It instead searches a prune-only residual calibration that:

- uses the officially stronger V12 selected predictions as its teacher;
- cannot add artery or vein positives absent from that teacher;
- protects the teacher skeleton and a configurable corridor around it; and
- still requires fold-stable pixel gains plus sampled COR/INF path gains.

This is a selector repair, not permission to submit a failed candidate. If no
R5 candidate passes, V12 remains the submission.

R5.1 extends the calibration boundary because both R5 tasks selected the
largest tested threshold (`0.575`). It searches `0.575` through `0.625` with a
narrow temperature sweep and keeps the same sensitivity, topology, and
three-fold stability gates. No model retraining is required.

## Release contract

The notebook creates one root-layout ZIP only after both segmentation tasks
and vein density pass. AVR may remain frozen because Task 3 writes rejected
targets byte-for-byte from the proven source. The final release decision also
requires local Task 1/2 scores of at least `8.7`, a conservative overall
projection of at least `7.7`, and a valid ZIP SHA256. The projection starts
from the official V12 score (`7.5341`) and discounts segmentation gains using
the observed V8-to-V12 local-to-official transfer ratio for each task.

`READY_FOR_ONE_CAUTIOUS_SUBMISSION` is evidence to test one candidate, not a
guarantee of an official score. `DO_NOT_SUBMIT` means keep V12 on the board.
