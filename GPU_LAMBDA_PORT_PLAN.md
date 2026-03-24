# GPU CCSD Lambda Port Plan

## Summary
- Implement a repo-local hybrid GPU lambda solver for analytic CCSD gradients.
- Port the highest-cost lambda intermediates and blocked update contractions to CuPy first.
- Keep a narrow CPU fallback surface for the missing `vvvv` path and DIIS vector packing.

## Cost Ranking
- `High`: `make_intermediates_gpu()` construction of `woooo`, `wvooo`, `wVOov`, `wvOOv`, `wvvov`
- `High`: `update_lambda_gpu()` blocked `ovvv` / `voov` contractions and contractions against the stored intermediates
- `High`: full GPU replacement for the `vvvv` contribution currently delegated to CPU `_add_vvvv`
- `Medium`: GPU memory-aware block-size estimator and recomputation strategy
- `Medium`: DIIS entirely on GPU instead of CPU vector packing once per outer iteration
- `Low`: outer iteration driver, convergence norm, denominator application, final lambda symmetrization
- `Low`: plumbing into the local gradient example and test flow

## Implementation Notes
- Mirror `pyscf.cc.ccsd_lambda`, not the gradient code.
- Reuse the local PySCF CCSD object for AO2MO, denominators, vector packing, and the temporary CPU `vvvv` fallback.
- Keep `t1`, `t2`, `l1`, `l2`, and the dominant lambda intermediates on GPU during the iterative solve.
- Integrate the new solver first into the local example and tests before trying to patch upstream `gpu4pyscf`.

## Acceptance Criteria
- Lambda amplitudes match CPU PySCF within tight tolerance on small RHF systems.
- The example uses the hybrid GPU lambda path automatically when CuPy sees a GPU.
- Gradient results remain consistent with the existing CPU-lambda reference path.
