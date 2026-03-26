Current status for the GPU CCSD lambda/gradient work.

## Confirmed results

### Lambda timing instrumentation

`ccsd_gpu/gpu_ccsd_lambda.py` now logs per-iteration timings for:

- `preamble`
- `vvvv`
- `setup`
- `ovvv_m4`
- `voov_m4`
- `other_blocks`
- `tail`
- `total`

### H2O / cc-pVTZ

Validated timing logs:

- `26430457`
- `26431682`

Observed lambda timing:

- `total`: about `0.56s` to `0.66s` per iteration
- `vvvv`: about `0.54s` to `0.65s`
- all other buckets are small

Conclusion:

- even on H2O, `vvvv` is the dominant lambda cost

### Aniline / cc-pVTZ

Validated timing log:

- `26430459`

Observed lambda timing:

- `total`: about `145s` to `147s` per iteration
- `vvvv`: about `139s` to `141s`
- `voov_m4`: about `3.3s` to `3.9s`
- `ovvv_m4`: about `1.4s` to `2.2s`
- `other_blocks`: about `1.0s` to `1.4s`

Conclusion:

- `vvvv` is the bottleneck by far for aniline / cc-pVTZ
- it is not primarily CPU-GPU transfer and not disk I/O

### Aniline / def2-SVP parity

Validated parity log:

- `26430084`

Observed result:

- `lambda_mode=gpu-hybrid`
- `lambda_l1_err=2.574e-12`
- `lambda_l2_err=1.564e-10`
- `gradient_max_err=5.124e-09`

Conclusion:

- the GPU lambda/gradient path is numerically correct on the smaller aniline basis

## vvvv microbenchmark

Standalone benchmark file:

- `benchmarks/benchmark_vvvv_random.py`

Validated random-tensor run:

- `26439866`

Case:

- `nocc=8`
- `nvir=96`
- `dtype=float32`

Observed results:

- `einsum`: mean `3.258 ms`, min `1.890 ms`
- `matmul`: mean `2.191 ms`, min `1.321 ms`
- `matmul_prepacked`: mean `0.761 ms`, min `0.753 ms`
- `matmul_blocked`: slower and numerically worse

Conclusion:

- for explicit dense `vvvv`, the best kernel tested so far is `matmul_prepacked`

## Code changes already made

### Lambda

- added a lambda-specific AO-direct GPU `vvvv` path
- added an explicit dense-`vvvv` prepacked GEMM path in `ccsd_gpu/gpu_ccsd_lambda.py`
- explicit dense `vvvv` now uses cached prepacked RHS when `eris.vvvv` exists and fits
- AO-direct helper remains the fallback for large/direct cases

Important limitation:

- the aniline / cc-pVTZ bottleneck is still the AO-direct path, not the explicit dense-`vvvv` path

### Gradient

- implemented a blocked `dm2` gradient path in `ccsd_gpu/gpu_ccsd_grad.py`
- large cases now route away from the impossible full in-core GPU `dvvvv` / `dm2` model
- added regression coverage for blocked-path routing in `tests/test_gpu_ccsd_grad.py`

## Latest aniline / cc-pVTZ end-to-end test

Blocked-gradient validation job:

- `26438180`

Observed behavior:

- `RHF` completed
- `CCSD` completed
- `Lambda` completed
- gradient still failed with GPU OOM

New OOM location:

- `ccsd_gpu/gpu_ccsd_grad.py:676`
- inside `ccsd_gpu/cuda/int2e_gpu.py:76`
- allocation:
  `eri0_gpu = cupy.zeros((nf, nao, nao_pair), dtype=numpy.float64)`

Conclusion:

- the old `dvvvv`/full-`dm2` OOM has been bypassed
- the next memory wall is the GPU AO integral buffer size in the blocked gradient path
- this OOM is not in the response equations

Follow-up chunked-integral validation job:

- `26440408`

Observed behavior:

- `RHF` completed
- `CCSD` completed
- `Lambda` completed
- gradient still failed with GPU OOM in the same AO integral allocation path

Conclusion:

- internal shell-range subdivision in `compute_int2e_gpu` / `compute_int2e_ip1_gpu` was not sufficient
- the next memory fix must force a much smaller effective `nf` at the actual integral buffer allocation site, or stream tiles directly into the contraction without materializing a large `(nf, nao, nao_pair)` slab

## CPU vs GPU lambda on aniline / cc-pVTZ

Comparison benchmark file:

- `benchmarks/benchmark_aniline_lambda_compare.py`

Validated run:

- `26440413`

Observed result:

- `cpu_lambda_time=1101.113s`
- `gpu_lambda_time=1560.279s`
- `gpu_lambda_mode=gpu-hybrid`
- `lambda_l1_err=8.695e-12`
- `lambda_l2_err=2.877e-09`
- `gpu_speedup_vs_cpu=0.706x`

Conclusion:

- for aniline / cc-pVTZ, the current GPU lambda implementation is slower than CPU
- correctness is good, but performance is not yet competitive

## Updated checkpoint-driven lambda result

Checkpoint generation file:

- `benchmarks/generate_lambda_checkpoint.py`

Checkpoint-driven benchmark files:

- `benchmarks/benchmark_lambda_from_checkpoint.py`
- `benchmarks/benchmark_lambda_iteration_from_checkpoint.py`
- `benchmarks/benchmark_vvvv_from_checkpoint.py`

Validated checkpoint build:

- `26449401`

Observed result:

- checkpoint `benchmark_data/aniline_ccpvtz_lambda_checkpoint.npz` created successfully
- `reference_ccsd_done mode=gpu`

Validated full-lambda benchmark:

- `26449403`

Observed result:

- `cpu_lambda_time=1341.766s`
- `gpu_lambda_time=567.978s`
- `gpu_lambda_mode=gpu-hybrid`
- `gpu_speedup_vs_cpu=2.362x`
- `lambda_l1_err=2.406e-13`
- `lambda_l2_err=8.616e-15`

Important detail:

- this winning configuration used the hybrid path with `--use-cpu-vvvv`
- GPU still handles the dominant intermediate builds and non-`vvvv` contractions
- CPU handles the large AO-direct `vvvv` contribution

Conclusion:

- for aniline / cc-pVTZ, the best validated lambda path is now faster than CPU
- the benchmark-proven winner is the hybrid mode with CPU `vvvv`
- this mode is now the production default for large AO-direct systems

## Interpretation

For aniline / cc-pVTZ, the current state is:

1. lambda is stable and the best validated production path is now `gpu-hybrid-cpu-vvvv`
2. blocked gradient progressed past the old `dvvvv` failure
3. the new gradient bottleneck is GPU allocation for `(nf, nao, nao_pair)` AO integral blocks
4. the remaining performance target is a pure-GPU `vvvv` path that can beat the current hybrid winner, not just beat CPU

## Next plan

### Goal

Reach an end-to-end aniline / cc-pVTZ path that:

- does not OOM
- preserves or improves on the current `2.362x` lambda speedup over CPU
- can be benchmarked without rerunning RHF and CCSD every iteration of development

### Priority 1: make the gradient AO integral path memory-safe

The current chunking still allows an oversized `eri0_gpu` allocation. The next implementation step should be a streamed integral-contraction path:

- enforce a hard cap on the actual AO-function tile size that reaches `_compute_cart`
- split oversized shell blocks again at the point where `nf` is known, not only at the public entry point
- avoid concatenating large intermediate tensors when possible
- contract each AO tile into the target accumulator immediately, then release the tile

Expected tradeoff:

- this will likely be slower than the current failing path
- but a slower completed gradient is preferable to an OOM
- streaming is the only realistic route to recover both safety and performance later

### Priority 2: replace the hybrid winner only if a pure-GPU path can beat it

The random-tensor result shows that `matmul_prepacked` is best for explicit dense `vvvv`, but aniline / cc-pVTZ still spends its time in the AO-direct `vvvv` path. The current production winner uses CPU `vvvv`, so the next lambda target is stricter than before: a pure-GPU `vvvv` algorithm must beat the hybrid path, not just beat CPU.

Candidate directions:

- keep iteration-invariant ERI transforms resident on GPU when possible
- prepack and cache any reusable RHS/LHS layouts once, not per lambda iteration
- reformulate the AO-direct `vvvv` route into tiled GEMM/cuTensor-friendly contractions
- avoid repeated host-to-device staging inside the lambda loop
- evaluate whether DF/Cholesky `vvvv` is acceptable for the target accuracy and would materially reduce both memory and compute

Expected outcome:

- minor einsum rewrites will not be enough
- beating the current hybrid winner likely requires a different `vvvv` algorithm for large systems

### Priority 3: preserve the best explicit-`vvvv` kernel where it applies

Retain `matmul_prepacked` for cases where explicit `eris.vvvv` exists and fits, but treat it as a small/explicit-tensor fast path rather than the solution for aniline / cc-pVTZ.

## Benchmark strategy

The current workflow is too expensive because full RHF + CCSD is being repeated before every lambda experiment. The benchmark plan should shift to cached and isolated measurements.

### A. Reusable precomputed benchmark inputs

Create a benchmark-data generation script for aniline / cc-pVTZ that runs once and writes:

- molecule metadata
- converged RHF object data needed to rebuild MO spaces
- converged CCSD amplitudes `t1`, `t2`
- ERI/intermediate data needed for lambda benchmarking

The goal is to let future benchmark jobs start directly from a saved post-CCSD state and time only:

- lambda iterations
- individual `vvvv` kernels
- gradient AO integral contractions

Status:

- implemented
- current checkpoint: `benchmark_data/aniline_ccpvtz_lambda_checkpoint.npz`

### B. Individual lambda microbenchmarks

Build separate benchmark files for:

- AO-direct `vvvv` only
- explicit dense `vvvv` only
- `ovvv_m4`
- `voov_m4`
- one full lambda iteration from a saved checkpoint

Each benchmark should:

- use fixed saved tensors where possible
- report both wall time and peak GPU memory
- allow repeated short runs on the same data

### C. Full-lambda benchmark without CCSD

Add a checkpoint-driven lambda benchmark that:

- loads saved `t1`, `t2`, and ERIs/intermediates
- runs `solve_lambda_gpu` only
- compares against CPU `solve_lambda`
- prints speed ratio and error norms

This should become the default performance benchmark for aniline / cc-pVTZ.

### D. Gradient-kernel benchmark without full gradient

Add a focused AO integral benchmark that:

- loads the saved molecule and density intermediates
- times only the GPU integral tile generation and contraction
- sweeps tile-size limits
- reports OOM threshold, runtime, and effective bandwidth

This is the fastest way to tune the non-OOM gradient path.

## Useful logs

- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26430084.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26430457.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26430459.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26431682.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26438180.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26439866.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26440408.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26440413.txt`
