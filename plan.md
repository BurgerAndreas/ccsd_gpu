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

## Interpretation

For aniline / cc-pVTZ, the current state is:

1. lambda is stable but dominated by AO-direct `vvvv`
2. blocked gradient progressed past the old `dvvvv` failure
3. the new gradient bottleneck is GPU allocation for `(nf, nao, nao_pair)` AO integral blocks

## Next plan

### Priority 1: fix the gradient OOM

Implement a memory-safe AO integral path for the blocked gradient:

- reduce or cap `nf` more aggressively
- or subdivide shell blocks again inside `compute_int2e_gpu`
- likely do the same for `compute_int2e_ip1_gpu`
- keep the existing blocked `dm2` path

Preferred approach:

- make the integral kernel path memory-safe regardless of caller block choice

### Priority 2: keep the best explicit-`vvvv` kernel

Retain `matmul_prepacked` for cases where explicit `eris.vvvv` exists and fits.

### Priority 3: attack the real lambda bottleneck

If the target remains aniline / cc-pVTZ, the next serious lambda optimization is not another local einsum rewrite. It is to improve or replace the AO-direct `vvvv` algorithm itself, because that is still about 95% of lambda iteration time.

## Useful logs

- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26430084.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26430457.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26430459.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26431682.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26438180.txt`
- `/lustre/fsw/portfolios/nvr/users/anburger/outslurm/slurm-26439866.txt`
