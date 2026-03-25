# ccsd_gpu

Standalone extraction of the current GPU CCSD gradient prototype from Egsmole.

What is included:
- GPU-accelerated CCSD gradient assembly in `ccsd_gpu/gpu_ccsd_grad.py`
- Custom CUDA-backed `int2e` and `int2e_ip1` kernels in `ccsd_gpu/cuda/`
- A single-file H2O example in `examples/h2o_ccsd_grad.py`

GPU session setup on this cluster:
1. Create the environment with `uv sync`.
2. Source `scripts/setup_gpu_env.sh` before running any example or importing GPU code.
3. Run the example with `uv run python examples/h2o_ccsd_grad.py`.

The setup script exports `CUDA_PATH`, `CUDA_HOME`, and `LD_LIBRARY_PATH` from the CUDA/NVIDIA wheels installed inside `.venv`. This is required on our cluster because the GPU allocation is present, but those user-space CUDA library paths are not available by default in the shell.

Current execution split:
- RHF reference: CPU PySCF in the example for robustness
- CCSD T solve: GPU4PySCF when available, otherwise CPU PySCF
- Lambda solve: local hybrid GPU solver for the dominant contractions, with CPU fallback for unsupported pieces
- Gradient assembly: custom GPU implementation in this package

For analytic CCSD gradients, the steps are:
1. SCF/HF solve: Get the reference orbitals and MO integrals.
2. CCSD amplitude solve: Solve for the cluster amplitudes `T` (`t1`, `t2`) and compute the CCSD energy.
3. Lambda solve: Solve the left-hand / adjoint / de-excitation equations for `Lambda` (`l1`, `l2`). This is not another SCF. It is a linear equation system built from the converged CCSD state.
4. Response / orbital-relaxation solve: Solve the orbital response equations, often described as CPHF/CPKS, Z-vector, or more generally response equations. This accounts for how the HF orbitals change when nuclei move.

Currently:
1. SCF/HF: can be on GPU via `GPU4PySCF`.
2. CCSD T solve: can be on GPU via `cc = gpu4pyscf.cc.ccsd_incore` when available.
3. Lambda solve: dominant intermediate builds and update contractions can run on GPU via `ccsd_gpu.gpu_ccsd_lambda.solve_lambda_gpu`; the `vvvv` contribution still falls back to CPU.
4. Response equations in the gradient path: partially GPU-accelerated in your custom `gpu_ccsd_grad` code.

`lambda` is usually the same scaling and about as much work as the CCSD amplitude solve, since it requires another iterative tensor-equation solve with expensive contractions.

## Timings

Interactive GPU timings measured on March 25, 2026 on an NVIDIA A100-SXM4-80GB.

| Molecule | Basis | RHF (s) | CCSD (s) | Lambda (s) | Gradient (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| H2O | cc-pVDZ | 2.28 | 5.43 | 3.03 | 1.13 |
| H2O | cc-pVTZ | 1.86 | 10.38 | 7.41 | 1.82 |
| Aniline | def2-SVP | 13.85 | 46.17 | 20.22 | 5.02 |
| Aniline | cc-pVTZ | running | running | running | running |
