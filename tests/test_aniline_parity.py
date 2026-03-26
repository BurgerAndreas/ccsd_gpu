"""Permanent CPU-vs-GPU parity tests for aniline CCSD lambda and gradients."""

import argparse
import numpy as np
from pyscf import cc as pyscf_cc
from pyscf import gto, scf
from pyscf.grad import ccsd as ccsd_grad
from pyscf.lib import logger

from ccsd_gpu.gpu_ccsd_grad import compute_ml_ccsd_forces_gpu
from ccsd_gpu.gpu_ccsd_lambda import solve_lambda_gpu


ANILINE = """
C   1.3960   0.0000   0.0000
C   0.6980   1.2090   0.0000
C  -0.6980   1.2090   0.0000
C  -1.3960   0.0000   0.0000
C  -0.6980  -1.2090   0.0000
C   0.6980  -1.2090   0.0000
N   2.7200   0.0000   0.0000
H   1.2400   2.1480   0.0000
H  -1.2400   2.1480   0.0000
H  -2.4800   0.0000   0.0000
H  -1.2400  -2.1480   0.0000
H   1.2400  -2.1480   0.0000
H   3.1100   0.9400   0.0000
H   3.1100  -0.9400   0.0000
"""


def _make_aniline(basis):
    return gto.M(atom=ANILINE, basis=basis, verbose=0)


def _run_reference_ccsd(mol):
    mf = scf.RHF(mol).run()
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    return mf, mycc


def _run_cpu_lambda(mf, base_cc):
    cpu = pyscf_cc.CCSD(mf)
    cpu.verbose = 0
    cpu.t1, cpu.t2 = base_cc.t1, base_cc.t2
    cpu.e_corr = base_cc.e_corr
    cpu.converged = True
    cpu.solve_lambda()
    return cpu


def _run_gpu_lambda(mf, base_cc):
    gpu = pyscf_cc.CCSD(mf)
    gpu.verbose = logger.INFO
    gpu.t1, gpu.t2 = base_cc.t1, base_cc.t2
    gpu.e_corr = base_cc.e_corr
    gpu.converged = True
    solve_lambda_gpu(gpu, fallback_to_cpu=False)
    return gpu


def _compute_cpu_forces(mf, base_cc, cpu_lambda_cc):
    cpu_grad_cc = pyscf_cc.CCSD(mf)
    cpu_grad_cc.verbose = 0
    cpu_grad_cc.t1, cpu_grad_cc.t2 = base_cc.t1, base_cc.t2
    cpu_grad_cc.l1, cpu_grad_cc.l2 = cpu_lambda_cc.l1, cpu_lambda_cc.l2
    cpu_grad_cc.converged = True
    cpu_grad_cc.converged_lambda = True
    grad = ccsd_grad.Gradients(cpu_grad_cc)
    grad.verbose = 0
    return np.array(grad.kernel())


def run_aniline_parity(basis, atol=1e-6):
    mol = _make_aniline(basis)
    mf, base_cc = _run_reference_ccsd(mol)
    cpu = _run_cpu_lambda(mf, base_cc)
    gpu = _run_gpu_lambda(mf, base_cc)

    l1_err = float(np.max(np.abs(cpu.l1 - gpu.l1)))
    l2_err = float(np.max(np.abs(cpu.l2 - gpu.l2)))
    cpu_forces = _compute_cpu_forces(mf, base_cc, cpu)
    gpu_forces = compute_ml_ccsd_forces_gpu(mf, base_cc.t1, base_cc.t2, gpu.l1, gpu.l2)
    force_err = float(np.max(np.abs(cpu_forces - gpu_forces)))

    print(f"basis={basis} lambda_mode={gpu._lambda_solver_mode}")
    print(f"basis={basis} lambda_l1_err={l1_err:.3e}")
    print(f"basis={basis} lambda_l2_err={l2_err:.3e}")
    print(f"basis={basis} gradient_max_err={force_err:.3e}")

    if l1_err >= atol or l2_err >= atol or force_err >= atol:
        raise AssertionError(
            f"Parity failure for {basis}: l1={l1_err:.3e} l2={l2_err:.3e} grad={force_err:.3e}"
        )


def test_aniline_def2svp_parity():
    run_aniline_parity("def2-svp")


def test_aniline_ccpvtz_parity():
    run_aniline_parity("cc-pvtz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", choices=["def2-svp", "cc-pvtz"], required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    run_aniline_parity(args.basis, atol=args.atol)


if __name__ == "__main__":
    main()
