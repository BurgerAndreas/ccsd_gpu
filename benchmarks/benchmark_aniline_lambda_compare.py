"""Benchmark CPU vs GPU lambda solve for aniline."""

from __future__ import annotations

import argparse
import time

import numpy as np
from pyscf import cc as pyscf_cc
from pyscf import gto, scf

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


def build_molecule(basis: str):
    return gto.M(atom=ANILINE, basis=basis, verbose=0)


def build_reference_ccsd(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    return mf, mycc


def clone_cc(mf, ref):
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.max_memory = ref.max_memory
    mycc.t1 = np.asarray(ref.t1)
    mycc.t2 = np.asarray(ref.t2)
    mycc.e_corr = float(ref.e_corr)
    mycc.converged = True
    return mycc


def run_cpu_lambda(mf, ref):
    cpu = clone_cc(mf, ref)
    t0 = time.perf_counter()
    cpu.solve_lambda()
    dt = time.perf_counter() - t0
    return cpu, dt


def run_gpu_lambda(mf, ref):
    gpu = clone_cc(mf, ref)
    t0 = time.perf_counter()
    solve_lambda_gpu(gpu)
    dt = time.perf_counter() - t0
    return gpu, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="cc-pvtz")
    args = parser.parse_args()

    mol = build_molecule(args.basis)
    print(f"aniline lambda compare basis={args.basis} nao={mol.nao_nr()}")
    mf, ref = build_reference_ccsd(mol)
    print("reference_ccsd_done")

    cpu, cpu_time = run_cpu_lambda(mf, ref)
    print(f"cpu_lambda_time={cpu_time:.3f}s")

    gpu, gpu_time = run_gpu_lambda(mf, ref)
    print(f"gpu_lambda_time={gpu_time:.3f}s")
    print(f"gpu_lambda_mode={getattr(gpu, '_lambda_solver_mode', 'unknown')}")

    l1_err = float(np.max(np.abs(np.asarray(cpu.l1) - np.asarray(gpu.l1))))
    l2_err = float(np.max(np.abs(np.asarray(cpu.l2) - np.asarray(gpu.l2))))
    print(f"lambda_l1_err={l1_err:.3e}")
    print(f"lambda_l2_err={l2_err:.3e}")
    print(f"gpu_speedup_vs_cpu={cpu_time / gpu_time:.3f}x")


if __name__ == "__main__":
    main()
