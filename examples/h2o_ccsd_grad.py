import argparse
import time

import numpy as np
from pyscf import gto, scf
from pyscf.lib import logger

from ccsd_gpu import (
    cc,
    compute_ml_ccsd_forces_gpu,
    has_gpu4pyscf,
    pyscf_cc,
    solve_lambda_gpu,
)

MAX_MEMORY_MB = 16000
MOLECULES = {
    "h2o": {
        "label": "H2O",
        "atom": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        "basis": "cc-pvdz",
        "spin": 0,
        "charge": 0,
    },
    "aniline": {
        "label": "Aniline",
        "atom": """
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
""",
        "basis": "def2-svp",
        "spin": 0,
        "charge": 0,
    },
}


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    get = getattr(x, "get", None)
    if callable(get):
        return get()
    return np.asarray(x)


def restore_ccsd_solver(mf, t1, t2, e_corr, max_memory_mb, l1=None, l2=None):
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = logger.INFO
    mycc.max_memory = max_memory_mb
    mycc.t1 = t1
    mycc.t2 = t2
    mycc.e_corr = float(e_corr)
    mycc.converged = True
    if l1 is not None:
        mycc.l1 = l1
    if l2 is not None:
        mycc.l2 = l2
    return mycc


def parse_args():
    parser = argparse.ArgumentParser(description="Run CCSD gradient example.")
    parser.add_argument(
        "--molecule",
        choices=sorted(MOLECULES),
        default="h2o",
        help="Molecule preset to run.",
    )
    parser.add_argument(
        "--basis",
        default=None,
        help="Basis set override. Defaults to the preset basis for the molecule.",
    )
    return parser.parse_args()


def count_heavy_atoms(atom_spec):
    heavy = 0
    for line in atom_spec.replace(";", "\n").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].upper() != "H":
            heavy += 1
    return heavy


def main():
    args = parse_args()
    preset = MOLECULES[args.molecule]
    basis = args.basis or preset["basis"]

    mol = gto.M(
        atom=preset["atom"],
        basis=basis,
        spin=preset["spin"],
        charge=preset["charge"],
        verbose=0,
    )

    print(
        f'{preset["label"]} / {basis}  heavy={count_heavy_atoms(preset["atom"])}  nao={mol.nao_nr()}'
    )

    t0 = time.perf_counter()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_memory = MAX_MEMORY_MB
    mf.kernel()
    print(f'RHF done in {time.perf_counter() - t0:.2f}s')
    if not mf.converged:
        raise RuntimeError('RHF did not converge.')

    used_gpu_ccsd = False
    cc_start = time.perf_counter()
    if has_gpu4pyscf:
        try:
            mycc_gpu = cc.CCSD(mf.to_gpu())
            mycc_gpu.verbose = 0
            mycc_gpu.max_memory = MAX_MEMORY_MB
            mycc_gpu.kernel()
            t1 = to_numpy(mycc_gpu.t1)
            t2 = to_numpy(mycc_gpu.t2)
            e_corr = float(mycc_gpu.e_corr)
            used_gpu_ccsd = True
        except Exception as exc:
            print(f'GPU CCSD failed ({exc}); falling back to CPU CCSD.')
    if not used_gpu_ccsd:
        mycc_cpu = pyscf_cc.CCSD(mf)
        mycc_cpu.verbose = 0
        mycc_cpu.max_memory = MAX_MEMORY_MB
        mycc_cpu.kernel()
        t1 = np.asarray(mycc_cpu.t1)
        t2 = np.asarray(mycc_cpu.t2)
        e_corr = float(mycc_cpu.e_corr)
    print(f"CCSD done ({'GPU' if used_gpu_ccsd else 'CPU'}) in {time.perf_counter() - cc_start:.2f}s")

    lambda_start = time.perf_counter()
    mycc_lambda = restore_ccsd_solver(mf, t1, t2, e_corr, MAX_MEMORY_MB)
    solve_lambda_gpu(mycc_lambda)
    if not mycc_lambda.converged_lambda:
        raise RuntimeError('Lambda equations did not converge.')
    l1 = np.asarray(mycc_lambda.l1)
    l2 = np.asarray(mycc_lambda.l2)
    lambda_mode = getattr(mycc_lambda, "_lambda_solver_mode", "cpu-fallback")
    print(f'Lambda done ({lambda_mode}) in {time.perf_counter() - lambda_start:.2f}s')

    grad_start = time.perf_counter()
    forces = compute_ml_ccsd_forces_gpu(mf, t1, t2, l1, l2)
    print(f'Gradient done (GPU path) in {time.perf_counter() - grad_start:.2f}s')

    print(f'Total CCSD energy: {mf.e_tot + e_corr:.12f} Ha')
    print('Forces (Ha/Bohr):')
    print(np.array2string(forces, precision=10, suppress_small=False))


if __name__ == '__main__':
    main()
