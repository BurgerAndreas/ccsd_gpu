"""Shared helpers for checkpoint-driven lambda benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyscf import gto, scf

from ccsd_gpu import cc, has_gpu4pyscf, pyscf_cc


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


MOLECULES = {
    "aniline": ANILINE,
}


def build_molecule(name: str, basis: str, charge: int = 0, spin: int = 0):
    atom = MOLECULES[name]
    return gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)


def build_reference_ccsd(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    if has_gpu4pyscf:
        try:
            mycc_gpu = cc.CCSD(mf.to_gpu())
            mycc_gpu.verbose = 0
            mycc_gpu.kernel()
            if not mycc_gpu.converged:
                raise RuntimeError("GPU CCSD did not converge")
            ref = pyscf_cc.CCSD(mf)
            ref.verbose = 0
            ref.t1 = np.asarray(mycc_gpu.t1)
            ref.t2 = np.asarray(mycc_gpu.t2)
            ref.e_corr = float(mycc_gpu.e_corr)
            ref.converged = True
            ref._benchmark_ccsd_mode = "gpu"
            return mf, ref
        except Exception:
            pass

    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    mycc._benchmark_ccsd_mode = "cpu"
    return mf, mycc


def clone_cc(mf, ref):
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.max_memory = ref.max_memory
    mycc.level_shift = ref.level_shift
    mycc.conv_tol_normt = ref.conv_tol_normt
    mycc.max_cycle = ref.max_cycle
    mycc.diis = ref.diis
    mycc.diis_space = ref.diis_space
    mycc.t1 = np.asarray(ref.t1)
    mycc.t2 = np.asarray(ref.t2)
    mycc.e_corr = float(ref.e_corr)
    mycc.converged = True
    return mycc


def clone_cc_gpu(mf, ref):
    if not has_gpu4pyscf:
        raise RuntimeError("gpu4pyscf unavailable")
    gpu_mf = mf.to_gpu()
    mycc = cc.CCSD(gpu_mf)
    mycc.verbose = 0
    mycc.max_memory = ref.max_memory
    mycc.level_shift = ref.level_shift
    mycc.conv_tol_normt = ref.conv_tol_normt
    mycc.max_cycle = ref.max_cycle
    mycc.diis = ref.diis
    mycc.diis_space = ref.diis_space
    mycc.t1 = np.asarray(ref.t1)
    mycc.t2 = np.asarray(ref.t2)
    mycc.e_corr = float(ref.e_corr)
    mycc.converged = True
    return mycc


def save_checkpoint(path: str | Path, molecule: str, basis: str, mf, mycc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": json.dumps(
            {
                "molecule": molecule,
                "basis": basis,
                "charge": int(mf.mol.charge),
                "spin": int(mf.mol.spin),
            }
        ),
        "mo_coeff": np.asarray(mf.mo_coeff),
        "mo_occ": np.asarray(mf.mo_occ),
        "mo_energy": np.asarray(mf.mo_energy),
        "t1": np.asarray(mycc.t1),
        "t2": np.asarray(mycc.t2),
        "e_tot": np.asarray(mf.e_tot),
        "e_corr": np.asarray(mycc.e_corr),
    }
    np.savez_compressed(path, **payload)


def load_checkpoint(path: str | Path):
    data = np.load(Path(path), allow_pickle=False)
    meta = json.loads(str(data["metadata"]))
    mol = build_molecule(
        meta["molecule"],
        meta["basis"],
        charge=meta["charge"],
        spin=meta["spin"],
    )

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.mo_coeff = np.asarray(data["mo_coeff"])
    mf.mo_occ = np.asarray(data["mo_occ"])
    mf.mo_energy = np.asarray(data["mo_energy"])
    mf.e_tot = float(np.asarray(data["e_tot"]))
    mf.converged = True

    ref = pyscf_cc.CCSD(mf)
    ref.verbose = 0
    ref.t1 = np.asarray(data["t1"])
    ref.t2 = np.asarray(data["t2"])
    ref.e_corr = float(np.asarray(data["e_corr"]))
    ref.converged = True
    return meta, mol, mf, ref
