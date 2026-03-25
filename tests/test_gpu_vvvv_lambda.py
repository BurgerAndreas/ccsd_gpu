"""Verify the GPU lambda ``vvvv`` path matches CPU ``_add_vvvv`` on a small system."""

import time

import cupy
import numpy as np
from pyscf import cc as pyscf_cc
from pyscf import gto, scf
from pyscf.lib import logger

from ccsd_gpu.gpu_ccsd_lambda import _add_vvvv_gpu


def make_test_molecule():
    return gto.M(atom="H 0 0 0; F 0 0 1.1", basis="sto-3g", verbose=0)


def main():
    mol = make_test_molecule()
    mf = scf.RHF(mol).run()
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    mycc.solve_lambda()
    eris = mycc.ao2mo()

    t0 = time.perf_counter()
    cpu_vvvv = mycc._add_vvvv(None, mycc.l2, eris, with_ovvv=False, t2sym="jiba")
    cpu_time = time.perf_counter() - t0

    log = logger.Logger(mycc.stdout, mycc.verbose)
    l2_gpu = cupy.asarray(mycc.l2)
    t0 = time.perf_counter()
    gpu_vvvv = cupy.asnumpy(_add_vvvv_gpu(mycc, l2_gpu, log))
    gpu_time = time.perf_counter() - t0

    max_err = np.max(np.abs(cpu_vvvv - gpu_vvvv))
    print(f"cpu_time={cpu_time:.6f}s")
    print(f"gpu_time={gpu_time:.6f}s")
    print(f"speedup={cpu_time / gpu_time:.3f}x")
    print(f"max_err={max_err:.3e}")
    if max_err >= 1e-7:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
