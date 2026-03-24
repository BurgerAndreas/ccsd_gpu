"""
Test the GPU int2e_ip1 kernel against the PySCF CPU reference.

Validates that compute_int2e_ip1_gpu matches mol.intor("int2e_ip1", ...) for
both mol.cart=True (Cartesian) and mol.cart=False (spherical, default PySCF).
The spherical path uses _cart_to_sph_eri1 post-processing inside the wrapper.

Usage:
    uv run python tests/test_int2e_ip1_gpu.py
"""

import time
import numpy as np
from pyscf import gto

from ccsd_gpu.cuda.int2e_ip1_gpu import compute_int2e_ip1_gpu
import cupy


def make_h2_mol():
    """H2 with STO-3G (s-functions only, simplest possible case)."""
    return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True, verbose=0)


def make_h2o_mol():
    """H2O with cc-pVDZ (includes p and d Cartesian functions)."""
    return gto.M(
        atom="O 0 0 0; H 0 0.96 0; H 0.96 0 0",
        basis="cc-pvdz",
        cart=True,
        verbose=0,
    )


def make_ch4_mol():
    """CH4 with STO-3G (p-functions, medium test)."""
    return gto.M(
        atom="""
            C  0.000  0.000  0.000
            H  0.629  0.629  0.629
            H -0.629 -0.629  0.629
            H -0.629  0.629 -0.629
            H  0.629 -0.629 -0.629
        """,
        basis="sto-3g",
        cart=True,
        verbose=0,
    )


# --- Spherical (mol.cart=False, PySCF default) ---

def make_h2o_sph_mol():
    """H2O with cc-pVDZ, spherical harmonics (PySCF default, cart=False)."""
    return gto.M(
        atom="O 0 0 0; H 0 0.96 0; H 0.96 0 0",
        basis="cc-pvdz",
        cart=False,
        verbose=0,
    )


def make_ch4_sph_mol():
    """CH4 with cc-pVDZ, spherical harmonics."""
    return gto.M(
        atom="""
            C  0.000  0.000  0.000
            H  0.629  0.629  0.629
            H -0.629 -0.629  0.629
            H -0.629  0.629 -0.629
            H  0.629 -0.629 -0.629
        """,
        basis="cc-pvdz",
        cart=False,
        verbose=0,
    )


def make_h2o_def2svp_sph_mol():
    """H2O with def2-SVP, spherical (l_max=2, no f-shells)."""
    return gto.M(
        atom="O 0 0 0; H 0 0.96 0; H 0.96 0 0",
        basis="def2-svp",
        cart=False,
        verbose=0,
    )


def make_h2o_pvtz_sph_mol():
    """H2O with cc-pVTZ, spherical — the key production molecule (l_max=3, has f-shells)."""
    return gto.M(
        atom="O 0 0 0; H 0 0.96 0; H 0.96 0 0",
        basis="cc-pvtz",
        cart=False,
        verbose=0,
    )


def make_h2o_pvqz_sph_mol():
    """H2O with cc-pVQZ, spherical — l_max=4, has g-shells on O (nroots up to 9)."""
    return gto.M(
        atom="O 0 0 0; H 0 0.96 0; H 0.96 0 0",
        basis="cc-pvqz",
        cart=False,
        verbose=0,
    )


def test_int2e_ip1_gpu(name, mol, atol=1e-10):
    """
    Compare GPU int2e_ip1 against mol.intor("int2e_ip1") for all shell blocks.
    """
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"  nao={mol.nao}, nbas={mol.nbas}, basis={mol.basis}, cart={mol.cart}")
    print(f"{'=' * 60}")

    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    nao = int(ao_loc[nbas])
    nao_pair = nao * (nao + 1) // 2

    # CPU reference: full int2e_ip1 for all shells
    t0 = time.perf_counter()
    eri1_cpu_full = mol.intor("int2e_ip1", comp=3, aosym="s2kl")
    cpu_time = time.perf_counter() - t0
    # shape: (3, nao, nao, nao_pair) — comp × i-AO × j-AO × kl-pair
    eri1_cpu_full = eri1_cpu_full.reshape(3, nao, nao, nao_pair)
    print(f"  CPU intor time: {cpu_time:.3f}s  shape={eri1_cpu_full.shape}")

    # GPU: process one shell at a time (b0, b1) = (ish, ish+1)
    max_err_total = 0.0
    t0 = time.perf_counter()
    for ish in range(nbas):
        b0, b1 = ish, ish + 1
        fi0 = int(ao_loc[b0])
        fi1 = int(ao_loc[b1])
        nf = fi1 - fi0

        eri1_gpu = compute_int2e_ip1_gpu(mol, b0, b1)  # (3, nf, nao, nao_pair)
        eri1_gpu_np = cupy.asnumpy(eri1_gpu)

        eri1_ref = eri1_cpu_full[:, fi0:fi1, :, :]  # (3, nf, nao, nao_pair)

        err = np.max(np.abs(eri1_gpu_np - eri1_ref))
        if err > max_err_total:
            max_err_total = err
        if err > atol:
            print(
                f"  FAIL ish={ish} (l={mol._bas[ish, 1]}) "
                f"max_err={err:.2e} >= {atol:.0e}"
            )
            # Print a few values for debugging
            idx = np.argmax(np.abs(eri1_gpu_np - eri1_ref))
            flat_shape = 3 * nf * nao * nao_pair
            c = idx // (nf * nao * nao_pair)
            remainder = idx % (nf * nao * nao_pair)
            fi = remainder // (nao * nao_pair)
            remainder2 = remainder % (nao * nao_pair)
            fj = remainder2 // nao_pair
            kl = remainder2 % nao_pair
            print(
                f"    worst at comp={c}, fi_local={fi}, fj_abs={fj}, kl={kl}: "
                f"gpu={eri1_gpu_np.flat[idx]:.6e} cpu={eri1_ref.flat[idx]:.6e}"
            )

    gpu_time = time.perf_counter() - t0
    print(f"  GPU  total time: {gpu_time:.3f}s")
    print(f"  Max error across all shells: {max_err_total:.2e}")
    print(f"  Speedup: {cpu_time / gpu_time:.2f}x")

    passed = max_err_total < atol
    if passed:
        print(f"  PASSED (max_err={max_err_total:.2e} < {atol:.0e})")
    else:
        print(f"  FAILED (max_err={max_err_total:.2e} >= {atol:.0e})")
    return passed


if __name__ == "__main__":
    results = []

    print("\n--- Cartesian (mol.cart=True) ---")
    results.append(test_int2e_ip1_gpu("H2/STO-3G cart", make_h2_mol()))
    results.append(test_int2e_ip1_gpu("CH4/STO-3G cart", make_ch4_mol()))
    results.append(test_int2e_ip1_gpu("H2O/cc-pVDZ cart", make_h2o_mol()))

    print("\n--- Spherical (mol.cart=False, PySCF default) ---")
    results.append(test_int2e_ip1_gpu("H2O/cc-pVDZ sph", make_h2o_sph_mol()))
    results.append(test_int2e_ip1_gpu("CH4/cc-pVDZ sph", make_ch4_sph_mol()))
    results.append(test_int2e_ip1_gpu("H2O/def2-SVP sph", make_h2o_def2svp_sph_mol()))
    results.append(test_int2e_ip1_gpu("H2O/cc-pVTZ sph", make_h2o_pvtz_sph_mol()))
    results.append(test_int2e_ip1_gpu("H2O/cc-pVQZ sph", make_h2o_pvqz_sph_mol()))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass}/{n_total} passed")
    if n_pass < n_total:
        exit(1)
