"""
Test GPU CCSD gradient against CPU PySCF reference.

Validates correctness and measures speedup.

Usage:
    uv run python tests/test_gpu_ccsd_grad.py
"""

import time
import numpy as np
from pyscf import gto, scf, cc as pyscf_cc
from pyscf.grad import ccsd as ccsd_grad

import ccsd_gpu.gpu_ccsd_grad as gpu_ccsd_grad
from ccsd_gpu.gpu_ccsd_grad import compute_ml_ccsd_forces_gpu
from ccsd_gpu.gpu_ccsd_lambda import solve_lambda_gpu


def make_hf_molecule():
    """H-F molecule with STO-3G basis (tiny, fast)."""
    mol = gto.M(
        atom="H 0 0 0; F 0 0 1.1",
        basis="sto-3g",
        verbose=0,
    )
    return mol


def make_ch4_molecule():
    """CH4 molecule with cc-pVDZ basis (QM7-representative)."""
    mol = gto.M(
        atom="""
            C  0.000  0.000  0.000
            H  0.629  0.629  0.629
            H -0.629 -0.629  0.629
            H -0.629  0.629 -0.629
            H  0.629 -0.629 -0.629
        """,
        basis="cc-pvdz",
        verbose=0,
    )
    return mol


def run_ccsd_and_lambda(mol):
    """Run HF -> CCSD -> Lambda equations, return (mf, t1, t2, l1, l2)."""
    mf = scf.RHF(mol).run()
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    solve_lambda_gpu(mycc)
    return mf, mycc.t1, mycc.t2, mycc.l1, mycc.l2


def run_ccsd_and_lambda_cpu(mol):
    """CPU reference lambda path."""
    mf = scf.RHF(mol).run()
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    mycc.solve_lambda()
    return mf, mycc.t1, mycc.t2, mycc.l1, mycc.l2


def compute_cpu_forces(mf, t1, t2, l1, l2):
    """CPU reference: PySCF CCSD gradient."""
    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.t1, mycc.t2, mycc.l1, mycc.l2 = t1, t2, l1, l2
    mycc.converged = True
    mycc.converged_lambda = True
    grad_calc = ccsd_grad.Gradients(mycc)
    grad_calc.verbose = 0
    return np.array(grad_calc.kernel())


def test_lambda_matches_cpu_reference():
    """Hybrid GPU lambda should match CPU PySCF on a small system."""
    mol = make_hf_molecule()
    _, t1_cpu, t2_cpu, l1_cpu, l2_cpu = run_ccsd_and_lambda_cpu(mol)
    _, t1_gpu, t2_gpu, l1_gpu, l2_gpu = run_ccsd_and_lambda(mol)

    assert np.max(np.abs(t1_gpu - t1_cpu)) < 1e-10
    assert np.max(np.abs(t2_gpu - t2_cpu)) < 1e-10
    assert np.max(np.abs(l1_gpu - l1_cpu)) < 1e-7
    assert np.max(np.abs(l2_gpu - l2_cpu)) < 1e-7


def test_compute_ml_ccsd_forces_gpu_falls_back_to_blocked(monkeypatch):
    mol = make_hf_molecule()
    mf, t1, t2, l1, l2 = run_ccsd_and_lambda_cpu(mol)
    expected = np.array(mf.nuc_grad_method().grad_nuc())

    monkeypatch.setattr(
        gpu_ccsd_grad,
        "_should_fallback_grad_to_cpu",
        lambda t1_: (True, "forced test fallback", {"peak_bytes": 1}),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GPU gradient path should not be called when fallback is forced")

    monkeypatch.setattr(gpu_ccsd_grad, "grad_elec_gpu", fail_if_called)
    calls = {"count": 0}

    def fake_blocked(*args, **kwargs):
        calls["count"] += 1
        return np.zeros((mf.mol.natm, 3))

    monkeypatch.setattr(gpu_ccsd_grad, "grad_elec_gpu_blocked", fake_blocked)

    forces = compute_ml_ccsd_forces_gpu(mf, t1, t2, l1, l2)

    assert calls["count"] == 1
    assert np.allclose(forces, expected)


def test_molecule(name, mol, atol=1e-6):
    """Test GPU forces match CPU forces for a molecule."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"  Atoms: {mol.natm}, AOs: {mol.nao}, Basis: {mol.basis}")
    print(f"{'=' * 60}")

    # Run CCSD + lambda
    print("  Running CCSD + lambda equations...")
    mf, t1, t2, l1, l2 = run_ccsd_and_lambda(mol)
    nocc, nvir = t1.shape
    print(f"  nocc={nocc}, nvir={nvir}")

    # CPU reference
    print("  Computing CPU forces...")
    t0 = time.perf_counter()
    forces_cpu = compute_cpu_forces(mf, t1, t2, l1, l2)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU time: {cpu_time:.3f}s")

    # GPU implementation
    print("  Computing GPU forces...")
    t0 = time.perf_counter()
    forces_gpu = compute_ml_ccsd_forces_gpu(mf, t1, t2, l1, l2)
    gpu_time = time.perf_counter() - t0
    print(f"  GPU time: {gpu_time:.3f}s")

    # Compare
    max_err = np.max(np.abs(forces_gpu - forces_cpu))
    mae = np.mean(np.abs(forces_gpu - forces_cpu))
    print(f"\n  Max error: {max_err:.2e} Ha/Bohr")
    print(f"  MAE:       {mae:.2e} Ha/Bohr")
    print(f"  Speedup:   {cpu_time / gpu_time:.2f}x")

    print(f"\n  CPU forces:\n{forces_cpu}")
    print(f"  GPU forces:\n{forces_gpu}")

    if max_err < atol:
        print(f"\n  PASSED (max_err={max_err:.2e} < {atol:.0e})")
    else:
        print(f"\n  FAILED (max_err={max_err:.2e} >= {atol:.0e})")

    return max_err < atol, cpu_time, gpu_time


def main():
    print("GPU CCSD Gradient Validation Test")
    print("=" * 60)

    results = []

    # Test 1: H-F (STO-3G) — tiny molecule, strict tolerance
    mol_hf = make_hf_molecule()
    passed, cpu_t, gpu_t = test_molecule("H-F / STO-3G", mol_hf, atol=1e-6)
    results.append(("H-F/STO-3G", passed, cpu_t, gpu_t))

    # Test 2: CH4 (cc-pVDZ) — QM7-representative size
    mol_ch4 = make_ch4_molecule()
    passed, cpu_t, gpu_t = test_molecule("CH4 / cc-pVDZ", mol_ch4, atol=1e-6)
    results.append(("CH4/cc-pVDZ", passed, cpu_t, gpu_t))

    # Test 3: Water (cc-pVTZ) — larger basis to stress-test
    mol_h2o = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="cc-pvtz",
        verbose=0,
    )
    passed, cpu_t, gpu_t = test_molecule("H2O / cc-pVTZ", mol_h2o, atol=1e-6)
    results.append(("H2O/cc-pVTZ", passed, cpu_t, gpu_t))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Molecule':<20} {'Status':<10} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10}"
    )
    print("-" * 60)
    all_passed = True
    for name, passed, cpu_t, gpu_t in results:
        status = "PASS" if passed else "FAIL"
        speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
        print(
            f"{name:<20} {status:<10} {cpu_t:<10.3f} {gpu_t:<10.3f} {speedup:<10.2f}x"
        )
        all_passed = all_passed and passed

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
