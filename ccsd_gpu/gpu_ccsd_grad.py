"""
GPU-accelerated CCSD analytical nuclear gradients.

Drop-in replacement for the CPU PySCF CCSD gradient, using CuPy for
tensor algebra while keeping integral computation on CPU.

See docs/gpu_ccsd_forces.md for design rationale.

Implementation phases (matching the guide):
  Phase 1: RDM construction on GPU (CuPy einsum)
  Phase 2: MO->AO transformation on GPU
  Phase 3: Gradient contraction loop (CPU integrals, GPU contractions)
  Phase 4: Orbital response / CPHF on GPU
  Phase 5: One-electron and nuclear terms
"""

import numpy
from functools import reduce

import cupy

# gpu4pyscf JK is only faster than CPU JK for large molecules.
# Below this nao threshold the CPU fvind path wins due to GPU kernel-launch
# overhead over ~20 CPHF iterations.
_GPU_JK_NAO_THRESHOLD = 200

from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange, _index_frozen_active, has_frozen_orbitals

# Module-level cache for gpu4pyscf JK availability.
# None = not yet tested; True = available; False = permanently unavailable.
_GPU_JK_AVAILABLE = None
_gpu_jk = None

# Module-level cache for GPU int2e_ip1 kernel availability.
# Only use the GPU int2e_ip1 kernel when nao >= this threshold; below it the
# per-shell kernel-launch overhead exceeds the benefit (measured crossover ~40).
_GPU_INT2E_IP1_NAO_THRESHOLD = 50
_GPU_INT2E_IP1_AVAILABLE = None
_gpu_int2e_ip1 = None

# Module-level cache for GPU int2e kernel availability.
# Only use the GPU int2e kernel when nao >= this threshold; below it the
# kernel launch overhead exceeds the benefit of GPU-side computation.
_GPU_INT2E_NAO_THRESHOLD = 200
_GPU_INT2E_AVAILABLE = None
_gpu_int2e = None


def _check_gpu_jk():
    """Test whether gpu4pyscf JK can be imported. Result is cached globally."""
    global _GPU_JK_AVAILABLE, _gpu_jk
    if _GPU_JK_AVAILABLE is not None:
        return _GPU_JK_AVAILABLE
    try:
        from gpu4pyscf.scf import jk as _jk_mod
        _jk_mod._VHFOpt  # noqa: just check attribute exists
        _gpu_jk = _jk_mod
        _GPU_JK_AVAILABLE = True
    except Exception:
        _GPU_JK_AVAILABLE = False
    return _GPU_JK_AVAILABLE


def _check_gpu_int2e_ip1():
    """Test whether the GPU int2e_ip1 kernel can be compiled. Result is cached globally."""
    global _GPU_INT2E_IP1_AVAILABLE, _gpu_int2e_ip1
    if _GPU_INT2E_IP1_AVAILABLE is not None:
        return _GPU_INT2E_IP1_AVAILABLE
    try:
        from ccsd_gpu.cuda import int2e_ip1_gpu as _mod
        _gpu_int2e_ip1 = _mod
        _GPU_INT2E_IP1_AVAILABLE = True
    except Exception:
        _GPU_INT2E_IP1_AVAILABLE = False
    return _GPU_INT2E_IP1_AVAILABLE


def _check_gpu_int2e():
    """Test whether the GPU int2e kernel can be compiled. Result is cached globally."""
    global _GPU_INT2E_AVAILABLE, _gpu_int2e
    if _GPU_INT2E_AVAILABLE is not None:
        return _GPU_INT2E_AVAILABLE
    try:
        from ccsd_gpu.cuda import int2e_gpu as _mod
        _gpu_int2e = _mod
        _GPU_INT2E_AVAILABLE = True
    except Exception:
        _GPU_INT2E_AVAILABLE = False
    return _GPU_INT2E_AVAILABLE


# ---------------------------------------------------------------------------
# Phase 1: RDM construction on GPU
# ---------------------------------------------------------------------------


def _gamma1_intermediates_gpu(mycc, t1, t2, l1, l2):
    """Port of pyscf.cc.ccsd_rdm._gamma1_intermediates to CuPy.

    Returns four CuPy arrays (doo, dov, dvo, dvv) on GPU.
    """
    t1 = cupy.asarray(t1)
    t2 = cupy.asarray(t2)
    l1 = cupy.asarray(l1)
    l2 = cupy.asarray(l2)

    doo = -cupy.einsum("ja,ia->ij", t1, l1)
    dvv = cupy.einsum("ia,ib->ab", t1, l1)
    xtv = cupy.einsum("ie,me->im", t1, l1)
    dvo = t1.T - cupy.einsum("im,ma->ai", xtv, t1)

    theta = t2 * 2 - t2.transpose(0, 1, 3, 2)
    doo -= cupy.einsum("jkab,ikab->ij", theta, l2)
    dvv += cupy.einsum("jica,jicb->ab", theta, l2)
    xt1 = cupy.einsum("mnef,inef->mi", l2, theta)
    xt2 = cupy.einsum("mnaf,mnef->ea", l2, theta)
    dvo += cupy.einsum("imae,me->ai", theta, l1)
    dvo -= cupy.einsum("mi,ma->ai", xt1, t1)
    dvo -= cupy.einsum("ie,ae->ai", t1, xt2)
    dov = l1

    return doo, dov, dvo, dvv


def _gamma2_intermediates_gpu(mycc, t1, t2, l1, l2):
    """Port of pyscf.cc.ccsd_rdm._gamma2_outcore to CuPy (in-core on GPU).

    For QM7-sized molecules the full 2-RDM fits in GPU memory, so we
    keep everything in-core as CuPy arrays instead of writing to HDF5.

    Returns tuple of 8 CuPy arrays:
        (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)
    where dvvov is None (same as CPU code).
    """
    t1 = cupy.asarray(t1)
    t2 = cupy.asarray(t2)
    l1 = cupy.asarray(l1)
    l2 = cupy.asarray(l2)
    nocc, nvir = t1.shape

    # --- Pass 1: pvOOv, pvoOV, goooo, gooov ---
    pvOOv = cupy.einsum("ikca,jkcb->aijb", l2, t2)
    moo = cupy.einsum("dljd->jl", pvOOv) * 2
    mvv = cupy.einsum("blld->db", pvOOv) * 2
    gooov = cupy.einsum("kc,cija->jkia", t1, pvOOv)

    pvoOV = -cupy.einsum("ikca,jkbc->aijb", l2, t2)
    theta = t2 * 2 - t2.transpose(0, 1, 3, 2)
    pvoOV += cupy.einsum("ikac,jkbc->aijb", l2, theta)
    moo += cupy.einsum("dljd->jl", pvoOV)
    mvv += cupy.einsum("blld->db", pvoOV)
    gooov -= cupy.einsum("jc,cika->jkia", t1, pvoOV)

    mia = cupy.einsum("kc,ikac->ia", l1, t2) * 2 - cupy.einsum("kc,ikca->ia", l1, t2)
    mab = cupy.einsum("kc,kb->cb", l1, t1)
    mij = cupy.einsum("kc,jc->jk", l1, t1) + moo * 0.5

    tau = cupy.einsum("ia,jb->ijab", t1, t1) + t2
    goooo = cupy.einsum("ijab,klab->ijkl", tau, l2) * 0.5
    doooo = (goooo.transpose(0, 2, 1, 3) * 2 - goooo.transpose(0, 3, 1, 2)).conj()

    gooov += cupy.einsum("ji,ka->jkia", -0.5 * moo, t1)
    gooov += cupy.einsum("la,jkil->jkia", 2 * t1, goooo)
    gooov -= cupy.einsum("ib,jkba->jkia", l1, tau)
    gooov = gooov.conj()
    gooov -= cupy.einsum("jkba,ib->jkia", l2, t1)
    dooov = gooov.transpose(0, 2, 1, 3) * 2 - gooov.transpose(1, 2, 0, 3)
    del tau

    # --- Pass 2: goovv, dovvo, doovv ---
    goovv = cupy.einsum("ia,jb->ijab", mia.conj(), t1.conj())
    dovvo = cupy.empty((nocc, nvir, nvir, nocc), dtype=t1.dtype)
    doovv = cupy.empty((nocc, nocc, nvir, nvir), dtype=t1.dtype)

    for p0, p1 in lib.prange(0, nvir, max(1, nvir)):
        tau = cupy.einsum("ia,jb->ijab", t1[:, p0:p1], t1) + t2[:, :, p0:p1]
        tmpoovv = cupy.einsum("ijkl,klab->ijab", goooo, tau)
        tmpoovv -= cupy.einsum("jk,ikab->ijab", mij, tau)
        tmpoovv -= cupy.einsum("cb,ijac->ijab", mab, t2[:, :, p0:p1])
        tmpoovv -= cupy.einsum("bd,ijad->ijab", mvv * 0.5, tau)
        tmpoovv += 0.5 * tau
        tmpoovv = tmpoovv.conj()
        tmpoovv += 0.5 * l2[:, :, p0:p1]
        goovv[:, :, p0:p1] += tmpoovv

        pvOOv_blk = pvOOv[p0:p1]
        pvoOV_blk = pvoOV[p0:p1]
        gOvvO = cupy.einsum("kiac,jc,kb->iabj", l2[:, :, p0:p1], t1, t1)
        gOvvO += cupy.einsum("aijb->iabj", pvOOv_blk)
        govVO = cupy.einsum("ia,jb->iabj", l1[:, p0:p1], t1)
        govVO -= cupy.einsum("ikac,jc,kb->iabj", l2[:, :, p0:p1], t1, t1)
        govVO += cupy.einsum("aijb->iabj", pvoOV_blk)
        dovvo[:, p0:p1] = 2 * govVO + gOvvO
        doovv[:, :, p0:p1] = (-2 * gOvvO - govVO).transpose(3, 0, 1, 2).conj()

        tau -= t2[:, :, p0:p1] * 0.5
        for q0, q1 in lib.prange(0, nvir, max(1, nvir)):
            goovv[:, :, q0:q1, :] += cupy.einsum(
                "dlib,jlda->ijab", pvOOv_blk, tau[:, :, :, q0:q1]
            ).conj()
            goovv[:, :, :, q0:q1] -= cupy.einsum(
                "dlia,jldb->ijab", pvoOV_blk, tau[:, :, :, q0:q1]
            ).conj()
            tmp = pvoOV_blk[:, :, :, q0:q1] + pvOOv_blk[:, :, :, q0:q1] * 0.5
            goovv[:, :, q0:q1, :] += cupy.einsum(
                "dlia,jlbd->ijab", tmp, t2[:, :, :, p0:p1]
            ).conj()

    dovov = goovv.transpose(0, 2, 1, 3) * 2 - goovv.transpose(1, 2, 0, 3)
    del goovv, goooo

    # --- Pass 3: dvvvv, dovvv ---
    dvvvv = cupy.empty((nvir, nvir, nvir, nvir), dtype=t1.dtype)
    dovvv = cupy.empty((nocc, nvir, nvir, nvir), dtype=t1.dtype)

    for p0, p1 in lib.prange(0, nvir, max(1, nvir)):
        l2tmp = l2[:, :, p0:p1]
        gvvvv = cupy.einsum("ijab,ijcd->abcd", l2tmp, t2)
        jabc = cupy.einsum("ijab,ic->jabc", l2tmp, t1)
        gvvvv += cupy.einsum("jabc,jd->abcd", jabc, t1)

        for i in range(p0, p1):
            vvv = gvvvv[i - p0].conj().transpose(1, 0, 2)
            dvvvv[i] = vvv - vvv.transpose(2, 1, 0) * 0.5

        gvovv = cupy.einsum("adbc,id->aibc", gvvvv, -t1)
        del gvvvv

        gvovv += cupy.einsum("akic,kb->aibc", pvoOV[p0:p1], t1)
        gvovv -= cupy.einsum("akib,kc->aibc", pvOOv[p0:p1], t1)
        gvovv += cupy.einsum("ja,jibc->aibc", l1[:, p0:p1], t2)
        gvovv += cupy.einsum("ja,jb,ic->aibc", l1[:, p0:p1], t1, t1)
        gvovv += cupy.einsum("ba,ic->aibc", mvv[:, p0:p1] * 0.5, t1)
        gvovv = gvovv.conj()
        gvovv += cupy.einsum("ja,jibc->aibc", t1[:, p0:p1], l2)

        dovvv[:, :, p0:p1] = gvovv.transpose(1, 3, 0, 2) * 2 - gvovv.transpose(
            1, 2, 0, 3
        )

    dvvov = None
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)


# ---------------------------------------------------------------------------
# Phase 2: MO->AO transformation on GPU
# ---------------------------------------------------------------------------


def _rdm2_mo2ao_gpu(mycc, d2, mo_coeff):
    """MO->AO transform of the 2-RDM entirely on GPU.

    Assembles the full 2-RDM in MO basis from the intermediate blocks
    (following pyscf.cc.ccsd_rdm._make_rdm2 with with_dm1=False),
    does a 4-index transform via CuPy tensordot (cuBLAS), symmetrizes,
    and packs to lower-triangular.

    Returns dm2_tril as a CuPy array of shape (nao_pair, nao_pair).
    """
    log = logger.Logger(mycc.stdout, mycc.verbose)
    time1 = logger.process_clock(), logger.perf_counter()

    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nmo = nocc + nvir

    # --- Assemble full dm2 in MO basis (physics notation) ---
    dm2 = cupy.zeros((nmo, nmo, nmo, nmo), dtype=dovov.dtype)

    dm2[:nocc, nocc:, :nocc, nocc:] = dovov
    dm2[:nocc, nocc:, :nocc, nocc:] += dovov.transpose(2, 3, 0, 1)
    dm2[nocc:, :nocc, nocc:, :nocc] = dm2[:nocc, nocc:, :nocc, nocc:].transpose(
        1, 0, 3, 2
    )

    dm2[:nocc, :nocc, nocc:, nocc:] = doovv
    dm2[:nocc, :nocc, nocc:, nocc:] += doovv.transpose(1, 0, 3, 2)
    dm2[nocc:, nocc:, :nocc, :nocc] = dm2[:nocc, :nocc, nocc:, nocc:].transpose(
        2, 3, 0, 1
    )

    dm2[:nocc, nocc:, nocc:, :nocc] = dovvo
    dm2[:nocc, nocc:, nocc:, :nocc] += dovvo.transpose(3, 2, 1, 0)
    dm2[nocc:, :nocc, :nocc, nocc:] = dm2[:nocc, nocc:, nocc:, :nocc].transpose(
        1, 0, 3, 2
    )

    dm2[nocc:, nocc:, nocc:, nocc:] = dvvvv
    dm2[nocc:, nocc:, nocc:, nocc:] += dvvvv.transpose(1, 0, 3, 2)
    dm2[nocc:, nocc:, nocc:, nocc:] *= 2

    dm2[:nocc, :nocc, :nocc, :nocc] = doooo
    dm2[:nocc, :nocc, :nocc, :nocc] += doooo.transpose(1, 0, 3, 2)
    dm2[:nocc, :nocc, :nocc, :nocc] *= 2

    dm2[:nocc, nocc:, nocc:, nocc:] = dovvv
    dm2[nocc:, nocc:, :nocc, nocc:] = dovvv.transpose(2, 3, 0, 1)
    dm2[nocc:, nocc:, nocc:, :nocc] = dovvv.transpose(3, 2, 1, 0)
    dm2[nocc:, :nocc, nocc:, nocc:] = dovvv.transpose(1, 0, 3, 2)

    dm2[:nocc, :nocc, :nocc, nocc:] = dooov
    dm2[:nocc, nocc:, :nocc, :nocc] = dooov.transpose(2, 3, 0, 1)
    dm2[:nocc, :nocc, nocc:, :nocc] = dooov.transpose(1, 0, 3, 2)
    dm2[nocc:, :nocc, :nocc, :nocc] = dooov.transpose(3, 2, 1, 0)

    # Convert to chemist's notation
    dm2 = dm2.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1("assemble dm2 MO (GPU)", *time1)

    # --- 4-index MO -> AO transform via tensordot (cuBLAS) ---
    C = cupy.asarray(mo_coeff)
    nao = C.shape[0]
    dm2 = cupy.tensordot(C, dm2, axes=(1, 0))
    dm2 = cupy.tensordot(C, dm2, axes=(1, 1)).transpose(1, 0, 2, 3)
    dm2 = cupy.tensordot(C, dm2, axes=(1, 2)).transpose(1, 2, 0, 3)
    dm2 = cupy.tensordot(C, dm2, axes=(1, 3)).transpose(1, 2, 3, 0)
    time1 = log.timer_debug1("4-index MO->AO (GPU)", *time1)

    # --- Symmetrize ---
    dm2 = dm2 + dm2.transpose(1, 0, 2, 3)
    dm2 = dm2 + dm2.transpose(0, 1, 3, 2)
    dm2 *= 0.5
    time1 = log.timer_debug1("symmetrize dm2 (GPU)", *time1)

    # --- Pack to lower triangular (keep on GPU) ---
    nao_pair = nao * (nao + 1) // 2
    rows, cols = cupy.tril_indices(nao)
    pair_idx = rows * nao + cols
    dm2_tril = dm2.reshape(nao * nao, nao * nao)[pair_idx[:, None], pair_idx[None, :]]
    del dm2
    time1 = log.timer_debug1("pack tril (GPU)", *time1)
    return dm2_tril


# ---------------------------------------------------------------------------
# Phase 3-5: Gradient computation
# ---------------------------------------------------------------------------


def grad_elec_gpu(mycc, t1, t2, l1, l2, atmlst=None, verbose=logger.INFO):
    """GPU-accelerated CCSD electronic gradient.

    Port of pyscf.grad.ccsd.grad_elec with:
    - Phase 1: RDM construction on GPU
    - Phase 2: MO->AO transform on GPU, dm2 stays on GPU (no HDF5)
    - Phase 3: Gradient loop fully on GPU (integrals CPU, contractions+vhf GPU)
    - Phase 4: CPHF with GPU J/K (currently CPU fallback for nao < 200)
    - Phase 5: One-electron terms (CPU, they're fast)
    """
    log = logger.new_logger(mycc, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    # Phase 1: Build RDMs on GPU
    log.debug("Build ccsd rdm1 intermediates (GPU)")
    d1 = _gamma1_intermediates_gpu(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    time1 = log.timer_debug1("rdm1 intermediates (GPU)", *time0)

    log.debug("Build ccsd rdm2 intermediates (GPU)")
    d2 = _gamma2_intermediates_gpu(mycc, t1, t2, l1, l2)
    time1 = log.timer_debug1("rdm2 intermediates (GPU)", *time1)

    # Phase 2: MO->AO transformation
    mol = mycc.mol
    mo_coeff = mycc.mo_coeff
    mo_energy = mycc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(mycc.mo_occ > 0)
    with_frozen = has_frozen_orbitals(mycc)
    OA, VA, OF, VF = _index_frozen_active(mycc.get_frozen_mask(), mycc.mo_occ)

    log.debug("symmetrized rdm2 and MO->AO transformation")
    mo_active = mo_coeff[:, numpy.hstack((OA, VA))]
    dm2_tril_gpu = _rdm2_mo2ao_gpu(mycc, d2, mo_active)
    time1 = log.timer_debug1("MO->AO transformation", *time1)

    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    hf_dm1_gpu = cupy.asarray(hf_dm1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = numpy.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    diagidx_gpu = cupy.asarray(diagidx)
    de = numpy.zeros((len(atmlst), 3))
    vhf1 = numpy.zeros((len(atmlst), 3, nao, nao))

    # Precompute tril indices for unpacking s2kl integrals on GPU
    tril_row, tril_col = cupy.tril_indices(nao)
    nao_pair = nao * (nao + 1) // 2
    offdiag_p = tril_row != tril_col   # (nao_pair,) bool

    # Indicator matrices for exchange VHF contractions — avoid 4D eri1_full.
    # I_row[k,p]=1 iff tril_row[p]==k; I_col[l,p]=1 iff tril_col[p]==l.
    _pidx = cupy.arange(nao_pair)
    I_row = cupy.zeros((nao, nao_pair), dtype=cupy.float64)
    I_row[tril_row, _pidx] = 1.0
    I_col = cupy.zeros((nao, nao_pair), dtype=cupy.float64)
    I_col[tril_col, _pidx] = 1.0
    I_col_od = I_col * offdiag_p[None, :]   # off-diagonal pairs only

    # Packed hf_dm1 for Coulomb contraction C3 ("ijkl,kl->ij"):
    # v_hf_dm[p] = P[k,l]+P[l,k] for off-diagonal, P[k,k] for diagonal.
    v_hf_dm = hf_dm1_gpu[tril_row, tril_col].copy()
    v_hf_dm[offdiag_p] += hf_dm1_gpu[tril_col[offdiag_p], tril_row[offdiag_p]]

    # Column-gathered hf_dm1 for exchange contraction X4 ("ijkl,jk->il"):
    P_krow = hf_dm1_gpu[:, tril_row]   # (nao, nao_pair): P[j, k_p=tril_row[p]]
    P_lcol = hf_dm1_gpu[:, tril_col]   # (nao, nao_pair): P[j, l_p=tril_col[p]]

    # Phase 3: 2e AO integrals dot 2pdm (CPU integrals, GPU contractions)
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))

    # Accumulate Imat entirely on GPU; transfer once after the loop.
    Imat_gpu = cupy.zeros((nao, nao))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf_gpu = cupy.zeros((3, nao, nao))
        de_k_gpu = cupy.zeros(3)
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf_gpu = _load_block_tril_gpu(dm2_tril_gpu, ip0, ip1, nao)
            dm2buf_gpu[:, :, diagidx_gpu] *= 0.5

            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)

            # Imat contraction: "ipx,iqx->pq" via GEMM
            if _check_gpu_int2e() and nao >= _GPU_INT2E_NAO_THRESHOLD and _gpu_int2e._check_mol(mol):
                eri0_gpu = _gpu_int2e.compute_int2e_gpu(mol, b0, b1)  # (nf, nao, nao_pair)
            else:
                eri0 = mol.intor("int2e", aosym="s2kl", shls_slice=shls_slice)
                eri0_gpu = cupy.asarray(eri0.reshape(nf, nao, -1))
                eri0 = None
            nao_pair = eri0_gpu.shape[2]
            eri0_2d = eri0_gpu.transpose(1, 0, 2).reshape(nao, nf * nao_pair)
            dm2_2d = dm2buf_gpu.transpose(1, 0, 2).reshape(nao, nf * nao_pair)
            Imat_gpu += eri0_2d @ dm2_2d.T
            eri0_gpu = eri0_2d = dm2_2d = None

            # de[k] + HF vhf: compute int2e_ip1 and keep on GPU for both
            if _check_gpu_int2e_ip1() and nao >= _GPU_INT2E_IP1_NAO_THRESHOLD and _gpu_int2e_ip1._check_mol(mol):
                eri1_gpu = _gpu_int2e_ip1.compute_int2e_ip1_gpu(mol, b0, b1)
                eri1_gpu = eri1_gpu.reshape(3, nf, nao, -1)
            else:
                eri1 = mol.intor(
                    "int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice
                ).reshape(3, nf, nao, -1)
                eri1_gpu = cupy.asarray(eri1)
                eri1 = None

            de_k_gpu -= eri1_gpu.reshape(3, -1) @ dm2buf_gpu.reshape(-1) * 2
            dm2buf_gpu = None

            # HF vhf on GPU: packed contractions without 4D eri1_full allocation
            D = hf_dm1_gpu[ip0:ip1]           # (nf, nao)
            D_col = D[:, tril_col]            # (nf, nao_pair): D[i, l_p=tril_col[p]]
            D_row = D[:, tril_row]            # (nf, nao_pair): D[i, k_p=tril_row[p]]
            for comp in range(3):
                ec = eri1_gpu[comp]           # (nf, nao, nao_pair)

                # C1: vhf[k,l] += sum_{i,j} ec[i,j,p(k,l)] * D[i,j]
                W = ec.reshape(nf * nao, nao_pair).T @ D.reshape(nf * nao)
                R = cupy.zeros((nao, nao), dtype=cupy.float64)
                R[tril_row, tril_col] = W
                R[tril_col, tril_row] = W     # diagonal written twice with same value: OK
                vhf_gpu[comp] += R

                # X2: vhf[k,j] -= 0.5 * sum_{i,l} ec[i,j,p(k,l)] * D[i,l]
                A = cupy.einsum("ijp,ip->jp", ec, D_col)  # (nao, nao_pair)
                B = cupy.einsum("ijp,ip->jp", ec, D_row)  # (nao, nao_pair)
                vhf_gpu[comp] -= 0.5 * (I_row @ A.T + I_col_od @ B.T)

                # C3: vhf[i,j] += sum_{k,l} ec[i,j,p] * v_hf_dm[p]
                vhf_gpu[comp, ip0:ip1] += (
                    ec.reshape(nf * nao, nao_pair) @ v_hf_dm
                ).reshape(nf, nao)

                # X4: vhf[i,l] -= 0.5 * sum_{j,k} ec[i,j,p(k,l)] * P[j,k]
                E = cupy.einsum("ijp,jp->ip", ec, P_krow)  # (nf, nao_pair)
                F = cupy.einsum("ijp,jp->ip", ec, P_lcol)  # (nf, nao_pair)
                vhf_gpu[comp, ip0:ip1] -= 0.5 * (
                    E @ I_col.T + (F * offdiag_p[None, :]) @ I_row.T
                )
            eri1_gpu = None
        de[k] += cupy.asnumpy(de_k_gpu)
        vhf1[k] = cupy.asnumpy(vhf_gpu)
        log.debug("2e-part grad of atom %d %s = %s", ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer_debug1("2e-part grad of atom %d" % ia, *time1)

    Imat = cupy.asnumpy(Imat_gpu)
    Imat_gpu = dm2_tril_gpu = None

    # Transfer RDM1 blocks to CPU for the remaining steps
    doo_np = cupy.asnumpy(doo)
    dov_np = cupy.asnumpy(dov)
    dvo_np = cupy.asnumpy(dvo)
    dvv_np = cupy.asnumpy(dvv)

    Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mycc._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = numpy.zeros((nmo, nmo))
    if with_frozen:
        dco = Imat[OF[:, None], OA] / (mo_energy[OF, None] - mo_energy[OA])
        dfv = Imat[VF[:, None], VA] / (mo_energy[VF, None] - mo_energy[VA])
        dm1mo[OA[:, None], OA] = doo_np + doo_np.T
        dm1mo[OF[:, None], OA] = dco
        dm1mo[OA[:, None], OF] = dco.T
        dm1mo[VA[:, None], VA] = dvv_np + dvv_np.T
        dm1mo[VF[:, None], VA] = dfv
        dm1mo[VA[:, None], VF] = dfv.T
    else:
        dm1mo[:nocc, :nocc] = doo_np + doo_np.T
        dm1mo[nocc:, nocc:] = dvv_np + dvv_np.T

    # Phase 4: CPHF orbital response
    # Try GPU JK (gpu4pyscf); fall back to CPU get_veff if unavailable.
    # _check_gpu_jk() caches the import check; build failures also update
    # the cache so the overhead is paid at most once per process.
    global _gpu_jk, _GPU_JK_AVAILABLE
    if _check_gpu_jk() and nao >= _GPU_JK_NAO_THRESHOLD:
        try:
            _vhfopt = _gpu_jk._VHFOpt(mol).build()
        except Exception:
            _vhfopt = None
            _GPU_JK_AVAILABLE = False  # don't retry in future calls
    else:
        _vhfopt = None

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    if _vhfopt is not None:
        vj, vk = _gpu_jk.get_jk(mol, cupy.asarray(dm1), hermi=1, vhfopt=_vhfopt)
        vhf = cupy.asnumpy(vj - vk * 0.5) * 2
        vj = vk = None
    else:
        vhf = mycc._scf.get_veff(mycc.mol, dm1) * 2
    Xvo = reduce(numpy.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    Xvo += Imat[:nocc, nocc:].T - Imat[nocc:, :nocc]

    dm1mo += _response_dm1_gpu(mycc, Xvo, _vhfopt)
    time1 = log.timer_debug1("response_rdm1 intermediates", *time1)

    Imat[nocc:, :nocc] = Imat[:nocc, nocc:].T
    im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))
    time1 = log.timer_debug1("response_rdm1", *time1)

    # Phase 5: One-electron and nuclear terms (CPU, fast)
    log.debug("h1 and JK1")
    mf_grad = mycc._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    zeta = lib.direct_sum("i+j->ij", mo_energy, mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[:nocc].reshape(-1, 1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta * dm1mo, mo_coeff.T))

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
    vhf_s1occ = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1 + dm1.T), p1))
    time1 = log.timer_debug1("h1 and JK1", *time1)

    dm1p = hf_dm1 + dm1 * 2
    dm1 += hf_dm1
    zeta += rhf_grad.make_rdm1e(mo_energy, mo_coeff, mycc.mo_occ)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        de[k] += numpy.einsum("xij,ij->x", s1[:, p0:p1], im1[p0:p1])
        de[k] += numpy.einsum("xji,ij->x", s1[:, p0:p1], im1[:, p0:p1])
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum("xij,ji->x", h1ao, dm1)
        de[k] -= numpy.einsum("xij,ij->x", s1[:, p0:p1], zeta[p0:p1])
        de[k] -= numpy.einsum("xji,ij->x", s1[:, p0:p1], zeta[:, p0:p1])
        de[k] -= numpy.einsum("xij,ij->x", s1[:, p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] -= numpy.einsum("xij,ij->x", vhf1[k], dm1p)

    log.timer("%s GPU gradients" % mycc.__class__.__name__, *time0)
    return de


def _response_dm1_cpu(mycc, Xvo, eris=None):
    """CPU CPHF solver — same as pyscf.grad.ccsd._response_dm1."""
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    with_frozen = has_frozen_orbitals(mycc)
    if eris is None or with_frozen:
        mo_energy = mycc._scf.mo_energy
        mo_occ = mycc.mo_occ
        mo_coeff = mycc.mo_coeff

        def fvind(x):
            x = x.reshape(Xvo.shape)
            dm = reduce(numpy.dot, (mo_coeff[:, nocc:], x, mo_coeff[:, :nocc].T))
            v = mycc._scf.get_veff(mycc.mol, dm + dm.T)
            v = reduce(numpy.dot, (mo_coeff[:, nocc:].T, v, mo_coeff[:, :nocc]))
            return v * 2
    else:
        mo_energy = eris.mo_energy
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        ovvo = numpy.empty((nocc, nvir, nvir, nocc))
        for i in range(nocc):
            ovvo[i] = eris.ovvo[i]
            ovvo[i] = ovvo[i] * 4 - ovvo[i].transpose(1, 0, 2)
            ovvo[i] -= eris.oovv[i].transpose(2, 1, 0)

        def fvind(x):
            return numpy.einsum("iabj,bj->ai", ovvo, x.reshape(Xvo.shape))

    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo, nmo))
    dm1[nocc:, :nocc] = dvo
    dm1[:nocc, nocc:] = dvo.T
    return dm1


def _response_dm1_gpu(mycc, Xvo, vhfopt=None):
    """GPU-accelerated CPHF solver.

    If vhfopt is provided (gpu4pyscf _VHFOpt), each fvind call builds J/K on
    GPU, reusing the pre-built vhfopt to avoid per-iteration overhead.
    Falls back to the CPU fvind (same as _response_dm1_cpu) if vhfopt is None.
    """
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    mol = mycc.mol
    mo_energy = mycc._scf.mo_energy
    mo_occ = mycc.mo_occ

    if vhfopt is not None:
        from gpu4pyscf.scf import jk as _gpu_jk

        mo_vir_gpu = cupy.asarray(mycc.mo_coeff[:, nocc:])   # (nao, nvir)
        mo_occ_gpu = cupy.asarray(mycc.mo_coeff[:, :nocc])   # (nao, nocc)

        def fvind(x):
            x_gpu = cupy.asarray(x.reshape(nvir, nocc))
            dm_gpu = mo_vir_gpu @ x_gpu @ mo_occ_gpu.T       # (nao, nao)
            dm_sym = dm_gpu + dm_gpu.T
            vj, vk = _gpu_jk.get_jk(mol, dm_sym, hermi=1, vhfopt=vhfopt)
            v_gpu = (vj - vk * 0.5) @ mo_occ_gpu             # (nao, nocc)
            v_gpu = mo_vir_gpu.T @ v_gpu                      # (nvir, nocc)
            return cupy.asnumpy(v_gpu * 2).ravel()
    else:
        mo_coeff = mycc.mo_coeff

        def fvind(x):
            x = x.reshape(Xvo.shape)
            dm = reduce(numpy.dot, (mo_coeff[:, nocc:], x, mo_coeff[:, :nocc].T))
            v = mycc._scf.get_veff(mycc.mol, dm + dm.T)
            v = reduce(numpy.dot, (mo_coeff[:, nocc:].T, v, mo_coeff[:, :nocc]))
            return v * 2

    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo, nmo))
    dm1[nocc:, :nocc] = dvo
    dm1[:nocc, nocc:] = dvo.T
    return dm1


def _load_block_tril_gpu(dm2_tril, row0, row1, nao):
    """Load a block from the lower-triangular dm2 on GPU.

    GPU equivalent of _load_block_tril: given dm2_tril of shape
    (nao_pair, nao_pair) as a CuPy array, extract the block for
    AO indices [row0, row1) and return shape (nf, nao, nao_pair).
    """
    nao_pair = nao * (nao + 1) // 2
    i_idx = cupy.arange(row0, row1, dtype=cupy.int64)[:, None]  # (nf, 1)
    j_idx = cupy.arange(nao, dtype=cupy.int64)[None, :]         # (1, nao)
    imax = cupy.maximum(i_idx, j_idx)
    imin = cupy.minimum(i_idx, j_idx)
    pair_idx = imax * (imax + 1) // 2 + imin  # (nf, nao)
    return dm2_tril[pair_idx.ravel()].reshape(row1 - row0, nao, nao_pair)


def _load_block_tril(h5dat, row0, row1, nao, out=None):
    """Load a block from the lower-triangular dm2 stored in HDF5.

    Same as pyscf.grad.ccsd._load_block_tril.
    """
    nao_pair = nao * (nao + 1) // 2
    if out is None:
        out = numpy.ndarray((row1 - row0, nao, nao_pair))
    dat = h5dat[row0 * (row0 + 1) // 2 : row1 * (row1 + 1) // 2]
    p1 = 0
    for i in range(row0, row1):
        p0, p1 = p1, p1 + i + 1
        out[i - row0, : i + 1] = dat[p0:p1]
        for j in range(row0, i):
            out[j - row0, i] = out[i - row0, j]
    for i in range(row1, nao):
        i2 = i * (i + 1) // 2
        out[:, i] = h5dat[i2 + row0 : i2 + row1]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ml_ccsd_forces_gpu(mf, t1, t2, l1, l2):
    """Drop-in GPU replacement for eval_forces.compute_ml_ccsd_forces.

    Args:
        mf: Converged PySCF RHF object.
        t1, t2: CCSD T-amplitudes (numpy arrays).
        l1, l2: CCSD lambda amplitudes (numpy arrays).

    Returns:
        forces: numpy array of shape (natm, 3) in Ha/Bohr.
    """
    from pyscf import cc as pyscf_cc

    mycc = pyscf_cc.CCSD(mf)
    mycc.verbose = 0
    mycc.t1 = t1
    mycc.t2 = t2
    mycc.l1 = l1
    mycc.l2 = l2
    mycc.converged = True
    mycc.converged_lambda = True

    de = grad_elec_gpu(mycc, t1, t2, l1, l2, verbose=0)

    # Add nuclear repulsion gradient
    mf_grad = mf.nuc_grad_method()
    de += mf_grad.grad_nuc()

    return numpy.array(de)
