"""Hybrid GPU CCSD lambda solver.

Ports the dominant-cost parts of ``pyscf.cc.ccsd_lambda`` to CuPy while
keeping a small CPU fallback surface for pieces that are still missing from
``gpu4pyscf``:

- The ``vvvv`` contribution prefers the GPU4PySCF direct AO path and falls
  back to the CPU ``_add_vvvv`` helper only when needed.
- DIIS packs amplitudes on CPU once per outer iteration.

This keeps the expensive intermediate builds and blocked lambda contractions
on GPU and provides a drop-in local path for analytic gradient work.
"""

from functools import reduce
import time

import numpy

from pyscf import ao2mo, lib
from pyscf.cc import ccsd
from pyscf.lib import logger

try:
    import cupy
except Exception:  # pragma: no cover - exercised through CPU fallback
    cupy = None


_GPU_CCSD_INCORE = None


def _has_usable_gpu():
    if cupy is None:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _to_cpu(a):
    get = getattr(a, "get", None)
    if callable(get):
        return get()
    return numpy.asarray(a)


def _to_gpu(a):
    if isinstance(a, numpy.ndarray):
        return cupy.asarray(a)
    return cupy.asarray(a)


def _as_cupy_c_order(a):
    return cupy.asarray(numpy.asarray(a, order="C"))


def _get_gpu_ccsd_incore():
    global _GPU_CCSD_INCORE
    if _GPU_CCSD_INCORE is not None:
        return _GPU_CCSD_INCORE
    try:
        from gpu4pyscf.cc import ccsd_incore as gpu_ccsd_incore

        _GPU_CCSD_INCORE = gpu_ccsd_incore
    except Exception:
        _GPU_CCSD_INCORE = False
    return _GPU_CCSD_INCORE


def _get_gpu_direct_vvvv():
    gpu_ccsd_incore = _get_gpu_ccsd_incore()
    if gpu_ccsd_incore is False:
        return False
    return gpu_ccsd_incore._direct_ovvv_vvvv


def _gpu_block_size(mycc, nocc, nvir, tensor_factor):
    """Conservative virtual/orbital block size for GPU temporary tensors."""
    free_gpu_bytes = int(cupy.cuda.runtime.memGetInfo()[0] * 0.45)
    cpu_budget_bytes = int(max(0, mycc.max_memory - lib.current_memory()[0]) * 0.95e6)
    budget = free_gpu_bytes if cpu_budget_bytes <= 0 else min(free_gpu_bytes, cpu_budget_bytes)
    unit = max(1, 8 * tensor_factor * max(1, nocc) * max(1, nvir))
    blksize = max(1, budget // unit)
    return min(nvir, max(ccsd.BLKMIN, int(blksize)))


class _IMDS:
    pass


def _build_dense_vvvv_prepacked_rhs(eris, nvir, dtype):
    vvvv = numpy.asarray(eris.vvvv)
    if vvvv.ndim == 2:
        vvvv = ao2mo.restore(1, vvvv, nvir)
    elif vvvv.ndim != 4:
        raise RuntimeError(f"Unsupported explicit vvvv rank: {vvvv.ndim}")
    rhs = vvvv.transpose(1, 2, 0, 3).reshape(nvir * nvir, nvir * nvir)
    return cupy.asarray(numpy.asarray(rhs, order="C"), dtype=dtype)


def _add_vvvv_gpu_dense_prepacked(mycc, l2, eris, imds=None):
    nocc, nvir = l2.shape[0], l2.shape[2]
    itemsize = numpy.dtype(l2.dtype).itemsize
    rhs_bytes = nvir**4 * itemsize
    free_gpu_bytes = int(cupy.cuda.runtime.memGetInfo()[0])
    cpu_budget_bytes = int(max(0, mycc.max_memory - lib.current_memory()[0]) * 0.95e6)
    if rhs_bytes > free_gpu_bytes * 0.30:
        raise RuntimeError("explicit dense-prepacked vvvv exceeds safe GPU memory budget")
    if cpu_budget_bytes > 0 and rhs_bytes > cpu_budget_bytes * 0.50:
        raise RuntimeError("explicit dense-prepacked vvvv exceeds safe CPU memory budget")

    rhs = None if imds is None else getattr(imds, "dense_vvvv_rhs", None)
    rhs_shape = (nvir * nvir, nvir * nvir)
    if rhs is None or rhs.shape != rhs_shape or rhs.dtype != l2.dtype:
        rhs = _build_dense_vvvv_prepacked_rhs(eris, nvir, l2.dtype)
        if imds is not None:
            imds.dense_vvvv_rhs = rhs

    lhs = l2.reshape(nocc * nocc, nvir * nvir)
    return (lhs @ rhs).reshape(nocc, nocc, nvir, nvir)


def _build_lambda_vvvv_context(mycc, l2, gpu_ccsd_incore):
    nocc = l2.shape[0]
    nao_cart = mycc.mol.nao_nr(cart=True)
    max_memory = max(gpu_ccsd_incore.MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = ((max_memory * 0.9e6 - l2.size * 4 * 8) / 8 / nao_cart**2 / 3.5) ** 0.5
    nocc2 = nocc * (nocc + 1) // 2
    ht2_mem = nocc2 * nao_cart**2 * 8 * 2
    mem_avail = int(cupy.cuda.runtime.memGetInfo()[0] * 0.75)
    if mem_avail * 0.9 < ht2_mem:
        raise RuntimeError(
            f"Not enough GPU memory. Available {mem_avail*1e-6} MB, required {ht2_mem/.9e-6} MB"
        )
    cupy.get_default_memory_pool().set_limit(mem_avail)
    blksize = max(
        gpu_ccsd_incore.BLKMIN,
        int(
            min(
                (nao_cart + 3) / 4,
                blksize,
                ((mem_avail - ht2_mem) * 0.5 / 8 / nao_cart**2) ** 0.5,
            )
        ),
    )
    vhfopt = gpu_ccsd_incore.int4c2e._VHFOpt(mycc.mol, "int2e")
    vhfopt.build(group_size=blksize, diag_block_with_triu=True)
    mo = vhfopt.coeff.dot(cupy.asarray(mycc.mo_coeff))
    cp_idx, cp_jdx = numpy.tril_indices(len(vhfopt.uniq_l_ctr))
    return {
        "nocc": nocc,
        "nvir": l2.shape[2],
        "vhfopt": vhfopt,
        "mo": mo,
        "orbv": cupy.asarray(mo[:, nocc:]),
        "ao_loc": vhfopt.mol.ao_loc,
        "cp_idx": cp_idx,
        "cp_jdx": cp_jdx,
    }


def _add_vvvv_gpu_lambda_only(mycc, l2, imds=None):
    gpu_ccsd_incore = _get_gpu_ccsd_incore()
    if gpu_ccsd_incore is False:
        raise RuntimeError("gpu4pyscf direct vvvv helper is unavailable")

    nocc = l2.shape[0]
    nvir = l2.shape[2]
    ctx = None if imds is None else getattr(imds, "lambda_vvvv_ctx", None)
    if ctx is None or ctx["nocc"] != nocc or ctx["nvir"] != nvir:
        ctx = _build_lambda_vvvv_context(mycc, l2, gpu_ccsd_incore)
        if imds is not None:
            imds.lambda_vvvv_ctx = ctx

    vhfopt = ctx["vhfopt"]
    mo = ctx["mo"]
    orbv = ctx["orbv"]
    ao_loc = ctx["ao_loc"]
    cp_idx = ctx["cp_idx"]
    cp_jdx = ctx["cp_jdx"]
    nao = mo.shape[0]
    nocc2 = nocc * (nocc + 1) // 2

    tau = cupy.asarray(l2)[cupy.tril_indices(nocc)]
    x2 = cupy.einsum("xab,pa->xpb", tau, orbv)
    x2 = cupy.einsum("xpb,qb->xpq", x2, orbv)
    x2 = cupy.asarray(x2, order="C")
    ht2ao = cupy.zeros_like(x2)

    handle = cupy.cuda.device.get_cublas_handle()
    dgemm = cupy.cuda.cublas.dgemm
    op_n = cupy.cuda.cublas.CUBLAS_OP_N
    op_t = cupy.cuda.cublas.CUBLAS_OP_T
    one = numpy.ones(1)
    one_ptr = one.ctypes.data
    x2_ptr = int(x2.data.ptr)
    ht2ao_ptr = int(ht2ao.data.ptr)
    nao2 = nao * nao

    def contract_vvvv_block(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        eri = eri.reshape(-1, jc * nao)
        dgemm(
            handle,
            op_n,
            op_n,
            jc * nao,
            nocc2,
            ic * nao,
            one_ptr,
            eri.data.ptr,
            jc * nao,
            x2_ptr + i0 * nao * 8,
            nao2,
            one_ptr,
            ht2ao_ptr + j0 * nao * 8,
            nao2,
        )
        if i0 > j0:
            dgemm(
                handle,
                op_t,
                op_n,
                ic * nao,
                nocc2,
                jc * nao,
                one_ptr,
                eri.data.ptr,
                jc * nao,
                x2_ptr + j0 * nao * 8,
                nao2,
                one_ptr,
                ht2ao_ptr + i0 * nao * 8,
                nao2,
            )

    if vhfopt.uniq_l_ctr[:, 0].max() > gpu_ccsd_incore.int4c2e.LMAX_ON_GPU:
        raise RuntimeError("lambda-only gpu direct vvvv does not support host integral fallback")

    idx, idy = cupy.tril_indices(nao)
    l_ctr_offsets = vhfopt.l_ctr_offsets
    for cp_ij_id, log_q_ij in enumerate(vhfopt.log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi, 0]
        lj = vhfopt.uniq_l_ctr[cpj, 0]
        if li > gpu_ccsd_incore.int4c2e.LMAX_ON_GPU or lj > gpu_ccsd_incore.int4c2e.LMAX_ON_GPU:
            continue
        if log_q_ij.size == 0:
            continue
        ish0 = l_ctr_offsets[cpi]
        jsh0 = l_ctr_offsets[cpj]
        ish1 = l_ctr_offsets[cpi + 1]
        jsh1 = l_ctr_offsets[cpj + 1]
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        eri = cupy.zeros((i1 - i0, nao, j1 - j0, nao))
        strides = [1, (j1 - j0) * nao, (j1 - j0) * nao**2, nao]
        gpu_ccsd_incore._fill_eri_block(eri, strides, [0, 0, i0, j0], vhfopt, cp_ij_id)
        eri[:, idx, :, idy] = eri[:, idy, :, idx]
        contract_vvvv_block(eri, i0, i1, j0, j1)

    ht2tril = cupy.einsum("xpq,pa->xaq", ht2ao, orbv)
    ht2tril = cupy.einsum("xaq,qb->xab", ht2tril, orbv)
    if gpu_ccsd_incore.FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()
    return gpu_ccsd_incore._unpack_t2_tril(ht2tril, nocc, nvir)


def _add_vvvv_gpu(mycc, l2, eris, log, imds=None):
    """Return the lambda ``vvvv`` contraction on GPU.

    This reuses the GPU4PySCF direct AO ``vvvv`` path. Setting ``t1 = 0``
    removes the extra ``ovvv`` correction terms, leaving exactly the
    ``Ht2 = einsum('ijcd,acdb->ijab', l2, vvvv)`` piece required by the
    lambda equations.
    """
    if getattr(eris, "vvvv", None) is not None:
        try:
            return cupy.asarray(_add_vvvv_gpu_dense_prepacked(mycc, l2, eris, imds=imds))
        except Exception as exc:
            log.debug1("dense-prepacked gpu vvvv failed; trying AO-direct helper: %s", exc)

    if _get_gpu_direct_vvvv() is False:
        raise RuntimeError("gpu4pyscf direct vvvv helper is unavailable")

    try:
        l2new = _add_vvvv_gpu_lambda_only(mycc, l2, imds=imds)
    except Exception as exc:
        log.debug1("lambda-only gpu direct vvvv failed; trying full helper: %s", exc)
        direct_vvvv = _get_gpu_direct_vvvv()
        nocc, nvir = l2.shape[0], l2.shape[2]
        zero_t1 = cupy.zeros((nocc, nvir), dtype=l2.dtype)
        try:
            _, _, l2new, _, _ = direct_vvvv(mycc, zero_t1, l2)
        except Exception as exc2:
            log.debug1("gpu direct vvvv failed; falling back to CPU helper: %s", exc2)
            raise
    return cupy.asarray(l2new)


def _build_vvov_slice(ovvv_blocks_cpu, p0, p1, nvir, nocc, dtype):
    """Assemble one ``vvov`` slice on GPU without storing all blocks on device."""
    eris_vvov = cupy.empty((p1 - p0, nvir, nocc, nvir), dtype=dtype)
    q0 = 0
    for block in ovvv_blocks_cpu:
        q1 = q0 + block.shape[3]
        eris_vvov[:, :, :, q0:q1] = cupy.asarray(block[p0:p1], dtype=dtype)
        q0 = q1
    return eris_vvov


def _make_tau_gpu(t2, t1a, t1b, fac=1.0):
    tau = cupy.einsum("ia,jb->ijab", t1a * fac, t1b)
    tau += t2
    return tau


def _contract_m4_ovvv_blocked(l2, t1, eris_ovvv, occ_blksize):
    """Blocked form of einsum('jidc,kc,kadb->ijab', l2, t1, eris_ovvv)."""
    nocc = t1.shape[0]
    m4 = cupy.zeros(
        (l2.shape[0], l2.shape[1], eris_ovvv.shape[1], eris_ovvv.shape[3]),
        dtype=l2.dtype,
    )
    for k0, k1 in lib.prange(0, nocc, occ_blksize):
        m4 += cupy.einsum("jidc,kc,kadb->ijab", l2, t1[k0:k1], eris_ovvv[k0:k1])
    return m4


def _contract_m4_voov_blocked(l2, tau, eris_voov, occ_blksize):
    """Blocked form of einsum('ijcd,klcd,aklb->ijab', l2, tau, eris_voov)."""
    nocc = tau.shape[0]
    m4 = cupy.zeros((l2.shape[0], l2.shape[1], eris_voov.shape[0], eris_voov.shape[3]), dtype=l2.dtype)
    for k0, k1 in lib.prange(0, nocc, occ_blksize):
        for l0, l1 in lib.prange(0, nocc, occ_blksize):
            m4 += cupy.einsum(
                "ijcd,klcd,aklb->ijab",
                l2,
                tau[k0:k1, l0:l1],
                eris_voov[:, k0:k1, l0:l1],
            )
    return m4


def _should_use_occ_blocking(nocc, nvir):
    """Use occupied blocking only once the problem is large enough to justify it."""
    return nocc >= 10 or (nocc * nvir) >= 1000


def _prepare_virtual_block_cache(eris, nvir, blksize):
    """Cache one CPU C-order copy per virtual block/layout.

    This avoids repeated host-side slicing/transposition of the same ERI blocks
    across both ``make_intermediates_gpu`` and ``update_lambda_gpu``.
    """
    cache = []
    for p0, p1 in lib.prange(0, nvir, blksize):
        ovvv = numpy.asarray(eris.get_ovvv(slice(None), slice(p0, p1)), order="C")
        ovoo = numpy.asarray(eris.ovoo[:, p0:p1], order="C")
        ovvo = numpy.asarray(eris.ovvo[:, p0:p1], order="C")
        oovv = numpy.asarray(eris.oovv[:, :, p0:p1], order="C")
        cache.append(
            {
                "p0": p0,
                "p1": p1,
                "ovvv": ovvv,
                "vvov": ovvv.transpose(2, 3, 0, 1).copy(),
                "ovoo": ovoo,
                "ovvo_t": ovvo.transpose(1, 0, 3, 2).copy(),
                "oovv_t": oovv.transpose(2, 1, 0, 3).copy(),
            }
        )
    return cache


def make_intermediates_gpu(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    t1 = _to_gpu(t1)
    t2 = _to_gpu(t2)
    nocc, nvir = t1.shape
    fock = _to_gpu(eris.fock)
    foo = fock[:nocc, :nocc]
    fov = fock[:nocc, nocc:]
    fvv = fock[nocc:, nocc:]

    imds = _IMDS()
    imds.woooo = cupy.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)
    imds.wvooo = cupy.zeros((nvir, nocc, nocc, nocc), dtype=t1.dtype)
    imds.wVOov = cupy.zeros((nvir, nocc, nocc, nvir), dtype=t1.dtype)
    imds.wvOOv = cupy.zeros((nvir, nocc, nocc, nvir), dtype=t1.dtype)
    imds.wvvov = cupy.zeros((nvir, nvir, nocc, nvir), dtype=t1.dtype)
    imds.oovv = _as_cupy_c_order(eris.oovv)
    imds.ovoo = _as_cupy_c_order(eris.ovoo)

    w1 = fvv - cupy.einsum("ja,jb->ba", fov, t1)
    w2 = foo + cupy.einsum("ib,jb->ij", fov, t1)
    w3 = cupy.einsum("kc,jkbc->bj", fov, t2) * 2 + fov.T
    w3 -= cupy.einsum("kc,kjbc->bj", fov, t2)
    w3 += cupy.einsum("kc,kb,jc->bj", fov, t1, t1)
    w4 = fov.copy()

    blksize = _gpu_block_size(mycc, nocc, nvir, tensor_factor=24 * max(1, nocc))
    log.debug1(
        "gpu lambda make_intermediates: block size = %d, nvir = %d in %d blocks",
        blksize,
        nvir,
        int((nvir + blksize - 1) // blksize),
    )

    block_cache = _prepare_virtual_block_cache(eris, nvir, blksize)
    imds.block_cache = block_cache
    imds.vvov_blocks_cpu = [entry["vvov"] for entry in block_cache]

    woooo = cupy.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)
    wvooo = cupy.zeros((nvir, nocc, nocc, nocc), dtype=t1.dtype)
    tau_full = t2 + cupy.einsum("ia,jb->ijab", t1, t1)

    for block in block_cache:
        p0 = block["p0"]
        p1 = block["p1"]
        eris_ovvv = cupy.asarray(block["ovvv"])
        eris_vvov = _build_vvov_slice(imds.vvov_blocks_cpu, p0, p1, nvir, nocc, t1.dtype)

        w1 += cupy.einsum("jcba,jc->ba", eris_ovvv, t1[:, p0:p1] * 2)
        w1[:, p0:p1] -= cupy.einsum("jabc,jc->ba", eris_ovvv, t1)
        theta_p = t2[:, :, :, p0:p1] * 2 - t2[:, :, :, p0:p1].transpose(1, 0, 2, 3)
        w3 += cupy.einsum("jkcd,kdcb->bj", theta_p, eris_ovvv)
        wVOov = cupy.einsum("jbcd,kd->bjkc", eris_ovvv, t1)
        wvOOv = cupy.einsum("cbjd,kd->cjkb", eris_vvov, -t1)
        g2vovv = eris_vvov.transpose(0, 2, 1, 3) * 2 - eris_vvov.transpose(0, 2, 3, 1)
        for i0, i1 in lib.prange(0, nocc, blksize):
            tau = tau_full[:, i0:i1]
            wvooo[p0:p1, i0:i1] += cupy.einsum("cibd,jkbd->ckij", g2vovv, tau)

        wvvov = cupy.einsum("jabd,jkcd->abkc", eris_ovvv, t2) * -1.5
        wvvov += eris_vvov.transpose(0, 3, 2, 1) * 2
        wvvov -= eris_vvov

        g2vvov = eris_vvov * 2 - eris_ovvv.transpose(1, 2, 0, 3)
        for i0, i1 in lib.prange(0, nocc, blksize):
            theta = t2[i0:i1] * 2 - t2[i0:i1].transpose(0, 1, 3, 2)
            vackb = cupy.einsum("acjd,kjbd->ackb", g2vvov, theta)
            wvvov[:, :, i0:i1] += vackb.transpose(0, 3, 2, 1)
            wvvov[:, :, i0:i1] -= vackb * 0.5

        eris_ovoo = cupy.asarray(block["ovoo"])
        w2 += cupy.einsum("kbij,kb->ij", eris_ovoo, t1[:, p0:p1]) * 2
        w2 -= cupy.einsum("ibkj,kb->ij", eris_ovoo, t1[:, p0:p1])
        theta_o = t2[:, :, p0:p1].transpose(1, 0, 2, 3) * 2 - t2[:, :, p0:p1]
        w3 -= cupy.einsum("lckj,klcb->bj", eris_ovoo, theta_o)

        tmp = cupy.einsum("lc,jcik->ijkl", t1[:, p0:p1], eris_ovoo)
        woooo += tmp
        woooo += tmp.transpose(1, 0, 3, 2)

        wvOOv += cupy.einsum("lbjk,lc->bjkc", eris_ovoo, t1)
        wVOov -= cupy.einsum("jbkl,lc->bjkc", eris_ovoo, t1)
        wvooo[p0:p1] += eris_ovoo.transpose(1, 3, 2, 0) * 2
        wvooo[p0:p1] -= eris_ovoo.transpose(1, 0, 2, 3)
        wvooo -= cupy.einsum("klbc,iblj->ckij", t2[:, :, p0:p1], eris_ovoo * 1.5)

        g2ovoo = eris_ovoo * 2 - eris_ovoo.transpose(2, 1, 0, 3)
        vcjik = cupy.einsum("jlcb,lbki->cjki", theta_p, g2ovoo)
        wvooo += vcjik.transpose(0, 3, 2, 1)
        wvooo -= vcjik * 0.5

        eris_voov = cupy.asarray(block["ovvo_t"])
        tau = t2[:, :, p0:p1] + cupy.einsum("ia,jb->ijab", t1[:, p0:p1], t1)
        woooo += cupy.einsum("cijd,klcd->ijkl", eris_voov, tau)

        g2voov = eris_voov * 2 - eris_voov.transpose(0, 2, 1, 3)
        tmpw4 = cupy.einsum("ckld,ld->kc", g2voov, t1)
        w1 -= cupy.einsum("ckja,kjcb->ba", g2voov, t2[:, :, p0:p1])
        w1[:, p0:p1] -= cupy.einsum("ja,jb->ba", tmpw4, t1)
        w2 += cupy.einsum("jkbc,bikc->ij", t2[:, :, p0:p1], g2voov)
        w2 += cupy.einsum("ib,jb->ij", tmpw4, t1[:, p0:p1])
        w3 += reduce(cupy.dot, (t1.T, tmpw4, t1[:, p0:p1].T))
        w4[:, p0:p1] += tmpw4

        wvOOv += cupy.einsum("bljd,kd,lc->bjkc", eris_voov, t1, t1)
        wVOov -= cupy.einsum("bjld,kd,lc->bjkc", eris_voov, t1, t1)

        VOov = cupy.einsum("bjld,klcd->bjkc", g2voov, t2)
        VOov -= cupy.einsum("bjld,kldc->bjkc", eris_voov, t2)
        VOov += eris_voov
        vOOv = cupy.einsum("bljd,kldc->bjkc", eris_voov, t2)
        vOOv -= cupy.asarray(block["oovv_t"])
        wVOov += VOov
        wvOOv += vOOv
        imds.wVOov[p0:p1] = wVOov
        imds.wvOOv[p0:p1] = wvOOv

        ov1 = vOOv * 2 + VOov
        ov2 = VOov * 2 + vOOv
        wvooo -= cupy.einsum("jb,bikc->ckij", t1[:, p0:p1], ov1)
        wvooo += cupy.einsum("kb,bijc->ckij", t1[:, p0:p1], ov2)
        w3 += cupy.einsum("ckjb,kc->bj", ov2, t1[:, p0:p1])

        wvvov += cupy.einsum("ajkc,jb->abkc", ov1, t1)
        wvvov -= cupy.einsum("ajkb,jc->abkc", ov2, t1)

        g2ovoo = eris_ovoo * 2 - eris_ovoo.transpose(2, 1, 0, 3)
        wvvov += cupy.einsum("laki,klbc->abic", g2ovoo, tau_full)
        imds.wvvov[p0:p1] = wvvov

    woooo += _as_cupy_c_order(eris.oooo).transpose(0, 2, 1, 3)
    imds.woooo = woooo
    imds.wvooo = wvooo

    w3 += cupy.einsum("bc,jc->bj", w1, t1)
    w3 -= cupy.einsum("kj,kb->bj", w2, t1)

    imds.w1 = w1
    imds.w2 = w2
    imds.w3 = w3
    imds.w4 = w4
    return imds


def update_lambda_gpu(mycc, t1, t2, l1, l2, eris=None, imds=None, use_cpu_vvvv=False):
    if imds is None:
        imds = make_intermediates_gpu(mycc, t1, t2, eris)
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    wall0 = time.perf_counter()

    t1 = _to_gpu(t1)
    t2 = _to_gpu(t2)
    l1 = _to_gpu(l1)
    l2 = _to_gpu(l2)
    nocc, nvir = t1.shape
    fov = _to_gpu(eris.fock[:nocc, nocc:])
    mo_e_o = _to_gpu(eris.mo_energy[:nocc])
    mo_e_v = _to_gpu(eris.mo_energy[nocc:] + mycc.level_shift)

    theta = t2 * 2 - t2.transpose(0, 1, 3, 2)
    mba = cupy.einsum("klca,klcb->ba", l2, theta)
    mij = cupy.einsum("ikcd,jkcd->ij", l2, theta)
    mba1 = cupy.einsum("jc,jb->bc", l1, t1) + mba
    mij1 = cupy.einsum("kb,jb->kj", l1, t1) + mij
    mia1 = t1 + cupy.einsum("kc,jkbc->jb", l1, t2) * 2
    mia1 -= cupy.einsum("kc,jkcb->jb", l1, t2)
    mia1 -= reduce(cupy.dot, (t1, l1.T, t1))
    mia1 -= cupy.einsum("bd,jd->jb", mba, t1)
    mia1 -= cupy.einsum("lj,lb->jb", mij, t1)
    t_preamble = time.perf_counter()

    if use_cpu_vvvv:
        l2new = cupy.asarray(
            mycc._add_vvvv(None, _to_cpu(l2), eris, with_ovvv=False, t2sym="jiba")
        )
    else:
        try:
            l2new = _add_vvvv_gpu(mycc, l2, eris, log, imds=imds)
        except Exception:
            l2new = cupy.asarray(
                mycc._add_vvvv(None, _to_cpu(l2), eris, with_ovvv=False, t2sym="jiba")
            )
    t_vvvv = time.perf_counter()
    l1new = cupy.einsum("ijab,jb->ia", l2new, t1) * 2
    l1new -= cupy.einsum("jiab,jb->ia", l2new, t1)
    l2new *= 0.5

    w1 = imds.w1 - cupy.diag(mo_e_v)
    w2 = imds.w2 - cupy.diag(mo_e_o)

    l1new += fov
    l1new += cupy.einsum("ib,ba->ia", l1, w1)
    l1new -= cupy.einsum("ja,ij->ia", l1, w2)
    l1new -= cupy.einsum("ik,ka->ia", mij, imds.w4)
    l1new -= cupy.einsum("ca,ic->ia", mba, imds.w4)
    l1new += cupy.einsum("ijab,bj->ia", l2, imds.w3) * 2
    l1new -= cupy.einsum("ijba,bj->ia", l2, imds.w3)

    l2new += cupy.einsum("ia,jb->ijab", l1, imds.w4)
    l2new += cupy.einsum("jibc,ca->jiba", l2, w1)
    l2new -= cupy.einsum("jk,kiba->jiba", w2, l2)

    eris_ovoo = imds.ovoo
    l1new -= cupy.einsum("iajk,kj->ia", eris_ovoo, mij1) * 2
    l1new += cupy.einsum("jaik,kj->ia", eris_ovoo, mij1)
    l2new -= cupy.einsum("jbki,ka->jiba", eris_ovoo, l1)

    tau = _make_tau_gpu(t2, t1, t1)
    l2_perm = l2.transpose(0, 2, 1, 3) - l2.transpose(0, 3, 1, 2) * 0.5
    t_setup = time.perf_counter()

    blksize = _gpu_block_size(mycc, nocc, nvir, tensor_factor=10 * max(1, nocc))
    use_occ_blocking = _should_use_occ_blocking(nocc, nvir)
    occ_blksize = min(
        nocc, _gpu_block_size(mycc, nocc, nocc, tensor_factor=6 * max(1, nvir))
    )
    log.debug1(
        "gpu lambda update: vir block size = %d, occ block size = %d, use_occ_blocking = %s, nvir = %d in %d blocks",
        blksize,
        occ_blksize,
        use_occ_blocking,
        nvir,
        int((nvir + blksize - 1) // blksize),
    )

    l1new -= cupy.einsum("jb,jiab->ia", l1, imds.oovv)
    ovvv_m4_time = 0.0
    voov_m4_time = 0.0
    other_block_time = 0.0
    for block in imds.block_cache:
        block0 = time.perf_counter()
        p0 = block["p0"]
        p1 = block["p1"]
        eris_ovvv = cupy.asarray(block["ovvv"])
        l1new[:, p0:p1] += cupy.einsum("iabc,bc->ia", eris_ovvv, mba1) * 2
        l1new -= cupy.einsum("ibca,bc->ia", eris_ovvv, mba1[p0:p1])
        l2new[:, :, p0:p1] += cupy.einsum("jbac,ic->jiba", eris_ovvv, l1)
        m4_t0 = time.perf_counter()
        if use_occ_blocking:
            m4 = _contract_m4_ovvv_blocked(l2, t1, eris_ovvv, occ_blksize)
        else:
            m4 = cupy.einsum("jidc,kc,kadb->ijab", l2, t1, eris_ovvv)
        ovvv_m4_time += time.perf_counter() - m4_t0
        l2new[:, :, p0:p1] -= m4
        l1new[:, p0:p1] -= cupy.einsum("ijab,jb->ia", m4, t1) * 2
        l1new -= cupy.einsum("ijab,ia->jb", m4, t1[:, p0:p1]) * 2
        l1new[:, p0:p1] += cupy.einsum("jiab,jb->ia", m4, t1)
        l1new += cupy.einsum("jiab,ia->jb", m4, t1[:, p0:p1])

        eris_voov = cupy.asarray(block["ovvo_t"])
        l1new[:, p0:p1] += cupy.einsum("jb,aijb->ia", l1, eris_voov) * 2
        l2new[:, :, p0:p1] += eris_voov.transpose(1, 2, 0, 3) * 0.5
        l2new[:, :, p0:p1] -= cupy.einsum("bjic,ca->jiba", eris_voov, mba1)
        l2new[:, :, p0:p1] -= cupy.einsum("bjka,ik->jiba", eris_voov, mij1)
        l1new[:, p0:p1] += cupy.einsum("aijb,jb->ia", eris_voov, mia1) * 2
        l1new -= cupy.einsum("bija,jb->ia", eris_voov, mia1[:, p0:p1])
        m4_t0 = time.perf_counter()
        if use_occ_blocking:
            m4 = _contract_m4_voov_blocked(l2, tau, eris_voov, occ_blksize)
        else:
            m4 = cupy.einsum("ijcd,klcd,aklb->ijab", l2, tau, eris_voov)
        voov_m4_time += time.perf_counter() - m4_t0
        l2new[:, :, p0:p1] += m4 * 0.5
        l1new[:, p0:p1] += cupy.einsum("ijab,jb->ia", m4, t1) * 2
        l1new -= cupy.einsum("ijba,jb->ia", m4, t1[:, p0:p1])

        saved_wvooo = imds.wvooo[p0:p1]
        l1new -= cupy.einsum("ckij,jkca->ia", saved_wvooo, l2[:, :, p0:p1])
        saved_wvovv = imds.wvvov[p0:p1]
        l1new[:, p0:p1] += cupy.einsum("abkc,kibc->ia", saved_wvovv, l2)

        saved_wvOOv = imds.wvOOv[p0:p1]
        tmp_voov = imds.wVOov[p0:p1] * 2 + saved_wvOOv
        l2new[:, :, p0:p1] += cupy.einsum("iakc,bjkc->jiba", l2_perm, tmp_voov)

        tmp = cupy.einsum("jkca,bikc->jiba", l2, saved_wvOOv)
        l2new[:, :, p0:p1] += tmp
        l2new[:, :, p0:p1] += tmp.transpose(1, 0, 2, 3) * 0.5
        other_block_time += time.perf_counter() - block0

    saved_woooo = imds.woooo
    m3 = cupy.einsum("ijkl,klab->ijab", saved_woooo, l2)
    l2new += m3 * 0.5
    l1new += cupy.einsum("ijab,jb->ia", m3, t1) * 2
    l1new -= cupy.einsum("ijba,jb->ia", m3, t1)

    eia = mo_e_o[:, None] - mo_e_v
    l1new /= eia

    for i in range(nocc):
        if i > 0:
            l2new[i, :i] += l2new[:i, i].transpose(0, 2, 1)
            l2new[i, :i] /= eia[i][None, :, None] + eia[:i][:, None, :]
            l2new[:i, i] = l2new[i, :i].transpose(0, 2, 1)
        l2new[i, i] = l2new[i, i] + l2new[i, i].T
        l2new[i, i] /= eia[i][:, None] + eia[i][None, :]

    wall_end = time.perf_counter()
    log.info(
        "gpu lambda timing: preamble=%.3fs vvvv=%.3fs setup=%.3fs ovvv_m4=%.3fs voov_m4=%.3fs other_blocks=%.3fs tail=%.3fs total=%.3fs",
        t_preamble - wall0,
        t_vvvv - t_preamble,
        t_setup - t_vvvv,
        ovvv_m4_time,
        voov_m4_time,
        max(0.0, other_block_time - ovvv_m4_time - voov_m4_time),
        wall_end - (t_setup + other_block_time),
        wall_end - wall0,
    )
    log.timer_debug1("gpu update l1 l2", *time0)
    return l1new, l2new


def kernel_gpu(
    mycc,
    eris=None,
    t1=None,
    t2=None,
    l1=None,
    l2=None,
    max_cycle=None,
    tol=None,
    verbose=logger.INFO,
    use_cpu_vvvv=False,
):
    if eris is None:
        eris = mycc.ao2mo()
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)

    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        l1 = t1
    if l2 is None:
        l2 = t2
    if max_cycle is None:
        max_cycle = mycc.max_cycle
    if tol is None:
        tol = mycc.conv_tol_normt

    t1_gpu = _to_gpu(t1)
    t2_gpu = _to_gpu(t2)
    l1_gpu = _to_gpu(l1)
    l2_gpu = _to_gpu(l2)
    imds = make_intermediates_gpu(mycc, t1_gpu, t2_gpu, eris)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput0 = log.timer(f"{mycc.__class__.__name__} gpu lambda initialization", *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new_gpu, l2new_gpu = update_lambda_gpu(
            mycc, t1_gpu, t2_gpu, l1_gpu, l2_gpu, eris, imds, use_cpu_vvvv=use_cpu_vvvv
        )
        l1new_cpu = _to_cpu(l1new_gpu)
        l2new_cpu = _to_cpu(l2new_gpu)
        l1_cpu = _to_cpu(l1_gpu)
        l2_cpu = _to_cpu(l2_gpu)
        normt = numpy.linalg.norm(
            mycc.amplitudes_to_vector(l1new_cpu, l2new_cpu)
            - mycc.amplitudes_to_vector(l1_cpu, l2_cpu)
        )
        if adiis is not None:
            l1new_cpu, l2new_cpu = mycc.run_diis(l1new_cpu, l2new_cpu, istep, normt, 0, adiis)
        l1_gpu = cupy.asarray(l1new_cpu)
        l2_gpu = cupy.asarray(l2new_cpu)
        log.info("cycle = %d  norm(lambda1,lambda2) = %.6g", istep + 1, normt)
        cput0 = log.timer(f"{mycc.__class__.__name__} gpu lambda iter", *cput0)
        if normt < tol:
            conv = True
            break

    return conv, _to_cpu(l1_gpu), _to_cpu(l2_gpu)


def solve_lambda_gpu(
    mycc,
    t1=None,
    t2=None,
    l1=None,
    l2=None,
    eris=None,
    use_cpu_vvvv=False,
    fallback_to_cpu=True,
):
    """Solve CCSD lambda with GPU-ported dominant contractions.

    The solver updates ``mycc.converged_lambda``, ``mycc.l1`` and ``mycc.l2``
    to match the standard PySCF ``solve_lambda`` side effects.
    """
    if _has_usable_gpu():
        try:
            conv, l1_out, l2_out = kernel_gpu(
                mycc,
                eris=eris,
                t1=t1,
                t2=t2,
                l1=l1,
                l2=l2,
                max_cycle=mycc.max_cycle,
                tol=mycc.conv_tol_normt,
                verbose=mycc.verbose,
                use_cpu_vvvv=use_cpu_vvvv,
            )
            mycc.converged_lambda = conv
            mycc.l1 = l1_out
            mycc.l2 = l2_out
            mycc._lambda_solver_mode = "gpu-hybrid"
            return mycc.l1, mycc.l2
        except Exception:
            if not fallback_to_cpu:
                raise

    if not fallback_to_cpu:
        raise RuntimeError("No usable CuPy GPU runtime found for lambda solve.")
    mycc._lambda_solver_mode = "cpu-fallback"
    mycc.l1, mycc.l2 = mycc.solve_lambda(t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)
    return mycc.l1, mycc.l2
