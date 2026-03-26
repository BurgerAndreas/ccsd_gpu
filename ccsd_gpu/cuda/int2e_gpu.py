"""
GPU-accelerated int2e (4-centre undifferentiated ERI).

compute_int2e_gpu(mol, b0, b1) is a drop-in replacement for

    mol.intor('int2e', aosym='s2kl',
              shls_slice=(b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas))

and returns a CuPy array of shape (nf, nao, nao_pair) where
nf = ao_loc[b1] - ao_loc[b0] and nao_pair = nao*(nao+1)//2.

Supports both mol.cart == True (direct Cartesian kernel) and mol.cart == False
(Cartesian kernel followed by cart→sph post-processing via mol.cart2sph_coeff()).
Angular momenta up to l=4 (g-shells) are supported (nroots <= 9).
"""

import numpy
import cupy

from ccsd_gpu.cuda.int2e_module import get_kernel
# Reuse _check_mol and _get_basis_cached — they are identical for both kernels
# since both share the same _L_MAX=4 and shell conventions.
from ccsd_gpu.cuda.int2e_ip1_gpu import _check_mol, _get_basis_cached, _split_shell_range_for_memory

# Rys quadrature max nroots supported by this kernel
_NROOTS_MAX = 9
# Max angular momentum supported (nroots <= _NROOTS_MAX requires l <= 4)
_L_MAX = 4

# PySCF mol._bas slot indices (must match CUDA kernel defines)
_ANG_OF   = 1
_NPRIM_OF = 2


# ---------------------------------------------------------------------------
# Cartesian kernel path
# ---------------------------------------------------------------------------

def _compute_cart(mol, b0, b1):
    """
    GPU kernel path for Cartesian basis (mol.cart must be True).

    Handles generally contracted shells (nctr > 1) by decontracting them
    into separate nctr=1 shells before launching the kernel.

    Parameters
    ----------
    mol : pyscf.gto.Mole  (mol.cart must be True)
    b0, b1 : int
        Shell range for the first (i) index.

    Returns
    -------
    eri0_gpu : cupy.ndarray, shape (nf_cart, nao_cart, nao_pair_cart), dtype float64
    """
    kern = get_kernel()

    (dc_bas, dc_ao_loc, orig2dc,
     bas_gpu, ao_loc_gpu, kl_pair_idx_gpu,
     n_kl_pairs, max_l, max_prim) = _get_basis_cached(mol)

    dc_nbas  = len(dc_bas)
    nao      = int(dc_ao_loc[dc_nbas])
    nao_pair = nao * (nao + 1) // 2

    # Map original shell range [b0, b1) to decontracted range
    dc_b0 = orig2dc[b0][0]
    dc_b1 = orig2dc[b1 - 1][-1] + 1
    nf    = int(dc_ao_loc[dc_b1]) - int(dc_ao_loc[dc_b0])

    # Upload only geometry-dependent arrays (coordinates live in mol._env)
    atm_gpu = cupy.asarray(mol._atm)
    env_gpu = cupy.asarray(mol._env)

    # Allocate output (zero-initialised; kernel writes to it)
    eri0_gpu = cupy.zeros((nf, nao, nao_pair), dtype=numpy.float64)

    block = (1, 32)  # blockDim.y = GOUT_STRIDE = 32
    grid  = (dc_nbas * n_kl_pairs,)

    for dc_ish in range(dc_b0, dc_b1):
        li    = int(dc_bas[dc_ish, _ANG_OF])
        iprim = int(dc_bas[dc_ish, _NPRIM_OF])
        fi_offset = int(dc_ao_loc[dc_ish]) - int(dc_ao_loc[dc_b0])

        # Shared memory: gx(3*g_size) + rw(2*NROOTS_MAX) + cicj(iprim*max_jprim)
        # For int2e: stride_j = li+1, so max g_size = (li+1)*(max_l+1)^3
        max_g_size = (li + 1) * (max_l + 1) ** 3
        smem_bytes = int((3 * max_g_size + 2 * _NROOTS_MAX + iprim * max_prim) * 8)

        kern(
            grid, block,
            (
                eri0_gpu,
                atm_gpu, bas_gpu, env_gpu, ao_loc_gpu,
                numpy.int32(dc_ish),
                numpy.int32(fi_offset),
                numpy.int32(nf),
                numpy.int32(dc_nbas),
                numpy.int32(nao),
                numpy.int32(nao_pair),
                kl_pair_idx_gpu,
                numpy.int32(n_kl_pairs),
            ),
            shared_mem=smem_bytes,
        )

    return eri0_gpu


# ---------------------------------------------------------------------------
# Cartesian → spherical post-processing
# ---------------------------------------------------------------------------

def _cart_to_sph_eri0(eri0_cart, mol, mol_cart, b0, b1):
    """
    Transform Cartesian int2e to spherical-harmonic basis via sequential
    CuPy contractions with mol.cart2sph_coeff().

    Parameters
    ----------
    eri0_cart : cupy.ndarray, shape (nf_cart, nao_cart, nao_pair_cart)
        Cartesian ERIs from _compute_cart(mol_cart, b0, b1).
    mol : pyscf.gto.Mole  (mol.cart == False, spherical target)
    mol_cart : pyscf.gto.Mole  (mol.cart == True, Cartesian copy)
    b0, b1 : int
        Shell range for the i-index.

    Returns
    -------
    eri0_sph : cupy.ndarray, shape (nf_sph, nao_sph, nao_pair_sph), dtype float64
    """
    # Block-diagonal cart→sph matrix: (nao_cart, nao_sph)
    C = cupy.asarray(mol.cart2sph_coeff())

    ao_loc_cart = mol_cart.ao_loc_nr()
    ao_loc_sph  = mol.ao_loc_nr()
    nao_cart = int(ao_loc_cart[-1])
    nao_sph  = int(ao_loc_sph[-1])

    # Sub-block of C for i-shells [b0, b1): (nf_cart_block, nf_sph_block)
    C_i = C[ao_loc_cart[b0]:ao_loc_cart[b1],
             ao_loc_sph[b0]:ao_loc_sph[b1]]
    nf_sph = int(C_i.shape[1])

    # Step 1 — transform i-index
    # (nf_cart, nao_cart, nao_pair_cart) × (nf_cart, nf_sph)
    # tensordot contracts axis 0 of eri0_cart with axis 0 of C_i:
    #   result axes: (nao_cart, nao_pair_cart, nf_sph) → transpose
    eri0 = cupy.tensordot(eri0_cart, C_i, axes=([0], [0]))  # (nao_cart, nao_pair_cart, nf_sph)
    eri0 = eri0.transpose(2, 0, 1)                          # (nf_sph, nao_cart, nao_pair_cart)

    # Step 2 — transform j-index
    # (nf_sph, nao_cart, nao_pair_cart) × (nao_cart, nao_sph)
    eri0 = cupy.tensordot(eri0, C, axes=([1], [0]))          # (nf_sph, nao_pair_cart, nao_sph)
    eri0 = eri0.transpose(0, 2, 1)                           # (nf_sph, nao_sph, nao_pair_cart)

    # Step 3 — unpack s2kl packed index to full (k_cart, l_cart) square
    # Both triangles must be filled so that the cart→sph contraction sums
    # over the full k_cart, l_cart space.
    k_c, l_c = cupy.tril_indices(nao_cart)
    eri0_kl = cupy.zeros(
        (nf_sph, nao_sph, nao_cart, nao_cart), dtype=numpy.float64
    )
    eri0_kl[:, :, k_c, l_c] = eri0    # lower triangle (k >= l)
    eri0_kl[:, :, l_c, k_c] = eri0    # upper triangle (s2kl symmetry)

    # Step 4 — transform k-index (axis 2)
    # tensordot contracts axis 2 with axis 0 of C:
    #   result: (nf_sph, nao_sph, nao_cart_l, nao_sph_k) → transpose k before l
    eri0_kl = cupy.tensordot(eri0_kl, C, axes=([2], [0]))    # (..., nao_cart_l, nao_sph_k)
    eri0_kl = eri0_kl.transpose(0, 1, 3, 2)                  # (..., nao_sph_k, nao_cart_l)

    # Step 5 — transform l-index (axis 3)
    eri0_kl = cupy.tensordot(eri0_kl, C, axes=([3], [0]))    # (nf_sph, nao_sph, nao_sph_k, nao_sph_l)

    # Step 6 — repack into spherical triangular (k' >= l')
    k_s, l_s = cupy.tril_indices(nao_sph)
    return eri0_kl[:, :, k_s, l_s]                           # (nf_sph, nao_sph, nao_pair_sph)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_int2e_gpu(mol, b0, b1):
    """
    GPU replacement for mol.intor('int2e', aosym='s2kl',
                                   shls_slice=(b0, b1, 0, nbas, 0, nbas, 0, nbas)).

    Supports mol.cart == True (direct Cartesian kernel) and mol.cart == False
    (Cartesian kernel + cart→sph post-processing). Max angular momentum l <= 4.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    b0, b1 : int
        Shell range for the first (i) index.

    Returns
    -------
    eri0_gpu : cupy.ndarray, shape (nf, nao, nao_pair), dtype float64
        eri0_gpu[fi_local, fj_abs, kl_packed]
          fi_local  = ao_loc[ish] - ao_loc[b0] (local to this shell block)
          fj_abs    = absolute AO index for j
          kl_packed = k_abs*(k_abs+1)//2 + l_abs  (k_abs >= l_abs)
        All AO indices are in the native basis (spherical or Cartesian).
    """
    chunks = _split_shell_range_for_memory(mol, b0, b1, comp=1)
    if len(chunks) > 1:
        return cupy.concatenate([compute_int2e_gpu(mol, c0, c1) for c0, c1 in chunks], axis=0)

    if mol.cart:
        return _compute_cart(mol, b0, b1)
    # Spherical: run kernel on Cartesian copy, then apply c2s post-processing.
    mol_cart = mol.copy()
    mol_cart.cart = True
    eri0_cart = _compute_cart(mol_cart, b0, b1)
    return _cart_to_sph_eri0(eri0_cart, mol, mol_cart, b0, b1)
