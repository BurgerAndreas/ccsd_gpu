"""
GPU-accelerated int2e_ip1 (4-centre ERI derivatives).

compute_int2e_ip1_gpu(mol, b0, b1) is a drop-in replacement for

    mol.intor('int2e_ip1', comp=3, aosym='s2kl',
              shls_slice=(b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas))

and returns a CuPy array of shape (3, nf, nao, nao_pair) where
nf = ao_loc[b1] - ao_loc[b0] and nao_pair = nao*(nao+1)//2.

Supports both mol.cart == True (direct Cartesian kernel) and mol.cart == False
(Cartesian kernel followed by cart→sph post-processing via mol.cart2sph_coeff()).
Angular momenta up to l=4 (g-shells) are supported (nroots <= 9).
"""

import numpy
import cupy

from ccsd_gpu.cuda.int2e_ip1_module import get_kernel

# Rys quadrature max nroots supported by this kernel
_NROOTS_MAX = 9
# Max angular momentum supported (nroots <= _NROOTS_MAX requires l <= 4)
_L_MAX = 4

# PySCF mol._bas slot indices (must match CUDA kernel defines)
_ATOM_OF   = 0
_ANG_OF    = 1
_NPRIM_OF  = 2
_NCTR_OF   = 3
_BAS_SLOTS = 8
_PTR_COEFF = 6


def _check_mol(mol):
    """Return True iff the GPU kernel can handle this mol."""
    if mol._bas[:, _ANG_OF].max() > _L_MAX:
        return False
    return True


def _decontract_bas(mol):
    """Expand generally contracted shells (nctr > 1) into nctr=1 shells.

    The CUDA kernel assumes nf = (l+1)*(l+2)/2 per shell (Cartesian),
    which only holds when nctr == 1.  Shells with nctr > 1 (general
    contractions, common in cc-pVXZ bases) are split into separate
    shells, each pointing to the appropriate contraction coefficients
    in mol._env.

    Returns
    -------
    dc_bas : ndarray, shape (dc_nbas, 8)
    dc_ao_loc : ndarray, shape (dc_nbas + 1,)
    orig2dc : list of lists
        orig2dc[ish] = [dc_idx0, dc_idx1, ...] mapping original shell
        index to decontracted indices.
    """
    new_bas_list = []
    orig2dc = []
    for ish in range(mol.nbas):
        nctr = int(mol._bas[ish, _NCTR_OF])
        nprim = int(mol._bas[ish, _NPRIM_OF])
        ptr_coeff = int(mol._bas[ish, _PTR_COEFF])
        dc_indices = []
        for ctr in range(nctr):
            row = mol._bas[ish].copy()
            row[_NCTR_OF] = 1
            row[_PTR_COEFF] = ptr_coeff + ctr * nprim
            dc_indices.append(len(new_bas_list))
            new_bas_list.append(row)
        orig2dc.append(dc_indices)
    dc_bas = numpy.array(new_bas_list, dtype=mol._bas.dtype)
    dc_ao_loc = numpy.zeros(len(dc_bas) + 1, dtype=numpy.int32)
    for i in range(len(dc_bas)):
        l = int(dc_bas[i, _ANG_OF])
        dc_ao_loc[i + 1] = dc_ao_loc[i] + (l + 1) * (l + 2) // 2
    return dc_bas, dc_ao_loc, orig2dc


# Module-level cache: basis topology + GPU arrays.
# Key: (mol._bas.tobytes(), device_id) — safe because dc_bas/ao_loc/kl_pairs
# depend only on basis topology, not on atom coordinates (which live in mol._env).
_basis_cache = {}


def _get_basis_cached(mol):
    """Return cached basis topology and pre-uploaded GPU arrays for mol.

    Caches decontracted basis, AO locations, and kl-pair index arrays on
    the current CUDA device.  Only geometry-independent data is cached;
    mol._atm and mol._env must still be uploaded per call.

    Returns
    -------
    dc_bas : ndarray (dc_nbas, 8)
    dc_ao_loc : ndarray (dc_nbas + 1,)
    orig2dc : list of lists
    bas_gpu : cupy.ndarray
    ao_loc_gpu : cupy.ndarray
    kl_pair_idx_gpu : cupy.ndarray  shape (2*n_kl_pairs,)
    n_kl_pairs : int
    max_l : int
    max_prim : int
    """
    key = (mol._bas.tobytes(), cupy.cuda.Device().id)
    if key in _basis_cache:
        return _basis_cache[key]

    has_genctr = numpy.any(mol._bas[:, _NCTR_OF] > 1)
    if has_genctr:
        dc_bas, dc_ao_loc, orig2dc = _decontract_bas(mol)
    else:
        dc_bas = mol._bas
        dc_ao_loc = mol.ao_loc_nr()
        orig2dc = [[i] for i in range(mol.nbas)]

    dc_nbas = len(dc_bas)

    bas_gpu    = cupy.asarray(dc_bas)
    ao_loc_gpu = cupy.asarray(dc_ao_loc)

    n_kl_pairs = dc_nbas * (dc_nbas + 1) // 2
    kl_pairs_np = numpy.empty(n_kl_pairs * 2, dtype=numpy.int32)
    idx = 0
    for ksh in range(dc_nbas):
        for lsh in range(ksh + 1):
            kl_pairs_np[idx]     = ksh
            kl_pairs_np[idx + 1] = lsh
            idx += 2
    kl_pair_idx_gpu = cupy.asarray(kl_pairs_np)

    max_l    = int(dc_bas[:, _ANG_OF].max())
    max_prim = int(dc_bas[:, _NPRIM_OF].max())

    result = (dc_bas, dc_ao_loc, orig2dc,
              bas_gpu, ao_loc_gpu, kl_pair_idx_gpu,
              n_kl_pairs, max_l, max_prim)
    _basis_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Main entry point
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
    eri1_gpu : cupy.ndarray, shape (3, nf_cart, nao_cart, nao_pair_cart), dtype float64
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
    eri1_gpu = cupy.zeros((3, nf, nao, nao_pair), dtype=numpy.float64)

    block = (1, 32)
    grid  = (dc_nbas * n_kl_pairs,)

    for dc_ish in range(dc_b0, dc_b1):
        li    = int(dc_bas[dc_ish, _ANG_OF])
        iprim = int(dc_bas[dc_ish, _NPRIM_OF])
        fi_offset = int(dc_ao_loc[dc_ish]) - int(dc_ao_loc[dc_b0])

        # Shared memory: gx(3*g_size) + rw(2*NROOTS_MAX) + cicj(iprim*max_jprim)
        max_g_size = (li + 2) * (max_l + 1) ** 3
        smem_bytes = int((3 * max_g_size + 2 * _NROOTS_MAX + iprim * max_prim) * 8)

        kern(
            grid, block,
            (
                eri1_gpu,
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

    return eri1_gpu


# ---------------------------------------------------------------------------
# Cartesian → spherical post-processing
# ---------------------------------------------------------------------------

def _cart_to_sph_eri1(eri1_cart, mol, mol_cart, b0, b1):
    """
    Transform Cartesian int2e_ip1 to spherical-harmonic basis via sequential
    CuPy contractions with mol.cart2sph_coeff().

    Parameters
    ----------
    eri1_cart : cupy.ndarray, shape (3, nf_cart, nao_cart, nao_pair_cart)
        Cartesian ERI derivatives from _compute_cart(mol_cart, b0, b1).
    mol : pyscf.gto.Mole  (mol.cart == False, spherical target)
    mol_cart : pyscf.gto.Mole  (mol.cart == True, Cartesian copy)
    b0, b1 : int
        Shell range for the i-index.

    Returns
    -------
    eri1_sph : cupy.ndarray, shape (3, nf_sph, nao_sph, nao_pair_sph), dtype float64
    """
    # Block-diagonal cart→sph matrix: (nao_cart, nao_sph)
    # Independent of mol.cart — purely a function of the basis angular momenta.
    C = cupy.asarray(mol.cart2sph_coeff())

    ao_loc_cart = mol_cart.ao_loc_nr()   # Cartesian shell boundaries
    ao_loc_sph  = mol.ao_loc_nr()        # Spherical shell boundaries
    nao_cart = int(ao_loc_cart[-1])
    nao_sph  = int(ao_loc_sph[-1])

    # Sub-block of C for i-shells [b0, b1): (nf_cart_block, nf_sph_block)
    C_i = C[ao_loc_cart[b0]:ao_loc_cart[b1],
             ao_loc_sph[b0]:ao_loc_sph[b1]]
    nf_sph = int(C_i.shape[1])

    # Step 1 — transform i-index
    # (3, nf_cart, nao_cart, nao_pair_cart) × (nf_cart, nf_sph)
    # tensordot contracts axis 1 of eri1_cart with axis 0 of C_i:
    #   result axes: (3, nao_cart, nao_pair_cart, nf_sph) → transpose
    eri1 = cupy.tensordot(eri1_cart, C_i, axes=([1], [0]))  # (3, nao_cart, nao_pair_cart, nf_sph)
    eri1 = eri1.transpose(0, 3, 1, 2)                       # (3, nf_sph, nao_cart, nao_pair_cart)

    # Step 2 — transform j-index
    # (3, nf_sph, nao_cart, nao_pair_cart) × (nao_cart, nao_sph)
    eri1 = cupy.tensordot(eri1, C, axes=([2], [0]))          # (3, nf_sph, nao_pair_cart, nao_sph)
    eri1 = eri1.transpose(0, 1, 3, 2)                        # (3, nf_sph, nao_sph, nao_pair_cart)

    # Step 3 — unpack s2kl packed index to full (k_cart, l_cart) square
    # Both triangles must be filled: C[k_c, k_s] sums over all k_c, including
    # those in the upper triangle of the original packed index.
    k_c, l_c = cupy.tril_indices(nao_cart)
    eri1_kl = cupy.zeros(
        (3, nf_sph, nao_sph, nao_cart, nao_cart), dtype=numpy.float64
    )
    eri1_kl[:, :, :, k_c, l_c] = eri1    # lower triangle (k >= l)
    eri1_kl[:, :, :, l_c, k_c] = eri1    # upper triangle (s2kl symmetry)

    # Step 4 — transform k-index (axis 3)
    # tensordot contracts axis 3 with axis 0 of C;
    # remaining axes of eri1_kl: (0,1,2,4), then C's axis 1 appended:
    #   (3, nf_sph, nao_sph, nao_cart_l, nao_sph_k) → transpose k before l
    eri1_kl = cupy.tensordot(eri1_kl, C, axes=([3], [0]))    # (..., nao_cart_l, nao_sph_k)
    eri1_kl = eri1_kl.transpose(0, 1, 2, 4, 3)               # (..., nao_sph_k, nao_cart_l)

    # Step 5 — transform l-index (axis 4)
    eri1_kl = cupy.tensordot(eri1_kl, C, axes=([4], [0]))    # (3, nf_sph, nao_sph, nao_sph_k, nao_sph_l)

    # Step 6 — repack into spherical triangular (k' >= l')
    k_s, l_s = cupy.tril_indices(nao_sph)
    return eri1_kl[:, :, :, k_s, l_s]                        # (3, nf_sph, nao_sph, nao_pair_sph)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_int2e_ip1_gpu(mol, b0, b1):
    """
    GPU replacement for mol.intor('int2e_ip1', comp=3, aosym='s2kl',
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
    eri1_gpu : cupy.ndarray, shape (3, nf, nao, nao_pair), dtype float64
        eri1_gpu[comp, fi_local, fj_abs, kl_packed]
          comp     = 0,1,2 (x,y,z derivative of the i-centre)
          fi_local = ao_loc[ish] - ao_loc[b0] (local to this shell block)
          fj_abs   = absolute AO index for j
          kl_packed = k_abs*(k_abs+1)//2 + l_abs  (k_abs >= l_abs)
        All AO indices are in the native basis (spherical or Cartesian).
    """
    if mol.cart:
        return _compute_cart(mol, b0, b1)
    # Spherical: run kernel on Cartesian copy, then apply c2s post-processing.
    mol_cart = mol.copy()
    mol_cart.cart = True
    eri1_cart = _compute_cart(mol_cart, b0, b1)
    return _cart_to_sph_eri1(eri1_cart, mol, mol_cart, b0, b1)
