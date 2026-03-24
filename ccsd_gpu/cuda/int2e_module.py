"""
CuPy RawModule for GPU-accelerated int2e (4-center undifferentiated ERI).

compute_int2e_gpu(mol, b0, b1) is a drop-in replacement for

    mol.intor('int2e', aosym='s2kl',
              shls_slice=(b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas))

and returns a CuPy array of shape (nf, nao, nao_pair) where
nf = ao_loc[b1] - ao_loc[b0] and nao_pair = nao*(nao+1)//2.

Supported: shells up to l=4 (nroots <= 9).

Differences from int2e_ip1_module:
  - lij = li+lj  (no +1 for nabla), so g_size = (li+1)*(lj+1)*(lk+1)*(ll+1)
  - nroots = (li+lj+lk+ll)//2 + 1
  - Accumulation: gout += gx[addrx]*gx[addry]*gx[addrz]  (no nabla terms)
  - Output shape: (nf, nao, nao_pair)  (no leading comp=3 axis)
  - VRR guarded with `if (lij > 0)` — when li=lj=0, g_size=1 and
    writing gx[1] would overflow into the rw[] region.
  - GWIDTH=32 (one gout[] array instead of three, fewer registers).
"""

import cupy
import numpy

# Re-use Rys data loading and CUDA constant strings from the ip1 module —
# both kernels share the same rys_data_n9.json, CART_XYZ, NROOTS_MAX,
# TEXTBOOK_PREFAC, SHELL_NORM, and rys_roots_simple.
from ccsd_gpu.cuda.int2e_ip1_module import (
    _load_or_extract_rys_data,
    _fmt_array,
    _RYS_ROOTS,
)

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------

# Number of gout register slots per thread.
# int2e has one output component (not 3), so we can use GWIDTH=32 while
# keeping register usage comparable to GWIDTH=16 for int2e_ip1.
# With gout_stride=32 threads: 32*GWIDTH = 1024 ijkl combos per batch.
# (g,g,g,g): nf=50625 -> 50 batches (vs 99 for ip1 with GWIDTH=16).
_GWIDTH = 32

_CUDA_PREFIX = r"""
/* ===================================================================
 * Standalone int2e CUDA kernel - ccsd_gpu.cuda
 * Rys quadrature + OS recurrence (4-centre undifferentiated ERI).
 * Closely follows int2e_ip1_module, but no derivative and no comp axis.
 * Supported: nroots = 1..9 (shells with l <= 4).
 * =================================================================== */

/* ---- Physical / algorithmic constants ---- */
#define SQRTPIE4   0.8862269254527580136
#define PIE4       0.7853981633974483096
#define TEXTBOOK_PREFAC 34.986836655249726
#define DEGREE     13
#define DEGREE1    14
#define INTERVALS  40

/* ---- PySCF mol._bas slot indices ---- */
#define ATOM_OF    0
#define ANG_OF     1
#define NPRIM_OF   2
#define PTR_EXP    5
#define PTR_COEFF  6
#define BAS_SLOTS  8

/* ---- PySCF mol._atm slot indices ---- */
#define ATM_SLOTS  6
#define PTR_COORD  1

/* ---- Cartesian (lx,ly,lz) for l=0..4 in lexicographic order.
 *     lex_xyz_offset(l) = l*(l+1)*(l+2)/2
 *     l=0 -> offset  0,  1 function  -> CART_XYZ[0..2]
 *     l=1 -> offset  3,  3 functions -> CART_XYZ[3..11]
 *     l=2 -> offset 12,  6 functions -> CART_XYZ[12..29]
 *     l=3 -> offset 30, 10 functions -> CART_XYZ[30..59]
 *     l=4 -> offset 60, 15 functions -> CART_XYZ[60..104]
 * ---- */
__device__ const int CART_XYZ[105] = {
    /* l=0 */ 0,0,0,
    /* l=1 */ 1,0,0, 0,1,0, 0,0,1,
    /* l=2 */ 2,0,0, 1,1,0, 1,0,1, 0,2,0, 0,1,1, 0,0,2,
    /* l=3 */ 3,0,0, 2,1,0, 2,0,1, 1,2,0, 1,1,1, 1,0,2, 0,3,0, 0,2,1, 0,1,2, 0,0,3,
    /* l=4 */ 4,0,0, 3,1,0, 3,0,1, 2,2,0, 2,1,1, 2,0,2, 1,3,0, 1,2,1, 1,1,2, 1,0,3, 0,4,0, 0,3,1, 0,2,2, 0,1,3, 0,0,4
};
#define NROOTS_MAX  9

/* ---- Per-shell normalization factors (CINTcommon_fac_sp) ---- */
__device__ const double SHELL_NORM[3] = {
    0.28209479177387814,    /* l=0: 1/(2*sqrt(pi)) */
    0.48860251190291992,    /* l=1: sqrt(3)/(2*sqrt(pi)) */
    1.0                     /* l>=2 */
};
"""

_KERNEL = r"""
/* ----------------------------------------------------------------
 * fill_int2e_kernel
 *
 * Computes raw Cartesian int2e for a fixed i-shell (ish) and all
 * combinations of (jsh in 0..nbas-1, ksh>=lsh).
 *
 * Output tensor eri0 has C-contiguous layout (nfi_block, nao, nao_pair)
 * corresponding to (i_local, j_abs, kl_packed) where:
 *   i_local   = fi_offset + fi  in [0, nfi_block)
 *   j_abs     = ao_loc[jsh] + fj  (absolute AO index)
 *   kl_packed = k_abs*(k_abs+1)/2 + l_abs  for k_abs >= l_abs
 *
 * Key differences from fill_int2e_ip1_kernel:
 *   - lij = li+lj  (no +1 for nabla)
 *   - stride_j = li+1  (not li+2)
 *   - nroots = (li+lj+lk+ll)/2 + 1
 *   - VRR is guarded: when li=lj=0, lij=0 and g_size=1,
 *     so we must NOT write gx[1] (would overflow into rw[]).
 *   - Accumulation: gout[n] += gx[x]*gx[y]*gx[z]  (no nabla)
 *   - Output: single value per ijkl combo (no 3x components)
 *
 * Grid:  dim3(n_jsh * n_kl_pairs)
 * Block: dim3(1, GOUT_STRIDE)
 * Shared: gx[3*g_size] + rw[2*NROOTS_MAX] + cicj_cache[iprim*jprim]
 * ---------------------------------------------------------------- */

#define GOUT_STRIDE 32
#define GWIDTH      """ + str(_GWIDTH) + r"""

__global__ void fill_int2e_kernel(
    double       * __restrict__ eri0,         /* (nfi_block, nao, nao_pair) */
    const int    * __restrict__ atm,
    const int    * __restrict__ bas,
    const double * __restrict__ env,
    const int    * __restrict__ ao_loc,
    int ish,
    int fi_offset,
    int nfi_block,
    int nbas, int nao, int nao_pair,
    const int * __restrict__ kl_pair_idx,
    int n_kl_pairs)
{
    int gout_id     = threadIdx.y;
    int gout_stride = blockDim.y;

    /* ---- map blockIdx -> (jsh, ksh, lsh) ---- */
    int block_id = blockIdx.x;
    int jsh      = block_id / n_kl_pairs;
    int kl_idx   = block_id % n_kl_pairs;
    if (jsh >= nbas) return;
    int ksh = kl_pair_idx[2 * kl_idx];
    int lsh = kl_pair_idx[2 * kl_idx + 1];

    /* ---- angular momenta ---- */
    int li = bas[ish * BAS_SLOTS + ANG_OF];
    int lj = bas[jsh * BAS_SLOTS + ANG_OF];
    int lk = bas[ksh * BAS_SLOTS + ANG_OF];
    int ll = bas[lsh * BAS_SLOTS + ANG_OF];

    /* ---- primitive counts and pointers ---- */
    int iprim  = bas[ish * BAS_SLOTS + NPRIM_OF];
    int jprim  = bas[jsh * BAS_SLOTS + NPRIM_OF];
    int kprim  = bas[ksh * BAS_SLOTS + NPRIM_OF];
    int lprim  = bas[lsh * BAS_SLOTS + NPRIM_OF];
    int expi   = bas[ish * BAS_SLOTS + PTR_EXP];
    int expj   = bas[jsh * BAS_SLOTS + PTR_EXP];
    int expk   = bas[ksh * BAS_SLOTS + PTR_EXP];
    int expl   = bas[lsh * BAS_SLOTS + PTR_EXP];
    int ci_ptr = bas[ish * BAS_SLOTS + PTR_COEFF];
    int cj_ptr = bas[jsh * BAS_SLOTS + PTR_COEFF];
    int ck_ptr = bas[ksh * BAS_SLOTS + PTR_COEFF];
    int cl_ptr = bas[lsh * BAS_SLOTS + PTR_COEFF];

    /* ---- shell centres ---- */
    int ri_p = atm[bas[ish*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
    int rj_p = atm[bas[jsh*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
    int rk_p = atm[bas[ksh*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
    int rl_p = atm[bas[lsh*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
    double rix=env[ri_p], riy=env[ri_p+1], riz=env[ri_p+2];
    double rjx=env[rj_p], rjy=env[rj_p+1], rjz=env[rj_p+2];
    double rkx=env[rk_p], rky=env[rk_p+1], rkz=env[rk_p+2];
    double rlx=env[rl_p], rly=env[rl_p+1], rlz=env[rl_p+2];
    double rjrix=rjx-rix, rjriy=rjy-riy, rjriz=rjz-riz;
    double rlrkx=rlx-rkx, rlrky=rly-rky, rlrkz=rlz-rkz;
    double rjri_rr = rjrix*rjrix + rjriy*rjriy + rjriz*rjriz;
    double rlrk_rr = rlrkx*rlrkx + rlrky*rlrky + rlrkz*rlrkz;

    /* ---- g-array dimensions for int2e (no nabla) ----
     *   lij = li+lj,  stride_j = li+1
     *   g_size = (li+1)*(lj+1)*(lk+1)*(ll+1)
     * Note: when li=lj=0, lij=0 and g_size=1 -- VRR must be guarded. */
    int lij      = li + lj;
    int lkl      = lk + ll;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int stride_l = stride_k * (lk + 1);
    int g_size   = stride_l * (ll + 1);
    int nroots   = (li + lj + lk + ll) / 2 + 1;

    /* ---- Cartesian function counts ---- */
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int nfl = (ll+1)*(ll+2)/2;
    int nf  = nfi * nfj * nfk * nfl;

    int off_i = li*(li+1)*(li+2)/2;
    int off_j = lj*(lj+1)*(lj+2)/2;
    int off_k = lk*(lk+1)*(lk+2)/2;
    int off_l = ll*(ll+1)*(ll+2)/2;

    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];

    /* ---- shared memory layout ----
     *  [0 .. 3*g_size)             gx:  g-array, 3 directions
     *  [3*g_size .. 3*g_size+2*NR) rw:  Rys roots then weights
     *  [3*g_size+2*NR .. ...]      cicj_cache[iprim*jprim]
     */
    extern __shared__ double shm[];
    double *gx         = shm;
    double *rw         = shm + 3 * g_size;
    double *cicj_cache = rw  + 2 * NROOTS_MAX;

    /* ---- Pre-compute cicj_cache ---- */
    for (int ij = gout_id; ij < iprim * jprim; ij += gout_stride) {
        int ip = ij / jprim, jp = ij % jprim;
        double ai = env[expi+ip], aj = env[expj+jp];
        double aij = ai + aj;
        double Kab = exp(-(ai*aj/aij) * rjri_rr);
        cicj_cache[ij] = env[ci_ptr+ip] * env[cj_ptr+jp] * Kab;
    }
    __syncthreads();

    /* ---- Per-shell normalization ---- */
    double shell_norm_fac = TEXTBOOK_PREFAC
                          * SHELL_NORM[li < 2 ? li : 2] * SHELL_NORM[lj < 2 ? lj : 2]
                          * SHELL_NORM[lk < 2 ? lk : 2] * SHELL_NORM[ll < 2 ? ll : 2];

    /* ---- register accumulator: single component (no 3x) ---- */
    double gout[GWIDTH];

    /* ====================================================
     * Batch over output ijkl combos
     * ==================================================== */
    for (int gout_start = 0; gout_start < nf; gout_start += gout_stride * GWIDTH) {

        for (int n = 0; n < GWIDTH; n++) gout[n] = 0.0;

        /* ---- primitive quartet loops ---- */
        for (int klp = 0; klp < kprim * lprim; klp++) {
            int kp = klp / lprim, lp = klp % lprim;
            double ak = env[expk+kp], al = env[expl+lp];
            double akl    = ak + al;
            double al_akl = al / akl;
            double Kcd    = exp(-(ak*al/akl) * rlrk_rr);
            double fac_kl = shell_norm_fac * env[ck_ptr+kp] * env[cl_ptr+lp] * Kcd;

            for (int ijp = 0; ijp < iprim * jprim; ijp++) {
                int ip = ijp / jprim, jp = ijp % jprim;
                double ai    = env[expi+ip];
                double aj    = env[expj+jp];
                double aij   = ai + aj;
                double aj_aij = aj / aij;

                double Px = rix + rjrix * aj_aij;
                double Py = riy + rjriy * aj_aij;
                double Pz = riz + rjriz * aj_aij;
                double Qx = rkx + rlrkx * al_akl;
                double Qy = rky + rlrky * al_akl;
                double Qz = rkz + rlrkz * al_akl;
                double xpq = Px - Qx, ypq = Py - Qy, zpq = Pz - Qz;
                double rr   = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                double T    = theta * rr;

                double cicj = cicj_cache[ijp] / (aij * akl * sqrt(aij + akl));

                if (gout_id == 0) {
                    rys_roots_simple(nroots, T, rw, rw + nroots);
                }
                __syncthreads();

                for (int irys = 0; irys < nroots; irys++) {
                    double rt     = rw[irys];
                    double wt     = rw[nroots + irys];
                    double rt_aa  = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00    = 0.5 * rt_aa;
                    double b10    = 0.5 / aij * (1.0 - rt_aij);
                    double b01    = 0.5 / akl  * (1.0 - rt_akl);

                    if (gout_id == 0) {
                        gx[0]        = fac_kl;
                        gx[g_size]   = cicj;
                        gx[2*g_size] = wt;
                    }
                    __syncthreads();

                    /* ---- VRR: fill gx[ix=0..lij] for each direction ----
                     * Guarded: when lij=0 (both i and j are s-shells),
                     * g_size=1 and gx[1] would write into rw[]. */
                    if (lij > 0) {
                        for (int nx = gout_id; nx < 3; nx += gout_stride) {
                            double *_gx = gx + nx * g_size;
                            double Rpq_n, rjri_n;
                            if (nx == 0) { Rpq_n = xpq; rjri_n = rjrix; }
                            else if (nx == 1) { Rpq_n = ypq; rjri_n = rjriy; }
                            else              { Rpq_n = zpq; rjri_n = rjriz; }
                            double c0 = rjri_n * aj_aij - rt_aij * Rpq_n;
                            double s0 = _gx[0];
                            double s1 = c0 * s0;
                            _gx[1] = s1;
                            for (int i = 1; i < lij; i++) {
                                double s2 = c0 * s1 + i * b10 * s0;
                                _gx[i+1] = s2;
                                s0 = s1; s1 = s2;
                            }
                        }
                        __syncthreads();
                    }

                    /* ---- TRR: fill gx[i,k] for i=0..lij, k=0..lkl ---- */
                    if (lkl > 0) {
                        int lij3 = (lij + 1) * 3;
                        for (int n = gout_id; n < lij3; n += gout_stride) {
                            int i = n / 3, nx = n % 3;
                            double *_gx = gx + i + nx * g_size;
                            double rlrk_n, Rpq_n;
                            if (nx==0){rlrk_n=rlrkx; Rpq_n=xpq;}
                            else if(nx==1){rlrk_n=rlrky; Rpq_n=ypq;}
                            else          {rlrk_n=rlrkz; Rpq_n=zpq;}
                            double cpx = rlrk_n * al_akl + rt_akl * Rpq_n;
                            double s0 = _gx[0];
                            double s1 = cpx * s0;
                            if (i > 0) s1 += i * b00 * _gx[-1];
                            _gx[stride_k] = s1;
                        }
                        for (int k = 1; k < lkl; k++) {
                            __syncthreads();
                            for (int n = gout_id; n < lij3; n += gout_stride) {
                                int i = n / 3, nx = n % 3;
                                double *_gx = gx + i + nx * g_size;
                                double rlrk_n, Rpq_n;
                                if (nx==0){rlrk_n=rlrkx; Rpq_n=xpq;}
                                else if(nx==1){rlrk_n=rlrky; Rpq_n=ypq;}
                                else          {rlrk_n=rlrkz; Rpq_n=zpq;}
                                double cpx = rlrk_n * al_akl + rt_akl * Rpq_n;
                                double s0 = _gx[(k-1) * stride_k];
                                double s1 = _gx[k * stride_k];
                                double s2 = cpx * s1 + k * b01 * s0;
                                if (i > 0) s2 += i * b00 * _gx[k * stride_k - 1];
                                _gx[(k+1) * stride_k] = s2;
                            }
                        }
                        __syncthreads();
                    }

                    /* ---- HRR_ij: (lij,0) -> (li,lj) via rjri ---- */
                    if (lj > 0) {
                        int lkl3 = (lkl + 1) * 3;
                        for (int m = gout_id; m < lkl3; m += gout_stride) {
                            int k  = m / 3, nx = m % 3;
                            double rjri_n;
                            if (nx==0) rjri_n=rjrix;
                            else if(nx==1) rjri_n=rjriy;
                            else rjri_n=rjriz;
                            double *_gx = gx + nx * g_size + k * stride_k;
                            for (int j = 0; j < lj; j++) {
                                int ij = (lij - j) + j * stride_j;
                                double s1 = _gx[ij];
                                for (--ij; ij >= j * stride_j; --ij) {
                                    double s0 = _gx[ij];
                                    _gx[ij + stride_j] = s1 - rjri_n * s0;
                                    s1 = s0;
                                }
                            }
                        }
                        __syncthreads();
                    }

                    /* ---- HRR_kl: (lkl,0) -> (lk,ll) via rlrk ---- */
                    if (ll > 0) {
                        int nkl_items = stride_k * 3;
                        for (int n = gout_id; n < nkl_items; n += gout_stride) {
                            int i  = n / 3, nx = n % 3;
                            double rlrk_n;
                            if (nx==0) rlrk_n=rlrkx;
                            else if(nx==1) rlrk_n=rlrky;
                            else rlrk_n=rlrkz;
                            double *_gx = gx + nx * g_size + i;
                            for (int l = 0; l < ll; l++) {
                                int kl = (lkl - l) * stride_k + l * stride_l;
                                double s1 = _gx[kl];
                                for (kl -= stride_k; kl >= l * stride_l; kl -= stride_k) {
                                    double s0 = _gx[kl];
                                    _gx[kl + stride_l] = s1 - rlrk_n * s0;
                                    s1 = s0;
                                }
                            }
                        }
                        __syncthreads();
                    }

                    /* ---- Accumulate: product of g-functions (no nabla) ---- */
                    for (int n = 0; n < GWIDTH; n++) {
                        int ijkl = gout_start + n * gout_stride + gout_id;
                        if (ijkl >= nf) break;

                        int jkl = ijkl / nfi,  fi = ijkl % nfi;
                        int kl  = jkl  / nfj,  fj = jkl  % nfj;
                        int fl  = kl   / nfk,  fk = kl   % nfk;

                        int fk_abs = k0 + fk, fl_abs = l0 + fl;
                        if (ksh == lsh && fl_abs > fk_abs) continue;

                        int ix=CART_XYZ[off_i+fi*3  ], iy=CART_XYZ[off_i+fi*3+1], iz=CART_XYZ[off_i+fi*3+2];
                        int jx=CART_XYZ[off_j+fj*3  ], jy=CART_XYZ[off_j+fj*3+1], jz=CART_XYZ[off_j+fj*3+2];
                        int kx=CART_XYZ[off_k+fk*3  ], ky=CART_XYZ[off_k+fk*3+1], kz=CART_XYZ[off_k+fk*3+2];
                        int lx=CART_XYZ[off_l+fl*3  ], ly=CART_XYZ[off_l+fl*3+1], lz=CART_XYZ[off_l+fl*3+2];

                        int addrx = ix + jx*stride_j + kx*stride_k + lx*stride_l;
                        int addry = iy + jy*stride_j + ky*stride_k + ly*stride_l + g_size;
                        int addrz = iz + jz*stride_j + kz*stride_k + lz*stride_l + g_size*2;

                        gout[n] += gx[addrx] * gx[addry] * gx[addrz];
                    }
                    __syncthreads();
                } /* irys */
            } /* ijp */
        } /* klp */

        /* ---- Write accumulated gout to eri0 ----
         * eri0 layout: (nfi_block, nao, nao_pair)
         * [fi_local * nao * nao_pair + fj_abs * nao_pair + kl_packed] */
        for (int n = 0; n < GWIDTH; n++) {
            int ijkl = gout_start + n * gout_stride + gout_id;
            if (ijkl >= nf) break;

            int jkl = ijkl / nfi,  fi = ijkl % nfi;
            int kl  = jkl  / nfj,  fj = jkl  % nfj;
            int fl  = kl   / nfk,  fk = kl   % nfk;

            int fk_abs = k0 + fk, fl_abs = l0 + fl;
            if (ksh == lsh && fl_abs > fk_abs) continue;

            int fj_abs    = j0 + fj;
            int kl_packed = fk_abs * (fk_abs + 1) / 2 + fl_abs;
            int fi_local  = fi_offset + fi;

            long long base = (long long)fi_local * nao * nao_pair
                           + (long long)fj_abs   * nao_pair
                           + kl_packed;

            eri0[base] = gout[n];
        }
        __syncthreads();

    } /* gout_start */
}
"""

# ---------------------------------------------------------------------------
# Build and cache
# ---------------------------------------------------------------------------

_NROOTS_MAX = 9


def _build_cuda_source(rys_data):
    tables = []
    for src, dst in [
        ("SMALLX_R0", "d_SMALLX_R0"),
        ("SMALLX_R1", "d_SMALLX_R1"),
        ("SMALLX_W0", "d_SMALLX_W0"),
        ("SMALLX_W1", "d_SMALLX_W1"),
        ("LARGEX_R",  "d_LARGEX_R"),
        ("LARGEX_W",  "d_LARGEX_W"),
    ]:
        tables.append(_fmt_array(dst, rys_data[src]))
    tables.append(_fmt_array("d_RW_DATA", rys_data["RW_DATA"], per_line=8))
    return (
        _CUDA_PREFIX
        + "\n".join(tables)
        + "\n"
        + _RYS_ROOTS
        + "\n"
        + _KERNEL
    )


_kernel_cache = {}


def get_kernel():
    """Return the compiled CuPy RawKernel for fill_int2e_kernel."""
    device = cupy.cuda.Device().id
    if device in _kernel_cache:
        return _kernel_cache[device]

    rys_data = _load_or_extract_rys_data()
    src = _build_cuda_source(rys_data)

    mod = cupy.RawModule(
        code=src,
        options=("--std=c++14",),
        name_expressions=["fill_int2e_kernel"],
    )
    kern = mod.get_function("fill_int2e_kernel")
    _kernel_cache[device] = kern
    return kern
