"""
CuPy RawModule for GPU-accelerated int2e_ip1 (4-center ERI derivatives).

The CUDA kernel computes Cartesian 4-center 2-electron repulsion integral
derivatives with respect to the center of the first basis function:

    int2e_ip1[x, μ, ν, λ, σ] = ∂/∂A_x (μν|λσ)
                               = 2α_μ (μ+x ν|λσ) - l_x(μ) (μ-x ν|λσ)

Uses Rys quadrature + Obara-Saika recurrence, closely following the
gpu4pyscf gvhf-rys/rys_contract_jk_ip1.cu implementation but outputting
raw Cartesian ERI values instead of contracting with density matrices.

Supported: shells up to l=4 (nroots <= 9).

IMPORTANT: This kernel outputs CARTESIAN integrals.  For molecules with
spherical-harmonic basis (mol.cart == False) and l >= 2 shells, the caller
is responsible for applying the cart->sph transformation.
"""

import json
import os

import cupy
import numpy

# ---------------------------------------------------------------------------
# Rys data management
# ---------------------------------------------------------------------------
_RYS_DATA_PATH = os.path.join(os.path.dirname(__file__), "rys_data_n9.json")


def _load_or_extract_rys_data():
    """Return dict with Rys polynomial tables for nroots=1..9."""
    if os.path.exists(_RYS_DATA_PATH):
        with open(_RYS_DATA_PATH) as f:
            return json.load(f)

    # Fall back to /tmp cache (written by previous extraction session)
    tmp_path = "/tmp/rys_data_n9.json"
    if os.path.exists(tmp_path):
        with open(tmp_path) as f:
            data = json.load(f)
        with open(_RYS_DATA_PATH, "w") as f:
            json.dump(data, f)
        return data

    # Try to extract from a cloned gpu4pyscf source tree
    for cu_path in [
        "/tmp/gpu4pyscf_src_new/gpu4pyscf/lib/gvhf-rys/rys_roots_dat.cu",
        "/tmp/gpu4pyscf_src/gpu4pyscf/lib/gvhf-rys/rys_roots_dat.cu",
    ]:
        if os.path.exists(cu_path):
            data = _extract_rys_from_cu(cu_path)
            with open(_RYS_DATA_PATH, "w") as f:
                json.dump(data, f)
            return data

    raise RuntimeError(
        f"Rys polynomial data not found.  Expected at {_RYS_DATA_PATH}.\n"
        "Run:  cd /tmp && git clone --depth=1 "
        "https://github.com/pyscf/gpu4pyscf.git gpu4pyscf_src_new"
    )


def _extract_rys_from_cu(path):
    import re

    with open(path) as f:
        txt = f.read()

    NROOTS_MAX, DEGREE1, INTERVALS = 9, 14, 40
    N_SMALLX = NROOTS_MAX * (NROOTS_MAX + 1) // 2   # cumulative entries for nroots 1..9 = 45

    def get(name):
        m = re.search(
            rf"__device__\s+double\s+{name}\[\]\s*=\s*\{{([^}}]+)\}}", txt, re.DOTALL
        )
        if not m:
            raise RuntimeError(f"{name} not found in {path}")
        body = re.sub(r"//[^\n]*", "", m.group(1))
        return [float(x.strip()) for x in body.split(",") if x.strip()]

    data = {}
    for src, dst in [
        ("ROOT_SMALLX_R0", "SMALLX_R0"),
        ("ROOT_SMALLX_R1", "SMALLX_R1"),
        ("ROOT_SMALLX_W0", "SMALLX_W0"),
        ("ROOT_SMALLX_W1", "SMALLX_W1"),
        ("ROOT_LARGEX_R_DATA", "LARGEX_R"),
        ("ROOT_LARGEX_W_DATA", "LARGEX_W"),
    ]:
        data[dst] = get(src)[:N_SMALLX]
    arr = get("ROOT_RW_DATA")
    n_rw = DEGREE1 * INTERVALS * NROOTS_MAX * (NROOTS_MAX + 1)
    data["RW_DATA"] = arr[:n_rw]
    return data


# ---------------------------------------------------------------------------
# CUDA source construction
# ---------------------------------------------------------------------------

def _fmt_array(name, vals, per_line=8):
    lines = [f"__device__ double {name}[] = {{"]
    for i in range(0, len(vals), per_line):
        chunk = vals[i : i + per_line]
        lines.append("    " + ", ".join(f"{v:.17e}" for v in chunk) + ",")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    return "\n".join(lines)


# Number of gout register slots per thread (batch size for AO combinations).
# With gout_stride=32 threads: processes 32*GWIDTH = 512 ijkl combos per batch.
# (p,p,p,p): nf=81 -> 1 batch; (d,d,d,d): nf=1296 -> 3 batches;
# (f,f,f,f): nf=10000 -> 20 batches; (g,g,g,g): nf=50625 -> 99 batches.
_GWIDTH = 16

_CUDA_PREFIX = r"""
/* ===================================================================
 * Standalone int2e_ip1 CUDA kernel - ccsd_gpu.cuda
 * Rys quadrature + OS recurrence (4-centre ERI ip1 derivative).
 * Closely follows gpu4pyscf gvhf-rys/rys_contract_jk_ip1.cu, but
 * outputs raw Cartesian ERI values instead of contracting with DMs.
 * Supported: nroots = 1..9 (shells with l <= 4).
 * =================================================================== */

/* ---- Physical / algorithmic constants ---- */
#define SQRTPIE4   0.8862269254527580136
#define PIE4       0.7853981633974483096
/* TEXTBOOK_PREFAC = 2*pi^(5/2).  Combined with per-shell SHELL_NORM factors
 * (matching libcint's CINTcommon_fac_sp), this gives the correct prefactor
 * for PySCF's gto_norm (radial-only) stored coefficients.
 * See docs/int2e_ip1_cuda_kernel.md Bug 4 for derivation. */
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

/* ---- Per-shell normalization factors (CINTcommon_fac_sp) ----
 * libcint applies a per-shell (not per-component) factor internally.
 * SHELL_NORM[l] matches CINTcommon_fac_sp from libcint:
 *   l=0: 1/(2*sqrt(pi))
 *   l=1: sqrt(3)/(2*sqrt(pi))
 *   l>=2: 1.0
 * Combined with TEXTBOOK_PREFAC, the product
 *   TEXTBOOK_PREFAC * SHELL_NORM[li] * SHELL_NORM[lj] * SHELL_NORM[lk] * SHELL_NORM[ll]
 * gives the correct prefactor for fac_kl, replacing the old PI_FAC * CART_NORM approach.
 */
__device__ const double SHELL_NORM[3] = {
    0.28209479177387814,    /* l=0: 1/(2*sqrt(pi)) */
    0.48860251190291992,    /* l=1: sqrt(3)/(2*sqrt(pi)) */
    1.0                     /* l>=2 */
};
"""

_RYS_ROOTS = r"""
/* ----------------------------------------------------------------
 * rys_roots_simple: compute Rys roots r[0..n-1] and weights
 * w[0..n-1] for argument x and nroots in [1..NROOTS_MAX].
 * ---------------------------------------------------------------- */
__device__ void rys_roots_simple(int nroots, double x,
                                  double *r, double *w)
{
    if (x < 3.e-7) {
        int off = nroots * (nroots - 1) / 2;
        for (int i = 0; i < nroots; i++) {
            r[i] = d_SMALLX_R0[off+i] + d_SMALLX_R1[off+i] * x;
            w[i] = d_SMALLX_W0[off+i] + d_SMALLX_W1[off+i] * x;
        }
        return;
    }
    if (x > 35.0 + nroots * 5) {
        int off = nroots * (nroots - 1) / 2;
        double t = sqrt(PIE4 / x);
        for (int i = 0; i < nroots; i++) {
            r[i] = d_LARGEX_R[off+i] / x;
            w[i] = d_LARGEX_W[off+i] * t;
        }
        return;
    }
    if (nroots == 1) {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        w[0] = fmt0;
        double e = exp(-x);
        double b = 0.5 / x;
        r[0] = b * (fmt0 - e) / fmt0;
        return;
    }
    /* General case: Chebyshev recurrence (DEGREE=13, INTERVALS=40). */
    double *datax = d_RW_DATA + (long long)DEGREE1 * INTERVALS * nroots * (nroots - 1);
    int it = (int)(x * 0.4);
    double u  = (x - it * 2.5) * 0.8 - 1.0;
    double u2 = u * 2.0;
    for (int i = 0; i < nroots * 2; i++) {
        double *c = datax + (long long)i * DEGREE1 * INTERVALS;
        double c0 = c[it + DEGREE    * INTERVALS];
        double c1 = c[it + (DEGREE-1)* INTERVALS];
        double c2, c3;
        for (int n = DEGREE - 2; n > 0; n -= 2) {
            c2 = c[it + n    * INTERVALS] - c1;
            c3 = c0 + c1 * u2;
            c1 = c2 + c3 * u2;
            c0 = c[it + (n-1)* INTERVALS] - c3;
        }
        /* DEGREE=13 is odd -> result = c0 + c1*u */
        double val = c0 + c1 * u;
        /* ROOT_RW_DATA is stored INTERLEAVED: r0,w0,r1,w1,...
         * i=0->r[0], i=1->w[0], i=2->r[1], i=3->w[1], ... */
        if (i % 2 == 0) r[i/2] = val;
        else            w[i/2] = val;
    }
}
"""

_KERNEL = r"""
/* ----------------------------------------------------------------
 * fill_int2e_ip1_kernel
 *
 * Computes raw Cartesian int2e_ip1 for a fixed i-shell (ish) and
 * all combinations of (jsh in 0..nbas-1, ksh>=lsh) passed via the
 * kl_pair_idx array.
 *
 * Output tensor eri1 has C-contiguous layout (3, nfi_block, nao, nao_pair)
 * corresponding to (comp, i_local, j_abs, kl_packed) where:
 *   comp      = 0,1,2  (x, y, z derivative)
 *   i_local   = fi_offset + fi  in [0, nfi_block)
 *   j_abs     = ao_loc[jsh] + fj  (absolute spherical/Cartesian AO)
 *   kl_packed = k_abs*(k_abs+1)/2 + l_abs  for k_abs >= l_abs
 *
 * Each block handles exactly ONE (ish, jsh, ksh, lsh) quartet.
 *
 * Grid:   dim3(n_jsh * n_kl_pairs)
 * Block:  dim3(1, GOUT_STRIDE)   -> sq_id=0, gout_id=0..GOUT_STRIDE-1
 * Shared: gx[3*g_size] + rw[2*NROOTS_MAX] + cicj_cache[iprim*jprim]
 *         Total <= ~(3*108 + 10 + ipmax*jpmax) doubles.  Sized at launch.
 * ---------------------------------------------------------------- */

#define GOUT_STRIDE 32
#define GWIDTH      """ + str(_GWIDTH) + r"""

__global__ void fill_int2e_ip1_kernel(
    double       * __restrict__ eri1,         /* (3, nfi_block, nao, nao_pair) */
    const int    * __restrict__ atm,
    const int    * __restrict__ bas,
    const double * __restrict__ env,
    const int    * __restrict__ ao_loc,
    int ish,          /* fixed i-shell index */
    int fi_offset,    /* ao_loc[ish] - ao_loc[b0]  (local fi start in eri1) */
    int nfi_block,    /* ao_loc[b1] - ao_loc[b0]  (total i-AOs in the block) */
    int nbas, int nao, int nao_pair,
    const int * __restrict__ kl_pair_idx,  /* [n_kl_pairs * 2]: (ksh, lsh) */
    int n_kl_pairs)
{
    int gout_id     = threadIdx.y;            /* 0 .. GOUT_STRIDE-1 */
    int gout_stride = blockDim.y;             /* = GOUT_STRIDE */

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

    /* ---- shell centres (from mol._atm and mol._env) ---- */
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

    /* ---- g-array dimensions for ip1 (ng=[1,0,0,0]) ----
     *   lij = li+lj+1  (extra +1 for nabla on i)
     *   stride_j = li+2  (room for ix = 0..li+1)
     *   g_size = (li+2)*(lj+1)*(lk+1)*(ll+1)            */
    int lij      = li + lj + 1;
    int lkl      = lk + ll;
    int stride_j = li + 2;
    int stride_k = stride_j * (lj + 1);
    int stride_l = stride_k * (lk + 1);
    int g_size   = stride_l * (ll + 1);
    int nroots   = (li + lj + lk + ll + 1) / 2 + 1;

    /* ---- Cartesian function counts ---- */
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int nfl = (ll+1)*(ll+2)/2;
    int nf  = nfi * nfj * nfk * nfl;

    /* lex_xyz_offset(l) = l*(l+1)*(l+2)/2 */
    int off_i = li*(li+1)*(li+2)/2;
    int off_j = lj*(lj+1)*(lj+2)/2;
    int off_k = lk*(lk+1)*(lk+2)/2;
    int off_l = ll*(ll+1)*(ll+2)/2;

    /* absolute AO base indices */
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];

    /* ---- shared memory layout ----
     *  [0 .. 3*g_size)              gx:  g-array, 3 directions interleaved
     *  [3*g_size .. 3*g_size+2*NR)  rw:  Rys roots then weights
     *  [3*g_size+2*NR .. ...]       cicj_cache[iprim*jprim]
     */
    extern __shared__ double shm[];
    double *gx         = shm;
    double *rw         = shm + 3 * g_size;
    double *cicj_cache = rw  + 2 * NROOTS_MAX;

    /* ---- Pre-compute cicj_cache in parallel ---- */
    for (int ij = gout_id; ij < iprim * jprim; ij += gout_stride) {
        int ip = ij / jprim, jp = ij % jprim;
        double ai = env[expi+ip], aj = env[expj+jp];
        double aij = ai + aj;
        double Kab = exp(-(ai*aj/aij) * rjri_rr);
        cicj_cache[ij] = env[ci_ptr+ip] * env[cj_ptr+jp] * Kab;
    }
    __syncthreads();

    /* ---- Per-shell normalization (constant for this quartet) ---- */
    double shell_norm_fac = TEXTBOOK_PREFAC
                          * SHELL_NORM[li < 2 ? li : 2] * SHELL_NORM[lj < 2 ? lj : 2]
                          * SHELL_NORM[lk < 2 ? lk : 2] * SHELL_NORM[ll < 2 ? ll : 2];

    /* ---- register accumulators: goutx/y/z[GWIDTH] ---- */
    double goutx[GWIDTH], gouty[GWIDTH], goutz[GWIDTH];

    /* ====================================================
     * Batch over output ijkl combos
     * ==================================================== */
    for (int gout_start = 0; gout_start < nf; gout_start += gout_stride * GWIDTH) {

        /* zero gout for this batch */
        for (int n = 0; n < GWIDTH; n++) {
            goutx[n] = 0.0; gouty[n] = 0.0; goutz[n] = 0.0;
        }

        /* ---- primitive quartet loops ---- */
        for (int klp = 0; klp < kprim * lprim; klp++) {
            int kp = klp / lprim, lp = klp % lprim;
            double ak = env[expk+kp], al = env[expl+lp];
            double akl     = ak + al;
            double al_akl  = al / akl;
            double Kcd     = exp(-(ak*al/akl) * rlrk_rr);
            double fac_kl  = shell_norm_fac * env[ck_ptr+kp] * env[cl_ptr+lp] * Kcd;

            for (int ijp = 0; ijp < iprim * jprim; ijp++) {
                int ip = ijp / jprim, jp = ijp % jprim;
                double ai    = env[expi+ip];
                double aj    = env[expj+jp];
                double aij   = ai + aj;
                double aj_aij = aj / aij;
                double ai2   = ai * 2.0;

                /* composite centres */
                double Px = rix + rjrix * aj_aij;
                double Py = riy + rjriy * aj_aij;
                double Pz = riz + rjriz * aj_aij;
                double Qx = rkx + rlrkx * al_akl;
                double Qy = rky + rlrky * al_akl;
                double Qz = rkz + rlrkz * al_akl;
                double xpq = Px - Qx, ypq = Py - Qy, zpq = Pz - Qz;
                double rr  = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                double T    = theta * rr;

                /* cicj = c_i * c_j * Kab / (aij * akl * sqrt(aij+akl))
                   cicj_cache already holds c_i*c_j*Kab               */
                double cicj = cicj_cache[ijp] / (aij * akl * sqrt(aij + akl));

                /* ---- Rys roots and weights (thread 0 computes, all read) ---- */
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

                    /* ---- initialise g-array seed values ----
                     * gx[0]        = fac_kl   (x-direction carries kl scale)
                     * gx[g_size]   = cicj     (y-direction carries ij/norm)
                     * gx[2*g_size] = wt       (z-direction carries Rys weight)
                     * VRR builds on these as the g[ix=0,...] starting values.
                     */
                    if (gout_id == 0) {
                        gx[0]          = fac_kl;
                        gx[g_size]     = cicj;
                        gx[2*g_size]   = wt;
                    }
                    __syncthreads();

                    /* ---- VRR: fill gx[ix=0..lij] for each direction ----
                     * Threads 0,1,2 each take one direction; 3..31 idle.
                     * g[n+1] = c0*g[n] + n*b10*g[n-1],  g[0] = seed above
                     */
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

                    /* ---- TRR: fill gx[i,k] for i=0..lij, k=0..lkl ----
                     * TRR formula:  g[i,k+1] = cp*g[i,k] + k*b01*g[i,k-1]
                     *                        + i*b00*g[i-1,k]
                     * Each thread handles one (i, direction) pair.
                     * __syncthreads() between each k-step so that
                     * g[i-1, k] written by thread(i-1) is visible to thread(i).
                     */
                    if (lkl > 0) {
                        /* TRR: extend g-array from (lij, 0) to (lij, lkl) in k.
                         * g[i,k+1] = cpx*g[i,k] + k*b01*g[i,k-1] + i*b00*g[i-1,k]
                         *
                         * Parallelise over lij3 = (lij+1)*3 items (i=0..lij, nx=0..2).
                         * __syncthreads() is kept UNCONDITIONAL (outside divergent
                         * branches) to avoid CUDA warp-divergence deadlock.
                         */
                        int lij3 = (lij + 1) * 3;
                        /* Pass 0: write k=1 layer using VRR k=0 data.
                         * VRR __syncthreads() already ensures g[k=0] is visible. */
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
                        /* Passes 1..lkl-1: write k+1 layer using k and k-1 layers.
                         * __syncthreads() is unconditional so all threads participate. */
                        for (int k = 1; k < lkl; k++) {
                            __syncthreads();  /* g[k*stride_k] must be visible */
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
                        __syncthreads();  /* all k layers written; HRR may read them */
                    }

                    /* ---- HRR_ij: (lij,0) -> (li,lj) via rjri ----
                     * g[i,j+1,k,l] = g[i+1,j,k,l] - rjri * g[i,j,k,l]
                     * Parallelise over (k-range * 3 directions).
                     */
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

                    /* ---- HRR_kl: (lkl,0) -> (lk,ll) via rlrk ----
                     * g[i,j,k,l+1] = g[i,j,k+1,l] - rlrk * g[i,j,k,l]
                     * Parallelise over (i-offset-in-stride_k * 3 directions).
                     */
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

                    /* ---- Accumulate ip1 derivative into gout ----
                     * PySCF int2e_ip1 = d/dr1 convention:
                     *   f_x = -2*ai * g[ix+1,...] + ix * g[ix-1,...]
                     * (Note: d/dA = +2ai ... ; d/dr1 = -d/dA => negate.)
                     * No per-component normalization needed: SHELL_NORM
                     * (applied in fac_kl) handles the full correction.
                     */
                    for (int n = 0; n < GWIDTH; n++) {
                        int ijkl = gout_start + n * gout_stride + gout_id;
                        if (ijkl >= nf) break;

                        /* decompose flat index -> (fi, fj, fk, fl) */
                        int jkl = ijkl / nfi,  fi = ijkl % nfi;
                        int kl  = jkl  / nfj,  fj = jkl  % nfj;
                        int fl  = kl   / nfk,  fk = kl   % nfk;

                        /* skip upper triangle (kl aosym='s2kl') */
                        int fk_abs = k0 + fk, fl_abs = l0 + fl;
                        if (ksh == lsh && fl_abs > fk_abs) continue;

                        /* Cartesian components of each function */
                        int ix=CART_XYZ[off_i+fi*3+0], iy=CART_XYZ[off_i+fi*3+1], iz=CART_XYZ[off_i+fi*3+2];
                        int jx=CART_XYZ[off_j+fj*3+0], jy=CART_XYZ[off_j+fj*3+1], jz=CART_XYZ[off_j+fj*3+2];
                        int kx=CART_XYZ[off_k+fk*3+0], ky=CART_XYZ[off_k+fk*3+1], kz=CART_XYZ[off_k+fk*3+2];
                        int lx=CART_XYZ[off_l+fl*3+0], ly=CART_XYZ[off_l+fl*3+1], lz=CART_XYZ[off_l+fl*3+2];

                        int addrx = ix + jx*stride_j + kx*stride_k + lx*stride_l;
                        int addry = iy + jy*stride_j + ky*stride_k + ly*stride_l + g_size;
                        int addrz = iz + jz*stride_j + kz*stride_k + lz*stride_l + g_size*2;

                        /* ip1 nabla with PySCF d/dr1 sign convention */
                        double fx = -ai2 * gx[addrx + 1];
                        double fy = -ai2 * gx[addry + 1];
                        double fz = -ai2 * gx[addrz + 1];
                        if (ix > 0) fx += (double)ix * gx[addrx - 1];
                        if (iy > 0) fy += (double)iy * gx[addry - 1];
                        if (iz > 0) fz += (double)iz * gx[addrz - 1];

                        goutx[n] += fx * gx[addry] * gx[addrz];
                        gouty[n] += fy * gx[addrx] * gx[addrz];
                        goutz[n] += fz * gx[addrx] * gx[addry];
                    }
                    __syncthreads();   /* ready for next irys */
                } /* irys */
            } /* ijp */
        } /* klp */

        /* ---- Write accumulated gout to eri1 ----
         * eri1 layout: (3, nfi_block, nao, nao_pair)
         * [comp * nfi_block * nao * nao_pair + fi_local * nao * nao_pair
         *  + fj_abs * nao_pair + kl_packed]                                */
        long long comp_stride = (long long)nfi_block * nao * nao_pair;
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

            eri1[base]                  = goutx[n];
            eri1[base + comp_stride]    = gouty[n];
            eri1[base + 2*comp_stride]  = goutz[n];
        }
        __syncthreads();   /* ready for next gout_start batch */

    } /* gout_start */
}
"""


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


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_kernel_cache = {}


def get_kernel():
    """Return the compiled CuPy RawKernel (JIT compiled on first call)."""
    device = cupy.cuda.Device().id
    if device in _kernel_cache:
        return _kernel_cache[device]

    rys_data = _load_or_extract_rys_data()
    src = _build_cuda_source(rys_data)

    mod = cupy.RawModule(
        code=src,
        options=("--std=c++14",),
        name_expressions=["fill_int2e_ip1_kernel"],
    )
    kern = mod.get_function("fill_int2e_ip1_kernel")
    _kernel_cache[device] = kern
    return kern
