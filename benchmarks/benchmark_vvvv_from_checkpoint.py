"""Benchmark isolated lambda vvvv strategies from a saved post-CCSD checkpoint."""

from __future__ import annotations

import argparse
import time

import numpy as np

from lambda_checkpoint import clone_cc_gpu, load_checkpoint
from ccsd_gpu.gpu_ccsd_lambda import (
    _add_vvvv_gpu,
    _add_vvvv_gpu_lambda_only,
    _get_gpu_direct_vvvv,
    _to_cpu,
    _to_gpu,
    make_intermediates_gpu,
)


def _sync():
    try:
        import cupy

        cupy.cuda.get_current_stream().synchronize()
    except Exception:
        pass


def bench_cpu(mycc, eris, l2):
    t0 = time.perf_counter()
    out = mycc._add_vvvv(None, _to_cpu(l2), eris, with_ovvv=False, t2sym="jiba")
    return out, time.perf_counter() - t0


def bench_lambda_only(mycc, l2, imds):
    _sync()
    t0 = time.perf_counter()
    out = _add_vvvv_gpu_lambda_only(mycc, _to_gpu(l2), imds=imds)
    _sync()
    return out, time.perf_counter() - t0


def bench_full_helper(mycc, l2):
    direct_vvvv = _get_gpu_direct_vvvv()
    if direct_vvvv is False:
        raise RuntimeError("gpu4pyscf direct helper unavailable")
    import cupy

    l2_gpu = _to_gpu(l2)
    zero_t1 = cupy.zeros((l2.shape[0], l2.shape[2]), dtype=l2_gpu.dtype)
    _sync()
    t0 = time.perf_counter()
    _, _, out, _, _ = direct_vvvv(mycc, zero_t1, l2_gpu)
    _sync()
    return out, time.perf_counter() - t0


def bench_auto(mycc, l2, eris, imds):
    from pyscf.lib import logger

    log = logger.Logger(mycc.stdout, 0)
    _sync()
    t0 = time.perf_counter()
    out = _add_vvvv_gpu(mycc, _to_gpu(l2), eris, log, imds=imds)
    _sync()
    return out, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="benchmark_data/aniline_ccpvtz_lambda_checkpoint.npz",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cc-object", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--free-gpu-cache",
        choices=["default", "true", "false"],
        default="default",
    )
    args = parser.parse_args()

    meta, mol, mf, ref = load_checkpoint(args.checkpoint)
    print(
        f"vvvv checkpoint benchmark molecule={meta['molecule']} "
        f"basis={meta['basis']} nao={mol.nao_nr()}"
    )
    work_cc = ref
    if args.cc_object == "gpu":
        work_cc = clone_cc_gpu(mf, ref)
    eris = work_cc.ao2mo()
    l2 = np.asarray(ref.t2)
    if args.free_gpu_cache != "default":
        work_cc._lambda_free_gpu_cache = args.free_gpu_cache == "true"
    imds = make_intermediates_gpu(work_cc, ref.t1, ref.t2, eris)
    print("eris_and_imds_built")
    print(f"gpu_cc_object={args.cc_object}")
    print(f"free_gpu_cache={args.free_gpu_cache}")

    ref_cpu, dt_cpu = bench_cpu(ref, eris, l2)
    print(f"cpu_vvvv_time={dt_cpu:.3f}s")

    results = [("cpu", dt_cpu, 0.0)]
    for name, fn in (
        ("lambda_only", lambda: bench_lambda_only(work_cc, l2, imds)),
        ("full_helper", lambda: bench_full_helper(work_cc, l2)),
        ("auto", lambda: bench_auto(work_cc, l2, eris, imds)),
    ):
        times = []
        max_err = 0.0
        for _ in range(args.repeat):
            out, dt = fn()
            times.append(dt)
            max_err = max(max_err, float(np.max(np.abs(_to_cpu(out) - ref_cpu))))
        mean_dt = float(np.mean(times))
        best_dt = float(np.min(times))
        print(f"{name}_mean_time={mean_dt:.3f}s")
        print(f"{name}_best_time={best_dt:.3f}s")
        print(f"{name}_max_err={max_err:.3e}")
        results.append((name, best_dt, max_err))

    best = min(results[1:], key=lambda item: item[1])
    print(f"best_gpu_vvvv={best[0]} best_gpu_vvvv_time={best[1]:.3f}s")


if __name__ == "__main__":
    main()
