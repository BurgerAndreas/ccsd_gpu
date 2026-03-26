"""Benchmark CPU and GPU lambda from a saved post-CCSD checkpoint."""

from __future__ import annotations

import argparse
import time

import numpy as np

from lambda_checkpoint import clone_cc, clone_cc_gpu, load_checkpoint
from ccsd_gpu.gpu_ccsd_lambda import solve_lambda_gpu


def build_cc_object(mf, ref, cc_object):
    if cc_object == "gpu":
        return clone_cc_gpu(mf, ref)
    return clone_cc(mf, ref)


def run_cpu_lambda(cpu, eris):
    t0 = time.perf_counter()
    cpu.solve_lambda(eris=eris)
    dt = time.perf_counter() - t0
    return cpu, dt


def run_gpu_lambda(
    gpu,
    eris,
    use_cpu_vvvv=False,
    vvvv_strategy=None,
    free_gpu_cache=None,
):
    if free_gpu_cache is not None:
        gpu._lambda_free_gpu_cache = free_gpu_cache
    t0 = time.perf_counter()
    solve_lambda_gpu(gpu, eris=eris, use_cpu_vvvv=use_cpu_vvvv, vvvv_strategy=vvvv_strategy)
    dt = time.perf_counter() - t0
    return gpu, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="benchmark_data/aniline_ccpvtz_lambda_checkpoint.npz",
    )
    parser.add_argument("--mode", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument(
        "--vvvv-strategy",
        choices=["auto", "dense-prepacked", "lambda-only", "full-helper"],
        default="auto",
    )
    parser.add_argument("--use-cpu-vvvv", action="store_true")
    parser.add_argument("--cc-object", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--free-gpu-cache",
        choices=["default", "true", "false"],
        default="default",
    )
    args = parser.parse_args()

    meta, mol, mf, ref = load_checkpoint(args.checkpoint)
    print(
        f"lambda checkpoint benchmark molecule={meta['molecule']} "
        f"basis={meta['basis']} nao={mol.nao_nr()}"
    )

    cpu = None
    if args.mode in ("cpu", "both"):
        cpu = build_cc_object(mf, ref, "cpu")
        cpu_eris = cpu.ao2mo()
        print("cpu_eris_built")
        cpu, cpu_time = run_cpu_lambda(cpu, cpu_eris)
        print(f"cpu_lambda_time={cpu_time:.3f}s")

    if args.mode in ("gpu", "both"):
        free_gpu_cache = None
        if args.free_gpu_cache != "default":
            free_gpu_cache = args.free_gpu_cache == "true"
        gpu = build_cc_object(mf, ref, args.cc_object)
        gpu_eris = gpu.ao2mo()
        print("gpu_eris_built")
        gpu, gpu_time = run_gpu_lambda(
            gpu,
            gpu_eris,
            use_cpu_vvvv=args.use_cpu_vvvv,
            vvvv_strategy=args.vvvv_strategy,
            free_gpu_cache=free_gpu_cache,
        )
        print(f"gpu_lambda_time={gpu_time:.3f}s")
        print(f"gpu_lambda_mode={getattr(gpu, '_lambda_solver_mode', 'unknown')}")
        print(f"gpu_cc_object={args.cc_object}")
        print(f"free_gpu_cache={args.free_gpu_cache}")
        if cpu is not None:
            l1_err = float(np.max(np.abs(np.asarray(cpu.l1) - np.asarray(gpu.l1))))
            l2_err = float(np.max(np.abs(np.asarray(cpu.l2) - np.asarray(gpu.l2))))
            print(f"lambda_l1_err={l1_err:.3e}")
            print(f"lambda_l2_err={l2_err:.3e}")
            print(f"gpu_speedup_vs_cpu={cpu_time / gpu_time:.3f}x")


if __name__ == "__main__":
    main()
