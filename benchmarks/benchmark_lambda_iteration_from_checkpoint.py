"""Benchmark one CPU/GPU lambda update iteration from a saved checkpoint."""

from __future__ import annotations

import argparse
import time

import numpy as np
from pyscf.cc import ccsd_lambda

from lambda_checkpoint import clone_cc, clone_cc_gpu, load_checkpoint
from ccsd_gpu.gpu_ccsd_lambda import make_intermediates_gpu, update_lambda_gpu


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
        f"lambda-iteration checkpoint benchmark molecule={meta['molecule']} "
        f"basis={meta['basis']} nao={mol.nao_nr()}"
    )
    if args.mode in ("cpu", "both"):
        cpu = clone_cc(mf, ref)
        eris_cpu = cpu.ao2mo()
        print("cpu_eris_built")
        t0 = time.perf_counter()
        imds_cpu = ccsd_lambda.make_intermediates(cpu, cpu.t1, cpu.t2, eris_cpu)
        cpu_imds_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        cpu_l1, cpu_l2 = ccsd_lambda.update_lambda(
            cpu, cpu.t1, cpu.t2, cpu.t1, cpu.t2, eris_cpu, imds_cpu
        )
        cpu_iter_time = time.perf_counter() - t0
        print(f"cpu_imds_time={cpu_imds_time:.3f}s")
        print(f"cpu_iter_time={cpu_iter_time:.3f}s")

    if args.mode in ("gpu", "both"):
        if args.cc_object == "gpu":
            gpu = clone_cc_gpu(mf, ref)
        else:
            gpu = clone_cc(mf, ref)
        eris_gpu = gpu.ao2mo()
        print("gpu_eris_built")
        gpu._lambda_vvvv_strategy = args.vvvv_strategy
        if args.free_gpu_cache != "default":
            gpu._lambda_free_gpu_cache = args.free_gpu_cache == "true"
        t0 = time.perf_counter()
        imds_gpu = make_intermediates_gpu(gpu, gpu.t1, gpu.t2, eris_gpu)
        gpu_imds_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        gpu_l1, gpu_l2 = update_lambda_gpu(
            gpu,
            gpu.t1,
            gpu.t2,
            gpu.t1,
            gpu.t2,
            eris_gpu,
            imds_gpu,
            use_cpu_vvvv=args.use_cpu_vvvv,
        )
        gpu_iter_time = time.perf_counter() - t0
        print(f"gpu_imds_time={gpu_imds_time:.3f}s")
        print(f"gpu_iter_time={gpu_iter_time:.3f}s")
        print(f"gpu_cc_object={args.cc_object}")
        print(f"gpu_vvvv_strategy={args.vvvv_strategy}")
        print(f"free_gpu_cache={args.free_gpu_cache}")
        if args.mode == "both":
            l1_err = float(np.max(np.abs(np.asarray(cpu_l1) - np.asarray(gpu_l1.get()))))
            l2_err = float(np.max(np.abs(np.asarray(cpu_l2) - np.asarray(gpu_l2.get()))))
            print(f"iter_l1_err={l1_err:.3e}")
            print(f"iter_l2_err={l2_err:.3e}")
            print(f"gpu_iter_speedup_vs_cpu={cpu_iter_time / gpu_iter_time:.3f}x")


if __name__ == "__main__":
    main()
