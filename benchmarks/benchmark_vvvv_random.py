"""Benchmark random-tensor vvvv contractions on GPU.

This isolates the dense contraction

    H[i,j,a,b] = einsum("ijcd,acdb->ijab", l2, vvvv)

from chemistry-specific AO-direct machinery so different tensor kernels can be
compared directly on random inputs.
"""

from __future__ import annotations

import argparse
import time

import cupy
import numpy as np


def _sync():
    cupy.cuda.get_current_stream().synchronize()


def _time_cuda(fn, repeat):
    times_ms = []
    out = None
    for _ in range(repeat):
        start = cupy.cuda.Event()
        end = cupy.cuda.Event()
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times_ms.append(cupy.cuda.get_elapsed_time(start, end))
    return out, np.array(times_ms)


def contract_einsum(l2, vvvv):
    return cupy.einsum("ijcd,acdb->ijab", l2, vvvv)


def contract_matmul(l2, vvvv):
    nocc = l2.shape[0]
    nvir = l2.shape[2]
    lhs = l2.reshape(nocc * nocc, nvir * nvir)
    rhs = vvvv.transpose(1, 2, 0, 3).reshape(nvir * nvir, nvir * nvir)
    return (lhs @ rhs).reshape(nocc, nocc, nvir, nvir)


def prepack_vvvv_rhs(vvvv):
    nvir = vvvv.shape[0]
    return cupy.asarray(
        vvvv.transpose(1, 2, 0, 3).reshape(nvir * nvir, nvir * nvir),
        order="C",
    )


def contract_matmul_prepacked(l2, rhs):
    nocc = l2.shape[0]
    nvir = l2.shape[2]
    lhs = l2.reshape(nocc * nocc, nvir * nvir)
    return (lhs @ rhs).reshape(nocc, nocc, nvir, nvir)


def contract_matmul_blocked(l2, vvvv, ab_block):
    nocc = l2.shape[0]
    nvir = l2.shape[2]
    lhs = l2.reshape(nocc * nocc, nvir * nvir)
    rhs = vvvv.transpose(1, 2, 0, 3).reshape(nvir * nvir, nvir * nvir)
    out = cupy.empty((nocc * nocc, nvir * nvir), dtype=l2.dtype)
    for ab0 in range(0, nvir * nvir, ab_block):
        ab1 = min(nvir * nvir, ab0 + ab_block)
        out[:, ab0:ab1] = lhs @ rhs[:, ab0:ab1]
    return out.reshape(nocc, nocc, nvir, nvir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nocc", type=int, default=8)
    parser.add_argument("--nvir", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--ab-block", type=int, default=4096)
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    rs = np.random.default_rng(args.seed)
    l2 = cupy.asarray(
        rs.standard_normal((args.nocc, args.nocc, args.nvir, args.nvir), dtype=dtype)
    )
    vvvv = cupy.asarray(
        rs.standard_normal((args.nvir, args.nvir, args.nvir, args.nvir), dtype=dtype)
    )

    print(
        f"random vvvv benchmark nocc={args.nocc} nvir={args.nvir} "
        f"dtype={args.dtype} repeat={args.repeat}"
    )
    free_b, total_b = cupy.cuda.runtime.memGetInfo()
    print(f"gpu_mem_free_gib={free_b / 1024**3:.2f} total_gib={total_b / 1024**3:.2f}")

    ref = contract_einsum(l2, vvvv)
    _sync()

    rhs_prepacked = prepack_vvvv_rhs(vvvv)

    candidates = {
        "einsum": lambda: contract_einsum(l2, vvvv),
        "matmul": lambda: contract_matmul(l2, vvvv),
        "matmul_prepacked": lambda: contract_matmul_prepacked(l2, rhs_prepacked),
        "matmul_blocked": lambda: contract_matmul_blocked(l2, vvvv, args.ab_block),
    }

    results = []
    for name, fn in candidates.items():
        out, times_ms = _time_cuda(fn, args.repeat)
        max_err = float(cupy.max(cupy.abs(out - ref)).get())
        mean_ms = float(times_ms.mean())
        min_ms = float(times_ms.min())
        results.append((name, mean_ms, min_ms, max_err))
        print(
            f"{name:16s} mean_ms={mean_ms:9.3f} min_ms={min_ms:9.3f} "
            f"max_err={max_err:.3e}"
        )

    best = min(results, key=lambda x: x[2])
    print(f"best_kernel={best[0]} best_min_ms={best[2]:.3f}")


if __name__ == "__main__":
    main()
