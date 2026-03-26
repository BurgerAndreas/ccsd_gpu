"""Summarize lambda benchmark key-value lines from Slurm logs."""

from __future__ import annotations

import argparse
from pathlib import Path


KEY_PREFIXES = (
    "cpu_lambda_time=",
    "gpu_lambda_time=",
    "gpu_lambda_mode=",
    "gpu_speedup_vs_cpu=",
    "lambda_l1_err=",
    "lambda_l2_err=",
    "cpu_imds_time=",
    "cpu_iter_time=",
    "gpu_imds_time=",
    "gpu_iter_time=",
    "gpu_iter_speedup_vs_cpu=",
    "gpu_vvvv_strategy=",
    "gpu_cc_object=",
    "free_gpu_cache=",
    "cpu_vvvv_time=",
    "lambda_only_mean_time=",
    "lambda_only_best_time=",
    "lambda_only_max_err=",
    "full_helper_mean_time=",
    "full_helper_best_time=",
    "full_helper_max_err=",
    "auto_mean_time=",
    "auto_best_time=",
    "auto_max_err=",
    "best_gpu_vvvv=",
    "best_gpu_vvvv_time=",
)


def summarize_log(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    picked = []
    for line in lines:
        if any(line.startswith(prefix) for prefix in KEY_PREFIXES):
            picked.append(line)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    for raw in args.logs:
        path = Path(raw)
        print(f"[{path}]")
        for line in summarize_log(path):
            print(line)
        print()


if __name__ == "__main__":
    main()
