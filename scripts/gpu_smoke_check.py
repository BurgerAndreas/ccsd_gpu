"""Minimal runtime check for CUDA availability on a Slurm node."""

import os
import subprocess

import cupy


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print("nvidia-smi:")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
    )

    count = cupy.cuda.runtime.getDeviceCount()
    print(f"device_count={count}")
    if count <= 0:
        raise RuntimeError("No CUDA-capable device visible to CuPy")
    print(f"runtime={cupy.cuda.runtime.runtimeGetVersion()}")
    print(f"driver={cupy.cuda.runtime.driverGetVersion()}")
    x = cupy.arange(1024, dtype=cupy.float64)
    print(f"sum={float(x.sum())}")


if __name__ == "__main__":
    main()
