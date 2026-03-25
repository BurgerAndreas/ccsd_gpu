"""Central GPU/PySCF availability detection and CUDA library preloading.

Provides:
    has_gpu4pyscf : bool - True when gpu4pyscf is importable and usable.
    cc            : module - gpu4pyscf.cc.ccsd_incore if available, else pyscf.cc.
    mp            : module - gpu4pyscf.mp if available, else pyscf.mp.
    pyscf_cc      : module - always pyscf.cc (needed for lambda equations / gradients).
"""

import ctypes
import os

from pyscf import cc as pyscf_cc
from pyscf import mp as pyscf_mp

has_gpu4pyscf = False
cc = pyscf_cc
mp = pyscf_mp


def _existing_path(path):
    return path if path and os.path.exists(path) else None


def _candidate_nvidia_roots():
    roots = []
    env_root = os.environ.get("CCSD_GPU_NVIDIA_ROOT")
    if env_root:
        roots.append(env_root)
    try:
        import site

        for base in site.getsitepackages():
            roots.append(os.path.join(base, "nvidia"))
    except Exception:
        pass
    return roots


def _candidate_driver_roots():
    roots = []
    env_root = os.environ.get("CCSD_GPU_DRIVER_ROOT")
    if env_root:
        roots.append(env_root)
    roots.extend(
        [
            "/cm/local/apps/cuda/libs/current/lib64",
            "/lib/x86_64-linux-gnu",
        ]
    )
    return roots


def _preload_cuda_libs():
    lib_paths = []
    for root in _candidate_driver_roots():
        lib_paths.extend(
            [
                _existing_path(os.path.join(root, "libcuda.so.1")),
                _existing_path(os.path.join(root, "libnvidia-ml.so.1")),
            ]
        )
    for root in _candidate_nvidia_roots():
        lib_paths.extend(
            [
                _existing_path(os.path.join(root, "cuda_runtime", "lib", "libcudart.so.12")),
                _existing_path(os.path.join(root, "cublas", "lib", "libcublas.so.12")),
                _existing_path(os.path.join(root, "cublas", "lib", "libcublasLt.so.12")),
                _existing_path(os.path.join(root, "cusparse", "lib", "libcusparse.so.12")),
                _existing_path(os.path.join(root, "cusolver", "lib", "libcusolver.so.11")),
                _existing_path(os.path.join(root, "cuda_nvrtc", "lib", "libnvrtc.so.12")),
                _existing_path(
                    os.path.join(root, "cuda_nvrtc", "lib", "libnvrtc-builtins.so.12.9")
                ),
                _existing_path(os.path.join(root, "nvjitlink", "lib", "libnvJitLink.so.12")),
            ]
        )

    seen = set()
    for lib_path in lib_paths:
        if not lib_path or lib_path in seen:
            continue
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        seen.add(lib_path)

try:
    _preload_cuda_libs()
    import cupy

    if cupy.cuda.runtime.getDeviceCount() <= 0:
        raise ImportError("No GPU found")

    from gpu4pyscf import mp as _gpu_mp
    from gpu4pyscf.cc import ccsd_incore as _gpu_cc

    has_gpu4pyscf = True
    cc = _gpu_cc
    mp = _gpu_mp
except (ImportError, Exception) as e:
    print(
        f"-\ngpu4pyscf not available ({type(e).__name__}), proceeding with regular pyscf.",
        flush=True,
    )
    print("-")
