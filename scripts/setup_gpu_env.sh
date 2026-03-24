#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead of executing it:"
    echo "  source scripts/setup_gpu_env.sh"
    exit 1
fi

_ccsd_gpu_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ccsd_gpu_repo_root="$(cd "${_ccsd_gpu_script_dir}/.." && pwd)"
_ccsd_gpu_venv_python="${_ccsd_gpu_repo_root}/.venv/bin/python"

if [[ ! -x "${_ccsd_gpu_venv_python}" ]]; then
    echo "Missing virtualenv Python at ${_ccsd_gpu_venv_python}"
    echo "Run 'uv sync' first."
    return 1
fi

_ccsd_gpu_site_packages="$("${_ccsd_gpu_venv_python}" - <<'PY'
import site

for path in site.getsitepackages():
    if path.endswith("site-packages"):
        print(path)
        break
else:
    raise SystemExit("Could not locate site-packages")
PY
)"

_ccsd_gpu_nvidia_root="${_ccsd_gpu_site_packages}/nvidia"
if [[ ! -d "${_ccsd_gpu_nvidia_root}" ]]; then
    echo "Missing NVIDIA wheel libraries under ${_ccsd_gpu_nvidia_root}"
    echo "Run 'uv sync' first."
    return 1
fi

export CUDA_PATH="${_ccsd_gpu_nvidia_root}/cuda_runtime"
export CUDA_HOME="${CUDA_PATH}"
export LD_LIBRARY_PATH="${_ccsd_gpu_nvidia_root}/cublas/lib:${_ccsd_gpu_nvidia_root}/cusolver/lib:${_ccsd_gpu_nvidia_root}/cusparse/lib:${_ccsd_gpu_nvidia_root}/cuda_nvrtc/lib:${_ccsd_gpu_nvidia_root}/cuda_runtime/lib:${_ccsd_gpu_nvidia_root}/nvjitlink/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="${_ccsd_gpu_nvidia_root}/cuda_runtime/lib/libcudart.so.12:${_ccsd_gpu_nvidia_root}/cublas/lib/libcublas.so.12:${_ccsd_gpu_nvidia_root}/cublas/lib/libcublasLt.so.12:${_ccsd_gpu_nvidia_root}/cusparse/lib/libcusparse.so.12:${_ccsd_gpu_nvidia_root}/cusolver/lib/libcusolver.so.11:${_ccsd_gpu_nvidia_root}/cuda_nvrtc/lib/libnvrtc.so.12:${_ccsd_gpu_nvidia_root}/nvjitlink/lib/libnvJitLink.so.12${LD_PRELOAD:+:${LD_PRELOAD}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

echo "Configured CUDA env for ccsd_gpu"
echo "CUDA_PATH=${CUDA_PATH}"
