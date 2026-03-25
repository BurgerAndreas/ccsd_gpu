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

_ccsd_gpu_driver_root="/cm/local/apps/cuda/libs/current/lib64"
if [[ ! -d "${_ccsd_gpu_driver_root}" ]]; then
    _ccsd_gpu_driver_root="/lib/x86_64-linux-gnu"
fi

_ccsd_gpu_join_by_colon() {
    local result=""
    local item
    for item in "$@"; do
        [[ -n "${item}" ]] || continue
        if [[ -z "${result}" ]]; then
            result="${item}"
        else
            result="${result}:${item}"
        fi
    done
    printf '%s' "${result}"
}

_ccsd_gpu_ld_paths=(
    "${_ccsd_gpu_driver_root}"
    "${_ccsd_gpu_nvidia_root}/cublas/lib"
    "${_ccsd_gpu_nvidia_root}/cusolver/lib"
    "${_ccsd_gpu_nvidia_root}/cusparse/lib"
    "${_ccsd_gpu_nvidia_root}/cuda_nvrtc/lib"
    "${_ccsd_gpu_nvidia_root}/cuda_runtime/lib"
    "${_ccsd_gpu_nvidia_root}/nvjitlink/lib"
)

_ccsd_gpu_preload_candidates=(
    "${_ccsd_gpu_driver_root}/libcuda.so.1"
    "${_ccsd_gpu_driver_root}/libnvidia-ml.so.1"
    "${_ccsd_gpu_nvidia_root}/cuda_runtime/lib/libcudart.so.12"
    "${_ccsd_gpu_nvidia_root}/cublas/lib/libcublas.so.12"
    "${_ccsd_gpu_nvidia_root}/cublas/lib/libcublasLt.so.12"
    "${_ccsd_gpu_nvidia_root}/cusparse/lib/libcusparse.so.12"
    "${_ccsd_gpu_nvidia_root}/cusolver/lib/libcusolver.so.11"
    "${_ccsd_gpu_nvidia_root}/cuda_nvrtc/lib/libnvrtc.so.12"
    "${_ccsd_gpu_nvidia_root}/cuda_nvrtc/lib/libnvrtc-builtins.so.12.9"
    "${_ccsd_gpu_nvidia_root}/nvjitlink/lib/libnvJitLink.so.12"
)

_ccsd_gpu_existing_ld_library_path="${LD_LIBRARY_PATH:-}"
_ccsd_gpu_existing_ld_preload="${LD_PRELOAD:-}"

_ccsd_gpu_preload_paths=()
for _ccsd_gpu_lib in "${_ccsd_gpu_preload_candidates[@]}"; do
    if [[ -f "${_ccsd_gpu_lib}" ]]; then
        _ccsd_gpu_preload_paths+=("${_ccsd_gpu_lib}")
    fi
done

export CUDA_PATH="${_ccsd_gpu_nvidia_root}/cuda_runtime"
export CUDA_HOME="${CUDA_PATH}"
export CCSD_GPU_NVIDIA_ROOT="${_ccsd_gpu_nvidia_root}"
export CCSD_GPU_DRIVER_ROOT="${_ccsd_gpu_driver_root}"
export LD_LIBRARY_PATH="$(_ccsd_gpu_join_by_colon "${_ccsd_gpu_ld_paths[@]}" "${_ccsd_gpu_existing_ld_library_path}")"
export LD_PRELOAD="$(_ccsd_gpu_join_by_colon "${_ccsd_gpu_preload_paths[@]}" "${_ccsd_gpu_existing_ld_preload}")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

ccsd_gpu_check_cuda_ready() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found in PATH"
        return 1
    fi

    echo "GPU visibility via nvidia-smi:"
    if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader; then
        echo "nvidia-smi could not communicate with the NVIDIA driver"
        return 1
    fi

    echo "GPU visibility via CuPy:"
    if ! uv run python - <<'PY'
import cupy

count = cupy.cuda.runtime.getDeviceCount()
print(f"device_count={count}")
if count <= 0:
    raise SystemExit("No CUDA device visible to CuPy")
print(f"runtime={cupy.cuda.runtime.runtimeGetVersion()}")
print(f"driver={cupy.cuda.runtime.driverGetVersion()}")
a = cupy.arange(8, dtype=cupy.float64)
print(f"sum={float(a.sum())}")
PY
    then
        echo "CuPy could not access a usable CUDA device"
        return 1
    fi
}

echo "Configured CUDA env for ccsd_gpu"
echo "CUDA_PATH=${CUDA_PATH}"
