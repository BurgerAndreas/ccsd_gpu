"""Central GPU/PySCF availability detection.

Provides:
    has_gpu4pyscf : bool - True when gpu4pyscf is importable and usable.
    cc            : module - gpu4pyscf.cc.ccsd_incore if available, else pyscf.cc.
    mp            : module - gpu4pyscf.mp if available, else pyscf.mp.
    pyscf_cc      : module - always pyscf.cc (needed for lambda equations / gradients).
"""

from pyscf import cc as pyscf_cc
from pyscf import mp as pyscf_mp

has_gpu4pyscf = False
cc = pyscf_cc
mp = pyscf_mp

try:
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
