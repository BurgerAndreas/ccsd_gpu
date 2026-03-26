"""Generate a reusable post-CCSD checkpoint for lambda benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from lambda_checkpoint import build_molecule, build_reference_ccsd, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", default="aniline")
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument(
        "--output",
        default="benchmark_data/aniline_ccpvtz_lambda_checkpoint.npz",
    )
    args = parser.parse_args()

    out = Path(args.output)
    print(f"generating checkpoint molecule={args.molecule} basis={args.basis} output={out}")
    mol = build_molecule(args.molecule, args.basis)
    print(f"nao={mol.nao_nr()}")
    mf, mycc = build_reference_ccsd(mol)
    print(f"reference_ccsd_done mode={getattr(mycc, '_benchmark_ccsd_mode', 'cpu')}")
    save_checkpoint(out, args.molecule, args.basis, mf, mycc)
    print(f"checkpoint_saved={out}")


if __name__ == "__main__":
    main()
