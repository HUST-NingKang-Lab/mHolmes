from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path
from importlib import resources


def _abs(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _run_packaged_script(filename: str) -> None:
    script_ref = resources.files("mHolmes").joinpath("scripts", filename)
    with resources.as_file(script_ref) as script_path:
        runpy.run_path(str(script_path), run_name="__main__")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="mhm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="mHolmes_foundation.py")
    p.add_argument("example_csv")
    p.add_argument("--export_path", required=True)

    p = sub.add_parser("tlpredict", help="mHolmes_transfer.py")
    p.add_argument("source_csv")
    p.add_argument("target_csv")
    p.add_argument("--export_path", required=True)

    p = sub.add_parser("shap", help="mHolmes_shap.py")
    p.add_argument("example_csv")
    p.add_argument("--export_path", required=True)

    p = sub.add_parser("tlshap", help="mHolmes_tlshap.py")
    p.add_argument("source_csv")
    p.add_argument("target_csv")
    p.add_argument("--export_path", required=True)

    p = sub.add_parser("mae", help="mHolmes_mae.py")
    p.add_argument("example_csv")
    p.add_argument("--export_path", required=True)

    p = sub.add_parser("tlmae", help="mHolmes_tlmae.py")
    p.add_argument("source_csv")
    p.add_argument("target_csv")
    p.add_argument("--export_path", required=True)

    p = sub.add_parser("mask", help="mHolmes_mask_exval.py")
    p.add_argument("source_csv")
    p.add_argument("target_csv")
    p.add_argument("external_val_csv")
    p.add_argument("--export_path", required=True)

    args = parser.parse_args(argv)

    # output directory for all commands
    os.environ["MHOLMES_EXPORT_PATH"] = _abs(args.export_path)

    # inputs
    if args.cmd in ("predict", "shap", "mae"):
        os.environ["MHOLMES_INPUT_CSV"] = _abs(args.example_csv)
    elif args.cmd in ("tlpredict", "tlshap", "tlmae"):
        os.environ["MHOLMES_SOURCE_CSV"] = _abs(args.source_csv)
        os.environ["MHOLMES_TARGET_CSV"] = _abs(args.target_csv)
    elif args.cmd == "mask":
        os.environ["MHOLMES_SOURCE_CSV"] = _abs(args.source_csv)
        os.environ["MHOLMES_TARGET_CSV"] = _abs(args.target_csv)
        os.environ["MHOLMES_EXTERNAL_CSV"] = _abs(args.external_val_csv)

    script_map = {
        "predict": "mHolmes_foundation.py",
        "tlpredict": "mHolmes_transfer.py",
        "shap": "mHolmes_shap.py",
        "tlshap": "mHolmes_tlshap.py",
        "mae": "mHolmes_mae.py",
        "tlmae": "mHolmes_tlmae.py",
        "mask": "mHolmes_mask_exval.py",
    }
    _run_packaged_script(script_map[args.cmd])
