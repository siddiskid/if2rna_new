#!/usr/bin/env python3
"""Run ROSIE evaluate.py over prepared H&E PNG inputs.

This script is a thin wrapper around ROSIE's evaluate.py to keep invocation
consistent within this repo.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROSIE conversion")
    parser.add_argument("--rosie_dir", default="models/rosie", help="Directory containing ROSIE code and weights")
    parser.add_argument("--input_dir", required=True, help="Input RGB image directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for predicted TIFFs")
    parser.add_argument("--model_path", default=None, help="ROSIE model path (defaults to <rosie_dir>/best_model_single.pth)")
    parser.add_argument("--postprocess_image", default=None, help="Optional ROSIE postprocess flag value")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    rosie_dir = (repo_root / args.rosie_dir).resolve() if not Path(args.rosie_dir).is_absolute() else Path(args.rosie_dir).resolve()
    eval_py = rosie_dir / "evaluate.py"
    if not eval_py.exists():
        print(f"ERROR: evaluate.py not found at {eval_py}")
        sys.exit(1)

    model_path = (
        (repo_root / args.model_path).resolve()
        if args.model_path and not Path(args.model_path).is_absolute()
        else (Path(args.model_path).resolve() if args.model_path else (rosie_dir / "best_model_single.pth").resolve())
    )
    if not model_path.exists():
        print(f"ERROR: model weights not found at {model_path}")
        sys.exit(1)

    input_dir = (repo_root / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()

    cmd = [
        sys.executable,
        "evaluate.py",
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--model_path", str(model_path),
    ]
    if args.postprocess_image is not None:
        cmd.extend(["--postprocess_image", args.postprocess_image])

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(rosie_dir))


if __name__ == "__main__":
    main()
