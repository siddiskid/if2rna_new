#!/usr/bin/env python3
"""Download ROSIE model repo from Hugging Face (gated).

Usage:
  python scripts/download_rosie_model.py --output_dir models/rosie
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ROSIE from Hugging Face")
    parser.add_argument("--repo_id", default="ericwu09/ROSIE", help="Hugging Face repo id")
    parser.add_argument("--output_dir", default="models/rosie", help="Local output directory")
    parser.add_argument("--token", default=None, help="HF token (optional; otherwise uses cached login)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(out),
            local_dir_use_symlinks=False,
            token=args.token,
        )
        print(f"Downloaded ROSIE repo to: {path}")
    except GatedRepoError:
        print("ERROR: ROSIE is a gated model repo.")
        print("1. Open https://huggingface.co/ericwu09/ROSIE and click 'Acknowledge license'.")
        print("2. Wait for access approval (repo says this may take 2-3 days).")
        print("3. Run: huggingface-cli login")
        print("4. Retry this script.")
        sys.exit(1)
    except RepositoryNotFoundError:
        print(f"ERROR: Repo not found: {args.repo_id}")
        sys.exit(1)
    except HfHubHTTPError as e:
        print(f"ERROR: Hugging Face download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
