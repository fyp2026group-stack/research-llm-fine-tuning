"""Save installed dependency versions under ignored artifacts (not a lock file)."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/environment/pip_freeze.txt"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    args.out.write_text(output, encoding="utf-8")
    print(f"Environment snapshot written to {args.out} (ignored by Git)")
    print("Review the file for paths or secrets before sharing.")


if __name__ == "__main__":
    main()
