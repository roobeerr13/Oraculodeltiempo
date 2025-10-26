"""Main entrypoint to run preprocessing and other pipeline phases.

Usage:
    python main.py

This will call the preprocessing pipeline implemented in
`scripts/preprocess_power_consumption.py` and produce outputs under `outputs/`.
"""
from __future__ import annotations
import sys

def main() -> int:
    try:
        from scripts.preprocess_power_consumption import main as preprocess_main
    except Exception as e:
        print("Failed to import preprocessing module:", e)
        return 2

    preprocess_main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
