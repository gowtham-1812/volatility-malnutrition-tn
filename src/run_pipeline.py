"""
src/run_pipeline.py
===================
Master script to run the full pipeline sequentially.
"""

import sys
import argparse
from pathlib import Path
from src.config import INTERIM

def main():
    parser = argparse.ArgumentParser(description="Run the full food price volatility pipeline.")
    parser.add_argument("--skip-export", action="store_true", default=True,
                        help="Skip building the SQLite database (default: True).")
    parser.add_argument("--export", action="store_false", dest="skip_export",
                        help="Build the SQLite database.")
    args = parser.parse_args()

    # Check interim files
    required_interim = ["prices_raw.csv", "nfhs_raw.csv"]
    for f in required_interim:
        if not (INTERIM / f).exists():
            print(f"Error: Required file missing in data/interim/: {f}")
            sys.exit(1)

    print("=== Running Pipeline ===")

    import src.districts
    src.districts.main()

    import src.prices
    src.prices.main()

    import src.features
    src.features.main()

    import src.nfhs
    src.nfhs.main()

    import src.assemble
    src.assemble.main()

    if not args.skip_export:
        import src.build_sqlite
        src.build_sqlite.main()
    else:
        print("Skipping SQLite build (--skip-export is True)")

    import src.analysis
    src.analysis.main()

    import src.ml
    src.ml.main()

    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
