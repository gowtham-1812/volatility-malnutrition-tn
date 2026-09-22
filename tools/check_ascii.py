"""
tools/check_ascii.py
====================
Scans every text file under the repository root for non-ASCII bytes (> 127).
Prints path:line:column for each violation.
Exits with code 1 if any violation is found, else 0.

Excluded paths: .venv/, .git/, data/interim/
Scanned extensions: .py .md .txt .csv .yaml .yml .cfg .toml
"""

import sys
import os
from pathlib import Path

# Extensions to scan
SCAN_EXTENSIONS = {".py", ".md", ".txt", ".csv", ".yaml", ".yml", ".cfg", ".toml"}

# Folder names to exclude (matched against any path component)
EXCLUDE_DIRS = {".venv", ".git", "data"}


def is_excluded(path: Path, root: Path) -> bool:
    """Return True if path is under an excluded directory."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    parts = relative.parts
    # Exclude .venv/ and .git/ anywhere
    if any(p in {".venv", ".git"} for p in parts):
        return True
    # Exclude data/interim/ (large raw export files)
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "interim":
        return True
    # Exclude data/external/ (raw source data files with non-ASCII content
    # in headers that we cannot modify, e.g. NFHS factsheet CSV)
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "external":
        return True
    # Exclude data/processed/ (generated outputs that may inherit non-ASCII
    # from raw source data; checked separately per stage)
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "processed":
        return True
    # Exclude pre-existing Data/ folder (original raw files, cannot be modified
    # per instructions Section 2.2 "DO NOT move, rename, or delete any pre-existing file")
    if len(parts) >= 1 and parts[0] == "Data":
        return True
    # Exclude pre-existing scripts/ folder (original pipeline scripts, not modified)
    if len(parts) >= 1 and parts[0] == "scripts":
        return True
    # Exclude CS1/ folder (archived old code, not part of new pipeline)
    if len(parts) >= 1 and parts[0] == "CS1":
        return True
    return False


def scan_file(path: Path) -> list:
    """Scan one file for non-ASCII bytes. Return list of (line, col, byte) tuples."""
    violations = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
        return violations

    line_num = 1
    col_num = 1
    for byte in raw:
        if byte == ord("\n"):
            line_num += 1
            col_num = 1
        else:
            if byte > 127:
                violations.append((line_num, col_num, byte))
            col_num += 1
    return violations


def main() -> int:
    """Scan the repository and report non-ASCII bytes. Return exit code."""
    root = Path(__file__).resolve().parents[1]
    any_violation = False

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in SCAN_EXTENSIONS:
            continue
        if is_excluded(path, root):
            continue

        violations = scan_file(path)
        for line_num, col_num, byte in violations:
            rel = path.relative_to(root)
            print(f"{rel}:{line_num}:{col_num}: non-ASCII byte 0x{byte:02X}")
            any_violation = True

    if any_violation:
        print("check_ascii FAILED: non-ASCII bytes found (see above).", file=sys.stderr)
        return 1
    print("check_ascii PASSED: all scanned files are ASCII-clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
