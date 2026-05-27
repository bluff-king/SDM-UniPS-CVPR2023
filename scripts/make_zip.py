"""
Pack the training code into a single .zip for SCP upload to Vast.ai.

Excludes: .git, __pycache__, checkpoints, data, weights, demo, figures,
.images, tmp, .vscode, *.pyc, *.zip.

Output: <repo_parent>/sdm_lvc_code.zip

Usage:
    python scripts/make_zip.py
"""

from __future__ import annotations
import os
import sys
import zipfile
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(REPO), "sdm_lvc_code.zip")

# Dirs excluded ANYWHERE in the tree
EXCLUDE_DIRS_ANY = {
    ".git", "__pycache__",
    ".idea", "node_modules", ".pytest_cache", ".vscode",
}
# Dirs excluded ONLY at repo root (don't accidentally skip src/data/)
EXCLUDE_DIRS_ROOT = {
    "checkpoints", "data", "weights",
    "demo", "figures", ".images", "tmp", "Stable3DGen",
    "launch_sh", "dl",
}
EXCLUDE_SUFFIX = (".pyc", ".zip", ".pth", ".pytmodel", ".exr", ".png", ".jpg", ".jpeg", ".gif", ".mp4")


def should_skip(path_parts: tuple) -> bool:
    """Skip if any path component is in exclude list."""
    if not path_parts:
        return False
    # Any-level exclusions
    for part in path_parts:
        if part in EXCLUDE_DIRS_ANY:
            return True
        if part.startswith(".") and part not in (".gitignore",):
            return True
    # Root-level only exclusions
    if path_parts[0] in EXCLUDE_DIRS_ROOT:
        return True
    return False


def main():
    print(f"[make_zip] packing  {REPO}")
    print(f"[make_zip] output   {OUT}")

    if os.path.exists(OUT):
        os.remove(OUT)

    n_files = 0
    total_size = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(REPO):
            # Prune excluded dirs in-place
            rel = os.path.relpath(root, REPO)
            parts = tuple(rel.split(os.sep)) if rel != "." else ()
            if should_skip(parts):
                dirs[:] = []
                continue
            # Filter children: any-level exclusions + root-level at top
            new_dirs = []
            for d in dirs:
                if d in EXCLUDE_DIRS_ANY:
                    continue
                if d.startswith(".") and d != ".gitignore":
                    continue
                # Root-level exclusions only apply when we're AT the root
                if not parts and d in EXCLUDE_DIRS_ROOT:
                    continue
                new_dirs.append(d)
            dirs[:] = new_dirs

            for fn in files:
                if fn.lower().endswith(EXCLUDE_SUFFIX):
                    continue
                src = os.path.join(root, fn)
                arc = os.path.relpath(src, REPO).replace(os.sep, "/")
                try:
                    zf.write(src, arcname=arc)
                    n_files += 1
                    total_size += os.path.getsize(src)
                except OSError as e:
                    print(f"  WARN  cannot read {src}: {e}")

    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    raw_mb = total_size / (1024 * 1024)
    print(f"[make_zip] {n_files} files, raw {raw_mb:.2f} MB -> zip {size_mb:.2f} MB")
    print(f"[make_zip] done  {OUT}")


if __name__ == "__main__":
    main()
