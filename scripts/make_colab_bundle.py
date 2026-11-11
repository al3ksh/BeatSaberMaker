"""Build bsmaker-code.zip (tracked files at HEAD) for the Colab training notebook."""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

PATHS = ["src", "pyproject.toml", "dataset_stats.json", "playability.toml"]
DEFAULT_OUT = Path(r"F:\BeatSaberMakerData\upload\bsmaker-code.zip")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout


def build_bundle(repo: Path, out: Path) -> str:
    """Write the bundle; returns the HEAD commit hash. Raises RuntimeError on a dirty src/."""
    dirty = _git(repo, "status", "--porcelain", "--", "src").decode("utf-8").splitlines()
    if dirty:
        raise RuntimeError("uncommitted changes under src/:\n" + "\n".join(dirty))
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = _git(repo, "archive", "--format=zip", "HEAD", "--", *PATHS)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(archive)
    with zipfile.ZipFile(tmp, "a") as zf:
        zf.writestr("BUNDLE_COMMIT.txt", commit + "\n")
    tmp.replace(out)
    return commit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = build_bundle(repo, args.out)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{args.out}  {args.out.stat().st_size / 1e6:.2f} MB  commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
