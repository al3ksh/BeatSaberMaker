"""Build bsmaker-code.zip (tracked files at HEAD) for the Colab training notebook."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PATHS = ["src", "pyproject.toml", "dataset_stats.json", "playability.toml"]
DEFAULT_OUT = Path(r"F:\BeatSaberMakerData\upload\bsmaker-code.zip")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout


def build_bundle(repo: Path, out: Path) -> str:
    """Write the bundle; returns the HEAD commit hash.

    Raises RuntimeError on uncommitted changes in, or untracked, bundled paths.
    """
    dirty = _git(repo, "status", "--porcelain", "--", *PATHS).decode("utf-8").splitlines()
    if dirty:
        raise RuntimeError("\n".join(["uncommitted changes in bundled paths:", *dirty]))
    tracked = set(_git(repo, "ls-files", "--", *PATHS).decode("utf-8").splitlines())
    for path in PATHS:
        if path not in tracked and not any(t.startswith(path + "/") for t in tracked):
            raise RuntimeError(f"bundled path not tracked by git: {path}")
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = _git(repo, "archive", "--format=zip", "HEAD", "--", *PATHS)
    tmp = out.with_suffix(".tmp")
    try:
        tmp.write_bytes(archive)
        with zipfile.ZipFile(tmp, "a") as zf:
            zf.writestr("BUNDLE_COMMIT.txt", commit + "\n")
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return commit


def _keys(index: Path) -> set[str]:
    return {str(r["map_key"]) for r in json.loads(index.read_text("utf-8"))}


def build_delta(base_index: Path, data_dir: Path, out_dir: Path) -> tuple[Path, int]:
    """Write bsmaker-train-delta.tar; returns (path, number of new keys).

    Holds events/ (whole) and features/<key>.npy + .json (dataset_stats.json travels in the code
    zip only)
    for every map key in data_dir/events/index.json that is absent from base_index.
    """
    out = out_dir / "bsmaker-train-delta.tar"
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    new_keys = sorted(_keys(data_dir / "events" / "index.json") - _keys(base_index))
    files: list[Path] = []
    for key in new_keys:
        for ext in (".npy", ".json"):
            f = data_dir / "features" / f"{key}{ext}"
            if not f.is_file():
                raise RuntimeError(f"feature file missing for map key {key}: {f}")
            files.append(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with tarfile.open(tmp, "w") as tf:
            tf.add(data_dir / "events", arcname="events", recursive=False)
            for f in sorted((data_dir / "events").iterdir()):
                if (f.is_file() and not f.is_symlink() and not f.name.startswith(".")
                        and not f.name.endswith((".tmp", ".partial"))):
                    tf.add(f, arcname=f"events/{f.name}")
            for f in files:
                tf.add(f, arcname=f"features/{f.name}")
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return out, len(new_keys)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--delta", action="store_true", help="build the delta data tar instead")
    ap.add_argument("--base-index", type=Path, help="events/index.json of the base tar")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    if args.delta:
        if not (args.base_index and args.data_dir and args.out_dir):
            ap.error("--delta needs --base-index, --data-dir and --out-dir")
        try:
            out, n = build_delta(args.base_index, args.data_dir, args.out_dir)
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            print(e, file=sys.stderr)
            return 1
        print(f"{out}  {n} new maps ({2 * n} feature files)  {out.stat().st_size / 1e6:.2f} MB")
        return 0
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = build_bundle(repo, args.out)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        msg = str(e).splitlines()
        print(msg[0] if msg else repr(e), file=sys.stderr)
        return 1
    print(f"{args.out}  {args.out.stat().st_size / 1e6:.2f} MB  commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
