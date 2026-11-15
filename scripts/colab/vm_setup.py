"""Prepare a Colab VM for training: unpack data and code, install deps, write bsmaker.toml.

Runs on the VM with the Python standard library only (it must not import bsmaker).
Order: base tar -> delta tar -> feature check -> code zip -> pip -> config. Safe to run twice.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

BASE_TAR = "bsmaker-train-data.tar"
DELTA_TAR = "bsmaker-train-delta.tar"
CODE_ZIP = "bsmaker-code.zip"
PIP_PACKAGES = ["librosa", "soundfile", "typer", "httpx"]


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract a tar into dest, refusing absolute/'..' paths and never creating links."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for m in tf:
            parts = Path(m.name.replace("\\", "/")).parts
            if m.name.startswith(("/", "\\")) or Path(m.name).is_absolute() or ".." in parts:
                raise SystemExit(f"unsafe path in {tar_path.name}: {m.name}")
            if m.issym() or m.islnk():
                raise SystemExit(f"link member in {tar_path.name}: {m.name}")
            tf.extract(m, dest, filter="data")


def unpack_base(tar: Path, work: Path) -> None:
    data, part = work / "data", work / "data.partial"
    marker = data / ".base.done"
    if marker.is_file() and not part.exists():
        print("base data already unpacked (.base.done), skipping")
        return
    for d in (data, part):  # leftovers of an interrupted or marker-less unpack
        shutil.rmtree(d, ignore_errors=True)
    print(f"unpacking {tar.name} ...", flush=True)
    safe_extract(tar, part)
    (part / ".base.done").write_text("ok")  # the rename below is the commit point
    part.rename(data)


def unpack_delta(tar: Path, work: Path) -> None:
    data = work / "data"
    marker = data / ".delta.done"
    st = tar.stat()
    stamp = f"{st.st_size} {st.st_mtime_ns}"  # a replaced delta tar differs in size or mtime
    if marker.is_file() and marker.read_text() == stamp:
        print("delta data already unpacked (.delta.done), skipping")
        return
    print(f"unpacking {tar.name} ...", flush=True)
    safe_extract(tar, data)
    marker.write_text(stamp)


def unpack_code(zip_path: Path, work: Path) -> Path:
    code = work / "bsmaker"
    shutil.rmtree(code, ignore_errors=True)
    code.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise SystemExit(f"unsafe path in {zip_path.name}: {name}")
        zf.extractall(code)
    return code


def check_features(data: Path) -> None:
    """Exit non-zero when an index entry has no features/<map_key>.npy."""
    index = data / "events" / "index.json"
    if not index.is_file():
        return
    keys = [str(r["map_key"]) for r in json.loads(index.read_text("utf-8"))]
    missing = [k for k in keys if not (data / "features" / f"{k}.npy").is_file()]
    if missing:
        print(f"vm_setup: {len(missing)} of {len(keys)} index entries have no feature file; "
              f"first {min(10, len(missing))}: {', '.join(missing[:10])}", flush=True)
        raise SystemExit(1)


def run_streamed(cmd: list[str]) -> int:
    """Run a child process, re-printing its output line by line (no inherited stdout)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = shlex.split(os.environ["BSM_ARGS"]) if "BSM_ARGS" in os.environ else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/BeatSaberMaker"))
    ap.add_argument("--work", type=Path, default=Path("/content"))
    ap.add_argument("--no-pip", action="store_true", help="skip the pip install")
    ap.add_argument("--no-delta", action="store_true", help="allow a missing delta tar")
    args = ap.parse_args(argv)
    drive, work = args.drive_dir, args.work

    base, delta, zip_path = drive / BASE_TAR, drive / DELTA_TAR, drive / CODE_ZIP
    if not base.is_file():
        raise SystemExit(f"missing {base} (is Drive mounted? is --drive-dir right?)")
    if not zip_path.is_file():
        raise SystemExit(f"missing {zip_path}")
    use_delta = not args.no_delta
    if use_delta and not delta.is_file():
        raise SystemExit(f"missing {delta} (looked in {drive}; pass --no-delta to run without it)")

    work.mkdir(parents=True, exist_ok=True)
    unpack_base(base, work)
    if use_delta:
        unpack_delta(delta, work)
    data = work / "data"
    check_features(data)
    code = unpack_code(zip_path, work)  # dataset_stats.json comes from the code zip only
    if not args.no_pip:
        code_ = run_streamed([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES])
        if code_ != 0:
            print(f"vm_setup: pip failed with exit code {code_}", flush=True)
            raise SystemExit(code_)
    (work / "none").mkdir(exist_ok=True)
    (code / "bsmaker.toml").write_text(
        f"data_dir = '{data.as_posix()}'\ncustom_levels = '{(work / 'none').as_posix()}'\n")

    index = data / "events" / "index.json"
    n_index = "missing"
    if index.is_file():
        n_index = str(len(json.loads(index.read_text("utf-8"))))
    commit = code / "BUNDLE_COMMIT.txt"
    print(f"feature files (.npy): {sum(1 for _ in (data / 'features').glob('*.npy'))}")
    print(f"index entries: {n_index}")
    print(f"bundle commit: {commit.read_text().strip() if commit.is_file() else 'unknown'}")
    print(f"free disk: {shutil.disk_usage(work).free / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    main()
