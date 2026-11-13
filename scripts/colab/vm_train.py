"""Run `bsmaker train <model>` on the Colab VM, logging to Drive. Standard library only.

Usage: vm_train.py --model timing|pattern --name RUN [--drive-dir D] [--work W] [--dry-run] -- <train options>
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def build_command(model: str, out: Path, extra: list[str]) -> list[str]:
    return [sys.executable, "-u", "-m", "bsmaker", "train", model, "--out", str(out), *extra]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["timing", "pattern"])
    ap.add_argument("--name", required=True, help="run name (folder under checkpoints/gen2)")
    ap.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/BeatSaberMaker"))
    ap.add_argument("--work", type=Path, default=Path("/content"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after `--`: options for bsmaker train")
    args = ap.parse_args(argv)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        ap.error("--name must match [A-Za-z0-9_.-]+")
    if any(a == "--out" or a.startswith("--out=") for a in extra):
        ap.error("do not pass --out; vm_train.py sets it")

    out = args.drive_dir / "checkpoints" / "gen2" / args.name
    cmd = build_command(args.model, out, extra)
    code_dir = args.work / "bsmaker"
    header = "$ " + " ".join(cmd)
    if args.dry_run:
        print(header)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(code_dir / "src"), PYTHONUNBUFFERED="1")
    with open(out / "console.log", "a", encoding="utf-8") as log:
        def emit(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        emit(header)
        proc = subprocess.Popen(cmd, cwd=code_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                emit(line.rstrip("\n"))
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        emit(f"vm_train: exit code {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
