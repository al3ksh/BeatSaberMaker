"""Fetch pinned yt-dlp and LGPL ffmpeg binaries with SHA256 verification."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def _default_opener(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "bsmaker-fetch-tools"})
    return urllib.request.urlopen(req, timeout=180)


def _download_verified(url: str, expected_sha256: str, opener) -> bytes:
    with opener(url) as fh:
        data = fh.read()
    got = hashlib.sha256(data).hexdigest()
    if got.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch for {url}: expected {expected_sha256}, got {got}"
        )
    return data


def fetch_tools(
    manifest_path: Path,
    out_dir: Path,
    *,
    opener=None,
) -> dict[str, Path]:
    """Download and verify yt-dlp.exe and LGPL ffmpeg.exe + ffprobe.exe into `out_dir`."""
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    open_fn = opener or _default_opener

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    yt_spec = manifest["yt_dlp"]
    ff_spec = manifest["ffmpeg_lgpl"]

    yt_bytes = _download_verified(yt_spec["url"], yt_spec["sha256"], open_fn)
    yt_out = out_dir / "yt-dlp.exe"
    yt_out.write_bytes(yt_bytes)

    ff_bytes = _download_verified(ff_spec["url"], ff_spec["sha256"], open_fn)
    ffmpeg_out = out_dir / "ffmpeg.exe"
    ffprobe_out = out_dir / "ffprobe.exe"

    found_ffmpeg = False
    found_ffprobe = False
    with zipfile.ZipFile(io.BytesIO(ff_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name.lower()
            if name == "ffmpeg.exe":
                with zf.open(member) as src, ffmpeg_out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                found_ffmpeg = True
            elif name == "ffprobe.exe":
                with zf.open(member) as src, ffprobe_out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                found_ffprobe = True

    if not found_ffmpeg or not found_ffprobe:
        raise RuntimeError("ffmpeg archive did not contain ffmpeg.exe and ffprobe.exe")

    return {"yt_dlp": yt_out, "ffmpeg": ffmpeg_out, "ffprobe": ffprobe_out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch pinned yt-dlp and LGPL ffmpeg tools.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "tools.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    res = fetch_tools(args.manifest, args.out_dir)
    print(f"fetched: {', '.join(f'{k}={v}' for k, v in res.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
