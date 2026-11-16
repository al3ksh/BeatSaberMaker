"""Build PyInstaller engine binary, assemble engine pack, and split into <=1.9 GB parts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CHUNK_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PART_BYTES = 1_900_000_000


def split_and_manifest(
    pack_dir: Path,
    out_dir: Path,
    engine_version: str,
    *,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    chunk_bytes: int = CHUNK_BYTES,
) -> Path:
    """Zip `pack_dir`, stream-split into `<= max_part_bytes` parts, delete zip, write manifest."""
    pack_dir = Path(pack_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"bsmaker-engine-v{engine_version}.zip"
    archive_path = out_dir / archive_name

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for file_path in sorted(pack_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(pack_dir).as_posix())

    total_size = archive_path.stat().st_size
    archive_hash = hashlib.sha256()
    parts: list[dict] = []

    try:
        with archive_path.open("rb") as src:
            part_index = 1
            while True:
                part_name = f"{archive_name}.{part_index:03d}"
                part_path = out_dir / part_name
                part_hash = hashlib.sha256()
                part_size = 0

                with part_path.open("wb") as dst:
                    while part_size < max_part_bytes:
                        to_read = min(chunk_bytes, max_part_bytes - part_size)
                        buf = src.read(to_read)
                        if not buf:
                            break
                        dst.write(buf)
                        archive_hash.update(buf)
                        part_hash.update(buf)
                        part_size += len(buf)

                if part_size == 0:
                    part_path.unlink(missing_ok=True)
                    break

                parts.append({
                    "name": part_name,
                    "size": part_size,
                    "sha256": part_hash.hexdigest(),
                })
                part_index += 1
    finally:
        archive_path.unlink(missing_ok=True)

    manifest = {
        "engineVersion": engine_version,
        "archiveSha256": archive_hash.hexdigest(),
        "totalSize": total_size,
        "parts": parts,
    }
    manifest_path = out_dir / "engine-pack-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _read_engine_version(repo_root: Path) -> str:
    cfg = json.loads((repo_root / "app" / "engine-version.json").read_text(encoding="utf-8"))
    return str(cfg["engineVersion"])


def build_engine_pack(
    *,
    repo_root: Path,
    checkpoints_dir: Path,
    tools_dir: Path,
    out_dir: Path,
    engine_version: str | None = None,
    playability_path: Path | None = None,
    layer_stats_path: Path | None = None,
) -> Path:
    """Run PyInstaller, stage the engine pack, and emit split parts + manifest."""
    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    version = engine_version or _read_engine_version(repo_root)

    work_dir = out_dir / "_pyinstaller_work"
    dist_dir = out_dir / "_pyinstaller_dist"
    stage_dir = out_dir / "_pack_stage"
    for d in (work_dir, dist_dir, stage_dir):
        shutil.rmtree(d, ignore_errors=True)

    spec_path = repo_root / "packaging" / "bsmaker-engine.spec"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--workpath",
            str(work_dir),
            "--distpath",
            str(dist_dir),
            str(spec_path),
        ],
        check=True,
    )
    shutil.rmtree(work_dir, ignore_errors=True)

    try:
        shutil.move(str(dist_dir / "bsmaker-engine"), str(stage_dir / "bsmaker-engine"))
        shutil.rmtree(dist_dir, ignore_errors=True)

        models_dst = stage_dir / "models"
        shutil.copytree(checkpoints_dir, models_dst / "checkpoints")
        if playability_path and playability_path.is_file():
            shutil.copy2(playability_path, models_dst / "playability.toml")
        if layer_stats_path and layer_stats_path.is_file():
            shutil.copy2(layer_stats_path, models_dst / "layer_stats.json")

        shutil.copytree(tools_dir, stage_dir / "tools")
        return split_and_manifest(stage_dir, out_dir, version)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build split PyInstaller engine pack.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine-version", type=str, default=None)
    parser.add_argument("--playability", type=Path, default=None)
    parser.add_argument("--layer-stats", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = build_engine_pack(
        repo_root=args.repo_root,
        checkpoints_dir=args.checkpoints_dir,
        tools_dir=args.tools_dir,
        out_dir=args.out_dir,
        engine_version=args.engine_version,
        playability_path=args.playability,
        layer_stats_path=args.layer_stats,
    )
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
