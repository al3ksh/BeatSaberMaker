import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
from build_engine_pack import split_and_manifest  # noqa: E402
from fetch_tools import fetch_tools  # noqa: E402


def test_split_and_manifest_roundtrips_zip(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    (pack / "bsmaker-engine").mkdir(parents=True)
    (pack / "bsmaker-engine" / "bsmaker-engine.exe").write_bytes(b"A" * 1500)
    (pack / "playability.toml").write_text("[mask]\n", encoding="utf-8")

    out = tmp_path / "dist"
    manifest_path = split_and_manifest(pack, out, engine_version="1", max_part_bytes=600)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["engineVersion"] == "1"
    assert len(manifest["parts"]) >= 2
    assert all(p["size"] <= 600 for p in manifest["parts"])

    assembled = b"".join((out / p["name"]).read_bytes() for p in manifest["parts"])
    assert hashlib.sha256(assembled).hexdigest() == manifest["archiveSha256"]
    with zipfile.ZipFile(io.BytesIO(assembled)) as zf:
        assert "playability.toml" in zf.namelist()
        assert "bsmaker-engine/bsmaker-engine.exe" in zf.namelist()


def test_fetch_tools_verifies_sha256_and_extracts_ffmpeg(tmp_path: Path) -> None:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("ffmpeg-build/bin/ffmpeg.exe", b"FFMPEG")
        zf.writestr("ffmpeg-build/bin/ffprobe.exe", b"FFPROBE")
    zbytes = zbuf.getvalue()
    ytbytes = b"YTDLP"
    manifest = tmp_path / "tools.json"
    manifest.write_text(json.dumps({
        "yt_dlp": {"url": "https://example.test/yt-dlp.exe", "sha256": hashlib.sha256(ytbytes).hexdigest()},
        "ffmpeg_lgpl": {"url": "https://example.test/ffmpeg.zip", "sha256": hashlib.sha256(zbytes).hexdigest()},
    }), encoding="utf-8")

    blobs = {"https://example.test/yt-dlp.exe": ytbytes, "https://example.test/ffmpeg.zip": zbytes}
    res = fetch_tools(manifest, tmp_path / "tools", opener=lambda url: io.BytesIO(blobs[url]))
    assert res["ffmpeg"].read_bytes() == b"FFMPEG"
    assert res["ffprobe"].read_bytes() == b"FFPROBE"
    assert res["yt_dlp"].read_bytes() == b"YTDLP"

    manifest.write_text(json.dumps({
        "yt_dlp": {"url": "https://example.test/yt-dlp.exe", "sha256": "0" * 64},
        "ffmpeg_lgpl": {"url": "https://example.test/ffmpeg.zip", "sha256": hashlib.sha256(zbytes).hexdigest()},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        fetch_tools(manifest, tmp_path / "tools_bad", opener=lambda url: io.BytesIO(blobs[url]))
