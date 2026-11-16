import io
import json
import subprocess
from pathlib import Path
import pytest

from bsmaker.errors import GenerateCancelled
from bsmaker.engine.youtube import (
    DownloadError, YtdlpMissingError, download_youtube, is_youtube_url,
    parse_artist_title, require_ytdlp, resolve_youtube, search_youtube,
)


def test_is_youtube_url_variants() -> None:
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123")
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ?t=10")
    assert is_youtube_url("https://music.youtube.com/watch?v=abc123")
    assert not is_youtube_url("Rick Astley Never Gonna Give You Up")
    assert not is_youtube_url("https://example.com/watch?v=1")


def test_parse_artist_title_splits_and_cleans() -> None:
    assert parse_artist_title("Camellia - GHOST (Official Audio)", "Camellia") == ("Camellia", "GHOST")
    assert parse_artist_title("K/DA – POP/STARS [Official Music Video]", "League") == ("K/DA", "POP/STARS")
    assert parse_artist_title("Only Song Name (Lyrics)", "UploaderName") == ("UploaderName", "Only Song Name")
    assert parse_artist_title("   ", "") == ("Unknown", "untitled")


def test_require_ytdlp_missing_and_explicit(tmp_path: Path) -> None:
    with pytest.raises(YtdlpMissingError):
        require_ytdlp(tmp_path / "nope.exe")
    exe = tmp_path / "yt-dlp.exe"
    exe.write_bytes(b"")
    assert require_ytdlp(exe) == str(exe)


def test_search_and_resolve_with_fake_runner(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"
    exe.write_bytes(b"")
    seen_cmds = []
    seen_kwargs = []

    def fake_runner(cmd, **kw):
        seen_cmds.append(cmd)
        seen_kwargs.append(kw)
        if any(a.startswith("ytsearch10:") for a in cmd):
            lines = "\n".join([
                json.dumps({"id": "1", "webpage_url": "https://youtu.be/1", "title": "A - B", "channel": "Ch", "duration": 185, "thumbnail": "https://i.ytimg.com/1.jpg"}),
                json.dumps({"id": "2", "title": "Live", "is_live": True}),
            ])
            return subprocess.CompletedProcess(cmd, 0, stdout=lines, stderr="")
        return subprocess.CompletedProcess(
            cmd, 0,
            stdout=json.dumps({"id": "9", "webpage_url": "https://youtu.be/9", "title": "X - Y (Official Video)", "uploader": "Up", "duration": 210}),
            stderr="",
        )

    items = search_youtube("test query", ytdlp_path=exe, runner=fake_runner)
    assert len(items) == 1
    assert items[0].artist == "A" and items[0].track_title == "B" and items[0].duration_s == 185
    assert "ytsearch10:test query" in seen_cmds[0]

    item = resolve_youtube("https://www.youtube.com/watch?v=9&list=PLx", ytdlp_path=exe, runner=fake_runner)
    assert item.artist == "X" and item.track_title == "Y"
    assert "--no-playlist" in seen_cmds[1]
    assert all(
        kw.get("encoding") == "utf-8" and kw.get("errors") == "replace" and kw.get("stdin") == subprocess.DEVNULL
        for kw in seen_kwargs
    )


def test_download_youtube_progress_and_cancel_cleanup(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"
    exe.write_bytes(b"")
    dest = tmp_path / "dl"
    pcts = []

    class FakeProc:
        def __init__(self, cmd, lines, returncode=0, create_files=True):
            self.cmd = cmd
            self.stdout = io.StringIO("".join(lines))
            self.returncode = returncode
            self.terminated = False
            if create_files:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "audio.webm").write_bytes(b"audio")
                (dest / "audio.jpg").write_bytes(b"\xff\xd8thumb")

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    res = download_youtube(
        "https://youtu.be/9", dest, ytdlp_path=exe,
        progress=pcts.append,
        popen=lambda cmd, **kw: FakeProc(cmd, ["[download]  12.5% of 4MiB\n", "[download] 100.0% of 4MiB\n"]),
    )
    assert pcts == [12, 100]
    assert res.audio_path.name == "audio.webm"
    assert res.cover_path is not None and res.cover_path.name == "audio.jpg"

    dest2 = tmp_path / "dl_cancel"
    with pytest.raises(GenerateCancelled):
        download_youtube(
            "https://youtu.be/9", dest2, ytdlp_path=exe,
            is_cancelled=lambda: True,
            popen=lambda cmd, **kw: FakeProc(cmd, ["[download]  10.0% of 4MiB\n"]),
        )
    assert not dest2.exists()

    with pytest.raises(DownloadError):
        download_youtube(
            "https://youtu.be/9", tmp_path / "dl_fail", ytdlp_path=exe,
            popen=lambda cmd, **kw: FakeProc(cmd, ["ERROR: video unavailable\n"], returncode=1, create_files=False),
        )
