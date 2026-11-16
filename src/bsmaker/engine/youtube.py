"""YouTube search, metadata resolution, and audio/thumbnail download via yt-dlp."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from bsmaker.errors import GenerateCancelled

_YOUTUBE_HOSTS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
})

_NOISE_TAIL_RE = re.compile(
    r"\s*[\(\[]"
    r"(?:official(?:\s+(?:music\s+)?(?:video|audio|lyric\s+video|visualizer))?"
    r"|lyrics?(?:\s+video)?|audio|music\s+video|visualizer|hd|4k|hq)"
    r"[^\)\]]*[\)\]]\s*$",
    re.IGNORECASE,
)

_PROGRESS_RE = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
_SKIP_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".part", ".ytdl", ".tmp", ".json"})


class YtdlpMissingError(Exception):
    """yt-dlp binary could not be found."""


class DownloadError(Exception):
    """YouTube search, resolve, or download failed."""


@dataclass(frozen=True)
class YouTubeItem:
    id: str
    url: str
    title: str
    artist: str
    track_title: str
    channel: str
    duration_s: float | None
    thumbnail_url: str | None


@dataclass(frozen=True)
class DownloadResult:
    audio_path: Path
    cover_path: Path | None
    title: str
    artist: str


def is_youtube_url(text: str) -> bool:
    """Return True when `text` is an http/https URL on a YouTube host."""
    raw = text.strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    if parsed.scheme.lower() not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host in _YOUTUBE_HOSTS


def parse_artist_title(video_title: str, uploader: str = "") -> tuple[str, str]:
    """Split a YouTube video title into `(artist, track_title)` and strip noise tags."""
    cleaned = video_title.strip()
    while cleaned:
        stripped = _NOISE_TAIL_RE.sub("", cleaned).strip()
        if stripped == cleaned:
            break
        cleaned = stripped

    for sep in (" - ", " \u2013 ", " \u2014 "):
        if sep in cleaned:
            left, right = cleaned.split(sep, 1)
            left_c, right_c = left.strip(), right.strip()
            if left_c and right_c:
                return left_c, right_c

    artist = uploader.strip() or "Unknown"
    track = cleaned or "untitled"
    return artist, track


def require_ytdlp(ytdlp_path: Path | None = None) -> str:
    """Return a usable yt-dlp executable path or raise `YtdlpMissingError`."""
    if ytdlp_path is not None:
        p = Path(ytdlp_path)
        if not p.is_file():
            raise YtdlpMissingError(f"yt-dlp not found: {p}")
        return str(p)
    found = shutil.which("yt-dlp")
    if found is None:
        raise YtdlpMissingError("yt-dlp not found on PATH")
    return found


def _subproc_kwargs(*, timeout_s: float | None = None) -> dict:
    kw: dict = {
        "shell": False,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if timeout_s is not None:
        kw["timeout"] = timeout_s
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kw


def _to_item(data: dict) -> YouTubeItem:
    vid = str(data.get("id") or "")
    url = str(data.get("webpage_url") or data.get("url") or "")
    if url and not url.startswith(("http://", "https://")) and vid:
        url = f"https://www.youtube.com/watch?v={vid}"
    raw_title = str(data.get("title") or "")
    channel = str(data.get("channel") or data.get("uploader") or "")
    uploader = str(data.get("uploader") or channel)
    artist, track_title = parse_artist_title(raw_title, uploader)
    dur_raw = data.get("duration")
    duration_s = float(dur_raw) if isinstance(dur_raw, (int, float)) else None
    thumb = data.get("thumbnail")
    if not thumb and isinstance(data.get("thumbnails"), list) and data["thumbnails"]:
        last = data["thumbnails"][-1]
        if isinstance(last, dict):
            thumb = last.get("url")
    return YouTubeItem(
        id=vid,
        url=url,
        title=raw_title,
        artist=artist,
        track_title=track_title,
        channel=channel,
        duration_s=duration_s,
        thumbnail_url=str(thumb) if thumb else None,
    )


def search_youtube(
    query: str,
    *,
    ytdlp_path: Path | None = None,
    limit: int = 10,
    timeout_s: float = 30.0,
    runner=subprocess.run,
) -> list[YouTubeItem]:
    """Search YouTube via yt-dlp `ytsearchN:` and return non-live items."""
    q = query.strip()
    if not q:
        return []
    exe = require_ytdlp(ytdlp_path)
    cmd = [
        exe,
        "--dump-json",
        "--flat-playlist",
        "--no-warnings",
        "--ignore-errors",
        f"ytsearch{limit}:{q}",
    ]
    try:
        res = runner(cmd, capture_output=True, **_subproc_kwargs(timeout_s=timeout_s))
    except subprocess.TimeoutExpired as e:
        raise DownloadError(f"YouTube search timed out after {timeout_s:g} s") from e
    except OSError as e:
        raise DownloadError(f"yt-dlp failed to start: {e}") from e

    items: list[YouTubeItem] = []
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("is_live"):
            continue
        if not (data.get("id") and (data.get("webpage_url") or data.get("url"))):
            continue
        items.append(_to_item(data))

    if res.returncode != 0 and not items:
        err = (res.stderr or "").strip().splitlines()
        msg = err[-1] if err else f"yt-dlp search failed (exit {res.returncode})"
        raise DownloadError(msg)
    return items


def resolve_youtube(
    url: str,
    *,
    ytdlp_path: Path | None = None,
    timeout_s: float = 30.0,
    runner=subprocess.run,
) -> YouTubeItem:
    """Resolve metadata for a single YouTube video URL."""
    if not is_youtube_url(url):
        raise DownloadError(f"invalid YouTube URL: {url}")
    exe = require_ytdlp(ytdlp_path)
    cmd = [exe, "--dump-json", "--no-playlist", "--no-warnings", url]
    try:
        res = runner(cmd, capture_output=True, **_subproc_kwargs(timeout_s=timeout_s))
    except subprocess.TimeoutExpired as e:
        raise DownloadError(f"YouTube metadata fetch timed out after {timeout_s:g} s") from e
    except OSError as e:
        raise DownloadError(f"yt-dlp failed to start: {e}") from e

    if res.returncode != 0:
        err = (res.stderr or "").strip().splitlines()
        msg = err[-1] if err else f"yt-dlp failed (exit {res.returncode})"
        raise DownloadError(msg)

    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise DownloadError(f"invalid metadata JSON from yt-dlp: {e}") from e
        if isinstance(data, dict):
            return _to_item(data)
    raise DownloadError("yt-dlp returned no metadata")


def download_youtube(
    url: str,
    dest_dir: Path,
    *,
    ytdlp_path: Path | None = None,
    ffmpeg_path: Path | None = None,
    progress: Callable[[int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    popen=subprocess.Popen,
) -> DownloadResult:
    """Download best audio and JPG thumbnail for a YouTube video into `dest_dir`."""
    if not is_youtube_url(url):
        raise DownloadError(f"invalid YouTube URL: {url}")
    exe = require_ytdlp(ytdlp_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "-f",
        "bestaudio/best",
        "--no-playlist",
        "--newline",
        "--progress",
        "--write-thumbnail",
        "--convert-thumbnails",
        "jpg",
        "-o",
        str(dest_dir / "audio.%(ext)s"),
    ]
    if ffmpeg_path is not None:
        cmd.extend(["--ffmpeg-location", str(ffmpeg_path)])
    cmd.append(url)

    kw = _subproc_kwargs()
    kw["stdout"] = subprocess.PIPE
    kw["stderr"] = subprocess.STDOUT
    try:
        proc = popen(cmd, **kw)
    except OSError as e:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise DownloadError(f"yt-dlp failed to start: {e}") from e

    last_line = ""
    try:
        if proc.stdout is not None:
            for raw_line in proc.stdout:
                if is_cancelled is not None and is_cancelled():
                    _terminate_proc(proc)
                    shutil.rmtree(dest_dir, ignore_errors=True)
                    raise GenerateCancelled("cancelled")
                line = raw_line.strip()
                if line:
                    last_line = line
                m = _PROGRESS_RE.search(line)
                if m and progress is not None:
                    pct = max(0, min(100, int(float(m.group(1)))))
                    progress(pct)
        if is_cancelled is not None and is_cancelled():
            _terminate_proc(proc)
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise GenerateCancelled("cancelled")
        rc = proc.wait()
    except GenerateCancelled:
        raise
    except Exception as e:
        _terminate_proc(proc)
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise DownloadError(f"yt-dlp download failed: {e}") from e

    if rc != 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise DownloadError(last_line or f"yt-dlp download failed (exit {rc})")

    audio_path: Path | None = None
    for p in sorted(dest_dir.iterdir()):
        if p.is_file() and p.name.startswith("audio.") and p.suffix.lower() not in _SKIP_EXTS:
            audio_path = p
            break
    if audio_path is None:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise DownloadError("yt-dlp finished without producing an audio file")

    cover_path: Path | None = None
    for ext in _IMAGE_EXTS:
        cand = dest_dir / f"audio{ext}"
        if cand.is_file():
            cover_path = cand
            break

    return DownloadResult(
        audio_path=audio_path,
        cover_path=cover_path,
        title="untitled",
        artist="Unknown",
    )


def _terminate_proc(proc) -> None:
    try:
        proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except OSError:
            pass
