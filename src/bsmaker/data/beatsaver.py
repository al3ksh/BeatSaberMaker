"""Minimal BeatSaver API client. API responses are untrusted data: parsed, never executed."""
from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

USER_AGENT = "BeatSaberMaker/0.1"
MIN_INTERVAL_FLOOR_S = 0.25
BATCH_SIZE = 50
DOWNLOAD_HOSTS = ("r2cdn.beatsaver.com",)
MAX_BACKOFF_S = 60.0


class BeatSaverError(Exception):
    """Any API, parsing or download failure."""


@dataclass(frozen=True)
class MapInfo:
    key: str
    hash: str
    download_url: str
    title: str
    mapper: str
    tags: tuple[str, ...]
    score: float
    upvotes: int
    ranked: bool
    bl_ranked: bool
    curated: bool
    automapper: bool


def parse_map_info(doc: dict) -> MapInfo:
    """Build a MapInfo from a BeatSaver map document; BeatSaverError if essentials are missing."""
    try:
        if not isinstance(doc, dict):
            raise BeatSaverError("map document is not an object")
        key = doc.get("id")
        versions = doc.get("versions")
        if not isinstance(key, str) or not key:
            raise BeatSaverError("map document has no id")
        if not isinstance(versions, list):
            raise BeatSaverError(f"map {key}: no versions")
        usable = [v for v in versions if isinstance(v, dict) and v.get("state") == "Published"
                  and v.get("hash") and v.get("downloadURL")]
        if not usable:
            raise BeatSaverError(f"map {key}: no Published version with hash and downloadURL")
        # newest by creation timestamp; list order breaks ties (max keeps the later one)
        ver = max(enumerate(usable), key=lambda iv: (str(iv[1].get("createdAt") or ""), iv[0]))[1]
        stats = doc.get("stats") if isinstance(doc.get("stats"), dict) else {}
        uploader = doc.get("uploader") if isinstance(doc.get("uploader"), dict) else {}
        tags = doc.get("tags") if isinstance(doc.get("tags"), list) else []
        ai = doc.get("declaredAi")
        return MapInfo(
            key=key, hash=str(ver["hash"]), download_url=str(ver["downloadURL"]),
            title=str(doc.get("name") or ""), mapper=str(uploader.get("name") or ""),
            tags=tuple(t for t in tags if isinstance(t, str)),
            score=float(stats.get("score") or 0.0), upvotes=int(stats.get("upvotes") or 0),
            ranked=bool(doc.get("ranked")), bl_ranked=bool(doc.get("blRanked")),
            curated=bool(doc.get("curatedAt")),
            automapper=bool(doc.get("automapper")) or (isinstance(ai, str) and ai not in ("", "None")),
        )
    except BeatSaverError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise BeatSaverError(f"malformed map document: {exc}") from exc


class BeatSaverClient:
    def __init__(self, base_url: str = "https://api.beatsaver.com", min_interval_s: float = 0.25,
                 max_retries: int = 5, transport: httpx.BaseTransport | None = None) -> None:
        if min_interval_s < MIN_INTERVAL_FLOOR_S:
            raise ValueError(f"min_interval_s must be >= {MIN_INTERVAL_FLOOR_S}")
        self._min_interval = min_interval_s
        self._max_retries = max_retries
        self._last = 0.0
        self._lock = threading.Lock()
        self.n_requests = 0
        self.n_429 = 0
        self._http = httpx.Client(base_url=base_url, headers={"User-Agent": USER_AGENT},
                                  transport=transport, timeout=30.0, follow_redirects=False)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> BeatSaverClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        with self._lock:  # holding the lock while sleeping keeps the interval across threads
            wait = self._min_interval - (time.monotonic() - self._last)
            if self._last and wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1

    @staticmethod
    def _backoff(resp: httpx.Response | None, attempt: int) -> float:
        if resp is not None:
            try:
                return min(max(float(resp.headers["Retry-After"]), 0.0), MAX_BACKOFF_S)
            except (KeyError, ValueError):
                pass
        return min(2.0 ** attempt, MAX_BACKOFF_S)

    def _get_json(self, path: str, params: dict | None = None) -> object:
        last: str = ""
        for attempt in range(self._max_retries + 1):
            self._throttle()
            resp: httpx.Response | None = None
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                last = f"transport error: {exc}"
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    if resp.status_code == 429:
                        self.n_429 += 1
                    last = f"HTTP {resp.status_code}"
                elif resp.status_code != 200:
                    raise BeatSaverError(f"GET {path}: HTTP {resp.status_code}")
                else:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise BeatSaverError(f"GET {path}: invalid JSON") from exc
            if attempt < self._max_retries:
                time.sleep(self._backoff(resp, attempt))
        raise BeatSaverError(f"GET {path} failed after {self._max_retries} retries ({last})")

    def get_maps_by_ids(self, keys: Sequence[str]) -> dict[str, MapInfo]:
        """Look up maps by BeatSaver id; ids the API does not return are absent from the result."""
        out: dict[str, MapInfo] = {}
        for i in range(0, len(keys), BATCH_SIZE):
            batch = [k.lower() for k in keys[i:i + BATCH_SIZE]]
            data = self._get_json("/maps/ids/" + ",".join(batch))
            if not isinstance(data, dict):
                raise BeatSaverError("by-ids response is not an object")
            if "id" in data and "versions" in data:  # bare document instead of {id: doc}
                data = {str(data["id"]): data}
            for doc in data.values():
                try:
                    info = parse_map_info(doc)
                except BeatSaverError:
                    continue  # unusable document: that key counts as not returned
                out[info.key] = info
        return out

    def search(self, page: int, **filters: object) -> list[MapInfo]:
        data = self._get_json(f"/search/text/{int(page)}", params=filters or None)
        docs = data.get("docs") if isinstance(data, dict) else None
        if not isinstance(docs, list):
            raise BeatSaverError("search response has no docs list")
        out: list[MapInfo] = []
        for d in docs:
            try:
                out.append(parse_map_info(d))
            except BeatSaverError:
                continue  # one unusable document must not abort the page
        return out

    def download(self, url: str, dest: Path, max_bytes: int = 200 * 1024 * 1024) -> None:
        """Stream `url` (https, BeatSaver download host only) to `dest` atomically."""
        try:
            parts = urlsplit(url)
            ok = (parts.scheme == "https" and parts.hostname in DOWNLOAD_HOSTS
                  and parts.username is None and parts.password is None
                  and parts.port in (None, 443))
        except ValueError:
            ok = False
        if not ok:
            raise BeatSaverError(f"refusing to download from {url!r}")
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".part")
        try:
            self._throttle()
            with self._http.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise BeatSaverError(f"download {url}: HTTP {resp.status_code}")
                declared = resp.headers.get("Content-Length")
                if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                    raise BeatSaverError(f"download {url}: {declared} bytes exceeds {max_bytes}")
                total = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise BeatSaverError(f"download {url}: exceeds {max_bytes} bytes")
                        fh.write(chunk)
            tmp.replace(dest)
        except BeatSaverError:
            tmp.unlink(missing_ok=True)
            raise
        except (httpx.HTTPError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise BeatSaverError(f"download {url} failed: {exc}") from exc
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
