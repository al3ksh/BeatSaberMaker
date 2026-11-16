"""Long-lived JSON-lines engine server over stdin/stdout."""
from __future__ import annotations

import json
import queue
import re
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import TextIO

import torch

from bsmaker.audio.decode import AudioDecodeError, FfmpegMissingError
from bsmaker.config import Config
from bsmaker.data.styles import STYLES
from bsmaker.engine.export import export_map_zip
from bsmaker.engine.steam import detect_custom_levels
from bsmaker.engine.youtube import (
    DownloadError,
    YtdlpMissingError,
    download_youtube,
    resolve_youtube,
    search_youtube,
)
from bsmaker.errors import GenerateCancelled
from bsmaker.formats.layer_tokens import SUPPORTED_ENVIRONMENTS
from bsmaker.generate import GenerateError
from bsmaker.pipeline import (
    LoadedPipeline,
    MissingModelsError,
    load_pipeline,
    pipeline_to_cpu,
    pipeline_to_device,
    resolve_device,
    run_generate,
)

_LAYER_SLOTS = ("ornaments", "decor-when", "decor-what", "lights-when", "lights-what")

_MESSAGES_PL: dict[str, str] = {
    "ffmpeg_missing": "Brak programu ffmpeg wymaganego do dekodowania dźwięku.",
    "ytdlp_missing": "Brak narzędzia yt-dlp wymaganego do pobierania z YouTube.",
    "download_failed": "Nie udało się pobrać utworu z YouTube.",
    "no_models": "Brak modeli generatora w paczce silnika.",
    "cancelled": "Zadanie zostało anulowane.",
    "audio_too_short": "Utwór jest za krótki (minimum 30 sekund).",
    "audio_silent": "Plik audio jest cichy lub nie zawiera dźwięku.",
    "invalid_audio": "Nie można odczytać pliku audio.",
    "file_exists": "Folder docelowy mapy już istnieje.",
    "not_found": "Nie znaleziono wskazanego pliku lub folderu.",
    "invalid_args": "Nieprawidłowe parametry zadania.",
    "busy": "Silnik wykonuje już inne zadanie.",
    "internal": "Wystąpił wewnętrzny błąd silnika. Szczegóły zapisano w logach.",
}


def _get_version() -> str:
    try:
        return pkg_version("bsmaker")
    except PackageNotFoundError:
        return "0.1.0"


def _clean_text(raw: str | None, fallback: str) -> str:
    if raw is None:
        return fallback
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", str(raw)).strip()[:120]
    return cleaned or fallback


@dataclass(frozen=True)
class EngineConfig:
    checkpoints_dir: Path
    playability_path: Path | None = None
    layer_stats_path: Path | None = None
    ffmpeg_path: Path | None = None
    ytdlp_path: Path | None = None
    device: str | None = None
    log_path: Path | None = None


class EngineServer:
    """Reads JSON-line requests from `stdin` and writes JSON-line events to `stdout`."""

    def __init__(
        self,
        cfg: EngineConfig,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.cfg = cfg
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        self._write_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._job_queue: queue.Queue[tuple[str, dict] | None] = queue.Queue()
        self._pipeline: LoadedPipeline | None = None

    def _emit(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._write_lock:
            self.stdout.write(line)
            self.stdout.flush()

    def _emit_error(self, req_id: str, code: str, detail: str = "") -> None:
        self._emit({
            "id": req_id,
            "type": "error",
            "code": code,
            "message_pl": _MESSAGES_PL.get(code, _MESSAGES_PL["internal"]),
            "detail": detail,
        })

    def _log_traceback(self, context: str, exc: BaseException) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        entry = f"[{stamp}] {context}: {tb}\n"
        if self.cfg.log_path is not None:
            try:
                self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cfg.log_path.open("a", encoding="utf-8", errors="replace") as fh:
                    fh.write(entry)
                return
            except OSError:
                pass
        try:
            self.stderr.write(entry)
            self.stderr.flush()
        except OSError:
            pass

    def _reader_loop(self) -> None:
        try:
            while True:
                line = self.stdin.readline()
                if not line:
                    self._job_queue.put(None)
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    req = json.loads(text)
                except json.JSONDecodeError as e:
                    self._emit_error("", "invalid_args", str(e))
                    continue
                if not isinstance(req, dict):
                    self._emit_error("", "invalid_args", "request must be a JSON object")
                    continue
                req_id = str(req.get("id") or "")
                cmd = req.get("cmd")
                if cmd == "cancel":
                    self._cancel_event.set()
                    self._emit({"id": req_id, "type": "result", "data": {"cancelled": True}})
                    continue
                self._job_queue.put((req_id, req))
                if cmd == "shutdown":
                    break
        except Exception as e:  # noqa: BLE001
            self._log_traceback("stdin_reader", e)
            self._job_queue.put(None)

    def serve(self) -> int:
        reader = threading.Thread(target=self._reader_loop, daemon=True)
        reader.start()

        while True:
            item = self._job_queue.get()
            if item is None:
                break
            req_id, req = item
            cmd = req.get("cmd")
            if cmd == "shutdown":
                self._emit({"id": req_id, "type": "result", "data": {"shutdown": True}})
                break
            try:
                data = self._dispatch(req_id, req)
                self._emit({"id": req_id, "type": "result", "data": data})
            except GenerateCancelled as e:
                self._emit_error(req_id, "cancelled", str(e))
            except FfmpegMissingError as e:
                self._emit_error(req_id, "ffmpeg_missing", str(e))
            except YtdlpMissingError as e:
                self._emit_error(req_id, "ytdlp_missing", str(e))
            except DownloadError as e:
                self._emit_error(req_id, "download_failed", str(e))
            except MissingModelsError as e:
                self._emit_error(req_id, "no_models", str(e))
            except AudioDecodeError as e:
                self._emit_error(req_id, "invalid_audio", str(e))
            except GenerateError as e:
                msg_low = str(e).lower()
                if "shorter" in msg_low:
                    code = "audio_too_short"
                elif "silent" in msg_low:
                    code = "audio_silent"
                else:
                    code = "invalid_audio"
                self._emit_error(req_id, code, str(e))
            except FileExistsError as e:
                self._emit_error(req_id, "file_exists", str(e))
            except FileNotFoundError as e:
                self._emit_error(req_id, "not_found", str(e))
            except ValueError as e:
                self._emit_error(req_id, "invalid_args", str(e))
            except Exception as e:  # noqa: BLE001
                self._log_traceback(f"cmd={cmd}", e)
                self._emit_error(req_id, "internal", "")
            finally:
                self._cancel_event.clear()

        return 0

    def _dispatch(self, req_id: str, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "info":
            return self._handle_info(req)
        if cmd == "search":
            return self._handle_search(req)
        if cmd == "resolve":
            return self._handle_resolve(req)
        if cmd == "download":
            return self._handle_download(req_id, req)
        if cmd == "generate":
            return self._handle_generate(req_id, req)
        if cmd == "export_zip":
            return self._handle_export_zip(req)
        raise ValueError(f"unknown command: {cmd!r}")

    def _handle_info(self, req: dict) -> dict:
        cuda_ok = bool(torch.cuda.is_available())
        pref = req.get("device") or self.cfg.device
        if pref in (None, "auto"):
            active_dev = "cuda" if cuda_ok else "cpu"
        elif pref in ("cuda", "cpu"):
            active_dev = "cuda" if (pref == "cuda" and cuda_ok) else "cpu"
        else:
            raise ValueError(f"unsupported device {pref!r}; expected auto, cuda, or cpu")

        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        detected = detect_custom_levels()
        layers_dir = Path(self.cfg.checkpoints_dir) / "layers"
        missing_layers = [
            slot for slot in _LAYER_SLOTS if not (layers_dir / slot / "best.pt").is_file()
        ]
        return {
            "version": _get_version(),
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "active_device": active_dev,
            "detected_custom_levels": str(detected) if detected is not None else None,
            "has_layer_models": len(missing_layers) < len(_LAYER_SLOTS),
            "missing_layer_models": missing_layers,
            "environments": list(SUPPORTED_ENVIRONMENTS),
            "styles": list(STYLES),
        }

    def _handle_search(self, req: dict) -> dict:
        if "query" not in req or not isinstance(req["query"], str):
            raise ValueError("search requires string 'query'")
        limit_raw = req.get("limit", 10)
        limit = int(limit_raw)
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        items = search_youtube(req["query"], ytdlp_path=self.cfg.ytdlp_path, limit=limit)
        return {"items": [asdict(it) for it in items]}

    def _handle_resolve(self, req: dict) -> dict:
        if "url" not in req or not isinstance(req["url"], str):
            raise ValueError("resolve requires string 'url'")
        item = resolve_youtube(req["url"], ytdlp_path=self.cfg.ytdlp_path)
        return asdict(item)

    def _handle_download(self, req_id: str, req: dict) -> dict:
        if "url" not in req or not isinstance(req["url"], str):
            raise ValueError("download requires string 'url'")
        if "dest_dir" not in req or not isinstance(req["dest_dir"], str):
            raise ValueError("download requires string 'dest_dir'")
        res = download_youtube(
            req["url"],
            Path(req["dest_dir"]),
            ytdlp_path=self.cfg.ytdlp_path,
            ffmpeg_path=self.cfg.ffmpeg_path,
            progress=lambda pct: self._emit({
                "id": req_id,
                "type": "progress",
                "stage": "downloading",
                "pct": pct,
            }),
            is_cancelled=self._cancel_event.is_set,
        )
        return {
            "audio_path": str(res.audio_path),
            "cover_path": str(res.cover_path) if res.cover_path is not None else None,
            "title": res.title,
            "artist": res.artist,
        }

    def _handle_generate(self, req_id: str, req: dict) -> dict:
        if "audio_path" not in req or not isinstance(req["audio_path"], str):
            raise ValueError("generate requires string 'audio_path'")
        if "out_dir" not in req or not isinstance(req["out_dir"], str):
            raise ValueError("generate requires string 'out_dir'")

        audio_path = Path(req["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        out_dir = Path(req["out_dir"])

        if self._pipeline is None:
            cfg = Config.from_paths(
                out_dir,
                checkpoints_dir=self.cfg.checkpoints_dir,
                playability_path=self.cfg.playability_path,
                layer_stats_path=self.cfg.layer_stats_path,
            )
            self._pipeline = load_pipeline(cfg, device="cpu", load_layers=True)

        req_dev = req.get("device")
        if req_dev is None:
            req_dev = self.cfg.device
        if req_dev == "auto":
            req_dev = None
        target_dev = resolve_device(req_dev)

        if self._pipeline.device != target_dev:
            pipeline_to_device(self._pipeline, target_dev)

        raw_title = req.get("title")
        raw_artist = req.get("artist")
        title_out = _clean_text(raw_title if raw_title is not None else audio_path.stem, "untitled")
        artist_out = _clean_text(raw_artist, "Unknown")

        cover_raw = req.get("cover_src")
        cover_src = Path(cover_raw) if cover_raw else None

        try:
            res = run_generate(
                self._pipeline,
                audio_path,
                out_dir=out_dir,
                nps=req.get("nps"),
                levels=req.get("levels"),
                styles=tuple(req.get("styles") or ()),
                seed=int(req.get("seed", 0)),
                overwrite=bool(req.get("overwrite", False)),
                unique_folder=bool(req.get("unique_folder", False)),
                title=title_out,
                artist=artist_out,
                cover_src=cover_src,
                ornaments=req.get("ornaments"),
                lights=int(req.get("lights", 0)),
                environment=str(req.get("environment") or "DefaultEnvironment"),
                lights_rules=bool(req.get("lights_rules", False)),
                ffmpeg_path=self.cfg.ffmpeg_path,
                progress=lambda stage, pct: self._emit({
                    "id": req_id,
                    "type": "progress",
                    "stage": stage,
                    "pct": pct,
                }),
                is_cancelled=self._cancel_event.is_set,
            )
        finally:
            pipeline_to_cpu(self._pipeline)

        used_rule_lights = bool(res.lights_report and res.lights_report.get("used_rules"))
        return {
            "folder": str(res.folder),
            "title": title_out,
            "artist": artist_out,
            "bpm": res.bpm,
            "grid_steady": res.grid_steady,
            "n_notes": res.n_notes,
            "nps": res.nps,
            "rank": res.rank,
            "levels": [
                {
                    "rank": lv.rank,
                    "target_nps": lv.target_nps,
                    "n_notes": lv.n_notes,
                    "nps": lv.nps,
                    "n_arcs": lv.n_arcs,
                    "n_chains": lv.n_chains,
                }
                for lv in res.levels
            ],
            "n_walls": res.n_walls,
            "n_bombs": res.n_bombs,
            "n_lights": res.n_lights,
            "environment": res.environment,
            "warnings": list(res.warnings),
            "used_rule_lights": used_rule_lights,
            "device_used": target_dev,
        }

    def _handle_export_zip(self, req: dict) -> dict:
        if "folder" not in req or not isinstance(req["folder"], str):
            raise ValueError("export_zip requires string 'folder'")
        if "out_zip" not in req or not isinstance(req["out_zip"], str):
            raise ValueError("export_zip requires string 'out_zip'")
        out = export_map_zip(Path(req["folder"]), Path(req["out_zip"]))
        return {"out_zip": str(out)}


def run_engine(cfg: EngineConfig) -> int:
    """Start the blocking JSON-lines engine server."""
    return EngineServer(cfg).serve()
