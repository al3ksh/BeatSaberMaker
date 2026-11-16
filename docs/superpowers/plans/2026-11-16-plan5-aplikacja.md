# BeatSaberMaker Desktop App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the existing `bsmaker` map generator a Windows 10/11 desktop app (Electron + React/TypeScript) with a 3-step creation wizard, map library, Steam `CustomLevels` auto-detection, YouTube/local audio input, split-archive first-run engine-pack download, NSIS installer, and GitHub Releases auto-updates.

**Architecture:** The Python `bsmaker` package gets a thin engine layer (`bsmaker engine`) speaking UTF-8 JSON-lines over stdin/stdout, backed by a shared model-loading/generation pipeline refactored out of the CLI, progress/cancel hooks, cover-image writing, `yt-dlp` search/resolve/download, ZIP export, and Steam `CustomLevels` discovery. An Electron + TypeScript + React + Tailwind + Framer Motion desktop app in `app/` manages the long-lived engine child process, serializes jobs, stores `%APPDATA%\BeatSaberMaker\library.json` atomically, downloads and verifies the ~2.5 GB CUDA PyInstaller engine pack on first run, and ships via NSIS + `electron-updater` built on GitHub Actions tag `v*` releases.

**Tech Stack:** Python 3.13, PyTorch, numpy, librosa, soundfile, typer, pytest, PyInstaller; Electron, TypeScript, React, Vite, Tailwind CSS, Framer Motion, Lucide React, Vitest, Testing Library, Playwright, electron-builder, electron-updater.

**Spec:** `docs/superpowers/specs/2026-11-16-aplikacja-desktop-design.md` (Polish; binding authority).

## Global Constraints

- Platform: Windows 10/11 x64 only (no macOS/Linux). Single app instance (`app.requestSingleInstanceLock()`). No code signing.
- Worktree & branch: `plan5-aplikacja`, branched from `plan4-warstwy`, at `C:\Users\alex\Desktop\BeatSaberMaker-wt\plan5`. All file paths in tasks are relative to that worktree.
- Python execution rules:
  - Always use the repo's existing venv (`C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe`) with `PYTHONPATH=src` so tests import the worktree `src/` rather than the main checkout.
  - Prefix every Python/pytest invocation with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` (PowerShell: `$env:OMP_NUM_THREADS="4"; $env:MKL_NUM_THREADS="4"; $env:PYTHONPATH="src"; ...`).
  - Run only focused tests (`-m pytest tests/path.py::test_name -v`), never the full suite (the controller runs the full suite).
- Node/npm tooling lives in `app/`.
- Tests must NOT require real network, YouTube, Steam installation, or GPU: use injected runners/mocks for `yt-dlp`, registry, local HTTP server for engine-pack download, and a fake stdio engine for Playwright Electron E2E.
- Existing `bsmaker` CLI behaviour and all existing CLI/writer/generate tests must stay unchanged.
- Backwards compatibility: `generate_map` hooks (`progress`, `is_cancelled`, `cover_src`, `unique_folder`, `ffmpeg_path`) default so existing callers produce byte-identical outputs for `cover.png`, `Info.dat`, and `.dat` files.
- GPU memory: the engine keeps parsed models in CPU RAM between jobs for fast re-generation, moves models to the active device only while generating, and releases VRAM (`model.to("cpu")` + `torch.cuda.empty_cache()`) immediately when idle.
- `CustomLevels` safety: never write outside `BSM - ...` folders inside `CustomLevels`; deletions move folders to the Windows Recycle Bin (`shell.trashItem`), and only when the path is tracked in `library.json`, lies inside `CustomLevels`, and its folder name starts with `BSM - `; missing folders are dropped from `library.json` without error.
- Re-generation (`wygeneruj ponownie`): generates the new map into a distinct folder alongside the old one (`unique_folder=True`), and moves the old folder to the Recycle Bin only after the new map succeeds.
- Error presentation: users never see Python tracebacks or internal English engine strings; engine emits error codes mapped to Polish copy in one file (`app/src/i18n/pl.ts`), while stack traces go only to `%APPDATA%\BeatSaberMaker\logs`.
- Visual direction: corporate dark theme with Beat Saber red/blue accents, Framer Motion transitions, Lucide icons, Polish UI strings in one file (`app/src/i18n/pl.ts`).
- Pinned binaries (`ffmpeg` LGPL, `yt-dlp`) are fetched in CI via URL + SHA256 in `packaging/tools.json`, never committed to git. Engine pack parts are `<= 1_900_000_000` bytes (under GitHub's 2 GB limit), verified with SHA256, and re-downloaded only when `engineVersion` changes.
- Windows UTF-8 / `cp1250` safety: engine entry points (`src/bsmaker/engine/__main__.py`, `src/bsmaker/cli.py` `engine` command, `packaging/engine_entry.py`) must call `sys.stdin.reconfigure(encoding="utf-8", errors="replace")` and `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at startup; `EngineManager` must spawn with `PYTHONUTF8="1"` and `windowsHide: true`; Python `subprocess.run`/`Popen` calls (`youtube.py`, `decode.py`, `writer.py`) must pass `encoding="utf-8", errors="replace"` (when decoding text) and `creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)` on Windows.
- Windows process-tree & binary-lock safety: `EngineManager` force-kill on Windows must terminate the full process tree (`taskkill /PID <pid> /T /F`) so child `yt-dlp.exe`/`ffmpeg.exe` processes never linger; `bsm:installEnginePack` must `await engineManager.stop()` before `enginePackInstaller.install()` so `bsmaker-engine.exe` is never file-locked (`EPERM`/`EBUSY`).
- Commits: `git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "<msg>" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"`. Never push.

## Review Focus

1. **YouTube titles with Windows-illegal chars, non-CP1250 Unicode (emoji/CJK/en-dash/Polish), no `" - "` separator, playlist query params, or non-JPG thumbnails** — `parse_artist_title` / `safe_title` / `download_youtube` (`--no-playlist`, `--convert-thumbnails jpg`, UTF-8 subprocess decoding) / UTF-8 stdio reconfiguration (`PYTHONUTF8="1"`, `reconfigure`) / `write_map_multi` must produce a clean folder name, valid `cover.jpg` + `_coverImageFilename: "cover.jpg"` in `Info.dat`, never crash with `UnicodeEncodeError`/`UnicodeDecodeError` on Windows `cp1250`, and never download a whole playlist (pinned in Tasks 2, 4, 6, 9, 15).
2. **Re-generation folder collision or mid-run cancellation** — clicking "Wygeneruj ponownie" with identical title/levels/style must write `BSM - ... (2)` alongside the old map (`unique_folder=True`) and trash the old map only on success (never trashing a newly generated map if the old folder was already deleted or is a symlink/junction); cancelling mid-run must abort promptly across notes, ornaments, and layer models, delete `.tmp-writing` / partial downloads, release VRAM, and keep any previous map intact (pinned in Tasks 2, 6, 8).
3. **Interrupted or corrupted first-run engine-pack download** — partial downloads must resume with HTTP `Range`, fall back cleanly on `200 OK`, reject SHA256 mismatches before extraction, stop any running engine before upgrading, and never overwrite an existing working engine folder with a broken extract (pinned in Task 10).
4. **Steam `libraryfolders.vdf` on non-default drives & missing/crafted paths on delete** — Steam detector must find Beat Saber on secondary library drives; `LibraryStore.deleteEntry` must silently clean missing folders and refuse to trash any folder outside `CustomLevels`, lacking `BSM - `, or being a symlink/junction (pinned in Tasks 5, 8).
5. **Unhandled Python exceptions, hung engine cancel (lingering child process trees), or absent layer checkpoints** — engine must log tracebacks privately and return `error {code: "internal"}` without leaking tracebacks; `EngineManager` must force-kill the Windows process tree (`taskkill /PID <pid> /T /F`) and restart an unresponsive engine after cancel timeout without flashing console windows; missing layer checkpoints must show as informational UI notices rather than errors (pinned in Tasks 4, 6, 9, 13, 14).

---

### Task 1: Parameterized `Config` and explicit `ffmpeg_path`

**Files:**
- Modify: `src/bsmaker/config.py`
- Modify: `src/bsmaker/audio/decode.py`
- Modify: `src/bsmaker/formats/writer.py` (`_encode_audio`, `write_map_multi`)
- Test: `tests/test_config.py`, `tests/audio/test_decode.py`, `tests/formats/test_writer.py`

**Interfaces:**
- Consumes: existing `Config`, `load_config`, `require_ffmpeg`, `decode_mono`, `write_compact`, `write_map_multi`.
- Produces:
  - `Config.from_paths(custom_levels: Path, *, checkpoints_dir: Path, playability_path: Path | None = None, layer_stats_path: Path | None = None, data_dir: Path | None = None, base_dir: Path | None = None) -> Config`
  - `Config._layer_stats_path: Path | None` (raw override path preserved even when file is absent on disk, consumed by Task 3 `load_pipeline` for `layer_stats_missing_path`).
  - `Config.layer_stats_path -> Path | None` property (returns override if configured and `is_file()`, else `self.base_dir / "layer_stats.json"` if `is_file()`, else `None`).
  - `Config.checkpoints_dir` and `Config.playability_path` properties respect optional overrides on `Config`.
  - `require_ffmpeg(ffmpeg_path: Path | None = None) -> str`
  - `decode_mono(path: Path, *, ffmpeg_path: Path | None = None) -> np.ndarray`
  - `write_compact(path_in: Path, path_out: Path, *, ffmpeg_path: Path | None = None) -> None`
  - `write_map_multi(..., ffmpeg_path: Path | None = None) -> Path`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:
```python
def test_config_from_paths_overrides_resources(tmp_path: Path) -> None:
    levels = tmp_path / "CustomLevels"
    ckpts = tmp_path / "pack" / "checkpoints"
    play = tmp_path / "pack" / "playability.toml"
    lstats = tmp_path / "pack" / "layer_stats.json"
    play.parent.mkdir(parents=True)
    play.write_text("[mask]\n", encoding="utf-8")
    lstats.write_text("{}", encoding="utf-8")

    cfg = Config.from_paths(
        levels, checkpoints_dir=ckpts, playability_path=play, layer_stats_path=lstats,
    )
    assert cfg.custom_levels == levels
    assert cfg.checkpoints_dir == ckpts
    assert cfg.playability_path == play
    assert cfg.layer_stats_path == lstats

    missing_cfg = Config.from_paths(
        levels, checkpoints_dir=ckpts,
        playability_path=tmp_path / "nope.toml",
        layer_stats_path=tmp_path / "nope.json",
    )
    assert missing_cfg.playability_path is None
    assert missing_cfg.layer_stats_path is None
    assert missing_cfg._layer_stats_path == tmp_path / "nope.json"
```

Add to `tests/audio/test_decode.py`:
```python
def test_explicit_ffmpeg_path_used_and_missing_raises(tmp_path, monkeypatch):
    import subprocess
    from bsmaker.audio.decode import FfmpegMissingError, require_ffmpeg

    missing = tmp_path / "bin" / "ffmpeg.exe"
    with pytest.raises(FfmpegMissingError):
        require_ffmpeg(missing)

    fake_bin = tmp_path / "bin" / "ffmpeg.exe"
    fake_bin.parent.mkdir()
    fake_bin.write_bytes(b"")
    assert require_ffmpeg(fake_bin) == str(fake_bin)

    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=b"\x00\x00\x80\x3f", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    decode_mono(tmp_path / "song.wav", ffmpeg_path=fake_bin)
    assert seen["cmd"][0] == str(fake_bin)
```

Add to `tests/formats/test_writer.py`:
```python
def test_writer_forwards_explicit_ffmpeg_path(tmp_path, monkeypatch):
    from bsmaker.formats import writer as W

    fake_bin = tmp_path / "ffmpeg.exe"
    fake_bin.write_bytes(b"")
    seen = {}

    def fake_encode(src, dst, *, ffmpeg_path=None):
        seen["ffmpeg_path"] = ffmpeg_path
        dst.write_bytes(b"OggS")

    monkeypatch.setattr(W, "_encode_audio", fake_encode)
    write_map_multi(
        tmp_path / "BSM - f", title="T", artist="A", bpm=120.0,
        levels=[_easy()], audio_src=AUDIO, ffmpeg_path=fake_bin,
    )
    assert seen["ffmpeg_path"] == fake_bin
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/test_config.py::test_config_from_paths_overrides_resources tests/audio/test_decode.py::test_explicit_ffmpeg_path_used_and_missing_raises tests/formats/test_writer.py::test_writer_forwards_explicit_ffmpeg_path -v
```
Expected: FAIL (`AttributeError` / `TypeError` for `Config.from_paths`, `ffmpeg_path`).

- [ ] **Step 3: Implement `Config.from_paths` and `ffmpeg_path` threading**

1. In `src/bsmaker/config.py`: add optional private/default-None override fields on `Config`: `_checkpoints_dir: Path | None = None`, `_playability_path: Path | None = None`, `_layer_stats_path: Path | None = None` (trailing defaults keep `Config(custom_levels, data_dir, base_dir)` 100% compatible). Update `checkpoints_dir` and `playability_path` properties, add `layer_stats_path` property, and add `@classmethod def from_paths(cls, custom_levels: Path, *, checkpoints_dir: Path, playability_path: Path | None = None, layer_stats_path: Path | None = None, data_dir: Path | None = None, base_dir: Path | None = None) -> Config`.
2. In `src/bsmaker/audio/decode.py`:
   - `require_ffmpeg(ffmpeg_path: Path | None = None) -> str`: if `ffmpeg_path is None`, preserve existing `shutil.which` check for `"ffmpeg"` and `"ffprobe"` and return `"ffmpeg"`; if provided, verify `Path(ffmpeg_path).is_file()` else raise `FfmpegMissingError(f"ffmpeg not found: {ffmpeg_path}")`, and return `str(ffmpeg_path)`.
   - Update `_run(args: list[str], path: Path, *, ffmpeg_path: Path | None = None) -> bytes` to use `str(ffmpeg_path) if ffmpeg_path is not None else "ffmpeg"` as `cmd[0]`, and pass `creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)` on Windows (`sys.platform == "win32"`). Thread `ffmpeg_path: Path | None = None` into `decode_mono` and `write_compact`.
3. In `src/bsmaker/formats/writer.py`: update `_encode_audio(src: Path, dst: Path, *, ffmpeg_path: Path | None = None) -> None` to use `str(ffmpeg_path) if ffmpeg_path is not None else "ffmpeg"` as `cmd[0]` (passing `creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)` on Windows), and add `ffmpeg_path: Path | None = None` to `write_map_multi`.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/test_config.py tests/audio/test_decode.py tests/formats/test_writer.py -v
```
Expected: PASS on all tests in those three files.

- [ ] **Step 5: Commit**

```bash
git add src/bsmaker/config.py src/bsmaker/audio/decode.py src/bsmaker/formats/writer.py tests/test_config.py tests/audio/test_decode.py tests/formats/test_writer.py
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(core): parameterize Config resources and explicit ffmpeg path" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 2: Writer cover image & `generate_map` hooks (`artist`, `cover_src`, `unique_folder`, `progress`, `is_cancelled`)

**Files:**
- Create: `src/bsmaker/errors.py`
- Modify: `src/bsmaker/formats/writer.py`
- Modify: `src/bsmaker/models/pattern/sample.py`
- Modify: `src/bsmaker/models/ornaments/sample.py`
- Modify: `src/bsmaker/models/layers/what.py`
- Modify: `src/bsmaker/layers_generate.py`
- Modify: `src/bsmaker/generate.py`
- Test: `tests/formats/test_writer.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: Task 1 (`ffmpeg_path` in `write_map_multi`, `decode_mono`), `sample_notes`, `sample_ornaments`, `sample_layer_values`, `generate_decor`, `generate_lights`.
- Produces:
  - `src/bsmaker/errors.py`: `class GenerateCancelled(Exception): """Generation or download was cancelled by the caller."""`
  - `write_map_multi(..., cover_src: Path | None = None, ffmpeg_path: Path | None = None) -> Path`
  - `sample_notes(..., is_cancelled: Callable[[], bool] | None = None, on_progress: Callable[[int, int], None] | None = None)`
  - `sample_ornaments(..., is_cancelled: Callable[[], bool] | None = None) -> tuple[list[Arc], list[Chain], OrnamentReport]`
  - `sample_layer_values(..., is_cancelled: Callable[[], bool] | None = None)`
  - `generate_decor(..., is_cancelled: Callable[[], bool] | None = None) -> tuple[list[Obstacle], list[Bomb], dict]`
  - `generate_lights(..., is_cancelled: Callable[[], bool] | None = None) -> tuple[list[LightEvent], dict]`
  - `src/bsmaker/generate.py`: re-exports `GenerateCancelled`; defines `ProgressCallback = Callable[[str, int], None]`, `CancelCheck = Callable[[], bool]`; extends `generate_map(..., artist: str = "Unknown", cover_src: Path | None = None, unique_folder: bool = False, ffmpeg_path: Path | None = None, progress: ProgressCallback | None = None, is_cancelled: CancelCheck | None = None) -> GenerateResult`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/formats/test_writer.py`:
```python
def test_writer_custom_jpg_cover_and_non_jpg_conversion(tmp_path, monkeypatch):
    jpg = tmp_path / "thumb.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0fakejpg")
    folder = write_map_multi(
        tmp_path / "BSM - cov", title="Title", artist="Artist", bpm=120.0,
        levels=[_easy()], audio_src=AUDIO, cover_src=jpg,
    )
    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_songAuthorName"] == "Artist"
    assert info["_coverImageFilename"] == "cover.jpg"
    assert (folder / "cover.jpg").read_bytes() == jpg.read_bytes()
    assert not (folder / "cover.png").exists()

    png = tmp_path / "thumb.webp"
    png.write_bytes(b"RIFFfakewebp")
    from bsmaker.formats import writer as W
    monkeypatch.setattr(W, "_convert_cover", lambda src, dst, ffmpeg_path=None: dst.write_bytes(b"\xff\xd8converted"))
    folder2 = write_map_multi(
        tmp_path / "BSM - cov2", title="Title", artist="Artist", bpm=120.0,
        levels=[_easy()], audio_src=AUDIO, cover_src=png,
    )
    assert (folder2 / "cover.jpg").read_bytes() == b"\xff\xd8converted"

    with pytest.raises(ValueError, match="cover"):
        write_map_multi(
            tmp_path / "BSM - badcov", title="T", artist="A", bpm=120.0,
            levels=[_easy()], audio_src=AUDIO, cover_src=tmp_path / "missing.jpg",
        )
```

Add to `tests/test_generate.py`:
```python
def test_generate_map_artist_cover_unique_folder_and_progress(models, tmp_path):
    audio = write_audio(tmp_path / "My Song.wav")
    cover = tmp_path / "art.jpg"
    cover.write_bytes(b"\xff\xd8\xff\xe0thumb")
    events: list[tuple[str, int]] = []

    res1 = _gen(
        models, audio, tmp_path / "out",
        title="My: Song?", artist="Cool/Band", cover_src=cover,
        unique_folder=True, progress=lambda s, p: events.append((s, p)),
    )
    info = json.loads((res1.folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_songAuthorName"] == "Cool/Band"
    assert info["_coverImageFilename"] == "cover.jpg"
    assert (res1.folder / "cover.jpg").read_bytes() == cover.read_bytes()
    stages = [s for s, _ in events]
    assert stages[0] == "analyzing" and "notes" in stages and "layers" in stages and stages[-1] == "writing"
    pcts = [p for _, p in events]
    assert pcts == sorted(pcts) and pcts[0] >= 0 and pcts[-1] == 100

    res2 = _gen(
        models, audio, tmp_path / "out",
        title="My: Song?", artist="Cool/Band", unique_folder=True,
    )
    assert res2.folder != res1.folder
    assert res2.folder.name == f"{res1.folder.name} (2)"
    assert res1.folder.is_dir() and res2.folder.is_dir()


def test_generate_map_cancel_mid_sampling_cleans_up(models, tmp_path):
    from bsmaker.generate import GenerateCancelled

    audio = write_audio(tmp_path / "a.wav")
    out = tmp_path / "out"
    calls = 0

    def cancel_after_notes_start() -> bool:
        nonlocal calls
        calls += 1
        return calls > 4

    with pytest.raises(GenerateCancelled):
        _gen(models, audio, out, is_cancelled=cancel_after_notes_start)
    assert not out.exists() or list(out.iterdir()) == []


def test_generate_map_monotonic_progress_across_density_rounds_and_layer_cancel(models, tmp_path, monkeypatch):
    from bsmaker import generate as G
    from bsmaker.generate import GenerateCancelled
    from bsmaker.models.layers import what as W
    from bsmaker.models.pattern.sample import SampleReport

    audio = write_audio(tmp_path / "rounds.wav")
    orig_sample = G.sample_notes
    calls = 0

    def two_round_sample(pattern, planned, mel, target_nps, styles, seed, mask_cfg, device, *, is_cancelled=None, on_progress=None):
        nonlocal calls
        calls += 1
        if on_progress is not None:
            on_progress(1, 2)
            on_progress(2, 2)
        notes, rep = orig_sample(pattern, planned, mel, target_nps, styles, seed, mask_cfg, device)
        if calls == 1:
            return notes[:2], SampleReport(len(planned), len(notes[:2]), 0, 0, 0)
        return notes, rep

    monkeypatch.setattr(G, "sample_notes", two_round_sample)
    events: list[tuple[str, int]] = []
    _gen(models, audio, tmp_path / "out_rounds", progress=lambda s, p: events.append((s, p)))
    pcts = [p for _, p in events]
    assert calls >= 2
    assert pcts == sorted(pcts)

    seen_cancel_args = []
    monkeypatch.setattr(G, "sample_notes", orig_sample)
    orig_generate_lights = G.generate_lights

    def spy_lights(*a, is_cancelled=None, **kw):
        seen_cancel_args.append(is_cancelled)
        if is_cancelled and is_cancelled():
            raise GenerateCancelled("cancelled")
        return orig_generate_lights(*a, is_cancelled=is_cancelled, **kw)

    monkeypatch.setattr(G, "generate_lights", spy_lights)
    cancel_fn = lambda: True
    # Also verify sample_layer_values raises GenerateCancelled directly when cancelled
    import numpy as np
    from bsmaker.models.layers.what import sample_layer_values
    lm = make_layer_models()
    with pytest.raises(GenerateCancelled):
        sample_layer_values(
            lm.lights_what, np.array([1.0]), np.array([0]), np.zeros((100, 80), dtype=np.float32),
            [], 3.0, 0, "cpu", is_cancelled=cancel_fn,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/formats/test_writer.py::test_writer_custom_jpg_cover_and_non_jpg_conversion tests/test_generate.py::test_generate_map_artist_cover_unique_folder_and_progress tests/test_generate.py::test_generate_map_cancel_mid_sampling_cleans_up -v
```
Expected: FAIL (`TypeError` / `ImportError`).

- [ ] **Step 3: Implement `GenerateCancelled`, writer cover support, and `generate_map` hooks**

1. Create `src/bsmaker/errors.py` containing `class GenerateCancelled(Exception): pass`.
2. In `src/bsmaker/formats/writer.py`:
   - Add `_convert_cover(src: Path, dst: Path, *, ffmpeg_path: Path | None = None) -> None` running `ffmpeg -nostdin -y -loglevel error -i <src> -frames:v 1 -q:v 2 <dst>` with `creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)` on Windows.
   - Extend `write_map_multi(..., cover_src: Path | None = None, ffmpeg_path: Path | None = None) -> Path`:
     - If `cover_src is not None`: validate `Path(cover_src).is_file()` (else raise `ValueError(f"cover file not found: {cover_src}")`), set `cover_name = "cover.jpg"`.
     - If `cover_src is None`: keep `cover_name = "cover.png"` and `(tmp / "cover.png").write_bytes(_png(256, (40, 90, 160)))` (exact pre-change bytes).
     - Inside `tmp` block when `cover_src is not None`: if suffix lower in `(".jpg", ".jpeg")`, `shutil.copyfile(cover_src, tmp / "cover.jpg")`; else call `_convert_cover(Path(cover_src), tmp / "cover.jpg", ffmpeg_path=ffmpeg_path)`.
3. In `src/bsmaker/models/pattern/sample.py` (`sample_notes`) and `src/bsmaker/models/layers/what.py` (`sample_layer_values`):
   - Add `is_cancelled: Callable[[], bool] | None = None` (and `on_progress: Callable[[int, int], None] | None = None` on `sample_notes`). Check `is_cancelled` before entering the loop, every 32 iterations, and before returning: if `is_cancelled is not None and is_cancelled(): raise GenerateCancelled("cancelled")`; call `on_progress(i + 1, len(times))` every 16 iterations or on last item when provided.
4. In `src/bsmaker/models/ornaments/sample.py` (`sample_ornaments`, `_probabilities`):
   - Add `is_cancelled: Callable[[], bool] | None = None`, checking it before/per model chunk in `_probabilities` and every 32 notes in `sample_ornaments`, raising `GenerateCancelled("cancelled")` when triggered.
5. In `src/bsmaker/layers_generate.py` (`generate_decor`, `generate_lights`):
   - Add `is_cancelled: Callable[[], bool] | None = None`, check before/after `predict_layer_events`, and thread `is_cancelled=is_cancelled` into `sample_layer_values(...)`.
6. In `src/bsmaker/generate.py`:
   - Re-export `GenerateCancelled`. Define `ProgressCallback = Callable[[str, int], None]`, `CancelCheck = Callable[[], bool]`.
   - Helper `_check_cancel(is_cancelled)` and monotonic emitter `_ProgressTracker` (or closure) tracking `last_pct = 0` and clamping every emitted value with `pct = max(last_pct, computed_pct)` so density re-rounds inside `_generate_level_notes` (`MAX_DENSITY_ROUNDS = 3`) never emit decreasing percentages.
   - Stage percentage bands:
     - `"analyzing"`: 0 on entry -> 5 after `decode_mono` -> 10 after `log_mel` -> 15 after `detect_grid`.
     - `"notes"`: spans `15..75` across `len(targets)` levels (forwarding `sample_notes` `on_progress` scaled into the current level's sub-range and clamped via `max(last_pct, pct)`).
     - `"layers"`: 75 -> 80 after ornaments -> 86 after decor -> 92 after lights & `check_layers`.
     - `"writing"`: 92 before `write_map_multi` -> 100 after `write_map_multi`.
   - Sanitize `artist`: `clean_artist = re.sub(r"[\x00-\x1f\x7f]", "", artist).strip()[:120] or "Unknown"`.
   - Pass `ffmpeg_path` to `decode_mono` and `write_map_multi`.
   - Thread `is_cancelled` into `_generate_level_notes` -> `sample_notes`, `sample_ornaments`, `generate_decor`, and `generate_lights`.
   - If `unique_folder=True` and `not overwrite` and `folder.exists()`: find first `n >= 2` where `Path(out_dir) / f"{base_name} ({n})"` does not exist.
   - Check `is_cancelled` before every stage and before `write_map_multi`.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/formats/test_writer.py tests/test_generate.py -v
```
Expected: PASS (including `test_all_layers_off_is_byte_identical_to_pre_change_output` and `test_notes_only_output_is_byte_identical_to_the_old_writer`).

- [ ] **Step 5: Commit**

```bash
git add src/bsmaker/errors.py src/bsmaker/formats/writer.py src/bsmaker/models/pattern/sample.py src/bsmaker/models/ornaments/sample.py src/bsmaker/models/layers/what.py src/bsmaker/layers_generate.py src/bsmaker/generate.py tests/formats/test_writer.py tests/test_generate.py
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(generate): cover image, artist, unique folder, progress and cancel hooks" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 3: Shared generation pipeline (`src/bsmaker/pipeline.py`) and CLI refactor

**Files:**
- Create: `src/bsmaker/pipeline.py`
- Modify: `src/bsmaker/cli.py` (`generate_cmd`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1 (`Config`, `Config._layer_stats_path`, `require_ffmpeg`), Task 2 (`generate_map`, `GenerateCancelled`, `ProgressCallback`, `CancelCheck`), `load_timing`, `load_pattern`, `load_layer_models`, `load_layer_stats`, `load_playability_config`, `auto_ornaments_level`, `parse_levels`, `require_stats`, `GenerateError`.
- Produces:
  - `MissingModelsError(GenerateError)`
  - `LoadedPipeline` dataclass: `timing: LoadedTiming`, `pattern: LoadedPattern`, `mask_cfg: PlayabilityConfig`, `layer_models: LayerModels`, `layer_warnings: tuple[str, ...]`, `layer_stats: dict | None`, `layer_stats_missing_path: Path | None`, `playability_missing: bool`, `device: str`
  - `resolve_device(device: str | None) -> str` (validates `None | "cuda" | "cpu"`; checks `torch.cuda.is_available()`, raising `GenerateError` if `"cuda"` unavailable)
  - `load_pipeline(cfg: Config, device: str | None = None, *, timing_ckpt: Path | None = None, pattern_ckpt: Path | None = None, load_layers: bool = True) -> LoadedPipeline`
  - `pipeline_to_device(pipeline: LoadedPipeline, device: str) -> None`
  - `pipeline_to_cpu(pipeline: LoadedPipeline) -> None`
  - `run_generate(pipeline: LoadedPipeline, audio_path: Path, *, out_dir: Path, nps: float | None = None, levels: Sequence[str] | None = None, styles: Sequence[str] = (), seed: int = 0, overwrite: bool = False, unique_folder: bool = False, title: str | None = None, artist: str = "Unknown", cover_src: Path | None = None, ornaments: int | None = None, lights: int = 0, environment: str = "DefaultEnvironment", lights_rules: bool = False, ffmpeg_path: Path | None = None, progress: ProgressCallback | None = None, is_cancelled: CancelCheck | None = None) -> GenerateResult`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:
```python
from dataclasses import asdict
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_generate import LAYER_STATS, STATS, make_layer_models, make_models, write_audio  # noqa: E402
from bsmaker.config import Config
from bsmaker.data.layer_stats import write_layer_stats
from bsmaker.models.common import save_checkpoint
from bsmaker.pipeline import (
    MissingModelsError, load_pipeline, pipeline_to_cpu, pipeline_to_device, run_generate,
)


def _setup_pack(tmp_path: Path, *, with_layers: bool = False) -> Config:
    ckpts = tmp_path / "checkpoints"
    t, p = make_models()
    for name, loaded in (("timing", t), ("pattern", p)):
        save_checkpoint(ckpts / name / "best.pt", {
            "model": loaded.model.state_dict(), "model_config": asdict(loaded.config),
            "stats": STATS, "step": 1, "best": {},
        })
    play = tmp_path / "playability.toml"
    play.write_text("[mask]\n[data]\n", encoding="utf-8")
    lstats = tmp_path / "layer_stats.json"
    write_layer_stats(LAYER_STATS, lstats)
    if with_layers:
        lm = make_layer_models()
        save_checkpoint(ckpts / "layers" / "ornaments" / "best.pt", {
            "model": lm.ornaments.model.state_dict(),
            "model_config": asdict(lm.ornaments.config), "stats": STATS,
        })
    return Config.from_paths(
        tmp_path / "CustomLevels", checkpoints_dir=ckpts,
        playability_path=play, layer_stats_path=lstats,
    )


def test_load_pipeline_missing_core_checkpoint_raises(tmp_path: Path) -> None:
    cfg = Config.from_paths(tmp_path / "CustomLevels", checkpoints_dir=tmp_path / "empty")
    with pytest.raises(MissingModelsError, match="checkpoint not found"):
        load_pipeline(cfg, device="cpu")


def test_load_pipeline_records_custom_missing_layer_stats_path(tmp_path: Path) -> None:
    cfg = _setup_pack(tmp_path, with_layers=False)
    missing_stats = tmp_path / "custom" / "my_layer_stats.json"
    cfg_custom = Config.from_paths(
        cfg.custom_levels,
        checkpoints_dir=cfg.checkpoints_dir,
        playability_path=cfg.playability_path,
        layer_stats_path=missing_stats,
    )
    pipe = load_pipeline(cfg_custom, device="cpu", load_layers=True)
    assert pipe.layer_stats is None
    assert pipe.layer_stats_missing_path == missing_stats


def test_run_generate_filters_layer_warnings_by_options(tmp_path: Path) -> None:
    cfg = _setup_pack(tmp_path, with_layers=False)
    pipe = load_pipeline(cfg, device="cpu", load_layers=True)
    audio = write_audio(tmp_path / "song.wav")

    res_off = run_generate(
        pipe, audio, out_dir=tmp_path / "out1", nps=3.0, ornaments=0, lights=0,
    )
    assert not any("checkpoint" in w for w in res_off.warnings)

    res_on = run_generate(
        pipe, audio, out_dir=tmp_path / "out2", nps=3.0, ornaments=2, lights=2,
    )
    assert any("ornaments" in w for w in res_on.warnings)
    assert any("lights-when" in w for w in res_on.warnings)
    assert res_on.lights_report is not None and res_on.lights_report["used_rules"] is True

    pipeline_to_cpu(pipe)
    pipeline_to_device(pipe, "cpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'bsmaker.pipeline'`).

- [ ] **Step 3: Implement `src/bsmaker/pipeline.py` and refactor `cli.py:generate_cmd`**

1. Create `src/bsmaker/pipeline.py`:
   - `load_pipeline` checks `timing_ck` and `pattern_ck` (raising `MissingModelsError(f"checkpoint not found: {ck}; train the models first")`), validates `require_stats`, loads `mask_cfg`, records `playability_missing = cfg.playability_path is None`. When `load_layers=True`, calls `load_layer_models(cfg.checkpoints_dir, dev)` and loads `load_layer_stats(cfg.layer_stats_path)` if `cfg.layer_stats_path` is not None (else sets `layer_stats_missing_path = cfg._layer_stats_path or (cfg.base_dir / "layer_stats.json")`).
   - `run_generate` computes `target_nps_list` (from `levels` + `timing.stats["rank_median_nps"]` or `nps`), determines `need_orn_decor`, `need_lights_model`, `need_layer_stats`, slices `pipeline.layer_models` to needed slots, filters `pipeline.layer_warnings` to `needed_slots` (matching existing `cli.py` prefix filter), includes `_NO_PLAYABILITY_WARNING` if `playability_missing`, includes missing `layer_stats.json` warning if `need_layer_stats and pipeline.layer_stats is None`, and delegates to `generate_map(...)`.
   - `pipeline_to_device(pipeline, device)` moves `timing.model`, `pattern.model`, and any non-None models inside `layer_models` (`ornaments`, `decor_when`, `decor_what`, `lights_when`, `lights_what`) to `torch.device(device)` and updates `pipeline.device`. `pipeline_to_cpu(pipeline)` calls `pipeline_to_device(pipeline, "cpu")`, `gc.collect()`, and if `torch.cuda.is_available()`, `torch.cuda.empty_cache()`.
2. Update `src/bsmaker/cli.py` `generate_cmd` to use `load_pipeline(..., load_layers=need_orn_decor or need_lights_model)` and `run_generate(...)`, preserving the exact `_fail(...)` messages, exit codes (`1` vs `2`), stderr playability warning, and stdout summary lines.

- [ ] **Step 4: Run focused tests to verify they pass**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/test_pipeline.py tests/test_cli_generate.py tests/test_cli_layers.py -v
```
Expected: PASS on all tests across all three files.

- [ ] **Step 5: Commit**

```bash
git add src/bsmaker/pipeline.py src/bsmaker/cli.py tests/test_pipeline.py
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "refactor: extract shared generation pipeline from CLI" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 4: YouTube `yt-dlp` wrapper (`src/bsmaker/engine/youtube.py`)

**Files:**
- Create: `src/bsmaker/engine/__init__.py`
- Create: `src/bsmaker/engine/youtube.py`
- Test: `tests/engine/test_youtube.py`

**Interfaces:**
- Consumes: Task 2 (`GenerateCancelled`, `CancelCheck`).
- Produces:
  - `YtdlpMissingError(Exception)`, `DownloadError(Exception)`
  - `YouTubeItem(id: str, url: str, title: str, artist: str, track_title: str, channel: str, duration_s: float | None, thumbnail_url: str | None)` (frozen dataclass)
  - `DownloadResult(audio_path: Path, cover_path: Path | None, title: str, artist: str)` (frozen dataclass)
  - `is_youtube_url(text: str) -> bool`
  - `parse_artist_title(video_title: str, uploader: str = "") -> tuple[str, str]`
  - `require_ytdlp(ytdlp_path: Path | None = None) -> str`
  - `search_youtube(query: str, *, ytdlp_path: Path | None = None, limit: int = 10, timeout_s: float = 30.0, runner=...) -> list[YouTubeItem]`
  - `resolve_youtube(url: str, *, ytdlp_path: Path | None = None, timeout_s: float = 30.0, runner=...) -> YouTubeItem`
  - `download_youtube(url: str, dest_dir: Path, *, ytdlp_path: Path | None = None, ffmpeg_path: Path | None = None, progress: Callable[[int], None] | None = None, is_cancelled: CancelCheck | None = None, popen=...) -> DownloadResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/engine/test_youtube.py`:
```python
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
    assert all(kw.get("encoding") == "utf-8" and kw.get("errors") == "replace" for kw in seen_kwargs)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/engine/test_youtube.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'bsmaker.engine'`).

- [ ] **Step 3: Implement `src/bsmaker/engine/__init__.py` and `src/bsmaker/engine/youtube.py`**

- `is_youtube_url(text)`: parses URL scheme `http`/`https` and host in `{"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}`.
- `parse_artist_title(video_title, uploader="")`: strips trailing bracketed noise (`(Official Video)`, `(Official Audio)`, `(Official Music Video)`, `(Lyrics)`, `[Official ...]`, `(Audio)` case-insensitive), splits on first `" - "`, `" – "`, or `" — "` if non-empty on both sides; falls back to `(uploader.strip() or "Unknown", cleaned or "untitled")`.
- `require_ytdlp(ytdlp_path=None)`: returns `str(ytdlp_path)` if `is_file()`, else raises `YtdlpMissingError`; if `None`, resolves via `shutil.which("yt-dlp")`.
- `search_youtube`: invokes `[bin, "--dump-json", "--flat-playlist", "--no-warnings", "--ignore-errors", f"ytsearch{limit}:{query}"]` with `shell=False, text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)` (on Windows), skips live/unplayable entries (`is_live=True` or missing url/id), parses each JSON line into `YouTubeItem`.
- `resolve_youtube`: invokes `[bin, "--dump-json", "--no-playlist", "--no-warnings", url]` with `shell=False, text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)` (on Windows), raises `DownloadError` on invalid URL or non-zero exit / empty output.
- `download_youtube`: creates `dest_dir`, runs `popen` with `[bin, "-f", "bestaudio/best", "--no-playlist", "--newline", "--progress", "--write-thumbnail", "--convert-thumbnails", "jpg", "-o", str(dest_dir / "audio.%(ext)s"), ...(--ffmpeg-location if ffmpeg_path), url]` (`shell=False, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)` on Windows). Streams stdout lines matching `r"\[download\]\s+(\d+(?:\.\d+)?)%"` -> `progress(int(float(pct)))`, checks `is_cancelled()` every line -> terminates/kills process, cleans `dest_dir`, and raises `GenerateCancelled`. Finds downloaded audio file (`audio.*` excluding `.jpg`/`.jpeg`/`.png`/`.webp`/`.part`) and optional thumbnail (`audio.jpg`/`audio.png`/`audio.webp`).

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/engine/test_youtube.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bsmaker/engine/__init__.py src/bsmaker/engine/youtube.py tests/engine/test_youtube.py
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(engine): add yt-dlp YouTube search, resolve and download wrapper" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 5: ZIP export and Steam `CustomLevels` detector (`src/bsmaker/engine/export.py`, `src/bsmaker/engine/steam.py`)

**Files:**
- Create: `src/bsmaker/engine/export.py`
- Create: `src/bsmaker/engine/steam.py`
- Test: `tests/engine/test_export.py`, `tests/engine/test_steam.py`

**Interfaces:**
- Produces:
  - `export_map_zip(map_folder: Path, out_zip: Path) -> Path`
  - `parse_library_folders_vdf(text: str) -> list[Path]`
  - `detect_custom_levels(*, reg_paths: Sequence[Path] | None = None, steam_path_override: Path | None = None) -> Path | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/engine/test_export.py`:
```python
import zipfile
from pathlib import Path
import pytest

from bsmaker.engine.export import export_map_zip


def test_export_map_zip_places_files_at_root_and_rejects_non_map(tmp_path: Path) -> None:
    folder = tmp_path / "BSM - Test"
    folder.mkdir()
    (folder / "Info.dat").write_text("{}", encoding="utf-8")
    (folder / "ExpertStandard.dat").write_text("{}", encoding="utf-8")
    (folder / "song.egg").write_bytes(b"OggS")
    (folder / "ignored.tmp").write_bytes(b"x")

    out = export_map_zip(folder, tmp_path / "exports" / "Test.zip")
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["ExpertStandard.dat", "Info.dat", "song.egg"]

    not_map = tmp_path / "empty"
    not_map.mkdir()
    with pytest.raises(ValueError, match="Info.dat"):
        export_map_zip(not_map, tmp_path / "bad.zip")
```

Create `tests/engine/test_steam.py`:
```python
from pathlib import Path

from bsmaker.engine.steam import detect_custom_levels, parse_library_folders_vdf

VDF_SAMPLE = r'''
"libraryfolders"
{
	"0"
	{
		"path"		"C:\\Program Files (x86)\\Steam"
		"apps"
		{
			"228980"		"0"
		}
	}
	"1"
	{
		"path"		"D:\\SteamLibrary"
		"apps"
		{
			"620980"		"123456"
		}
	}
}
'''


def test_parse_library_folders_vdf_and_detect_secondary_drive(tmp_path: Path) -> None:
    paths = parse_library_folders_vdf(VDF_SAMPLE)
    assert paths == [Path(r"C:\Program Files (x86)\Steam"), Path(r"D:\SteamLibrary")]

    steam_root = tmp_path / "Steam"
    lib2 = tmp_path / "D_SteamLibrary"
    (steam_root / "steamapps").mkdir(parents=True)
    bs_data = lib2 / "steamapps" / "common" / "Beat Saber" / "Beat Saber_Data"
    bs_data.mkdir(parents=True)
    vdf = f'"libraryfolders"\n{{\n "0" {{ "path" "{str(steam_root).replace("\\", "\\\\")}" }}\n "1" {{ "path" "{str(lib2).replace("\\", "\\\\")}" }}\n}}\n'
    (steam_root / "steamapps" / "libraryfolders.vdf").write_text(vdf, encoding="utf-8")

    found = detect_custom_levels(reg_paths=[steam_root])
    assert found == bs_data / "CustomLevels"
    assert found.is_dir()

    assert detect_custom_levels(reg_paths=[tmp_path / "missing_steam"]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/engine/test_export.py tests/engine/test_steam.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `export.py` and `steam.py`**

1. `src/bsmaker/engine/export.py`: `export_map_zip` checks `map_folder.is_dir()` and `(map_folder / "Info.dat").is_file()` (or `info.dat`, case-insensitive), creates `out_zip.parent`, writes regular non-symlink files (excluding `.tmp*`/`.partial` suffixes) flat at zip root into `out_zip.with_name(out_zip.name + ".tmp")`, and `os.replace`s into `out_zip`.
2. `src/bsmaker/engine/steam.py`:
   - `parse_library_folders_vdf(text)`: matches `"path"\s+"([^"]+)"` (unescaping `\\` -> `\`) across VDF text, deduplicating in order.
   - `_read_registry_steam_paths()`: guarded by `import winreg` (returning `[]` on non-Windows or missing keys) querying `HKCU\Software\Valve\Steam` (`SteamPath`), `HKLM\SOFTWARE\WOW6432Node\Valve\Steam` (`InstallPath`), `HKLM\SOFTWARE\Valve\Steam` (`InstallPath`), plus `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App 620980` (`InstallLocation` direct game folder check).
   - `detect_custom_levels`: inspects `reg_paths` (or `_read_registry_steam_paths()` + default `Path(r"C:\Program Files (x86)\Steam")`), parses `<steam>/steamapps/libraryfolders.vdf` if present, checks `<lib>/steamapps/common/Beat Saber/Beat Saber_Data` (creating `CustomLevels` if `Beat Saber_Data` exists and returning it; or returning existing `CustomLevels`). Returns `None` if not found.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/engine/test_export.py tests/engine/test_steam.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bsmaker/engine/export.py src/bsmaker/engine/steam.py tests/engine/test_export.py tests/engine/test_steam.py
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(engine): add Beat Saber map ZIP export and Steam CustomLevels detector" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 6: JSON-lines Engine Server & `bsmaker engine` command

**Files:**
- Create: `src/bsmaker/engine/server.py`
- Create: `src/bsmaker/engine/__main__.py`
- Modify: `src/bsmaker/cli.py` (add `@app.command("engine")`)
- Test: `tests/engine/test_server.py`

**Interfaces:**
- Consumes: Tasks 1–5 (`Config.from_paths`, `load_pipeline`, `run_generate`, `pipeline_to_device`, `pipeline_to_cpu`, `GenerateCancelled`, `MissingModelsError`, `GenerateError`, `AudioDecodeError`, `FfmpegMissingError`, `search_youtube`, `resolve_youtube`, `download_youtube`, `YtdlpMissingError`, `DownloadError`, `export_map_zip`, `detect_custom_levels`, `STYLES`, `SUPPORTED_ENVIRONMENTS`).
- Produces:
  - `EngineConfig(checkpoints_dir: Path, playability_path: Path | None = None, layer_stats_path: Path | None = None, ffmpeg_path: Path | None = None, ytdlp_path: Path | None = None, device: str | None = None, log_path: Path | None = None)`
  - `EngineServer(cfg: EngineConfig, stdin=..., stdout=..., stderr=...)` with `.serve() -> int`
  - `run_engine(cfg: EngineConfig) -> int`
  - Wire protocol (UTF-8 JSON lines):
    - Request: `{"id": str, "cmd": "info"|"search"|"resolve"|"download"|"generate"|"export_zip"|"cancel"|"shutdown", ...}`
    - Command schemas (exact fields expected by Task 7/9):
      - `"info"`: req `{"id", "cmd": "info", "device"?: "auto"|"cuda"|"cpu"}` -> result `{"version": str, "cuda_available": bool, "gpu_name": str | None, "active_device": str, "detected_custom_levels": str | None, "has_layer_models": bool, "missing_layer_models": list[str], "environments": list[str], "styles": list[str]}` (matches `EngineInfo`).
      - `"search"`: req `{"id", "cmd": "search", "query": str, "limit"?: int}` -> result `{"items": list[YouTubeItem dict]}`.
      - `"resolve"`: req `{"id", "cmd": "resolve", "url": str}` -> result `YouTubeItem dict` (`{id, url, title, artist, track_title, channel, duration_s, thumbnail_url}`).
      - `"download"`: req `{"id", "cmd": "download", "url": str, "dest_dir": str}` -> progress `{"id", "type": "progress", "stage": "downloading", "pct": int}` -> result `{"audio_path": str, "cover_path": str | None, "title": str, "artist": str}`.
      - `"generate"`: req `{"id", "cmd": "generate", "audio_path": str, "out_dir": str, "levels"?: list[str] | None, "nps"?: float | None, "styles"?: list[str], "ornaments"?: int | None, "lights"?: int, "environment"?: str, "seed"?: int, "lights_rules"?: bool, "title"?: str | None, "artist"?: str, "cover_src"?: str | None, "unique_folder"?: bool, "overwrite"?: bool, "device"?: "auto"|"cuda"|"cpu"}` -> progress `{"id", "type": "progress", "stage": "analyzing"|"notes"|"layers"|"writing", "pct": int}` -> result `GenerateOutput dict` (`{folder, title, artist, bpm, grid_steady, n_notes, nps, rank, levels: [{rank, target_nps, n_notes, nps, n_arcs, n_chains}], n_walls, n_bombs, n_lights, environment, warnings, used_rule_lights, device_used}`).
      - `"export_zip"`: req `{"id", "cmd": "export_zip", "folder": str, "out_zip": str}` -> result `{"out_zip": str}`.
      - `"cancel"`: req `{"id", "cmd": "cancel"}` -> result `{"cancelled": True}`.
      - `"shutdown"`: req `{"id", "cmd": "shutdown"}` -> result `{"shutdown": True}` (emitted after prior queued jobs finish).
    - Events: `{"id": str, "type": "progress", "stage": str, "pct": int}`, `{"id": str, "type": "result", "data": dict}`, `{"id": str, "type": "error", "code": str, "message_pl": str, "detail": str}`
  - Error codes: `ffmpeg_missing`, `ytdlp_missing`, `download_failed`, `no_models`, `cancelled`, `audio_too_short`, `audio_silent`, `invalid_audio`, `file_exists`, `not_found`, `invalid_args`, `busy`, `internal`.
  - CLI: `bsmaker engine --checkpoints-dir PATH [--playability PATH] [--layer-stats PATH] [--ffmpeg PATH] [--ytdlp PATH] [--device auto|cuda|cpu] [--log-file PATH]`

- [ ] **Step 1: Write the failing test**

Create `tests/engine/test_server.py`:
```python
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from test_generate import LAYER_STATS, STATS, make_models, write_audio  # noqa: E402
from bsmaker.data.layer_stats import write_layer_stats
from bsmaker.engine.server import EngineConfig, EngineServer
from bsmaker.models.common import save_checkpoint


def _setup_engine_dir(tmp_path: Path) -> EngineConfig:
    ckpts = tmp_path / "checkpoints"
    t, p = make_models()
    for name, loaded in (("timing", t), ("pattern", p)):
        save_checkpoint(ckpts / name / "best.pt", {
            "model": loaded.model.state_dict(), "model_config": asdict(loaded.config),
            "stats": STATS, "step": 1, "best": {},
        })
    play = tmp_path / "playability.toml"
    play.write_text("[mask]\n[data]\n", encoding="utf-8")
    lstats = tmp_path / "layer_stats.json"
    write_layer_stats(LAYER_STATS, lstats)
    return EngineConfig(
        checkpoints_dir=ckpts, playability_path=play, layer_stats_path=lstats,
        device="cpu", log_path=tmp_path / "engine.log",
    )


def _run_cmds(cfg: EngineConfig, cmds: list[dict]) -> list[dict]:
    stdin = io.StringIO("\n".join(json.dumps(c) for c in cmds) + "\n")
    stdout = io.StringIO()
    stderr = io.StringIO()
    srv = EngineServer(cfg, stdin=stdin, stdout=stdout, stderr=stderr)
    assert srv.serve() == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_engine_info_and_shutdown(tmp_path: Path) -> None:
    cfg = _setup_engine_dir(tmp_path)
    msgs = _run_cmds(cfg, [{"id": "1", "cmd": "info"}, {"id": "2", "cmd": "shutdown"}])
    info = next(m for m in msgs if m["id"] == "1")
    assert info["type"] == "result"
    assert info["data"]["has_layer_models"] is False
    assert "DefaultEnvironment" in info["data"]["environments"]
    assert "tech" in info["data"]["styles"]


def test_engine_generate_progress_result_and_vram_release(tmp_path: Path, monkeypatch) -> None:
    cfg = _setup_engine_dir(tmp_path)
    audio = write_audio(tmp_path / "song.wav")
    out_dir = tmp_path / "CustomLevels"
    out_dir.mkdir()
    freed = []
    import bsmaker.engine.server as S
    orig_cpu = S.pipeline_to_cpu
    monkeypatch.setattr(S, "pipeline_to_cpu", lambda p: (freed.append(True), orig_cpu(p))[1])

    msgs = _run_cmds(cfg, [
        {"id": "g1", "cmd": "generate", "audio_path": str(audio), "out_dir": str(out_dir),
         "nps": 3.0, "styles": ["tech"], "seed": 1, "title": "Song", "artist": "Artist",
         "lights": 2, "environment": "DefaultEnvironment"},
        {"id": "s1", "cmd": "shutdown"},
    ])
    progress = [m for m in msgs if m["id"] == "g1" and m["type"] == "progress"]
    res = next(m for m in msgs if m["id"] == "g1" and m["type"] == "result")
    assert progress and progress[-1]["pct"] == 100
    assert Path(res["data"]["folder"]).is_dir()
    assert res["data"]["used_rule_lights"] is True
    assert freed == [True]


def test_engine_internal_exception_never_leaks_traceback(tmp_path: Path, monkeypatch) -> None:
    cfg = _setup_engine_dir(tmp_path)
    import bsmaker.engine.server as S
    monkeypatch.setattr(S, "search_youtube", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret stack detail")))

    res = _run_cmds(cfg, [
        {"id": "q1", "cmd": "search", "query": "hello"},
        {"id": "s1", "cmd": "shutdown"},
    ])
    err = next(m for m in res if m["id"] == "q1")
    assert err["type"] == "error" and err["code"] == "internal"
    assert "Traceback" not in json.dumps(err)
    assert "secret stack detail" not in err["message_pl"]
    assert "secret stack detail" in (tmp_path / "engine.log").read_text(encoding="utf-8")


def test_engine_cancel_mid_generate_returns_cancelled_and_releases_vram(tmp_path: Path, monkeypatch) -> None:
    import threading
    import time
    cfg = _setup_engine_dir(tmp_path)
    audio = write_audio(tmp_path / "song.wav")
    out_dir = tmp_path / "CustomLevels"
    out_dir.mkdir()

    import bsmaker.engine.server as S
    started = threading.Event()
    freed = []
    orig_cpu = S.pipeline_to_cpu
    monkeypatch.setattr(S, "pipeline_to_cpu", lambda p: (freed.append(True), orig_cpu(p))[1])

    def slow_generate(*a, is_cancelled=None, **kw):
        started.set()
        for _ in range(100):
            if is_cancelled and is_cancelled():
                from bsmaker.errors import GenerateCancelled
                raise GenerateCancelled("cancelled")
            time.sleep(0.01)
        raise AssertionError("cancel flag never observed")

    monkeypatch.setattr(S, "run_generate", slow_generate)

    r_read, w_write = io.Pipe if hasattr(io, "Pipe") else (None, None)
    del r_read, w_write
    import queue
    lines: queue.Queue[str | None] = queue.Queue()

    class QueueReader:
        def readline(self):
            item = lines.get()
            return "" if item is None else item

    stdout = io.StringIO()
    srv = EngineServer(cfg, stdin=QueueReader(), stdout=stdout, stderr=io.StringIO())
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
    lines.put(json.dumps({
        "id": "g1", "cmd": "generate", "audio_path": str(audio), "out_dir": str(out_dir),
        "nps": 3.0, "styles": ["tech"], "seed": 1, "title": "S", "artist": "A",
    }) + "\n")
    assert started.wait(timeout=5.0)
    lines.put(json.dumps({"id": "c1", "cmd": "cancel"}) + "\n")
    lines.put(json.dumps({"id": "s1", "cmd": "shutdown"}) + "\n")
    lines.put(None)
    t.join(timeout=5.0)
    msgs = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    g1_err = next(m for m in msgs if m["id"] == "g1" and m["type"] == "error")
    assert g1_err["code"] == "cancelled"
    assert freed == [True]
    assert list(out_dir.iterdir()) == []


def test_engine_pipelined_jobs_and_unicode_non_cp1250_stdio(tmp_path: Path, monkeypatch) -> None:
    cfg = _setup_engine_dir(tmp_path)
    audio = write_audio(tmp_path / "song.wav")
    out_dir = tmp_path / "CustomLevels"
    out_dir.mkdir()
    unicode_title = "Żółć – 日本語 🎵"
    msgs = _run_cmds(cfg, [
        {"id": "i1", "cmd": "info"},
        {"id": "g1", "cmd": "generate", "audio_path": str(audio), "out_dir": str(out_dir),
         "nps": 3.0, "styles": ["tech"], "seed": 1, "title": unicode_title, "artist": "Artysta – 🎤"},
        {"id": "s1", "cmd": "shutdown"},
    ])
    assert [m["id"] for m in msgs if m["type"] == "result"] == ["i1", "g1", "s1"]
    gen_res = next(m for m in msgs if m["id"] == "g1" and m["type"] == "result")
    assert gen_res["data"]["title"] == unicode_title
    assert gen_res["data"]["artist"] == "Artysta – 🎤"

    # Also verify CLI / __main__ UTF-8 reconfigure helper does not raise on TextIOWrapper
    raw_in = io.TextIOWrapper(io.BytesIO(b'{"id":"1","cmd":"shutdown"}\n'), encoding="cp1250")
    raw_out = io.TextIOWrapper(io.BytesIO(), encoding="cp1250")
    raw_in.reconfigure(encoding="utf-8", errors="replace")
    raw_out.reconfigure(encoding="utf-8", errors="replace")
    srv = EngineServer(cfg, stdin=raw_in, stdout=raw_out, stderr=io.StringIO())
    assert srv.serve() == 0
    raw_out.flush()
    assert json.loads(raw_out.buffer.getvalue().decode("utf-8").splitlines()[0])["id"] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/engine/test_server.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'bsmaker.engine.server'`).

- [ ] **Step 3: Implement `server.py`, `__main__.py`, and `bsmaker engine` CLI**

1. `src/bsmaker/engine/server.py`:
   - `EngineServer` uses a thread-safe `write_lock` around `stdout.write(json.dumps(obj, ensure_ascii=False) + "\n"); stdout.flush()`.
   - Stdin reader thread reads lines and parses JSON into an unbounded FIFO `queue.Queue()` (`self._job_queue`):
     - `"cancel"` immediately sets `self._cancel_event.set()` on the reader thread and emits `result {"cancelled": True}` (without waiting behind the active job).
     - `"shutdown"` enqueues `("shutdown", req_id)` (or `None` sentinel on stdin EOF) onto `self._job_queue` without setting an early abort flag, so the worker thread completes all prior queued jobs (`info`, `search`, `resolve`, `download`, `generate`, `export_zip`) in FIFO order, emits `{"id": req_id, "type": "result", "data": {"shutdown": True}}` for `"shutdown"`, and then exits `serve()`.
     - Before starting each new job on the worker thread, clear `self._cancel_event.clear()` only if no cancel for that queued job is pending (or clear on job completion in `finally:` after checking). Keep `"busy"` in the error code table for schema completeness, while FIFO queuing handles pipelined commands cleanly without spurious `"busy"` errors.
   - Maps exceptions cleanly:
     - `GenerateCancelled` -> `"cancelled"`
     - `FfmpegMissingError` -> `"ffmpeg_missing"`
     - `YtdlpMissingError` -> `"ytdlp_missing"`
     - `DownloadError` -> `"download_failed"`
     - `MissingModelsError` -> `"no_models"`
     - `AudioDecodeError` -> `"invalid_audio"`
     - `GenerateError`: substring `"shorter"` -> `"audio_too_short"`, `"silent"` -> `"audio_silent"`, else `"invalid_audio"`
     - `FileExistsError` -> `"file_exists"`
     - `FileNotFoundError` -> `"not_found"`
     - `ValueError` -> `"invalid_args"`
     - Any other `Exception`: log full traceback to `cfg.log_path` (or `stderr`), emit `"internal"` with generic Polish message `"Wystąpił wewnętrzny błąd silnika. Szczegóły zapisano w logach."` and `detail=""`.
   - `generate` handler lazily caches `self._pipeline` (CPU RAM), resolves device (`req.get("device") or self.cfg.device` -> `"auto"` becomes `"cuda"` if available else `"cpu"`), moves to target device via `pipeline_to_device`, executes `run_generate` with `progress` and `is_cancelled=self._cancel_event.is_set`, and in `finally:` calls `pipeline_to_cpu(self._pipeline)` so GPU memory is always released. Populates `title`, `artist`, `used_rule_lights`, `device_used`, and `levels` matching `GenerateOutput` in Task 7.
2. Create `src/bsmaker/engine/__main__.py` reconfiguring `sys.stdin` and `sys.stdout` at startup (`if hasattr(s, "reconfigure"): s.reconfigure(encoding="utf-8", errors="replace")`), parsing CLI flags (`--checkpoints-dir`, `--playability`, `--layer-stats`, `--ffmpeg`, `--ytdlp`, `--device`, `--log-file`) via `argparse`, and calling `run_engine`.
3. Add `@app.command("engine")` in `src/bsmaker/cli.py` calling the same `reconfigure(encoding="utf-8", errors="replace")` on `sys.stdin` / `sys.stdout` at startup, accepting the same optional flags (defaulting from `load_config()` if `--checkpoints-dir` is omitted), and invoking `run_engine`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/engine/test_server.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bsmaker/engine/server.py src/bsmaker/engine/__main__.py src/bsmaker/cli.py tests/engine/test_server.py
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(engine): add JSON-lines engine server and bsmaker engine command" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 7: Electron + React + Vite + Tailwind + Framer Motion Scaffold & Polish Copy

**Files:**
- Create: `app/package.json`
- Create: `app/tsconfig.json`, `app/tsconfig.electron.json`
- Create: `app/vite.config.ts`, `app/tailwind.config.js`, `app/postcss.config.js`, `app/index.html`
- Create: `app/vitest.config.ts`, `app/playwright.config.ts`
- Create: `app/engine-version.json` (`{"engineVersion": "1"}`)
- Create: `app/src/types.ts`
- Create: `app/src/i18n/pl.ts`
- Create: `app/src/index.css`
- Create: `app/src/main.tsx`, `app/src/App.tsx`
- Test: `app/src/i18n/pl.test.ts`

**Interfaces:**
- Produces:
  - Exact shared TS types in `app/src/types.ts`:
    - `export type DevicePref = "auto" | "cuda" | "cpu";`
    - `export type StageName = "downloading" | "analyzing" | "notes" | "layers" | "writing";`
    - `export interface Settings { customLevels: string; device: DevicePref; exportDir: string; engineVersionInstalled: string | null; }`
    - `export interface YouTubeItem { id: string; url: string; title: string; artist: string; track_title: string; channel: string; duration_s: number | null; thumbnail_url: string | null; }`
    - `export interface EngineInfo { version: string; cuda_available: boolean; gpu_name: string | null; active_device: string; detected_custom_levels: string | null; has_layer_models: boolean; missing_layer_models: string[]; environments: string[]; styles: string[]; }`
    - `export interface MapGenerationConfig { mode: "pack" | "custom"; levels: string[] | null; nps: number | null; styles: string[]; ornaments: number | null; lights: number; environment: string; seed: number; lightsRules: boolean; }`
    - `export interface StartGenerationRequest { sourceType: "youtube" | "file" | "cached"; sourcePathOrUrl: string; title: string; artist: string; coverPath: string | null; config: MapGenerationConfig; regenerateId?: string; }`
    - `export interface LevelSummary { rank: string; target_nps: number; n_notes: number; nps: number; n_arcs: number; n_chains: number; }`
    - `export interface GenerateOutput { folder: string; title: string; artist: string; bpm: number; grid_steady: boolean; n_notes: number; nps: number; rank: string; levels: LevelSummary[]; n_walls: number; n_bombs: number; n_lights: number; environment: string; warnings: string[]; used_rule_lights: boolean; device_used: string; }`
    - `export interface LibraryEntry { id: string; title: string; artist: string; folder: string; createdAt: string; bpm: number; levels: Array<{ rank: string; nps: number; nNotes: number; nArcs: number; nChains: number }>; nWalls: number; nBombs: number; nLights: number; environment: string; audioCachePath: string; coverCachePath: string | null; config: MapGenerationConfig; }`
    - `export interface ProgressEvent { stage: StageName; pct: number; }`
    - `export interface EnginePackProgress { stage: "downloading" | "verifying" | "extracting"; pct: number; bytesDownloaded: number; totalBytes: number; partIndex: number; partCount: number; }`
    - `export interface UpdateStatus { state: "idle" | "checking" | "available" | "not-available" | "downloading" | "downloaded" | "error"; version?: string; pct?: number; }`
    - `export interface StartupState { needsEnginePack: boolean; engineVersion: string; appVersion: string; settings: Settings; info: EngineInfo | null; }`
    - `export interface BsmApi { getStartupState(): Promise<StartupState>; installEnginePack(): Promise<void>; getInfo(): Promise<EngineInfo>; getSettings(): Promise<Settings>; saveSettings(partial: Partial<Settings>): Promise<Settings>; validateCustomLevels(folder: string): Promise<{ valid: boolean; resolvedPath: string | null }>; pickFolder(title?: string): Promise<string | null>; pickAudioFile(): Promise<string | null>; openPath(targetPath: string): Promise<void>; openLogs(): Promise<void>; readCoverDataUrl(filePath: string): Promise<string | null>; searchYouTube(query: string): Promise<YouTubeItem[]>; resolveYouTube(url: string): Promise<YouTubeItem>; startGeneration(req: StartGenerationRequest): Promise<{ result: GenerateOutput; entry: LibraryEntry }>; cancelActive(): Promise<void>; listLibrary(): Promise<LibraryEntry[]>; deleteLibraryEntry(id: string): Promise<LibraryEntry[]>; exportMapZip(folder: string, suggestedName?: string): Promise<string>; checkForUpdates(): Promise<void>; installUpdate(): Promise<void>; onProgress(cb: (ev: ProgressEvent) => void): () => void; onEnginePackProgress(cb: (ev: EnginePackProgress) => void): () => void; onUpdateStatus(cb: (ev: UpdateStatus) => void): () => void; }`
  - Single Polish localization file `app/src/i18n/pl.ts`: `PL`, `ERRORS_PL`, `ENVIRONMENT_LABELS_PL`, `formatErrorPl(code: string): string` (fallback to `ERRORS_PL.internal`). `PL` must define exact test-pinned strings:
    - `common.next`: `"Dalej"`, `common.cancel`: `"Anuluj"`
    - `firstRun.downloadEngine`: `"Pobierz silnik"`, `firstRun.pickFolder`: `"Wybierz folder"`
    - `settings.cpuNotice`: `"Brak karty NVIDIA — generowanie na CPU będzie działać wolniej."`, `settings.openLogs`: `"Otwórz logi"`, `settings.checkUpdates`: `"Sprawdź aktualizacje"`, `settings.devices.cpu`: `"CPU"`
    - `wizard.song.searchPlaceholder`: `"Szukaj na YouTube lub wklej link..."`
    - `wizard.map.regenerateNotice`: `"Powstanie nowa mapa, a poprzedni folder trafi do Kosza po sukcesie."`, `wizard.map.advanced`: `"Zaawansowane"`, `wizard.map.lightsRules`: `"Światła regułowe"`, `wizard.map.generate`: `"Generuj mapę"`
    - `wizard.done.successTitle`: `"Mapa jest w grze!"`, `wizard.done.openFolder`: `"Otwórz folder"`, `wizard.done.warnings.missingLayerModels`: `"Brak modeli warstw — użyto ustawień zastępczych."`
    - `library.regenerate`: `"Wygeneruj ponownie"`, `library.exportZip`: `"Eksportuj ZIP"`, `library.delete`: `"Usuń"`, `library.confirmDelete`: `"Potwierdź"`
  - `app/package.json`: `"main": "dist-electron/main.js"`, NO `"type": "module"` (CommonJS package root for Electron main/preload; Vite bundles `src/` as ESM), scripts: `dev`, `build` (`vite build && tsc -p tsconfig.electron.json`), `typecheck`, `test` (`vitest run`), `e2e` (`playwright test`).
  - `app/tsconfig.electron.json`: compiles `electron/**/*.ts` (excluding `**/*.test.ts`) with `"module": "CommonJS"`, `"moduleResolution": "Node"`, `"outDir": "dist-electron"`, `"rootDir": "electron"`.
  - Dependencies include `"extract-zip"` in `dependencies` and `"@types/extract-zip"` in `devDependencies` (used by Task 10).

- [ ] **Step 1: Create `app/package.json`, configs, `types.ts`, `pl.ts`, and write the failing test**

Create `app/package.json` with `"main": "dist-electron/main.js"`, NO `"type": "module"`, and dependencies:
- `dependencies`: `clsx`, `electron-updater`, `extract-zip`, `framer-motion`, `lucide-react`, `react`, `react-dom`, `tailwind-merge`
- `devDependencies`: `@playwright/test`, `@testing-library/jest-dom`, `@testing-library/react`, `@types/extract-zip`, `@types/node`, `@types/react`, `@types/react-dom`, `@vitejs/plugin-react`, `autoprefixer`, `electron`, `electron-builder`, `jsdom`, `postcss`, `tailwindcss`, `typescript`, `vite`, `vitest`

Create `app/engine-version.json` (`{"engineVersion": "1"}`), `app/src/types.ts` (matching Task 6 wire structs and Task 8–13 IPC contracts), and `app/src/i18n/pl.ts` with all UI labels, error mappings (`ffmpeg_missing`, `ytdlp_missing`, `download_failed`, `no_models`, `cancelled`, `audio_too_short`, `audio_silent`, `invalid_audio`, `file_exists`, `not_found`, `invalid_args`, `busy`, `internal`), and environment Polish labels.

Create `app/src/i18n/pl.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { ERRORS_PL, ENVIRONMENT_LABELS_PL, formatErrorPl, PL } from "./pl";

const EXPECTED_CODES = [
  "ffmpeg_missing", "ytdlp_missing", "download_failed", "no_models", "cancelled",
  "audio_too_short", "audio_silent", "invalid_audio", "file_exists", "not_found",
  "invalid_args", "busy", "internal",
];

describe("Polish i18n dictionary", () => {
  it("maps every engine error code to non-empty Polish text without traceback jargon", () => {
    for (const code of EXPECTED_CODES) {
      const msg = ERRORS_PL[code];
      expect(msg, code).toBeTruthy();
      expect(msg).not.toMatch(/traceback|exception|stack/i);
    }
    expect(formatErrorPl("unknown_future_code")).toBe(ERRORS_PL.internal);
  });

  it("includes labels for all supported environments and wizard steps", () => {
    expect(Object.keys(ENVIRONMENT_LABELS_PL)).toHaveLength(10);
    expect(PL.wizard.steps.song).toBe("Utwór");
    expect(PL.wizard.steps.map).toBe("Mapa");
    expect(PL.wizard.steps.done).toBe("Gotowe");
    expect(PL.settings.cpuNotice).toMatch(/brak karty nvidia|wolniej/i);
    expect(PL.wizard.done.successTitle).toMatch(/mapa jest w grze/i);
    expect(PL.wizard.done.warnings.missingLayerModels).toMatch(/brak modeli warstw/i);
    expect(PL.library.confirmDelete).toMatch(/potwierdź/i);
  });
});
```

- [ ] **Step 2: Install npm dependencies and run test to verify setup**

Run inside `app/`:
```bash
npm install
npx vitest run src/i18n/pl.test.ts
```
Expected: FAIL before `pl.ts` has all entries (or after creating stub `pl.ts`), then pass once implemented.

- [ ] **Step 3: Complete `tsconfig.json`, `tsconfig.electron.json`, `vite.config.ts`, `tailwind.config.js`, `index.html`, `index.css`, `main.tsx`, placeholder `App.tsx`, and complete `pl.ts`**

Ensure `app/tsconfig.electron.json` compiles `electron/` to `dist-electron/` with `"module": "CommonJS"` and `"moduleResolution": "Node"`, `vite.config.ts` sets `base: "./"` so packaged Electron loads `dist/index.html` over `file://` cleanly, and Tailwind uses dark slate/zinc surface palette (`#09090b` / `#18181b`) with Beat Saber red (`#ef4444`) and blue (`#3b82f6`) accents.

- [ ] **Step 4: Run `typecheck`, `vitest`, and `vite build`**

Run inside `app/`:
```bash
npm run typecheck
npx vitest run src/i18n/pl.test.ts
npx vite build
```
Expected: PASS, 0 TypeScript errors, `dist/` built cleanly.

- [ ] **Step 5: Commit**

```bash
git add app/
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(app): scaffold Electron React Tailwind Framer Motion shell and Polish copy" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 8: Main process persistence, logging, and Recycle Bin safety

**Files:**
- Create: `app/electron/logger.ts`
- Create: `app/electron/store.ts`
- Create: `app/electron/library.ts`
- Test: `app/electron/library.test.ts`

**Interfaces:**
- Consumes: Task 7 (`Settings`, `LibraryEntry`).
- Produces:
  - `createLogger(logsDir: string) -> AppLogger` (`info`, `warn`, `error`, `engineStderr`, `getLogsDir`)
  - `atomicWriteJson(filePath: string, data: unknown): Promise<void>`
  - `SettingsStore(filePath: string, defaultExportDir: string)`: `load() -> Promise<Settings>`, `save(partial: Partial<Settings>) -> Promise<Settings>`
  - `isSafeBsmMapFolder(folder: string, customLevelsDir: string): boolean`
  - `LibraryStore(filePath: string, mediaDir: string)`: `list() -> Promise<LibraryEntry[]>`, `get(id: string)`, `saveEntry(entry: LibraryEntry)`, `cacheMedia(id: string, audioSrc: string, coverSrc: string | null) -> Promise<{ audioCachePath: string; coverCachePath: string | null }>`, `deleteEntry(id: string, customLevelsDir: string, trashFn: (p: string) => Promise<void>) -> Promise<LibraryEntry[]>`, `replaceAfterRegenerate(id: string, newEntry: LibraryEntry, customLevelsDir: string, trashFn: (p: string) => Promise<void>) -> Promise<LibraryEntry[]>`

- [ ] **Step 1: Write the failing test**

Create `app/electron/library.test.ts`:
```ts
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { isSafeBsmMapFolder, LibraryStore } from "./library";
import { SettingsStore } from "./store";

describe("isSafeBsmMapFolder and LibraryStore", () => {
  it("only allows deleting direct BSM - subfolders inside CustomLevels", () => {
    const cl = path.join("C:", "Games", "Beat Saber_Data", "CustomLevels");
    expect(isSafeBsmMapFolder(path.join(cl, "BSM - Song [Hard]"), cl)).toBe(true);
    expect(isSafeBsmMapFolder(path.join(cl, "Other Map"), cl)).toBe(false);
    expect(isSafeBsmMapFolder(path.join(cl, "sub", "BSM - Nested"), cl)).toBe(false);
    expect(isSafeBsmMapFolder(cl, cl)).toBe(false);
    expect(isSafeBsmMapFolder(path.join("C:", "Windows", "BSM - Fake"), cl)).toBe(false);
  });

  it("trashes only safe existing folders, drops missing folders silently, and swaps on regenerate", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-lib-"));
    const cl = path.join(tmp, "CustomLevels");
    const map1 = path.join(cl, "BSM - First [Hard, tech]");
    const map2 = path.join(cl, "BSM - First [Hard, tech] (2)");
    const outsider = path.join(tmp, "NotInCustomLevels");
    await fs.mkdir(map1, { recursive: true });
    await fs.mkdir(map2, { recursive: true });
    await fs.mkdir(outsider, { recursive: true });

    const trashed: string[] = [];
    const trashFn = async (p: string) => { trashed.push(p); await fs.rm(p, { recursive: true, force: true }); };

    const store = new LibraryStore(path.join(tmp, "library.json"), path.join(tmp, "media"));
    const audioSrc = path.join(tmp, "song.wav");
    await fs.writeFile(audioSrc, "wav");
    const cached = await store.cacheMedia("id1", audioSrc, null);

    await store.saveEntry({
      id: "id1", title: "First", artist: "Art", folder: map1, createdAt: "2026-11-16T10:00:00Z",
      bpm: 120, levels: [{ rank: "Hard", nps: 3.5, nNotes: 200, nArcs: 5, nChains: 2 }],
      nWalls: 10, nBombs: 4, nLights: 50, environment: "DefaultEnvironment",
      audioCachePath: cached.audioCachePath, coverCachePath: cached.coverCachePath,
      config: { mode: "pack", levels: ["Hard", "Expert", "ExpertPlus"], nps: null, styles: [], ornaments: null, lights: 2, environment: "DefaultEnvironment", seed: 42, lightsRules: false },
    });

    await store.replaceAfterRegenerate("id1", {
      ...(await store.get("id1"))!,
      folder: map2,
    }, cl, trashFn);
    expect(trashed).toEqual([map1]);
    expect((await store.get("id1"))?.folder).toBe(map2);

    // Crafted outside folder must NOT be trashed
    await store.saveEntry({ ...(await store.get("id1"))!, id: "bad", folder: outsider });
    await store.deleteEntry("bad", cl, trashFn);
    expect(trashed).toEqual([map1]);

    // Missing map folder is removed from library without error
    await fs.rm(map2, { recursive: true, force: true });
    await store.deleteEntry("id1", cl, trashFn);
    expect(await store.list()).toEqual([]);

    // If old folder was deleted by user beforehand and regeneration lands at the exact same path,
    // replaceAfterRegenerate must NOT trash the newly generated folder!
    const samePath = path.join(cl, "BSM - SamePath");
    await fs.mkdir(samePath, { recursive: true });
    await store.saveEntry({
      id: "same1", title: "Same", artist: "Art", folder: samePath, createdAt: "2026-11-16T10:00:00Z",
      bpm: 120, levels: [{ rank: "Hard", nps: 3.5, nNotes: 200, nArcs: 5, nChains: 2 }],
      nWalls: 10, nBombs: 4, nLights: 50, environment: "DefaultEnvironment",
      audioCachePath: cached.audioCachePath, coverCachePath: cached.coverCachePath,
      config: { mode: "pack", levels: ["Hard", "Expert", "ExpertPlus"], nps: null, styles: [], ornaments: null, lights: 2, environment: "DefaultEnvironment", seed: 42, lightsRules: false },
    });
    const trashedBeforeSame = trashed.length;
    await store.replaceAfterRegenerate("same1", { ...(await store.get("same1"))!, folder: samePath }, cl, trashFn);
    expect(trashed.length).toBe(trashedBeforeSame);
    expect(await fs.stat(samePath)).toBeTruthy();

    // Symlink / junction inside CustomLevels must NEVER be followed/trashed
    const linkTarget = path.join(tmp, "RealTarget");
    const linkFolder = path.join(cl, "BSM - Junction");
    await fs.mkdir(linkTarget, { recursive: true });
    await fs.symlink(linkTarget, linkFolder, process.platform === "win32" ? "junction" : "dir");
    await store.saveEntry({ ...(await store.get("same1"))!, id: "sym1", folder: linkFolder });
    await store.deleteEntry("sym1", cl, trashFn);
    expect(trashed.length).toBe(trashedBeforeSame);
    expect(await fs.stat(linkTarget)).toBeTruthy();

    const sStore = new SettingsStore(path.join(tmp, "settings.json"), path.join(tmp, "Downloads"));
    const loaded = await sStore.load();
    expect(loaded.device).toBe("auto");
    await sStore.save({ customLevels: cl });
    expect((await sStore.load()).customLevels).toBe(cl);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run inside `app/`:
```bash
npx vitest run electron/library.test.ts
```
Expected: FAIL (`Cannot find module './library'`).

- [ ] **Step 3: Implement `logger.ts`, `store.ts`, and `library.ts`**

1. `app/electron/logger.ts`: writes timestamped lines to `<logsDir>/app.log` (creates `logsDir` on first write; rotates if > 5 MB to `app.log.1`).
2. `app/electron/store.ts`: `atomicWriteJson` writes `<filePath>.tmp` then `fs.rename`s atomically; `SettingsStore` loads `settings.json` with safe defaults `{ customLevels: "", device: "auto", exportDir: defaultExportDir, engineVersionInstalled: null }` and persists updates atomically.
3. `app/electron/library.ts`:
   - `isSafeBsmMapFolder(folder, customLevelsDir)`: verifies non-empty `customLevelsDir`, `path.dirname(resolvedFolder)` matches `resolvedCustomLevels` (case-insensitive on `process.platform === "win32"`), and `path.basename(resolvedFolder).startsWith("BSM - ")`.
   - `LibraryStore`: loads `library.json` (backing up corrupt JSON if unparseable instead of crashing), writes atomically via `atomicWriteJson`, copies audio + optional cover into `media/<id>/` on `cacheMedia`, deletes `media/<id>/` on `deleteEntry`, and uses a shared `_trashIfSafe(folder, customLevelsDir, trashFn)` helper enforcing:
     - `isSafeBsmMapFolder(folder, customLevelsDir)`
     - `fs.lstat(resolvedFolder)` exists AND `!(await fs.lstat(resolvedFolder)).isSymbolicLink()` (never follow junctions/symlinks inside `CustomLevels`).
     - In `replaceAfterRegenerate(id, newEntry, customLevelsDir, trashFn)`: only call `_trashIfSafe(oldEntry.folder, ...)` when `path.resolve(oldEntry.folder).toLowerCase() !== path.resolve(newEntry.folder).toLowerCase()` (so regenerating into the same path after the user manually deleted the old folder never trashes the new map).

- [ ] **Step 4: Run test to verify it passes**

Run inside `app/`:
```bash
npx vitest run electron/library.test.ts
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/electron/logger.ts app/electron/store.ts app/electron/library.ts app/electron/library.test.ts
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(app): add atomic store, logger and safe Recycle Bin library" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 9: Main process `EngineManager`, IPC wiring, single instance & preload

**Files:**
- Create: `app/electron/engineManager.ts`
- Create: `app/electron/ipc.ts`
- Create: `app/electron/preload.ts`
- Create: `app/electron/main.ts`
- Test: `app/electron/engineManager.test.ts`

**Interfaces:**
- Consumes: Task 6 (exact JSON-lines schemas for `info`, `search`, `resolve`, `download`, `generate`, `export_zip`, `cancel`, `shutdown`), Task 7 (`BsmApi`, `types.ts`, CommonJS `dist-electron/main.js` without `"type": "module"`), Task 8 (`AppLogger`, `SettingsStore`, `LibraryStore`, `isSafeBsmMapFolder`).
- Produces:
  - `EngineManager`: spawns `command` + `args` (supports `BSM_ENGINE_CMD` / `BSM_ENGINE_ARGS` override or packaged `<userData>/engine/bsmaker-engine/bsmaker-engine.exe` with `--log-file <userData>/logs/engine.log`), with `windowsHide: true` and `env: { ...process.env, PYTHONUTF8: "1" }`, sends JSON-line commands, routes `progress` events to callback, serializes active jobs, sends `cancel` immediately, and force-kills the process tree (`taskkill /PID <pid> /T /F` on Windows, `proc.kill()` fallback) + restarts if `cancel` does not finish within `cancelTimeoutMs` (default 5000 ms).
  - `validateCustomLevelsFolder(pickedPath: string) -> Promise<{ valid: boolean; resolvedPath: string | null }>` (accepts either `CustomLevels` directly or a `Beat Saber` root containing `Beat Saber_Data/CustomLevels`, creating `CustomLevels` if `Beat Saber_Data` exists).
  - `main.ts`: `app.requestSingleInstanceLock()`, focuses existing window on second launch, registers IPC handlers via `ipc.ts`, disables menu bar, handles clean shutdown on `before-quit`.

- [ ] **Step 1: Write the failing test**

Create `app/electron/engineManager.test.ts`:
```ts
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { EngineManager } from "./engineManager";

const FAKE_SCRIPT = `
const rl = require("readline").createInterface({ input: process.stdin });
let timer = null;
rl.on("line", (line) => {
  const msg = JSON.parse(line);
  if (msg.cmd === "info") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { pid: process.pid, pyutf8: process.env.PYTHONUTF8 } }));
  } else if (msg.cmd === "generate") {
    console.log(JSON.stringify({ id: msg.id, type: "progress", stage: "notes", pct: 40 }));
    if (msg.hangOnCancel) {
      timer = setTimeout(() => {}, 60000);
    } else {
      console.log(JSON.stringify({ id: msg.id, type: "result", data: { folder: "BSM - OK" } }));
    }
  } else if (msg.cmd === "cancel") {
    console.log(JSON.stringify({ id: msg.id, type: "result", data: { cancelled: true } }));
  } else if (msg.cmd === "shutdown") {
    process.exit(0);
  }
});
`;

describe("EngineManager", () => {
  it("streams progress, returns result, and force-restarts on hung cancel", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-em-"));
    const script = path.join(tmp, "fake.cjs");
    await fs.writeFile(script, FAKE_SCRIPT);

    const events: Array<{ stage: string; pct: number }> = [];
    const mgr = new EngineManager({
      resolveLaunch: () => ({ command: process.execPath, args: [script] }),
      cancelTimeoutMs: 150,
      onProgress: (ev) => events.push({ stage: ev.stage, pct: ev.pct }),
      onStderr: () => {},
    });

    const info1 = await mgr.request<{ pid: number; pyutf8?: string }>("info", {});
    expect(info1.pyutf8).toBe("1");
    const genOk = await mgr.request<{ folder: string }>("generate", {});
    expect(genOk.folder).toBe("BSM - OK");
    expect(events).toEqual([{ stage: "notes", pct: 40 }]);

    const hanging = mgr.request("generate", { hangOnCancel: true });
    await new Promise((r) => setTimeout(r, 40));
    await expect(mgr.cancelActive().then(() => hanging)).rejects.toMatchObject({ code: "cancelled" });

    // Transparently restarted with a fresh process
    const info2 = await mgr.request<{ pid: number }>("info", {});
    expect(info2.pid).not.toBe(info1.pid);
    await mgr.stop();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run inside `app/`:
```bash
npx vitest run electron/engineManager.test.ts
```
Expected: FAIL (`Cannot find module './engineManager'`).

- [ ] **Step 3: Implement `engineManager.ts`, `ipc.ts`, `preload.ts`, and `main.ts`**

1. `app/electron/engineManager.ts`:
   - `EngineError extends Error { constructor(public code: string, public detail = "") ... }`
   - Default `resolveLaunch` passes `--log-file <logsDir>/engine.log` to `bsmaker-engine.exe` (and dev launch) so engine tracebacks are written to `%APPDATA%\BeatSaberMaker\logs\engine.log`.
   - Uses `child_process.spawn` with `stdio: ["pipe", "pipe", "pipe"]`, `windowsHide: true`, and `env: { ...process.env, PYTHONUTF8: "1" }`, `readline` on stdout.
   - `request(cmd, params)` assigns incrementing string ID, queues non-`cancel` requests sequentially (or rejects `cancel` cleanly if idle), resolves/rejects on matching `result`/`error` line, forwards `progress` lines to `onProgress`.
   - `cancelActive()`: sends `{ id, cmd: "cancel" }` immediately; starts `cancelTimeoutMs` timer. If active job hasn't settled when timer fires, kills the entire process tree (`spawn("taskkill", ["/PID", String(proc.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" })` on `process.platform === "win32"`, with `proc.kill()` fallback), settles the active job with `EngineError("cancelled")`, clears state, and allows the next `request()` to spawn a fresh child automatically.
   - `stop()`: sends `"shutdown"` with short timeout and falls back to the same process-tree kill if still alive.
2. `app/electron/ipc.ts`:
   - Implements all `BsmApi` handlers from `app/src/types.ts`:
     - `bsm:searchYouTube`: calls `engine.request<{ items: YouTubeItem[] }>("search", { query })` and returns `res.items`.
     - `bsm:resolveYouTube`: calls `engine.request<YouTubeItem>("resolve", { url })` and returns the `YouTubeItem`.
     - `bsm:startGeneration`: calls engine `download` (if `req.sourceType === "youtube"`, into temp download folder inside `userData/tmp-downloads`), then `library.cacheMedia(id, audio, cover)`, then engine `generate` with explicitly mapped snake_case fields (NEVER spreading `req.config`):
       `{ audio_path: cached.audioCachePath, out_dir: settings.customLevels, levels: req.config.mode === "pack" ? (req.config.levels ?? ["Hard", "Expert", "ExpertPlus"]) : req.config.nps !== null ? null : req.config.levels, nps: req.config.mode === "pack" ? null : req.config.nps, styles: req.config.styles, ornaments: req.config.ornaments, lights: req.config.lights, environment: req.config.environment, seed: req.config.seed, lights_rules: req.config.lightsRules, title: req.title, artist: req.artist, cover_src: cached.coverCachePath, unique_folder: true, device: settings.device }`.
       Then if `req.regenerateId` calls `library.replaceAfterRegenerate(...)` (trashing old folder safely!) else `library.saveEntry(...)`, cleaning up temp download folder in `finally`.
     - `bsm:exportMapZip`: prompts `dialog.showSaveDialog` defaulting to `<settings.exportDir>/<suggestedName || folderBase>.zip` (honouring `process.env.BSM_TEST_EXPORT_PATH` only when set in tests), calls `engine.request("export_zip", { folder, out_zip })`.
     - `bsm:validateCustomLevels`: accepts `CustomLevels` directly or `<dir>/Beat Saber_Data/CustomLevels` (creating `CustomLevels` if `Beat Saber_Data` exists).
     - `bsm:readCoverDataUrl`: validates resolved `filePath` lies inside `userData/media` and reads PNG/JPEG as `data:image/...;base64,...` (returns `null` if missing or outside media dir).
     - Wraps any `EngineError` thrown by IPC cleanly as `Error(`BSM_CODE:${err.code}`)` so the renderer extracts `code` and maps via `formatErrorPl(code)` without displaying stack traces.
3. `app/electron/preload.ts`: `contextBridge.exposeInMainWorld("bsm", ...)` matching `BsmApi` strictly (compiled as CommonJS by `tsconfig.electron.json`).
4. `app/electron/main.ts`: sets `app.setName("BeatSaberMaker")`, honours `BSM_USER_DATA_DIR` (used in E2E tests), enforces `app.requestSingleInstanceLock()`, loads `dist/index.html` (or `VITE_DEV_SERVER_URL`), shuts down `EngineManager` on quit.

- [ ] **Step 4: Run test and Electron TypeScript compile to verify**

Run inside `app/`:
```bash
npx vitest run electron/engineManager.test.ts
npx tsc -p tsconfig.electron.json --noEmit
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/electron/engineManager.ts app/electron/ipc.ts app/electron/preload.ts app/electron/main.ts app/electron/engineManager.test.ts
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(app): add EngineManager, single-instance main process, preload and IPC" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 10: Engine pack downloader (split parts + resume + SHA256) and `electron-updater`

**Files:**
- Create: `app/electron/enginePack.ts`
- Create: `app/electron/updater.ts`
- Modify: `app/electron/ipc.ts`, `app/electron/main.ts`
- Test: `app/electron/enginePack.test.ts`

**Interfaces:**
- Consumes: Task 7 (`EnginePackProgress`, `UpdateStatus`, `engine-version.json`, `extract-zip`), Task 8 (`AppLogger`, `SettingsStore`), Task 9 (`EngineManager`, `EngineManager.stop()`).
- Produces:
  - `EnginePackManifest`: `{ engineVersion: string, archiveSha256: string, totalSize: number, parts: Array<{ name: string, size: number, sha256: string }> }`
  - `EnginePackInstaller(options)`: `needsInstall() -> Promise<boolean>`, `getInstalledVersion() -> Promise<string | null>`, `install(onProgress: (p: EnginePackProgress) => void) -> Promise<void>`
  - `setupUpdater(mainWindow, logger)`: returns `{ checkForUpdates: () => Promise<void>, installUpdate: () => void }` (startup check skipped when `!app.isPackaged || process.env.BSM_DISABLE_UPDATER === "1"`).

- [ ] **Step 1: Write the failing test**

Create `app/electron/enginePack.test.ts`:
```ts
import crypto from "node:crypto";
import fs from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { EnginePackInstaller } from "./enginePack";

function sha256(buf: Buffer): string {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (c ^ 0xffffffff) >>> 0;
}

function makeStoreZip(files: Record<string, string>): Buffer {
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;
  for (const [name, content] of Object.entries(files)) {
    const nb = Buffer.from(name, "utf8");
    const data = Buffer.from(content, "utf8");
    const crc = crc32(data);
    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0); lh.writeUInt16LE(20, 4); lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(data.length, 18); lh.writeUInt32LE(data.length, 22); lh.writeUInt16LE(nb.length, 26);
    locals.push(lh, nb, data);
    const ch = Buffer.alloc(46);
    ch.writeUInt32LE(0x02014b50, 0); ch.writeUInt16LE(20, 4); ch.writeUInt16LE(20, 6);
    ch.writeUInt32LE(crc, 16); ch.writeUInt32LE(data.length, 20); ch.writeUInt32LE(data.length, 24);
    ch.writeUInt16LE(nb.length, 28); ch.writeUInt32LE(offset, 42);
    centrals.push(ch, nb);
    offset += lh.length + nb.length + data.length;
  }
  const cd = Buffer.concat(centrals);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(Object.keys(files).length, 8);
  eocd.writeUInt16LE(Object.keys(files).length, 10);
  eocd.writeUInt32LE(cd.length, 12); eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, eocd]);
}

describe("EnginePackInstaller", () => {
  it("downloads split parts, resumes via Range, verifies SHA256, and falls back on 200", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "bsm-pack-"));
    const zip = makeStoreZip({
      "bsmaker-engine/bsmaker-engine.exe": "EXE",
      "models/timing/best.pt": "PT",
    });
    const split = Math.floor(zip.length / 2);
    const parts = [zip.subarray(0, split), zip.subarray(split)];
    const manifest = {
      engineVersion: "1",
      archiveSha256: sha256(zip),
      totalSize: zip.length,
      parts: parts.map((p, i) => ({ name: `engine.zip.00${i + 1}`, size: p.length, sha256: sha256(p) })),
    };

    let mode: "range" | "no_range" | "corrupt" = "range";
    const rangeHeaders: string[] = [];
    const srv = http.createServer((req, res) => {
      if (req.url === "/manifest.json") {
        res.end(JSON.stringify(manifest));
        return;
      }
      const idx = req.url?.endsWith(".001") ? 0 : 1;
      const body = mode === "corrupt" && idx === 0 ? Buffer.alloc(parts[0].length, 0xff) : parts[idx];
      const rng = req.headers.range;
      if (rng) rangeHeaders.push(rng);
      if (mode === "range" && rng?.startsWith("bytes=")) {
        const start = Number(rng.slice(6).replace("-", ""));
        res.writeHead(206, { "Content-Range": `bytes ${start}-${body.length - 1}/${body.length}` });
        res.end(body.subarray(start));
      } else {
        res.writeHead(200, { "Content-Length": body.length });
        res.end(body);
      }
    });
    await new Promise<void>((r) => srv.listen(0, "127.0.0.1", () => r()));
    const port = (srv.address() as { port: number }).port;

    try {
      const dlDir = path.join(tmp, "dl");
      const engDir = path.join(tmp, "engine");
      await fs.mkdir(dlDir, { recursive: true });
      await fs.writeFile(path.join(dlDir, `${manifest.parts[0].name}.part`), parts[0].subarray(0, 20));

      const installer = new EnginePackInstaller({
        engineDir: engDir, downloadDir: dlDir, expectedVersion: "1",
        manifestUrl: `http://127.0.0.1:${port}/manifest.json`,
      });
      expect(await installer.needsInstall()).toBe(true);
      await installer.install(() => {});
      expect(rangeHeaders).toContain("bytes=20-");
      expect(await installer.needsInstall()).toBe(false);
      expect(await fs.readFile(path.join(engDir, "bsmaker-engine", "bsmaker-engine.exe"), "utf8")).toBe("EXE");

      // Corrupt download must reject and preserve existing engineDir
      mode = "corrupt";
      const installerV2 = new EnginePackInstaller({
        engineDir: engDir, downloadDir: path.join(tmp, "dl2"), expectedVersion: "2",
        manifestUrl: `http://127.0.0.1:${port}/manifest.json`,
      });
      await expect(installerV2.install(() => {})).rejects.toThrow(/SHA256/i);
      expect(await fs.readFile(path.join(engDir, "bsmaker-engine", "bsmaker-engine.exe"), "utf8")).toBe("EXE");

      // Server ignoring Range (200 OK) resets partial file cleanly
      mode = "no_range";
      const dl3 = path.join(tmp, "dl3");
      await fs.mkdir(dl3, { recursive: true });
      await fs.writeFile(path.join(dl3, `${manifest.parts[0].name}.part`), parts[0].subarray(0, 10));
      const installer3 = new EnginePackInstaller({
        engineDir: path.join(tmp, "engine3"), downloadDir: dl3, expectedVersion: "1",
        manifestUrl: `http://127.0.0.1:${port}/manifest.json`,
      });
      await installer3.install(() => {});
      expect(await installer3.needsInstall()).toBe(false);
    } finally {
      srv.close();
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run inside `app/`:
```bash
npx vitest run electron/enginePack.test.ts
```
Expected: FAIL (`Cannot find module './enginePack'`).

- [ ] **Step 3: Implement `enginePack.ts`, `updater.ts`, and wire into `ipc.ts` / `main.ts`**

1. `app/electron/enginePack.ts`:
   - Resolves expected `engineVersion` from `engine-version.json` (`process.resourcesPath` when packaged, `app/` in dev).
   - `needsInstall()`: returns `false` if `process.env.BSM_SKIP_ENGINE_PACK === "1"` or `process.env.BSM_ENGINE_CMD`; otherwise checks `<engineDir>/installed-version.json` matches expected `engineVersion` AND `<engineDir>/bsmaker-engine/bsmaker-engine.exe` exists.
   - `install(onProgress)`:
     - Manifest URL: `process.env.BSM_ENGINE_MANIFEST_URL` or `https://github.com/al3ksh/BeatSaberMaker/releases/download/v${appVersion}/engine-pack-manifest.json`.
     - Downloads each part into `<downloadDir>/<part.name>.part`: if `.part` exists with `0 < size < part.size`, sends `Range: bytes=${size}-`. If response is `206`, appends chunks; if `200`, truncates and writes from byte 0; follows GitHub 302 redirects cleanly (`fetch` handles redirects automatically!).
     - Verifies each part's SHA256 against `part.sha256` (if mismatch, deletes `.part` and throws).
     - Streams concatenation of parts into `<downloadDir>/engine-pack.zip` while computing `archiveSha256`, verifies against `manifest.archiveSha256`.
     - Extracts ZIP via `extract-zip` (streaming `yauzl` ZIP64 support for the ~2.5 GB archive, zero shell) into `<userData>/engine.tmp`, writes `installed-version.json`, and swaps into `<userData>/engine` atomically.
2. `app/electron/updater.ts`: configures `autoUpdater.autoDownload = true`, `autoUpdater.autoInstallOnAppQuit = true`, forwards `checking-for-update`, `update-available`, `update-not-available`, `download-progress`, `update-downloaded`, `error` to `BrowserWindow.webContents.send("bsm:updateStatus", ...)`.
3. Wire `bsm:installEnginePack`, `bsm:checkForUpdates`, `bsm:installUpdate`, and startup state in `ipc.ts` / `main.ts`: inside `bsm:installEnginePack`, call `await engineManager.stop()` BEFORE `await enginePackInstaller.install(...)` so a running `bsmaker-engine.exe` never holds a Windows file lock (`EPERM`/`EBUSY`) during the directory swap.

- [ ] **Step 4: Run tests to verify they pass**

Run inside `app/`:
```bash
npm install
npx vitest run electron/enginePack.test.ts
npx tsc -p tsconfig.electron.json --noEmit
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/package.json app/package-lock.json app/electron/enginePack.ts app/electron/updater.ts app/electron/ipc.ts app/electron/main.ts app/electron/enginePack.test.ts
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(app): add resumable SHA256-verified engine-pack installer and updater" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 11: UI primitives, App Shell, First-Run Setup Gate, and Settings Modal

**Files:**
- Create: `app/src/components/ui/primitives.tsx`
- Create: `app/src/components/FirstRun.tsx`
- Create: `app/src/components/SettingsModal.tsx`
- Modify: `app/src/App.tsx`, `app/src/i18n/pl.ts`
- Test: `app/src/components/FirstRun.test.tsx`, `app/src/components/SettingsModal.test.tsx`

**Interfaces:**
- Consumes: Task 7 (`PL`, `types.ts`), `window.bsm`.
- Produces:
  - `ui/primitives.tsx`: `Button`, `Card`, `Badge`, `Input`, `Slider`, `Checkbox`, `ProgressBar`, `Modal` (corporate dark aesthetic, red/blue sword accents, Framer Motion animations, accessible labels/roles).
  - `FirstRun`: handles engine pack download UI (when `startup.needsEnginePack`), then auto-detects `info.detected_custom_levels` or blocks progression with a folder picker (`pickFolder` + `validateCustomLevels`) until `customLevels` is valid and saved.
  - `SettingsModal`: edits `customLevels` (manual pick + auto-detect button), `device` (`auto` / `cuda` / `cpu` with GPU name or CPU-slower notice), `exportDir`, displays app + engine versions, "Sprawdź aktualizacje" button + restart-to-update CTA, and "Otwórz logi" button.
  - `App.tsx`: top navigation (`Kreator` | `Biblioteka` | `Ustawienia`), subtle CPU-mode banner when `!info.cuda_available`, update-ready banner, FirstRun gate before rendering main views.

- [ ] **Step 1: Write the failing component tests**

Create `app/src/components/FirstRun.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FirstRun } from "./FirstRun";

describe("FirstRun", () => {
  it("downloads engine pack when needed and blocks CustomLevels step until valid folder selected", async () => {
    let packProgressCb: ((ev: unknown) => void) | null = null;
    const installEnginePack = vi.fn(async () => {
      packProgressCb?.({ stage: "downloading", pct: 50, bytesDownloaded: 50, totalBytes: 100, partIndex: 1, partCount: 2 });
    });
    const pickFolder = vi.fn(async () => "C:\\Games\\Beat Saber");
    const validateCustomLevels = vi.fn(async () => ({
      valid: true,
      resolvedPath: "C:\\Games\\Beat Saber\\Beat Saber_Data\\CustomLevels",
    }));
    const saveSettings = vi.fn(async (p) => ({ customLevels: p.customLevels, device: "auto", exportDir: "D:\\Dl", engineVersionInstalled: "1" }));
    const getInfo = vi.fn(async () => ({
      version: "0.1.0", cuda_available: true, gpu_name: "RTX 4060 Ti", active_device: "cuda",
      detected_custom_levels: null, has_layer_models: false, missing_layer_models: [], environments: [], styles: [],
    }));
    (window as unknown as { bsm: unknown }).bsm = {
      installEnginePack, pickFolder, validateCustomLevels, saveSettings, getInfo,
      onEnginePackProgress: (cb: (ev: unknown) => void) => { packProgressCb = cb; return () => {}; },
    };
    const onComplete = vi.fn();
    render(
      <FirstRun
        startup={{
          needsEnginePack: true, engineVersion: "1", appVersion: "0.1.0",
          settings: { customLevels: "", device: "auto", exportDir: "D:\\Dl", engineVersionInstalled: null },
          info: null,
        }}
        onComplete={onComplete}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /pobierz silnik/i }));
    await waitFor(() => expect(installEnginePack).toHaveBeenCalled());
    await waitFor(() => expect(getInfo).toHaveBeenCalled());

    const cont = screen.getByRole("button", { name: /dalej/i });
    expect(cont).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /wybierz folder/i }));
    await waitFor(() => expect(validateCustomLevels).toHaveBeenCalledWith("C:\\Games\\Beat Saber"));
    await waitFor(() => expect(cont).not.toBeDisabled());
    fireEvent.click(cont);
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith({ customLevels: "C:\\Games\\Beat Saber\\Beat Saber_Data\\CustomLevels" }));
    expect(onComplete).toHaveBeenCalled();
  });
});
```

Create `app/src/components/SettingsModal.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SettingsModal } from "./SettingsModal";

describe("SettingsModal", () => {
  it("shows CPU note when CUDA unavailable and wires save, updates, and logs", async () => {
    const saveSettings = vi.fn(async (p) => ({
      customLevels: "C:\\CL", device: p.device ?? "auto", exportDir: "C:\\Exp", engineVersionInstalled: "1",
    }));
    const openLogs = vi.fn(async () => {});
    const checkForUpdates = vi.fn(async () => {});
    (window as unknown as { bsm: unknown }).bsm = {
      saveSettings, openLogs, checkForUpdates,
      pickFolder: vi.fn(async () => null),
      validateCustomLevels: vi.fn(),
      getInfo: vi.fn(),
      installUpdate: vi.fn(),
    };
    render(
      <SettingsModal
        open
        settings={{ customLevels: "C:\\CL", device: "auto", exportDir: "C:\\Exp", engineVersionInstalled: "1" }}
        info={{
          version: "0.1.0", cuda_available: false, gpu_name: null, active_device: "cpu",
          detected_custom_levels: null, has_layer_models: true, missing_layer_models: [], environments: [], styles: [],
        }}
        appVersion="0.1.0"
        updateStatus={{ state: "idle" }}
        onClose={() => {}}
        onSettingsSaved={() => {}}
      />,
    );
    expect(screen.getByText(/brak karty nvidia|wolniej/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /otwórz logi/i }));
    expect(openLogs).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /sprawdź aktualizacje/i }));
    expect(checkForUpdates).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /cpu/i }));
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith({ device: "cpu" }));
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run inside `app/`:
```bash
npx vitest run src/components/FirstRun.test.tsx src/components/SettingsModal.test.tsx
```
Expected: FAIL (`Cannot find module './FirstRun'`).

- [ ] **Step 3: Implement `primitives.tsx`, `FirstRun.tsx`, `SettingsModal.tsx`, and shell in `App.tsx`**

Keep all user-facing strings sourced from `PL` in `app/src/i18n/pl.ts`. Ensure Framer Motion `AnimatePresence` transitions run smoothly without blocking DOM testing.

- [ ] **Step 4: Run component tests and typecheck**

Run inside `app/`:
```bash
npm run typecheck
npx vitest run src/components/FirstRun.test.tsx src/components/SettingsModal.test.tsx
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/components/ui/primitives.tsx app/src/components/FirstRun.tsx app/src/components/SettingsModal.tsx app/src/App.tsx app/src/i18n/pl.ts app/src/components/FirstRun.test.tsx app/src/components/SettingsModal.test.tsx
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(ui): add corporate dark primitives, FirstRun gate, Settings modal and App shell" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 12: Wizard Step 1 (Utwór) & Step 2 (Mapa)

**Files:**
- Create: `app/src/components/wizard/StepSong.tsx`
- Create: `app/src/components/wizard/StepMap.tsx`
- Modify: `app/src/i18n/pl.ts`
- Test: `app/src/components/wizard/StepSong.test.tsx`, `app/src/components/wizard/StepMap.test.tsx`

**Interfaces:**
- Consumes: Task 7 (`PL`, `ENVIRONMENT_LABELS_PL`, `types.ts`), Task 11 (`ui/primitives`), `window.bsm`.
- Produces:
  - `SongDraft`: `{ sourceType: "youtube" | "file" | "cached"; sourcePathOrUrl: string; title: string; artist: string; coverPath: string | null; thumbnailUrl: string | null }`
  - `StepSong({ song, onChange, onNext }: { song: SongDraft; onChange: (next: SongDraft) => void; onNext: () => void })`:
    - YouTube search/URL box: auto-detects YouTube links (calling `resolveYouTube`), otherwise calls `searchYouTube` on submit (`<form role="search">`) and renders scrollable result cards (thumbnail, title, channel, formatted `mm:ss` duration).
    - "Z pliku" tab/button calling `pickAudioFile()`, pre-filling `title` from filename stem and `artist` as `"Unknown"` (editable).
    - Editable **Tytuł** and **Artysta** fields + cover preview before advancing to Step 2.
  - `PACK_DEFAULT_CONFIG: MapGenerationConfig` (`mode: "pack"`, `levels: ["Hard", "Expert", "ExpertPlus"]`, `nps: null`, `styles: []`, `ornaments: null`, `lights: 2`, `environment: "DefaultEnvironment"`, `seed: 0`, `lightsRules: false`) and `StepMap({ config, isRegenerate, onChange, onBack, onGenerate }: { config: MapGenerationConfig; isRegenerate: boolean; onChange: (next: MapGenerationConfig) => void; onBack: () => void; onGenerate: () => void })`:
    - Segmented toggle: **Pakiet** (domyślny) vs **Własne**.
    - **Pakiet**: summarizes `Hard + Expert + Expert+`, światła `2`, ozdoby `auto`, otoczenie `DefaultEnvironment`, losowy seed (generated automatically on submit).
    - **Własne**: toggle between `Poziomy` (checkboxes `Easy`..`ExpertPlus`, requiring >= 1) and `Jedno NPS` (slider `0.5`–`12`, step `0.5`), style chips (`tech`, `dance`, `speed`, `accuracy`, `balanced`, `challenge`, `fitness`), `Ozdoby` (`auto` | `0` | `1` | `2` | `3`), `Światła` (`0` | `1` | `2` | `3`), `Otoczenie` dropdown (10 `ENVIRONMENT_LABELS_PL` entries), `Seed` numeric input + dice button, collapsible `Zaawansowane` section containing `Światła regułowe` toggle (`lightsRules`).
    - When `isRegenerate=true`, displays confirmation notice explaining that a new map will be created and the previous folder moved to Recycle Bin after success, and button asks confirmation or confirms clearly before invoking `onGenerate`.

- [ ] **Step 1: Write the failing tests**

Create `app/src/components/wizard/StepSong.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StepSong } from "./StepSong";

describe("StepSong", () => {
  it("searches YouTube, populates editable artist/title on pick, and enables Dalej", async () => {
    const searchYouTube = vi.fn(async () => [{
      id: "1", url: "https://youtu.be/1", title: "Camellia - GHOST",
      artist: "Camellia", track_title: "GHOST", channel: "Camellia", duration_s: 240, thumbnail_url: null,
    }]);
    (window as unknown as { bsm: unknown }).bsm = {
      searchYouTube, resolveYouTube: vi.fn(), pickAudioFile: vi.fn(),
    };
    const onChange = vi.fn();
    const { rerender } = render(
      <StepSong
        song={{ sourceType: "youtube", sourcePathOrUrl: "", title: "", artist: "", coverPath: null, thumbnailUrl: null }}
        onChange={onChange}
        onNext={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /dalej/i })).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText(/szukaj|link/i), { target: { value: "ghost" } });
    fireEvent.submit(screen.getByRole("search"));
    await waitFor(() => expect(searchYouTube).toHaveBeenCalledWith("ghost"));
    fireEvent.click(await screen.findByText("Camellia - GHOST"));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      sourceType: "youtube", sourcePathOrUrl: "https://youtu.be/1", title: "GHOST", artist: "Camellia",
    }));

    rerender(
      <StepSong
        song={{ sourceType: "youtube", sourcePathOrUrl: "https://youtu.be/1", title: "GHOST", artist: "Camellia", coverPath: null, thumbnailUrl: null }}
        onChange={onChange}
        onNext={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /dalej/i })).not.toBeDisabled();
  });
});
```

Create `app/src/components/wizard/StepMap.test.tsx`:
```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PACK_DEFAULT_CONFIG, StepMap } from "./StepMap";

describe("StepMap", () => {
  it("defaults to Pakiet, supports Własne controls and shows saved seed on regenerate", () => {
    const onChange = vi.fn();
    const onGenerate = vi.fn();
    render(
      <StepMap
        config={{ ...PACK_DEFAULT_CONFIG, mode: "custom", seed: 12345 }}
        isRegenerate
        onChange={onChange}
        onBack={() => {}}
        onGenerate={onGenerate}
      />,
    );
    expect(screen.getByDisplayValue("12345")).toBeInTheDocument();
    expect(screen.getByText(/kosza/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /zaawansowane/i }));
    fireEvent.click(screen.getByLabelText(/światła regułowe/i));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ lightsRules: true }));

    fireEvent.click(screen.getByRole("button", { name: /generuj/i }));
    expect(onGenerate).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run inside `app/`:
```bash
npx vitest run src/components/wizard/StepSong.test.tsx src/components/wizard/StepMap.test.tsx
```
Expected: FAIL.

- [ ] **Step 3: Implement `StepSong.tsx` and `StepMap.tsx`**

Ensure Pakiet assigns a fresh random integer seed `[1, 2_000_000_000]` on submit (and persists it into the `LibraryEntry.config.seed` so when opened in Własne on re-generation, that exact seed is visible!).

- [ ] **Step 4: Run tests to verify they pass**

Run inside `app/`:
```bash
npm run typecheck
npx vitest run src/components/wizard/StepSong.test.tsx src/components/wizard/StepMap.test.tsx
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/components/wizard/StepSong.tsx app/src/components/wizard/StepMap.tsx app/src/i18n/pl.ts app/src/components/wizard/StepSong.test.tsx app/src/components/wizard/StepMap.test.tsx
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(ui): implement Wizard Step 1 (Song) and Step 2 (Map)" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 13: Wizard Step 3 (Gotowe), Wizard Orchestrator & Library View

**Files:**
- Create: `app/src/components/wizard/StepProgressDone.tsx`
- Create: `app/src/components/wizard/Wizard.tsx`
- Create: `app/src/components/LibraryView.tsx`
- Modify: `app/src/App.tsx`, `app/src/i18n/pl.ts`
- Test: `app/src/components/wizard/StepProgressDone.test.tsx`, `app/src/components/LibraryView.test.tsx`

**Interfaces:**
- Consumes: Task 7 (`PL`, `formatErrorPl`, `types.ts`), Task 9 (`readCoverDataUrl`), Task 11 (`ui/primitives`), Task 12 (`StepSong`, `StepMap`), `window.bsm`.
- Produces:
  - `StepProgressDone`: live stage stepper (`downloading` [only when YT source] -> `analyzing` -> `notes` -> `layers` -> `writing`), progress bar (`pct`), Cancel button (`cancelActive()`), error screen using `formatErrorPl(errorCode)` + `Otwórz logi` button when `code === "internal"`, success screen `"Mapa jest w grze!"` with informational warnings banner (missing layer checkpoints / rule-lights / grid notices rendered as info pills, NEVER errors) and actions: **Otwórz folder**, **Eksportuj ZIP**, **Zrób kolejną**.
  - `Wizard`: manages steps 1 -> 2 -> 3, accepts optional `regenerateTarget: LibraryEntry | null` which opens directly on Step 2 prefilled with `entry.config` (`mode: "custom"` so the saved `seed` and settings are immediately visible/editable) and `sourceType: "cached"` using `entry.audioCachePath` / `entry.coverCachePath`.
  - `LibraryView`: grid of map cards (cover loaded via `readCoverDataUrl` or fallback icon, title, artist, difficulty pills, formatted creation date) + **Wygeneruj ponownie**, **Eksportuj ZIP**, **Usuń** (with confirmation dialog).

- [ ] **Step 1: Write the failing tests**

Create `app/src/components/wizard/StepProgressDone.test.tsx`:
```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ERRORS_PL } from "../../i18n/pl";
import { StepProgressDone } from "./StepProgressDone";

describe("StepProgressDone", () => {
  it("renders success with info warnings and masks internal errors without tracebacks", () => {
    const cancelActive = vi.fn();
    const openPath = vi.fn();
    const exportMapZip = vi.fn();
    const openLogs = vi.fn();
    (window as unknown as { bsm: unknown }).bsm = { cancelActive, openPath, exportMapZip, openLogs };

    const { rerender } = render(
      <StepProgressDone
        hasDownloadStage
        status="running"
        progress={{ stage: "notes", pct: 45 }}
        result={null}
        errorCode={null}
        onReset={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /anuluj/i }));
    expect(cancelActive).toHaveBeenCalled();

    rerender(
      <StepProgressDone
        hasDownloadStage={false}
        status="done"
        progress={{ stage: "writing", pct: 100 }}
        result={{
          folder: "C:\\CL\\BSM - Song", title: "Song", artist: "Artist", bpm: 120,
          grid_steady: true, n_notes: 300, nps: 4.0, rank: "Hard",
          levels: [{ rank: "Hard", target_nps: 4.0, n_notes: 300, nps: 4.0, n_arcs: 0, n_chains: 0 }],
          n_walls: 0, n_bombs: 0, n_lights: 40, environment: "DefaultEnvironment",
          warnings: ["missing ornaments checkpoint"], used_rule_lights: true, device_used: "cpu",
        }}
        errorCode={null}
        onReset={() => {}}
      />,
    );
    expect(screen.getByText(/mapa jest w grze/i)).toBeInTheDocument();
    expect(screen.getByText(/brak modeli warstw/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /otwórz folder/i }));
    expect(openPath).toHaveBeenCalledWith("C:\\CL\\BSM - Song");

    rerender(
      <StepProgressDone
        hasDownloadStage={false}
        status="error"
        progress={null}
        result={null}
        errorCode="internal"
        onReset={() => {}}
      />,
    );
    expect(screen.getByText(ERRORS_PL.internal)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /otwórz logi/i }));
    expect(openLogs).toHaveBeenCalled();
  });
});
```

Create `app/src/components/LibraryView.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LibraryView } from "./LibraryView";

describe("LibraryView", () => {
  it("lists cards and confirms before deleting", async () => {
    const entry = {
      id: "e1", title: "GHOST", artist: "Camellia", folder: "C:\\CL\\BSM - GHOST",
      createdAt: "2026-11-16T12:00:00Z", bpm: 180,
      levels: [{ rank: "Hard", nps: 4.0, nNotes: 400, nArcs: 10, nChains: 4 }],
      nWalls: 12, nBombs: 3, nLights: 80, environment: "DefaultEnvironment",
      audioCachePath: "C:\\media\\audio.ogg", coverCachePath: null,
      config: { mode: "pack" as const, levels: ["Hard", "Expert", "ExpertPlus"], nps: null, styles: [], ornaments: null, lights: 2, environment: "DefaultEnvironment", seed: 777, lightsRules: false },
    };
    const deleteLibraryEntry = vi.fn(async () => []);
    const exportMapZip = vi.fn(async () => "C:\\Exp\\GHOST.zip");
    (window as unknown as { bsm: unknown }).bsm = {
      listLibrary: vi.fn(async () => [entry]),
      deleteLibraryEntry,
      exportMapZip,
      readCoverDataUrl: vi.fn(async () => null),
    };
    const onRegenerate = vi.fn();
    render(<LibraryView onRegenerate={onRegenerate} />);
    expect(await screen.findByText("GHOST")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /wygeneruj ponownie/i }));
    expect(onRegenerate).toHaveBeenCalledWith(entry);

    fireEvent.click(screen.getByRole("button", { name: /eksportuj zip/i }));
    await waitFor(() => expect(exportMapZip).toHaveBeenCalledWith(entry.folder, entry.title));

    fireEvent.click(screen.getByRole("button", { name: /usuń/i }));
    expect(deleteLibraryEntry).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /potwierdź/i }));
    await waitFor(() => expect(deleteLibraryEntry).toHaveBeenCalledWith("e1"));
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run inside `app/`:
```bash
npx vitest run src/components/wizard/StepProgressDone.test.tsx src/components/LibraryView.test.tsx
```
Expected: FAIL.

- [ ] **Step 3: Implement `StepProgressDone.tsx`, `Wizard.tsx`, `LibraryView.tsx`, and wire into `App.tsx`**

Note: In `StepProgressDone.tsx`, detect layer-checkpoint warnings (`"checkpoint"` or `"layer_stats"`) and present them via friendly Polish info notices from `PL` (and show `"warstwy regułowe"` when `result.used_rule_lights` + warning occurred), never showing raw Python exceptions.

- [ ] **Step 4: Run tests and full renderer build**

Run inside `app/`:
```bash
npm run typecheck
npx vitest run src/components/wizard/StepProgressDone.test.tsx src/components/LibraryView.test.tsx
npm run build
```
Expected: PASS and full `dist/` + `dist-electron/` build succeeds.

- [ ] **Step 5: Commit**

```bash
git add app/src/ app/electron/
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(ui): implement Wizard Step 3, Library view and full app navigation" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 14: Playwright Electron E2E suite with fake engine

**Files:**
- Create: `app/e2e/fake-engine.mjs`
- Create: `app/e2e/app.spec.ts`
- Modify: `app/electron/ipc.ts` (optional test-safe `BSM_TEST_EXPORT_PATH` env override for `dialog.showSaveDialog` / `dialog.showOpenDialog` in Playwright if dialog mocking isn't used — or mock `dialog` via `electronApp.evaluate`!)

**Interfaces:**
- Consumes: Task 6 (exact JSON-lines schemas, e.g., `search` -> `{ items }`, `resolve` -> `YouTubeItem`), built app (`app/dist/index.html`, `app/dist-electron/main.js`), `BSM_ENGINE_CMD`, `BSM_USER_DATA_DIR`, `BSM_SKIP_ENGINE_PACK`, `BSM_DISABLE_UPDATER`.
- Produces: end-to-end Playwright test suite covering all core flows against the real compiled Electron binary + real preload + real IPC + real React UI, with a stdio JSON-lines `fake-engine.mjs`.

- [ ] **Step 1: Write `app/e2e/fake-engine.mjs` and `app/e2e/app.spec.ts`**

`fake-engine.mjs` implements all engine JSON-lines commands (`info`, `search`, `resolve`, `download`, `generate`, `export_zip`, `cancel`, `shutdown`) controlled via env vars (`BSM_FAKE_DETECTED_CL`, `BSM_FAKE_FAIL_CODE`, `BSM_FAKE_HANG_CANCEL`, `BSM_FAKE_LAYER_WARN`), creating realistic `BSM - ...` folders (including `(2)` when colliding!) and `.zip` files in temp test dirs.

`app.spec.ts` launches Electron via `_electron.launch(...)` and tests:
1. **First-run blocking & auto-detect**: with `BSM_FAKE_DETECTED_CL=""`, FirstRun blocks until a valid `CustomLevels` path is picked (`electronApp.evaluate` stubs `dialog.showOpenDialog`); with `BSM_FAKE_DETECTED_CL` set, advances directly.
2. **Happy path + Library lifecycle**: search `"Ghost"` -> select result -> edit title/artist -> **Pakiet** -> progress stages -> `"Mapa jest w grze!"` + missing-layer info notice -> switch to **Biblioteka** -> verify card visible -> **Eksportuj ZIP** -> **Wygeneruj ponownie** (Step 2 opens with saved non-zero seed, changes to custom/generates `(2)` folder, verifies old folder removed) -> **Usuń** (confirm modal, verifies folder gone & library empty).
3. **Cancel flow**: click **Anuluj** mid-generation -> returns cleanly without adding a broken entry.
4. **Internal error masking**: fake engine returns `{ code: "internal", detail: "raw" }` -> UI displays Polish error from `ERRORS_PL.internal` and never shows `"Traceback"`.

- [ ] **Step 2: Run E2E suite before/after any fixes in Electron main/preload**

Run inside `app/`:
```bash
npm run build
npx playwright test e2e/app.spec.ts
```
Fix any Electron IPC / preload / UI wiring bug revealed by the real Electron launch until all 4 E2E scenarios pass cleanly.

- [ ] **Step 3: Commit**

```bash
git add app/e2e/ app/electron/
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "test(e2e): add Playwright Electron end-to-end suite with fake engine" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 15: Checkpoint slimming, pinned tools fetcher, and split PyInstaller engine pack

**Files:**
- Create: `packaging/tools.json`
- Create: `packaging/slim_checkpoints.py`
- Create: `packaging/fetch_tools.py`
- Create: `packaging/engine_entry.py`
- Create: `packaging/bsmaker-engine.spec`
- Create: `packaging/build_engine_pack.py`
- Test: `tests/packaging/test_slim_checkpoints.py`, `tests/packaging/test_build_engine_pack.py`

**Interfaces:**
- Consumes: Task 6 (`EngineConfig`, `run_engine`), `load_checkpoint`, `save_checkpoint`.
- Produces:
  - `packaging/tools.json`: pinned official release URLs and SHA256 checksums for `yt-dlp` (`yt-dlp.exe`) and LGPL `ffmpeg` (`*-win64-lgpl.zip` from `BtbN/FFmpeg-Builds` — NOT GPL!). Implementer must look up current exact pinned tag URLs + SHA256s from the official release `SHA2-256SUMS` / `checksums.sha256` (never `"latest"` mutable URLs, never guessed hashes).
  - `slim_checkpoints(src_dir: Path, dst_dir: Path, *, allow_missing_layers: bool = True) -> dict`: keeps keys `("model", "model_config", "stats", "step", "best", "train_options")`, strips `"optimizer"` and `"scaler"`, requires `timing/best.pt` and `pattern/best.pt`, and slims present `layers/{ornaments,decor-when,decor-what,lights-when,lights-what}/best.pt`.
  - `fetch_tools(manifest_path: Path, out_dir: Path, *, opener=...) -> dict[str, Path]`: downloads `yt-dlp.exe` and LGPL `ffmpeg` archive, verifies SHA256 before writing/extracting `ffmpeg.exe` + `ffprobe.exe`.
  - `split_and_manifest(pack_dir: Path, out_dir: Path, engine_version: str, *, max_part_bytes: int = 1_900_000_000) -> Path`: creates ZIP of `pack_dir`, streams splitting into 64 MB chunks into `bsmaker-engine-v<engineVersion>.zip.001`, `.002`, ..., deletes the unsplit `.zip` immediately after splitting (never retaining both the 2.5 GB `.zip` and split parts simultaneously on disk), and writes `engine-pack-manifest.json` (`engineVersion`, `archiveSha256`, `totalSize`, `parts: [{name, size, sha256}]`).
  - `packaging/engine_entry.py` + `packaging/bsmaker-engine.spec`: PyInstaller `--onedir` build (`bsmaker-engine.exe`); `engine_entry.py` reconfigures `sys.stdin` and `sys.stdout` to `encoding="utf-8", errors="replace"` at startup; `bsmaker-engine.spec` uses `collect_data_files("librosa")` + `collect_submodules("librosa")` (required for `lazy_loader.attach_stub` `.pyi` stubs) and collects `bsmaker`, `torch`, `soundfile`, `_soundfile_data`, `lazy_loader`. Keep `packaging/` WITHOUT `packaging/__init__.py` so it never shadows third-party PyPA `packaging` used by `pytest`/`pyinstaller`.

- [ ] **Step 1: Write the failing tests**

Create `tests/packaging/test_slim_checkpoints.py`:
```python
from dataclasses import asdict
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
from test_generate import LAYER_STATS, STATS, make_layer_models, make_models  # noqa: E402
from bsmaker.models.common import load_checkpoint, save_checkpoint
from bsmaker.models.layers.what import load_what
from bsmaker.models.pattern.sample import load_pattern
from bsmaker.models.timing.decode import load_timing
from slim_checkpoints import slim_checkpoints


def test_slim_checkpoints_strips_optimizer_and_keeps_train_options(tmp_path: Path) -> None:
    src = tmp_path / "src_ckpt"
    dst = tmp_path / "dst_ckpt"
    t, p = make_models()
    for name, loaded in (("timing", t), ("pattern", p)):
        fake_opt = {"state": {0: {"exp_avg": loaded.model.state_dict()}}}
        save_checkpoint(src / name / "best.pt", {
            "model": loaded.model.state_dict(), "optimizer": fake_opt,
            "scaler": {"scale": 65536.0}, "step": 1000,
            "model_config": asdict(loaded.config), "stats": STATS, "best": {"f1": 0.8},
        })
    lm = make_layer_models()
    save_checkpoint(src / "layers" / "lights-what" / "best.pt", {
        "model": lm.lights_what.model.state_dict(),
        "optimizer": {"state": {0: {"exp_avg": lm.lights_what.model.state_dict()}}},
        "model_config": asdict(lm.lights_what.config),
        "stats": STATS,
        "train_options": {"layer": "lights"},
    })

    report = slim_checkpoints(src, dst, allow_missing_layers=True)
    assert set(report["slimmed"]) == {"timing", "pattern", "layers/lights-what"}
    assert len(report["missing_layers"]) == 4
    for name in ("timing", "pattern", "layers/lights-what"):
        raw = load_checkpoint(dst / name / "best.pt")
        assert "optimizer" not in raw and "scaler" not in raw
        assert (dst / name / "best.pt").stat().st_size < (src / name / "best.pt").stat().st_size
    load_timing(dst / "timing" / "best.pt", "cpu")
    load_pattern(dst / "pattern" / "best.pt", "cpu")
    loaded_lw = load_what(dst / "layers" / "lights-what" / "best.pt", "cpu")
    assert loaded_lw.layer == "lights"
```

Create `tests/packaging/test_build_engine_pack.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/packaging/test_slim_checkpoints.py tests/packaging/test_build_engine_pack.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'slim_checkpoints'`).

- [ ] **Step 3: Implement `packaging/` scripts, `tools.json`, and `bsmaker-engine.spec`**

1. Fetch exact current pinned release tag URL + SHA256 for `yt-dlp/yt-dlp` (`yt-dlp.exe` from `SHA2-256SUMS`) and `BtbN/FFmpeg-Builds` (`ffmpeg-n*-win64-lgpl-*.zip` or dated `autobuild-*` `-win64-lgpl.zip` from `checksums.sha256` — verify it is LGPL!) and record them in `packaging/tools.json`.
2. Implement:
   - `packaging/slim_checkpoints.py` (CLI `--src --dst [--strict-layers]`),
   - `packaging/fetch_tools.py` (CLI `--manifest --out-dir`),
   - `packaging/engine_entry.py` (calls `sys.stdin.reconfigure(encoding="utf-8", errors="replace")` and `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at startup, resolves bundled `checkpoints`, `playability.toml`, `layer_stats.json`, `tools/ffmpeg.exe`, `tools/yt-dlp.exe` relative to `Path(sys.executable).resolve().parent.parent` when `--resources-dir` / frozen, and calls `run_engine`),
   - `packaging/bsmaker-engine.spec` (imports `collect_data_files, collect_submodules` from `PyInstaller.utils.hooks`, adds `datas += collect_data_files("librosa")` and `hiddenimports += collect_submodules("librosa")` plus `bsmaker`, `torch`, `soundfile`, `_soundfile_data`, `lazy_loader` so `lazy_loader.attach_stub` does not fail in the frozen binary),
   - `packaging/build_engine_pack.py` (invokes PyInstaller, deletes PyInstaller's `build/` directory immediately afterward, assembles `pack_dir` with `bsmaker-engine/`, `checkpoints/`, `tools/`, `playability.toml`, `layer_stats.json`, streams `split_and_manifest` in 64 MB chunks, and deletes the unsplit `.zip` immediately once split).

- [ ] **Step 4: Run packaging unit tests to verify they pass**

Run:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src C:/Users/alex/Desktop/BeatSaberMaker/.venv/Scripts/python.exe -m pytest tests/packaging/test_slim_checkpoints.py tests/packaging/test_build_engine_pack.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packaging/ tests/packaging/
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "feat(packaging): add checkpoint slimmer, pinned tool fetcher and split engine pack builder" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```

---

### Task 16: `electron-builder` NSIS config & GitHub Actions CI / Release workflows

**Files:**
- Modify: `app/package.json` (add `"build"` config for `electron-builder` NSIS + GitHub `al3ksh/BeatSaberMaker` + `extraResources: ["engine-version.json"]`)
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: Task 7–14 (`app/` build + tests + E2E), Task 10 (`engine-version.json`, `engine-pack-manifest.json`), Task 15 (`packaging/fetch_tools.py`, `packaging/slim_checkpoints.py`, `packaging/build_engine_pack.py`).
- Produces:
  - `app/package.json`: preserves `"main": "dist-electron/main.js"` and omits `"type": "module"` (CommonJS Electron entry); `"build"` section: `appId: "com.beatsabermaker.app"`, `productName: "BeatSaberMaker"`, `directories: { "output": "release" }`, `files: ["dist/**/*", "dist-electron/**/*", "package.json"]`, `extraResources: ["engine-version.json"]`, `win: { "target": [{ "target": "nsis", "arch": ["x64"] }] }`, `nsis: { "oneClick": false, "allowToChangeInstallationDirectory": true, "perMachine": false }`, `publish: [{ "provider": "github", "owner": "al3ksh", "repo": "BeatSaberMaker" }]`.
  - `.github/workflows/ci.yml`: runs on `push` (branches) & `pull_request` on `windows-latest`: Python CPU unit tests (`pytest` with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`, no heavy CUDA torch build) + `app/` `npm ci`, `npm run typecheck`, `npm test`, `npm run build`, `npx playwright test`.
  - `.github/workflows/release.yml`: runs ONLY on tags `v*` on `windows-latest` (`permissions: contents: write`):
    1. Runs Python + `app/` unit & Playwright E2E tests.
    2. Sets up Python 3.13, installs CUDA PyTorch in its own step (`pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128`), and THEN in a separate step installs `bsmaker` and PyInstaller from PyPI (`pip install --no-cache-dir . pyinstaller`) so PyPI packages do not fail against the PyTorch index and pip cache never fills disk.
    3. Fetches pinned LGPL `ffmpeg` + `yt-dlp` via `packaging/fetch_tools.py`.
    4. Downloads checkpoints archive from secret `${{ secrets.CHECKPOINTS_ARCHIVE_URL }}` (failing clearly if unset), extracts & slims via `packaging/slim_checkpoints.py` (deleting the raw archive immediately after slimming).
    5. Runs `packaging/build_engine_pack.py` reading `engineVersion` from `app/engine-version.json`, producing `engine-pack-manifest.json` and `bsmaker-engine-v*.zip.*` (all `<= 1.9 GB`), with `build/` and the unsplit `.zip` cleaned immediately to stay within `windows-latest` free disk space.
    6. Builds Electron app (`npm run build`) and NSIS installer (`npx electron-builder --win --publish never`), producing `BeatSaberMaker-Setup-*.exe`, `*.blockmap`, `latest.yml`.
    7. Creates GitHub Release via `softprops/action-gh-release@v2` (or `gh release create`) attaching NSIS `.exe`, `.blockmap`, `latest.yml`, `engine-pack-manifest.json`, and all `bsmaker-engine-v*.zip.*` parts!

- [ ] **Step 1: Configure `"build"` in `app/package.json` and write `.github/workflows/ci.yml` + `.github/workflows/release.yml`**

Ensure `.github/workflows/release.yml` sets `OMP_NUM_THREADS: "4"` and `MKL_NUM_THREADS: "4"`, verifies tag matches `app/package.json` version (or syncs `npm version` cleanly / checks `v${version}`), and uses `softprops/action-gh-release@v2` with `files:` covering both installer artifacts and split engine pack parts.

- [ ] **Step 2: Verify unpacked `electron-builder` packaging and workflow files**

Run inside `app/`:
```bash
npm run build
npx electron-builder --win --dir --publish never
```
Verify `app/release/win-unpacked/BeatSaberMaker.exe` and `app/release/win-unpacked/resources/engine-version.json` exist, and verify `.github/workflows/ci.yml` and `.github/workflows/release.yml` have no tab indentation and valid required keys (`on:`, `jobs:`).
Expected: `win-unpacked/BeatSaberMaker.exe` and `win-unpacked/resources/engine-version.json` created with exit code 0.

- [ ] **Step 3: Clean `app/release/` if untracked or ensure `app/release/` is in `.gitignore`, and commit**

Add `app/node_modules/`, `app/dist/`, `app/dist-electron/`, `app/release/`, `app/test-results/` to `.gitignore`.

```bash
git add .gitignore app/package.json .github/workflows/ci.yml .github/workflows/release.yml
git -c user.name="Alex" -c user.email="alex.szotek@gmail.com" commit -m "chore(release): add electron-builder NSIS config and GitHub Actions workflows" -m "Co-Authored-By: Gemini 4 Hydrogen <noreply@google.com>"
```
