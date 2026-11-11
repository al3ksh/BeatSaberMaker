"""Command line interface."""
import json
import os
from collections import Counter
from pathlib import Path

import typer

from bsmaker.audio.decode import FfmpegMissingError, require_ffmpeg
from bsmaker.config import load_config
from bsmaker.formats.events import to_events
from bsmaker.formats.reader import read_map
from bsmaker.playability.calibrate import (
    HARD_RULE_NAMES,
    calibrate,
    collect_event_lists,
    render_report,
    write_playability_toml,
)
from bsmaker.playability.checker import check as check_events
from bsmaker.data.beatsaver import BeatSaverClient
from bsmaker.data.features import build_features
from bsmaker.data.fetch_metadata import fetch_metadata
from bsmaker.data.index_local import index_folder
from bsmaker.data.manifest import open_manifest
from bsmaker.data.songs import resplit
from bsmaker.data.styles import STYLES, styles_from_tags
from bsmaker.playability.config import PROFILES, load_playability_config

app = typer.Typer(help="BeatSaberMaker: Beat Saber map generator.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _require_ffmpeg() -> None:
    """Exit with code 2 and a clear message when ffmpeg or ffprobe is not on PATH."""
    try:
        require_ffmpeg()
    except FfmpegMissingError as e:
        typer.echo(f"{e}; install ffmpeg and put it on PATH", err=True)
        raise typer.Exit(2) from None


def _summarise(conn, s, report_name: str, cfg, source: str) -> None:
    per_rank = Counter(r["rank"] for r in conn.execute(
        "SELECT d.rank FROM difficulties d JOIN maps m ON m.key = d.map_key "
        "WHERE d.accepted = 1 AND m.source = ?", (source,)))
    lines = [f"n_maps (unique folders): {s.n_maps}",
             f"n_duplicate_keys skipped: {s.n_duplicate_keys}",
             f"unreadable: {s.reasons.get('unreadable', 0)}",
             f"n_difficulties: {s.n_difficulties}",
             f"n_accepted: {s.n_accepted}",
             "accepted per rank:"]
    lines += [f"  {k}: {v}" for k, v in sorted(per_rank.items())]
    lines.append("rejections by reason (a difficulty can have several):")
    lines += [f"  {k}: {v}" for k, v in sorted(s.reasons.items())]
    text = "\n".join(lines)
    print(text)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / report_name).write_text(text + "\n", encoding="utf-8")


@data_app.command("index-local")
def index_local() -> None:
    """Index the local CustomLevels folder into the manifest (data playability profile)."""
    _require_ffmpeg()
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    s = index_folder(conn, cfg.custom_levels,
                     load_playability_config(cfg.playability_path, "data"), progress_every=100)
    _summarise(conn, s, "index-local.txt", cfg, "local")


@data_app.command("index-raw")
def index_raw() -> None:
    """Index the downloaded BeatSaver maps in the raw folder (data playability profile)."""
    _require_ffmpeg()
    cfg = load_config()
    if not cfg.raw_dir.is_dir():
        print(f"raw folder does not exist: {cfg.raw_dir}")
        return
    conn = open_manifest(cfg.manifest_path)
    s = index_folder(conn, cfg.raw_dir, load_playability_config(cfg.playability_path, "data"),
                     source="beatsaver", progress_every=100)
    _summarise(conn, s, "index-raw.txt", cfg, "beatsaver")


@data_app.command("resplit")
def resplit_cmd() -> None:
    """Group maps into songs and re-assign train/val/test by song (safe to repeat)."""
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    before = {r[0]: r[1] for r in conn.execute(
        "SELECT split, COUNT(*) FROM difficulties GROUP BY split")}
    after = resplit(conn)
    n_maps, n_songs = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT song_id) FROM maps").fetchone()
    lines = [f"difficulties per split before: {dict(sorted(before.items(), key=str))}",
             f"difficulties per split after: {after}",
             f"maps: {n_maps}, songs: {n_songs}"]
    big = conn.execute(
        "SELECT song_id, COUNT(*) AS n FROM maps GROUP BY song_id ORDER BY n DESC, song_id "
        "LIMIT 5").fetchall()
    lines.append("largest groups:")
    for r in big:
        titles = [t[0] for t in conn.execute(
            "SELECT DISTINCT title FROM maps WHERE song_id = ? LIMIT 3", (r["song_id"],))]
        lines.append(f"  {r['song_id']}: {r['n']} maps, titles {titles}")
    text = "\n".join(lines)
    print(text)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "resplit.txt").write_text(text + "\n", encoding="utf-8")


@data_app.command("features")
def features(workers: int = typer.Option(
        max(1, min(8, (os.cpu_count() or 2) - 2)), "--workers",
        help="Worker processes for decoding (>= 1).")) -> None:
    """Compute log-mel features for every map with an accepted difficulty."""
    if workers < 1:
        raise typer.BadParameter("must be >= 1", param_hint="--workers")
    _require_ffmpeg()
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    print(f"workers: {workers}")
    computed, failed = build_features(conn, cfg.features_dir, progress_every=100, workers=workers)
    print(f"computed: {computed}, failed: {failed}")


@data_app.command("build-events")
def build_events() -> None:
    """Pack accepted maps into the event cache and write dataset_stats.json."""
    from bsmaker.data.events_cache import EventCache, build_event_cache
    from bsmaker.data.stats import compute_stats, write_stats

    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    left_out = build_event_cache(conn, cfg.events_dir, cfg.features_dir, progress_every=500)
    cache = EventCache.load(cfg.events_dir)
    print(f"cached: {len(cache.entries)}, left out: {left_out}", flush=True)
    stats = compute_stats(cache, cfg.features_dir)
    write_stats(stats, cfg.base_dir / "dataset_stats.json")
    print(f"wrote {cfg.base_dir / 'dataset_stats.json'}")


@data_app.command("fetch-metadata")
def fetch_metadata_cmd() -> None:
    """Fetch BeatSaver metadata (tags, ratings, flags) for indexed maps; safe to repeat."""
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    with BeatSaverClient() as client:
        updated, not_found = fetch_metadata(conn, client)
        n_requests = client.n_requests
    print(f"updated: {updated}, not found: {not_found}, requests: {n_requests}")
    per_style: Counter = Counter()
    n_styled = 0
    for r in conn.execute("SELECT bs_tags FROM maps WHERE bs_tags IS NOT NULL"):
        styles = [s for s in styles_from_tags(json.loads(r["bs_tags"])) if s in STYLES]
        n_styled += bool(styles)
        per_style.update(styles)
    print(f"maps with at least one style tag: {n_styled}")
    for s in STYLES:
        print(f"  {s}: {per_style[s]}")
    for col in ("ranked", "bl_ranked", "curated", "automapper"):
        n = conn.execute(f"SELECT COUNT(*) FROM maps WHERE bs_{col} = 1").fetchone()[0]
        print(f"{col}: {n}")


@data_app.command("calibrate-playability")
def calibrate_playability() -> None:
    """Calibrate both playability profiles on local maps; writes playability.toml and a report."""
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    print("reading calibration maps...", flush=True)
    lists = collect_event_lists(conn)
    print(f"{len(lists)} difficulties", flush=True)
    cal = calibrate(lists, progress=lambda m: print(m, flush=True))
    write_playability_toml(cal.mask.config, cal.data.config, cfg.base_dir / "playability.toml")
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "playability-calibration.md").write_text(render_report(cal), encoding="utf-8")
    print(f"data profile: pass_share={cal.data.pass_share:.4f} "
          f"(<=0.02: {cal.data.pass_share_02:.4f}); mask: pass_share={cal.mask.pass_share:.4f}")
    if cal.data.pass_share < 0.95:
        print("data profile pass_share still below 0.95; thresholds not loosened further.")
        raise typer.Exit(1)


def _mmss(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


@app.command("check")
def check_map(folder: str, profile: str = typer.Option(
        "mask", "--profile", help="Playability profile: mask or data.")) -> None:
    """Report playability rule violations for every difficulty of a map folder."""
    if profile not in PROFILES:
        typer.echo(f"unknown profile {profile!r}; expected one of {', '.join(PROFILES)}", err=True)
        raise typer.Exit(2)
    try:
        bm = read_map(Path(folder))
        if not bm.difficulties:
            raise ValueError("no readable difficulties")
    except Exception as e:  # noqa: BLE001 - untrusted input, one-line message only
        typer.echo(f"not a readable map: {folder}: {type(e).__name__}: {e}".splitlines()[0], err=True)
        raise typer.Exit(2) from None
    try:
        pcfg = load_playability_config(load_config().playability_path, profile)
    except FileNotFoundError:
        pcfg = load_playability_config(None, profile)
        typer.echo("warning: no playability.toml found; using built-in defaults "
                   "(no rare-token blocking)", err=True)
    except ValueError as e:
        typer.echo(str(e).splitlines()[0], err=True)
        raise typer.Exit(2) from None
    print(f"profile: {profile}")
    for d in bm.difficulties:
        events, _ = to_events(d.notes)
        report = check_events(events, pcfg)
        n_notes = len(d.notes)
        times = [n.time_s for n in d.notes]
        span = max(times) - min(times) if len(times) > 1 else 0.0
        nps = n_notes / span if span > 0 else 0.0
        counts = Counter(v.rule for v in report.hard)
        per_rule = ", ".join(f"{r}={counts.get(r, 0)}" for r in HARD_RULE_NAMES)
        print(f"{d.characteristic} {d.rank}: notes {n_notes}, events {len(events)}, nps {nps:.2f}, "
              f"hard rate {report.hard_rate:.4f}, {per_rule}")
        for v in sorted(report.hard, key=lambda v: v.time_s)[:20]:
            print(f"  {_mmss(v.time_s)} {'left' if v.hand == 0 else 'right'} {v.rule}")


@data_app.command("scrape")
def scrape_cmd(max_maps: int = typer.Option(20000, "--max-maps", help="Maximum maps to consider.")) -> None:
    """Download curated BeatSaver maps into the raw folder and index them; safe to repeat."""
    import time

    from bsmaker.data.scrape import scrape

    _require_ffmpeg()
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    t0 = time.monotonic()
    with BeatSaverClient() as client:
        s = scrape(conn, client, cfg, max_maps=max_maps)
        n_req, n_429 = client.n_requests, client.n_429
    print(f"downloaded: {s.downloaded}, skipped existing: {s.skipped_existing}, "
          f"failed: {s.failed}, stopped for low disk: {s.stopped_low_disk}")
    print(f"requests: {n_req}, HTTP 429: {n_429}, seconds: {time.monotonic() - t0:.0f}")


train_app = typer.Typer(help="Model training commands.", no_args_is_help=True)
app.add_typer(train_app, name="train")


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


@train_app.command("timing")
def train_timing(
        max_steps: int = typer.Option(40000, "--max-steps", help="Optimizer steps to train for."),
        batch_size: int = typer.Option(32, "--batch-size"),
        num_workers: int = typer.Option(4, "--num-workers"),
        device: str = typer.Option("cuda", "--device", help="cuda or cpu."),
        eval_every: int = typer.Option(2000, "--eval-every", help="Steps between evaluations."),
        save_every: int = typer.Option(1000, "--save-every", help="Steps between checkpoints."),
        out: Path | None = typer.Option(None, "--out", help="Checkpoint folder "
                                        "(default: <data_dir>/checkpoints/timing).")) -> None:
    """Train the timing model on the event cache; resumes from the checkpoint folder."""
    from dataclasses import asdict

    import torch

    from bsmaker.data.events_cache import EventCache
    from bsmaker.data.stats import load_stats
    from bsmaker.models.timing.dataset import TimingWindows
    from bsmaker.models.timing.decode import make_timing_eval, timing_better
    from bsmaker.models.timing.model import (
        TimingConfig, TimingModel, timing_eval, timing_step)
    from bsmaker.models.train_loop import TrainSettings, run_training

    cfg = load_config()
    stats_path = cfg.base_dir / "dataset_stats.json"
    if not (cfg.events_dir / "index.json").is_file() or not (cfg.events_dir / "events.npz").is_file():
        _fail(f"event cache not found in {cfg.events_dir}; run `bsmaker data build-events`")
    if not stats_path.is_file():
        _fail(f"{stats_path} not found; run `bsmaker data build-events`")
    if device == "cuda" and not torch.cuda.is_available():
        _fail("--device cuda requested but CUDA is not available; use --device cpu")
    cache = EventCache.load(cfg.events_dir)
    stats = load_stats(stats_path)
    mcfg = TimingConfig()
    train_entries = cache.select("train")
    val_all = cache.select("val")
    if not train_entries or not val_all:
        _fail("event cache has no train or val entries")
    step = max(1, len(val_all) // 512)
    val_entries = val_all[::step][:512]
    common = dict(features_dir=cfg.features_dir, stats=stats, window=mcfg.window,
                  events_dir=cfg.events_dir)
    train_ds = TimingWindows(cache, train_entries, seed=0, **common)
    val_ds = TimingWindows(cache, val_entries, seed=1, fixed=True, **common)
    settings = TrainSettings(max_steps=max_steps, batch_size=batch_size, num_workers=num_workers,
                             device=device, eval_every=eval_every, save_every=save_every)
    result = run_training(
        TimingModel(mcfg), train_ds, val_ds, timing_step,
        make_timing_eval(cache, val_all, cfg.features_dir, stats, timing_eval),
        out if out is not None else cfg.checkpoints_dir / "timing", settings, asdict(mcfg),
        timing_better,
        lambda m: print(m, flush=True), stats=stats)
    print(f"done at step {result['step']}, best {result['best']}")


@train_app.command("pattern")
def train_pattern(
        max_steps: int = typer.Option(60000, "--max-steps", help="Optimizer steps to train for."),
        batch_size: int = typer.Option(32, "--batch-size"),
        num_workers: int = typer.Option(4, "--num-workers"),
        device: str = typer.Option("cuda", "--device", help="cuda or cpu."),
        eval_every: int = typer.Option(2000, "--eval-every", help="Steps between evaluations."),
        save_every: int = typer.Option(1000, "--save-every", help="Steps between checkpoints."),
        out: Path | None = typer.Option(None, "--out", help="Checkpoint folder "
                                        "(default: <data_dir>/checkpoints/pattern).")) -> None:
    """Train the pattern model on the event cache; resumes from the checkpoint folder."""
    from dataclasses import asdict

    import torch

    from bsmaker.data.events_cache import EventCache
    from bsmaker.data.stats import load_stats
    from bsmaker.models.pattern.dataset import PatternSequences
    from bsmaker.models.pattern.model import (
        PatternConfig, PatternModel, pattern_better, pattern_eval, pattern_step)
    from bsmaker.models.train_loop import TrainSettings, run_training

    cfg = load_config()
    stats_path = cfg.base_dir / "dataset_stats.json"
    if not (cfg.events_dir / "index.json").is_file() or not (cfg.events_dir / "events.npz").is_file():
        _fail(f"event cache not found in {cfg.events_dir}; run `bsmaker data build-events`")
    if not stats_path.is_file():
        _fail(f"{stats_path} not found; run `bsmaker data build-events`")
    if device == "cuda" and not torch.cuda.is_available():
        _fail("--device cuda requested but CUDA is not available; use --device cpu")
    cache = EventCache.load(cfg.events_dir)
    stats = load_stats(stats_path)
    mcfg = PatternConfig()
    train_entries = cache.select("train")
    val_all = cache.select("val")
    if not train_entries or not val_all:
        _fail("event cache has no train or val entries")
    stride = max(1, len(val_all) // 256)
    val_entries = val_all[::stride][:256]
    common = dict(features_dir=cfg.features_dir, stats=stats, context=mcfg.context,
                  events_dir=cfg.events_dir)
    train_ds = PatternSequences(cache, train_entries, seed=0, **common)
    val_ds = PatternSequences(cache, val_entries, seed=1, fixed=True, **common)
    settings = TrainSettings(max_steps=max_steps, batch_size=batch_size, num_workers=num_workers,
                             device=device, eval_every=eval_every, save_every=save_every)
    result = run_training(
        PatternModel(mcfg), train_ds, val_ds, pattern_step, pattern_eval,
        out if out is not None else cfg.checkpoints_dir / "pattern", settings, asdict(mcfg),
        pattern_better, lambda m: print(m, flush=True), stats=stats)
    print(f"done at step {result['step']}, best {result['best']}")


@app.command("generate")
def generate_cmd(
        audio: Path = typer.Argument(..., help="Song file (mp3, ogg, wav, ...)."),
        nps: float = typer.Option(..., "--nps", help="Target notes per second (0.5-12)."),
        style: list[str] = typer.Option([], "--style", help="Style tag; repeat for several."),
        seed: int = typer.Option(0, "--seed"),
        out: Path | None = typer.Option(None, "--out", help="Output folder "
                                        "(default: the CustomLevels folder)."),
        overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing BSM map."),
        device: str | None = typer.Option(None, "--device", help="cuda or cpu.")) -> None:
    """Generate a Beat Saber map folder from a song."""
    if not 0.5 <= nps <= 12:
        typer.echo("--nps must be between 0.5 and 12", err=True)
        raise typer.Exit(2)
    unknown = [s for s in style if s not in STYLES]
    if unknown:
        typer.echo(f"unknown style {unknown[0]!r}; expected one of {', '.join(STYLES)}", err=True)
        raise typer.Exit(2)

    import torch

    from bsmaker.audio.decode import AudioDecodeError
    from bsmaker.generate import GenerateError, generate_map
    from bsmaker.models.pattern.sample import load_pattern
    from bsmaker.models.timing.decode import load_timing

    cfg = load_config()
    if not audio.is_file():
        _fail(f"audio file not found: {audio}")
    try:
        require_ffmpeg()
    except FfmpegMissingError as e:
        _fail(f"{e}; install ffmpeg and put it on PATH")
    timing_ck = cfg.checkpoints_dir / "timing" / "best.pt"
    pattern_ck = cfg.checkpoints_dir / "pattern" / "best.pt"
    for ck in (timing_ck, pattern_ck):
        if not ck.is_file():
            _fail(f"checkpoint not found: {ck}; train the models first")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        _fail("--device cuda requested but CUDA is not available; use --device cpu")
    try:
        timing = load_timing(timing_ck, dev)
        pattern = load_pattern(pattern_ck, dev)
        mask_cfg = load_playability_config(cfg.playability_path, "mask")
        result = generate_map(
            audio, nps=nps, styles=tuple(style), seed=seed,
            out_dir=out if out is not None else cfg.custom_levels, timing=timing,
            pattern=pattern, mask_cfg=mask_cfg, stats=timing.stats, device=dev,
            overwrite=overwrite)
    except (GenerateError, AudioDecodeError, FileExistsError, ValueError) as e:
        _fail(str(e).splitlines()[0] if str(e) else type(e).__name__)
    for w in result.warnings:
        typer.echo(f"warning: {w}")
    typer.echo(f"wrote {result.folder}")
    typer.echo(f"{result.n_notes} notes, {result.nps:.2f} nps, {result.rank}, "
               f"{result.bpm:.1f} bpm (grid {'steady' if result.grid_steady else 'not steady'}), "
               f"{result.density_rounds} density round(s)")
