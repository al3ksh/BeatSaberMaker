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
