"""Command line interface."""
from collections import Counter
from pathlib import Path

import typer

from bsmaker.config import load_config
from bsmaker.formats.events import to_events
from bsmaker.formats.reader import read_map
from bsmaker.playability.calibrate import (
    HARD_RULE_NAMES,
    calibrate,
    collect_event_lists,
    measure,
    render_report,
    write_playability_toml,
)
from bsmaker.playability.checker import check as check_events
from bsmaker.data.features import build_features
from bsmaker.data.index_local import index_folder
from bsmaker.data.manifest import open_manifest
from bsmaker.playability.config import PlayabilityConfig, load_playability_config

app = typer.Typer(help="BeatSaberMaker: Beat Saber map generator.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("index-local")
def index_local() -> None:
    """Index the local CustomLevels folder into the manifest."""
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    s = index_folder(conn, cfg.custom_levels,
                     load_playability_config(cfg.playability_path), progress_every=100)
    per_rank = Counter(r["rank"] for r in conn.execute(
        "SELECT rank FROM difficulties WHERE accepted = 1"))
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
    (cfg.reports_dir / "index-local.txt").write_text(text + "\n", encoding="utf-8")


@data_app.command("features")
def features() -> None:
    """Compute log-mel features for every map with an accepted difficulty."""
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    computed, failed = build_features(conn, cfg.features_dir, progress_every=100)
    print(f"computed: {computed}, failed: {failed}")


@data_app.command("calibrate-playability")
def calibrate_playability() -> None:
    """Calibrate the playability thresholds on local maps; writes playability.toml and a report."""
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    print("reading calibration maps...", flush=True)
    lists = collect_event_lists(conn)
    print(f"{len(lists)} difficulties; measuring with defaults", flush=True)
    start = PlayabilityConfig()
    before = measure(lists, start)
    print(f"before: pass_share={before.pass_share:.4f}", flush=True)
    after, steps = calibrate(lists, start, progress=lambda m: print(m, flush=True))
    write_playability_toml(after.config, cfg.base_dir / "playability.toml")
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "playability-calibration.md").write_text(
        render_report(before, after, steps), encoding="utf-8")
    print(f"after: pass_share={after.pass_share:.4f}")
    if after.pass_share < 0.95:
        print("pass_share still below 0.95; thresholds not loosened further.")
        raise typer.Exit(1)


def _mmss(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


@app.command("check")
def check_map(folder: str) -> None:
    """Report playability rule violations for every difficulty of a map folder."""
    try:
        bm = read_map(Path(folder))
        if not bm.difficulties:
            raise ValueError("no readable difficulties")
    except Exception as e:  # noqa: BLE001 - untrusted input, one-line message only
        typer.echo(f"not a readable map: {folder}: {type(e).__name__}: {e}".splitlines()[0], err=True)
        raise typer.Exit(2) from None
    try:
        pcfg = load_playability_config(load_config().playability_path)
    except FileNotFoundError:
        pcfg = PlayabilityConfig()
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
