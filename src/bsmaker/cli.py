"""Command line interface."""
from collections import Counter

import typer

from bsmaker.config import load_config
from bsmaker.data.features import build_features
from bsmaker.data.index_local import index_folder
from bsmaker.data.manifest import open_manifest
from bsmaker.playability.config import PlayabilityConfig

app = typer.Typer(help="BeatSaberMaker: Beat Saber map generator.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("index-local")
def index_local() -> None:
    """Index the local CustomLevels folder into the manifest."""
    cfg = load_config()
    conn = open_manifest(cfg.manifest_path)
    s = index_folder(conn, cfg.custom_levels, PlayabilityConfig(), progress_every=100)
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
