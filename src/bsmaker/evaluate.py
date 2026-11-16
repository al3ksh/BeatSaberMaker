"""Evaluate the trained models on held-out songs and render a markdown report."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date
import json
from pathlib import Path

import numpy as np

from bsmaker.data.align import onset_envelope
from bsmaker.data.events_cache import Entry, EventCache
from bsmaker.data.layer_stats import level_rate
from bsmaker.formats.events import FRAMES_PER_SECOND, Event, to_events
from bsmaker.formats.layer_tokens import (
    ARC_LENGTH_BEATS, LIGHT_CHANNELS, LIGHT_VOCAB, WALL_VOCAB, arc_length_bucket, cells_to_mask,
    light_event_to_token, wall_to_token)
from bsmaker.formats.model import Bomb, Layers, LightEvent, Note
from bsmaker.generate import check_exact
from bsmaker.layers_generate import LayerModels, generate_decor, generate_lights
from bsmaker.models.ornaments.sample import sample_ornaments
from bsmaker.obstacles_rules import gameplay_obstacles
from bsmaker.playability.layers import check_layers
from bsmaker.formats.tokens import NONE_TOKEN, decode_token
from bsmaker.models.pattern.sample import LoadedPattern, sample_notes
from bsmaker.models.timing.decode import (
    HOP, SAMPLE_RATE, LoadedTiming, frame_probs, pick_peaks, predict_events, threshold_for_count)
from bsmaker.models.timing.metrics import onset_f1
from bsmaker.playability.checker import check
from bsmaker.playability.config import PlayabilityConfig

DENSITY_TOLERANCE = 0.10
AXES = {"column": 4, "row": 3, "direction": 9, "hand": 2}
AXIS_LABELS = {
    "column": ("0", "1", "2", "3"),
    "row": ("0", "1", "2"),
    "direction": ("up", "down", "left", "right", "up-left", "up-right", "down-left",
                  "down-right", "dot"),
    "hand": ("left", "right"),
}


def _spaced(entries: Sequence[Entry], n: int) -> list[Entry]:
    """At most `n` entries, deterministic and evenly spaced."""
    entries = list(entries)
    if n <= 0 or not entries:
        return []
    if n >= len(entries):
        return entries
    return [entries[i] for i in np.unique(np.linspace(0, len(entries) - 1, n).astype(int))]


def load_entry_keys(path: Path) -> set[tuple[str, str, str]]:
    """(map_key, characteristic, rank) of every record of an event-cache index.json."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} is not an index.json (a JSON list of records)")
    try:
        return {(r["map_key"], r["characteristic"], r["rank"]) for r in raw}
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path} has records without map_key/characteristic/rank") from e


def select_entries(cache: EventCache, split: str,
                   keys: set[tuple[str, str, str]] | None = None) -> list[Entry]:
    """Difficulties of the split, restricted to `keys` when given (before any subsampling)."""
    entries = cache.select(split)
    if keys is None:
        return entries
    return [e for e in entries if (e.map_key, e.characteristic, e.rank) in keys]


def _measured_nps(notes) -> float:
    if len(notes) < 2:
        return 0.0
    span = notes[-1].time_s - notes[0].time_s
    return len(notes) / span if span > 0 else 0.0


def _one_per_song(entries: Sequence[Entry]) -> list[Entry]:
    """Per song the difficulty with the median nps (ties: the lower)."""
    by_song: dict[str, list[Entry]] = {}
    for e in entries:
        by_song.setdefault(e.map_key, []).append(e)
    out = []
    for group in by_song.values():
        group = sorted(group, key=lambda e: (e.nps, e.i))
        out.append(group[(len(group) - 1) // 2])
    return out


def _mel(features_dir: Path, e: Entry) -> np.ndarray:
    mel = np.load(Path(features_dir) / f"{e.map_key}.npy", allow_pickle=False)
    return mel[:min(int(e.n_frames), mel.shape[0])]


def human_events(cache: EventCache, e: Entry) -> list[Event]:
    t, left, right = cache.events(e.i)
    return [Event(float(a), int(b), int(c)) for a, b, c in zip(t, left, right)]


def _bin_label(edges: Sequence[float], nps: float) -> str:
    b = min(max(int(np.searchsorted(edges, nps, side="right")) - 1, 0), len(edges) - 2)
    return f"{edges[b]:g}-{edges[b + 1]:g}"


def eval_timing(loaded: LoadedTiming, cache: EventCache, features_dir: Path, split: str,
                n: int, device: str, entries: Sequence[Entry] | None = None) -> dict:
    """Mean onset P/R/F1 at the human event count (25 ms and 10 ms), overall and by nps bin."""
    pool = cache.select(split) if entries is None else list(entries)
    entries = _spaced(pool, n)
    edges = list(loaded.stats.get("nps_bins", []))
    scores: list[tuple[float, float, float, float]] = []
    labels: list[str] = []
    order: dict[str, int] = {}
    for e in entries:
        # same decode as the training-time whole-song eval; the predictions are scored twice
        mel = _mel(features_dir, e)
        true = np.asarray(cache.events(e.i)[0], dtype=np.float64)
        probs = frame_probs(loaded.model, mel, e.nps, e.styles, loaded.stats, device)
        pred = pick_peaks(probs, threshold_for_count(probs, len(true))).astype(np.float64) \
            * HOP / SAMPLE_RATE
        pr, rc, f1 = onset_f1(pred, true)
        scores.append((pr, rc, f1, onset_f1(pred, true, tol_s=0.010)[2]))
        lab = _bin_label(edges, e.nps) if len(edges) >= 2 else "all"
        order.setdefault(lab, len(edges) and int(np.searchsorted(edges, e.nps, side="right")))
        labels.append(lab)
    arr = np.asarray(scores, dtype=np.float64).reshape(-1, 4)
    by_bin: dict[str, dict] = {}
    for lab in sorted(set(labels), key=lambda k: order[k]):
        sel = np.asarray([x == lab for x in labels])
        by_bin[lab] = {"n": int(sel.sum()), "f1": float(arr[sel, 2].mean()),
                       "f1_10": float(arr[sel, 3].mean())}
    nan = float("nan")
    return {"n": len(entries), "precision": float(arr[:, 0].mean()) if len(arr) else nan,
            "recall": float(arr[:, 1].mean()) if len(arr) else nan,
            "f1": float(arr[:, 2].mean()) if len(arr) else nan,
            "f1_10": float(arr[:, 3].mean()) if len(arr) else nan, "f1_by_nps_bin": by_bin}


def eval_pattern(loaded: LoadedPattern, cache: EventCache, features_dir: Path, split: str,
                 n: int, device: str, entries: Sequence[Entry] | None = None) -> dict:
    """Cross-entropy on one fixed run per difficulty (same construction as validation)."""
    from torch.utils.data import DataLoader

    from bsmaker.models.pattern.dataset import PatternSequences
    from bsmaker.models.pattern.model import pattern_eval

    entries = _spaced(cache.select(split) if entries is None else list(entries), n)
    ds = PatternSequences(cache, entries, features_dir=features_dir, stats=loaded.stats,
                          context=loaded.config.context, seed=1, fixed=True)
    m = pattern_eval(loaded.model, DataLoader(ds, batch_size=16, num_workers=0))
    m = {k: float(v) for k, v in m.items()}
    m["n"] = len(entries)
    return m


def distribution(event_lists: Sequence[Sequence[Event]]) -> dict:
    """Shares over all notes by column, row, direction and hand, plus the double share."""
    counts = {a: np.zeros(k) for a, k in AXES.items()}
    n_events = n_double = 0
    for events in event_lists:
        for e in events:
            n_events += 1
            n_double += int(e.left != NONE_TOKEN and e.right != NONE_TOKEN)
            for hand, tok in enumerate((e.left, e.right)):
                if tok == NONE_TOKEN:
                    continue
                x, y, d = decode_token(tok)
                counts["column"][x] += 1
                counts["row"][y] += 1
                counts["direction"][d] += 1
                counts["hand"][hand] += 1
    out: dict = {}
    for a, c in counts.items():
        total = c.sum()
        out[a] = (c / total if total else c).tolist()
    out["n_notes"] = int(counts["hand"].sum())
    out["double_share"] = n_double / n_events if n_events else 0.0
    return out


def tv_distance(p: Sequence[float], q: Sequence[float]) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def _soft_rates(event_lists: Sequence[Sequence[Event]], cfg: PlayabilityConfig) -> tuple[dict, int, int]:
    """(soft violations per 1000 notes by rule, notes, hard violations)."""
    soft: Counter[str] = Counter()
    notes = hard = 0
    for ev in event_lists:
        rep = check(ev, cfg)
        notes += rep.n_notes
        hard += len(rep.hard)
        soft.update(v.rule for v in rep.soft)
    return ({r: 1000.0 * c / notes for r, c in sorted(soft.items())} if notes else {}), notes, hard


def eval_generated(timing: LoadedTiming, pattern: LoadedPattern, cache: EventCache,
                   features_dir: Path, mask_cfg: PlayabilityConfig, split: str, n: int,
                   seed: int, device: str, entries: Sequence[Entry] | None = None) -> dict:
    """Generate for `n` test difficulties (one per song) and compare with the human maps.

    No audio is decoded: the cached mel is used, there is no beat grid and no snapping, and
    there is a single density round.
    """
    pool = cache.select(split) if entries is None else list(entries)
    entries = _spaced(_one_per_song(pool), n)
    gen_lists: list[list[Event]] = []
    hum_lists: list[list[Event]] = []
    n_events = n_skipped = n_fallback = n_collisions = 0
    masses: list[float] = []
    hits = 0
    for e in entries:
        mel = _mel(features_dir, e)
        times = predict_events(timing, mel, e.nps, e.styles, device)
        notes, rep = sample_notes(pattern, times, mel, e.nps, e.styles, seed, mask_cfg, device)
        events, dropped = to_events(notes)
        if dropped:
            raise RuntimeError(f"{dropped} generated notes dropped by to_events in {e.map_key} (bug)")
        if check(events, mask_cfg).hard:
            raise RuntimeError(f"generated map for {e.map_key} breaks a hard rule "
                               f"({len(check(events, mask_cfg).hard)} violations; bug)")
        if check_exact(notes, mask_cfg).hard:
            raise RuntimeError(f"generated map for {e.map_key} breaks a hard rule on exact "
                               "times (bug)")
        gen_lists.append(events)
        hum_lists.append(human_events(cache, e))
        n_events += rep.n_events
        n_skipped += rep.n_skipped
        n_collisions += rep.n_frame_collisions
        n_fallback += rep.n_single_hand_fallback
        if rep.n_events:
            masses.append(rep.masked_mass_mean)
        m = _measured_nps(notes)
        hits += int(m > 0 and abs(m / e.nps - 1.0) <= DENSITY_TOLERANCE)
    gen_soft, gen_notes, gen_hard = _soft_rates(gen_lists, mask_cfg)
    hum_soft, hum_notes, hum_hard = _soft_rates(hum_lists, mask_cfg)
    gd, hd = distribution(gen_lists), distribution(hum_lists)
    return {
        "n": len(entries), "dropped_notes": 0,
        "nps_range": [float(x) for x in np.percentile([e.nps for e in entries], [0, 50, 100])]
        if entries else [0.0, 0.0, 0.0], "generated": gd, "human": hd,
        "tv": {a: tv_distance(gd[a], hd[a]) for a in AXES},
        "soft_per_1000": {"generated": gen_soft, "human": hum_soft},
        "hard_violations_generated": gen_hard,
        "hard_rate_human_per_1000": 1000.0 * hum_hard / hum_notes if hum_notes else 0.0,
        "skipped_rate": n_skipped / n_events if n_events else 0.0,
        "frame_collision_rate": n_collisions / n_events if n_events else 0.0,
        "single_hand_fallback": n_fallback,
        "masked_mass_mean": float(np.mean(masses)) if masses else 0.0,
        "density_hit_share": hits / len(entries) if entries else 0.0,
    }


UP_DIRS = (0, 4, 5)
DOWN_DIRS = (1, 6, 7)
DIR_NAMES = AXIS_LABELS["direction"]


def row_dir_shares(event_lists: Sequence[Sequence[Event]]) -> tuple[list[list[float]], int]:
    """Joint share table [row][direction] over all notes of both hands, and the note count."""
    counts = np.zeros((3, 9))
    for events in event_lists:
        for e in events:
            for tok in (e.left, e.right):
                if tok != NONE_TOKEN:
                    _, y, d = decode_token(tok)
                    counts[y, d] += 1
    total = counts.sum()
    return (counts / total if total else counts).tolist(), int(total)


def eval_pattern_sampled(pattern: LoadedPattern, cache: EventCache, features_dir: Path,
                         mask_cfg: PlayabilityConfig, split: str, n: int, seed: int, device: str,
                         entries: Sequence[Entry] | None = None) -> dict:
    """Sample notes on the human event times of `n` difficulties and compare row x direction."""
    pool = cache.select(split) if entries is None else list(entries)
    chosen = _spaced(_one_per_song(pool), n)
    gen_lists: list[list[Event]] = []
    hum_lists: list[list[Event]] = []
    for e in chosen:
        human = human_events(cache, e)
        mel = _mel(features_dir, e)
        times = np.asarray([ev.time_s for ev in human], dtype=np.float64)
        notes, _ = sample_notes(pattern, times, mel, e.nps, e.styles, seed, mask_cfg, device)
        events, dropped = to_events(notes)
        if dropped:
            raise RuntimeError(f"{dropped} sampled notes dropped by to_events in {e.map_key} (bug)")
        hard = check(events, mask_cfg).hard
        if hard:
            raise RuntimeError(f"sampled map for {e.map_key} breaks a hard rule "
                               f"({len(hard)} violations; bug)")
        gen_lists.append(events)
        hum_lists.append(human)
    gen, n_gen = row_dir_shares(gen_lists)
    hum, n_hum = row_dir_shares(hum_lists)

    def share(table: list[list[float]], row: int, dirs: tuple[int, ...]) -> float:
        return float(sum(table[row][d] for d in dirs))

    return {
        "n": len(chosen), "generated": gen, "human": hum,
        "n_notes_generated": n_gen, "n_notes_human": n_hum,
        "tv_row_dir": tv_distance(np.ravel(gen), np.ravel(hum)),
        "up_share_bottom_row": {"generated": share(gen, 0, UP_DIRS), "human": share(hum, 0, UP_DIRS)},
        "down_share_top_row": {"generated": share(gen, 2, DOWN_DIRS), "human": share(hum, 2, DOWN_DIRS)},
        "hard_violations": 0,
    }


LAYER_LEVELS = (1, 2, 3)
HUMAN_QUARTER_BEAT_SHARE = 0.76
HUMAN_ONSET_RATIO = 2.5
QUARTER_BEAT_TOL_S = 0.030


def _human_notes(layer_cache, i: int, bpm: float) -> list[Note]:
    t, hand, token = layer_cache.ornaments(i)[:3]
    out = []
    for a, h, tok in zip(t.tolist(), hand.tolist(), token.tolist()):
        try:
            x, y, d = decode_token(int(tok))
        except ValueError:  # unencodable note stored as token 0
            continue
        out.append(Note(float(a), float(a) * bpm / 60.0, x, y, int(h), d))
    return out


def _tv_counts(gen: Sequence[float], hum: Sequence[float]) -> float:
    """TV distance of two count vectors; 1.0 when exactly one side is empty, nan when both are."""
    g, h = np.asarray(gen, dtype=np.float64), np.asarray(hum, dtype=np.float64)
    if g.sum() == 0 and h.sum() == 0:
        return float("nan")
    if g.sum() == 0 or h.sum() == 0:
        return 1.0
    return tv_distance(g / g.sum(), h / h.sum())


def _nanmean(xs: Sequence[float]) -> float:
    a = np.asarray([x for x in xs if x == x], dtype=np.float64)
    return float(a.mean()) if len(a) else float("nan")


def _light_tokens(events: Sequence[LightEvent]) -> list[tuple[int, int]]:
    return [m for ev in events if (m := light_event_to_token(ev.type, ev.value)) is not None]


def _light_metrics(times: np.ndarray, bpm: float, env: np.ndarray) -> tuple[float, float]:
    """(share within 30 ms of a quarter beat, mean onset strength / mean envelope) of event times."""
    if len(times) == 0:
        return float("nan"), float("nan")
    q = 60.0 / bpm / 4.0
    off = np.abs(times / q - np.round(times / q)) * q
    share = float((off <= QUARTER_BEAT_TOL_S).mean())
    if env.size == 0 or env.mean() <= 0:
        return share, float("nan")
    frames = np.clip(np.round(times * FRAMES_PER_SECOND).astype(int), 0, env.size - 1)
    return share, float(env[frames].mean() / env.mean())


def _channel_tv(gen: Sequence[tuple[int, int]], hum: Sequence[tuple[int, int]]) -> float:
    tvs = []
    for ch in range(LIGHT_CHANNELS):
        g = np.bincount([t for c, t in gen if c == ch], minlength=LIGHT_VOCAB)
        h = np.bincount([t for c, t in hum if c == ch], minlength=LIGHT_VOCAB)
        tvs.append(_tv_counts(g, h))
    return _nanmean(tvs)


def _group_bombs(bombs: Sequence[Bomb]) -> list[int]:
    groups: dict[float, set] = {}
    for b in bombs:
        groups.setdefault(round(b.time_s, 4), set()).add((b.x, b.y))
    return [cells_to_mask(c) for c in groups.values()]


def _mean_over(rows: Sequence[dict], key: str) -> float:
    return _nanmean([r[key] for r in rows])


def eval_layers(models: LayerModels, layer_stats: dict, timing, pattern, event_cache: EventCache,
                layer_cache, features_dir: Path, mask_cfg, split: str, n: int, seed: int,
                device: str, entries: Sequence[Entry] | None = None) -> dict:
    """Generate every layer on the HUMAN notes of `n` difficulties (one per song) at levels 1-3.

    Reports per layer and level the elements per minute against the human percentile for the
    map's nps, distribution distances to the human layer, rule violations (must be 0) and rule
    skips. `timing`, `pattern` and `mask_cfg` are accepted for interface stability: the layers
    never touch the note models and the human notes need no mask.
    """
    pool = event_cache.select(split) if entries is None else list(entries)
    pool = [e for e in pool if e.n_frames > 0 and layer_cache.meta(e.i).get("readable")
            and layer_cache.meta(e.i).get("bpm", 0.0) > 0 and layer_cache.meta(e.i).get("n_notes", 0) > 0]
    chosen = _spaced(_one_per_song(pool), n)
    have_orn = models.ornaments is not None
    have_decor = models.decor_when is not None and models.decor_what is not None
    have_lights = models.lights_when is not None and models.lights_what is not None

    def status(ok: bool) -> str:
        return "ok" if ok else "not trained"

    out: dict = {
        "n": len(chosen), "seed": seed, "levels": list(LAYER_LEVELS),
        "reference": {"quarter_beat_share": HUMAN_QUARTER_BEAT_SHARE,
                      "onset_ratio": HUMAN_ONSET_RATIO, "tolerance_s": QUARTER_BEAT_TOL_S},
        "ornaments": {"status": status(have_orn), "levels": {}},
        "decor": {"status": status(have_decor), "levels": {}},
        "lights": {"status": status(have_lights), "levels": {}}}
    bomb_vocab = [int(m) for m in layer_stats.get("bomb_vocab", ())]
    bomb_index = {m: k for k, m in enumerate(bomb_vocab)}

    # human references (over the chosen difficulties)
    h_arc = np.zeros(len(ARC_LENGTH_BEATS))
    h_wall = np.zeros(WALL_VOCAB)
    h_bomb = np.zeros(len(bomb_vocab))
    h_light: list[tuple[int, int]] = []
    h_share: list[float] = []
    h_onset: list[float] = []
    for e in chosen:
        meta, mel = layer_cache.meta(e.i), _mel(features_dir, e)
        env = onset_envelope(mel)
        for v in layer_cache.ornaments(e.i)[4]:
            if v:
                h_arc[int(v) - 1] += 1
        for v in layer_cache.walls(e.i)[2]:
            h_wall[int(v)] += 1
        for m in layer_cache.bombs(e.i)[1]:
            if int(m) in bomb_index:
                h_bomb[bomb_index[int(m)]] += 1
        lt, lch, ltok = layer_cache.lights(e.i)
        if meta.get("lights_ok"):
            h_light += list(zip(lch.tolist(), ltok.tolist()))
            sh, on = _light_metrics(np.asarray(lt, dtype=np.float64), meta["bpm"], env)
            h_share.append(sh)
            h_onset.append(on)
    out["human"] = {"lights": {"quarter_beat_share": _nanmean(h_share),
                               "onset_ratio": _nanmean(h_onset)}}

    for level in LAYER_LEVELS:
        orn_rows: list[dict] = []
        decor_rows: list[dict] = []
        light_rows: dict[str, list[dict]] = {"model": [], "rules": []}
        g_arc = np.zeros(len(ARC_LENGTH_BEATS))
        g_wall = np.zeros(WALL_VOCAB)
        g_bomb = np.zeros(len(bomb_vocab))
        g_light: dict[str, list[tuple[int, int]]] = {"model": [], "rules": []}
        for e in chosen:
            meta, mel = layer_cache.meta(e.i), _mel(features_dir, e)
            bpm = float(meta["bpm"])
            minutes = e.n_frames / FRAMES_PER_SECOND / 60.0
            duration_s = minutes * 60.0
            notes = _human_notes(layer_cache, e.i, bpm)
            env = onset_envelope(mel)
            if have_orn:
                arcs, chains, rep = sample_ornaments(
                    models.ornaments, notes, mel, e.nps, level, layer_stats, bpm, seed, device)
                viol = check_layers(notes, Layers((), (), tuple(arcs), tuple(chains), (), False, 0))
                for a in arcs:
                    g_arc[arc_length_bucket(a.tail_beat - a.beat) - 1] += 1
                orn_rows.append({
                    "arcs_per_min": len(arcs) / minutes, "chains_per_min": len(chains) / minutes,
                    "arcs_target": level_rate(layer_stats, "arcs", e.nps, level),
                    "chains_target": level_rate(layer_stats, "chains", e.nps, level),
                    "violations": len(viol),
                    "rejected": rep.rejected_chain + rep.rejected_arc})
            if have_decor:
                obstacles, bombs, rep = generate_decor(
                    models, mel, notes, e.nps, bpm, duration_s, level, layer_stats, seed, device)
                viol = check_layers(notes, Layers(tuple(bombs), tuple(obstacles), (), (), (), False, 0))
                gp = set(gameplay_obstacles(notes, bpm, duration_s, level, seed))
                for ob in obstacles:
                    m = None if ob in gp else wall_to_token(ob, bpm)
                    if m is not None:
                        g_wall[m[1]] += 1
                for m in _group_bombs(bombs):
                    if m in bomb_index:
                        g_bomb[bomb_index[m]] += 1
                decor_rows.append({
                    "walls_per_min": rep["decor_walls_placed"] / minutes,
                    "bomb_groups_per_min": rep["bomb_groups_placed"] / minutes,
                    "walls_target": level_rate(layer_stats, "walls", e.nps, level),
                    "bomb_groups_target": level_rate(layer_stats, "bomb_groups", e.nps, level),
                    "violations": len(viol), "skipped_by_rules": rep["skipped_by_rules"]})
            target = sum(level_rate(layer_stats, f"light_ch{c}", e.nps, level)
                         for c in range(LIGHT_CHANNELS))
            for kind, use_rules, ok in (("model", False, have_lights), ("rules", True, True)):
                if not ok:
                    continue
                events, rep = generate_lights(models, mel, notes, e.nps, bpm, 0.0, duration_s,
                                              level, layer_stats, seed, device, use_rules=use_rules)
                share, onset = _light_metrics(
                    np.asarray([ev.time_s for ev in events], dtype=np.float64), bpm, env)
                g_light[kind] += _light_tokens(events)
                light_rows[kind].append({
                    "per_min": len(events) / minutes, "target": target,
                    "quarter_beat_share": share, "onset_ratio": onset,
                    "skipped_by_rules": rep["skipped_by_rules"]})
        if have_orn:
            out["ornaments"]["levels"][level] = {
                **{k: _mean_over(orn_rows, k) for k in (
                    "arcs_per_min", "chains_per_min", "arcs_target", "chains_target")},
                "tv_arc_length": _tv_counts(g_arc, h_arc),
                "violations": int(sum(r["violations"] for r in orn_rows)),
                "rejected": int(sum(r["rejected"] for r in orn_rows))}
        if have_decor:
            out["decor"]["levels"][level] = {
                **{k: _mean_over(decor_rows, k) for k in (
                    "walls_per_min", "bomb_groups_per_min", "walls_target", "bomb_groups_target")},
                "tv_wall_token": _tv_counts(g_wall, h_wall),
                "tv_bomb_mask": _tv_counts(g_bomb, h_bomb),
                "violations": int(sum(r["violations"] for r in decor_rows)),
                "skipped_by_rules": int(sum(r["skipped_by_rules"] for r in decor_rows))}
        lv: dict = {}
        for kind, rows in light_rows.items():
            if not rows:
                continue
            lv[kind] = {
                **{k: _mean_over(rows, k) for k in (
                    "per_min", "target", "quarter_beat_share", "onset_ratio")},
                "tv_channel_tokens": _channel_tv(g_light[kind], h_light),
                "skipped_by_rules": int(sum(r["skipped_by_rules"] for r in rows))}
        out["lights"]["levels"][level] = lv
    out["violations"] = int(
        sum(v["violations"] for v in out["ornaments"]["levels"].values())
        + sum(v["violations"] for v in out["decor"]["levels"].values()))
    return out


def render_layers_section(layers: dict | None) -> str:
    """Markdown "Layers" section; empty string when the layers were not evaluated."""
    if not layers:
        return ""
    ref = layers["reference"]
    h = layers["human"]["lights"]
    L = ["## Layers", "",
         f"{layers['n']} difficulties (one per song), layers generated on the human notes at levels "
         f"{', '.join(map(str, layers['levels']))}; per-minute rates are means over difficulties and "
         "`human` is the train percentile (25/50/75) of the map's nps bin.", ""]
    o = layers["ornaments"]
    L += ["### Ornaments", ""]
    if o["status"] != "ok":
        L += ["not trained", ""]
    else:
        L += ["| level | arcs/min | human | chains/min | human | TV arc length | rule violations | "
              "rejected notes |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {lv} | {_f(r['arcs_per_min'], 2)} | {_f(r['arcs_target'], 2)} | "
              f"{_f(r['chains_per_min'], 2)} | {_f(r['chains_target'], 2)} | "
              f"{_f(r['tv_arc_length'])} | {r['violations']} | {r['rejected']} |"
              for lv, r in o["levels"].items()] + [""]
    d = layers["decor"]
    L += ["### Decoration", ""]
    if d["status"] != "ok":
        L += ["not trained", ""]
    else:
        L += ["| level | walls/min | human | bomb groups/min | human | TV wall token | TV bomb mask | "
              "rule violations | skipped by rules |", "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {lv} | {_f(r['walls_per_min'], 2)} | {_f(r['walls_target'], 2)} | "
              f"{_f(r['bomb_groups_per_min'], 2)} | {_f(r['bomb_groups_target'], 2)} | "
              f"{_f(r['tv_wall_token'])} | {_f(r['tv_bomb_mask'])} | {r['violations']} | "
              f"{r['skipped_by_rules']} |" for lv, r in d["levels"].items()] + [""]
    li = layers["lights"]
    L += ["### Lights", ""]
    if li["status"] != "ok":
        L += ["Light model: not trained (rule baseline below).", ""]
    L += [f"Human reference: {ref['quarter_beat_share']:.2f} of events within "
          f"{1000 * ref['tolerance_s']:.0f} ms of a quarter beat, onset strength "
          f"{ref['onset_ratio']:.1f}x the mean (measured on these difficulties: "
          f"{_f(h['quarter_beat_share'])} and {_f(h['onset_ratio'], 2)}).", "",
          "| level | source | events/min | human | quarter-beat share | onset ratio | "
          "TV channel tokens | skipped by rules |", "|---|---|---|---|---|---|---|---|"]
    for lv, kinds in li["levels"].items():
        for kind, r in kinds.items():
            L.append(f"| {lv} | {kind} | {_f(r['per_min'], 1)} | {_f(r['target'], 1)} | "
                     f"{_f(r['quarter_beat_share'])} | {_f(r['onset_ratio'], 2)} | "
                     f"{_f(r['tv_channel_tokens'])} | {r['skipped_by_rules']} |")
    L += ["", f"- Hard rule violations in generated ornaments and decoration: {layers['violations']} "
              "(must be 0)", ""]
    return "\n".join(L)


def json_safe(x):
    """Recursively convert to JSON-serialisable values (NaN and inf become None)."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else None
    return x


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def _summary(r: dict) -> list[str]:
    t, p, g = r["timing"], r["pattern"], r["generated"]
    sp = r.get("pattern_sampled")
    extra = [f"- Timing onset F1 at 10 ms tolerance: {_f(t['f1_10'])}"]
    if sp:
        a, h = sp["up_share_bottom_row"]["generated"], sp["up_share_bottom_row"]["human"]
        extra.append(f"- Pattern on human event times ({sp['n']} difficulties): tv_row_dir "
                     f"{_f(sp['tv_row_dir'])}, upward cuts in the bottom row: generated "
                     f"{_f(a)} vs human {_f(h)}")
    return [
        "## Summary", "",
        f"- Timing onset F1 at 25 ms ({t['n']} songs, human event count): {_f(t['f1'])} "
        f"(precision {_f(t['precision'])}, recall {_f(t['recall'])})",
        f"- Pattern cross-entropy, both hands (joint, per event): {_f(p['ce'])} "
        f"(left hand {_f(p['ce_left'])}, right hand given left {_f(p['ce_right'])})",
        f"- Generated maps ({g['n']}): {g['hard_violations_generated']} hard violations, "
        f"density within 10% after a single round: {100 * g['density_hit_share']:.0f}%",
        f"- Double share: generated {_f(g['generated']['double_share'])} vs "
        f"human {_f(g['human']['double_share'])}",
        f"- Skipped-event rate: {100 * g['skipped_rate']:.1f}% "
        f"(frame collisions, counted separately: {100 * g['frame_collision_rate']:.1f}%)",
    ] + extra


def render_summary(results: dict) -> str:
    return "\n".join(_summary(results)) + "\n"


def render_eval_report(results: dict) -> str:
    t, p, g, s = results["timing"], results["pattern"], results["generated"], results["setup"]
    L = [f"# Evaluation report ({s['split']})", ""] + _summary(results) + [""]
    L += ["## Timing", "", "| | value |", "|---|---|",
          f"| precision | {_f(t['precision'])} |", f"| recall | {_f(t['recall'])} |",
          f"| F1 (25 ms) | {_f(t['f1'])} |", f"| F1 (10 ms) | {_f(t['f1_10'])} |", "",
          "F1 by notes-per-second bin:", "", "| nps bin | songs | F1 (25 ms) | F1 (10 ms) |",
          "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {_f(v['f1'])} | {_f(v['f1_10'])} |"
          for k, v in t["f1_by_nps_bin"].items()]
    L += ["", "## Pattern", "", "| | cross-entropy | perplexity |", "|---|---|---|"]
    for name, key in (("both hands (joint, per event)", "ce"), ("left hand", "ce_left"),
                      ("right hand given left", "ce_right")):
        v = p[key]
        L.append(f"| {name} | {_f(v)} | {_f(float(np.exp(v)) if v == v else v)} |")
    L += ["", "## Generated vs human", "",
          f"{g['n']} test difficulties, one per song. Generation uses the cached mel (no audio is "
          "decoded), no beat-grid detection and no snapping, and a single density round.", ""]
    for axis, labels in AXIS_LABELS.items():
        L += [f"| {axis} | generated | human |", "|---|---|---|"]
        L += [f"| {lab} | {_f(a)} | {_f(b)} |"
              for lab, a, b in zip(labels, g["generated"][axis], g["human"][axis])]
        L.append("")
    L += ["| axis | total-variation distance |", "|---|---|"]
    L += [f"| {a} | {_f(v)} |" for a, v in g["tv"].items()]
    L += ["", f"Double share: generated {_f(g['generated']['double_share'])}, "
              f"human {_f(g['human']['double_share'])}.", "",
          "Soft-rule violations per 1 000 notes:", "", "| rule | generated | human |", "|---|---|---|"]
    gs, hs = g["soft_per_1000"]["generated"], g["soft_per_1000"]["human"]
    L += [f"| {r} | {_f(gs.get(r, 0.0), 1)} | {_f(hs.get(r, 0.0), 1)} |"
          for r in sorted(set(gs) | set(hs))] or []
    L += ["", f"- Notes dropped by event grouping in generated maps: {g['dropped_notes']} (must be 0)",
          f"- Hard violations in generated maps: {g['hard_violations_generated']} (must be 0)",
          f"- Hard violations in human maps under the mask profile (information): "
          f"{_f(g['hard_rate_human_per_1000'], 1)} per 1 000 notes",
          f"- Skipped-event rate (playability mask): {100 * g['skipped_rate']:.2f}%",
          f"- Frame-collision rate (events sharing a mel frame, not mask skips): "
          f"{100 * g['frame_collision_rate']:.2f}%",
          f"- Mean probability mass removed by the mask: {_f(g['masked_mass_mean'], 4)}",
          f"- Share of maps within 10% of the target nps after a single round: "
          f"{100 * g['density_hit_share']:.0f}%", ""]
    sp = results.get("pattern_sampled")
    if sp:
        L += ["## Pattern on human event times", "",
              f"{sp['n']} difficulties (one per song); notes sampled with the production "
              "temperature and top-p on the human event times.", "",
              "| row / direction | " + " | ".join(DIR_NAMES) + " |",
              "|---|" + "---|" * 9]
        for side in ("generated", "human"):
            for row in (2, 1, 0):
                L.append(f"| {side} row {row} | " + " | ".join(_f(v) for v in sp[side][row]) + " |")
        L += ["", f"- tv_row_dir (27 cells): {_f(sp['tv_row_dir'])}",
              f"- Upward cuts (up, up-left, up-right) in the bottom row: generated "
              f"{_f(sp['up_share_bottom_row']['generated'])}, human "
              f"{_f(sp['up_share_bottom_row']['human'])}",
              f"- Downward cuts in the top row: generated "
              f"{_f(sp['down_share_top_row']['generated'])}, human "
              f"{_f(sp['down_share_top_row']['human'])}",
              f"- Notes: generated {sp['n_notes_generated']}, human {sp['n_notes_human']}",
              f"- Hard violations: {sp['hard_violations']} (must be 0)", ""]
    lo, med, hi = g["nps_range"]

    def ck(kind: str) -> list[str]:
        out = [f"- {kind.capitalize()} checkpoint: {s[kind + '_checkpoint']} "
               f"(step {s[kind + '_step']})"]
        for key in ("model_config", "train_options"):
            if s.get(f"{kind}_{key}"):
                out.append(f"  - {key}: {json.dumps(json_safe(s[f'{kind}_{key}']), sort_keys=True)}")
        return out

    section = render_layers_section(results.get("layers"))
    if section:
        L += section.split("\n")
    L += ["## Setup", ""] + ck("timing") + ck("pattern")
    if s.get("entries_from"):
        L.append(f"- Entries from: {s['entries_from']} (kept {s['entries_kept']} of "
                 f"{s['entries_in_split']} {s['split']} difficulties)")
    L += [f"- Split: {s['split']}", f"- n (timing, pattern): {s['n']}; n generated: {s['n_generated']}; "
          f"n sampled pattern: {s.get('n_sampled', 0)}",
          f"- Generated difficulties' nps (min / median / max): {lo:.2f} / {med:.2f} / {hi:.2f}",
          f"- Seed: {s['seed']}", f"- Date: {s.get('date', date.today().isoformat())}", ""]
    return "\n".join(L)
