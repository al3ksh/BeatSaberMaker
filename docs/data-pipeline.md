# Data pipeline notes

Commands: `bsmaker data index-local`, `index-raw`, `scrape`, `features`, `fetch-metadata`,
`calibrate-playability`.

## Concurrency rule

The manifest (`manifest.sqlite`) is opened in WAL mode with a 120 s busy timeout. This gives:

- **Any number of readers.** A reader (a training loader, `iter_accepted`, an ad-hoc query, the
  calibration read) never blocks the writer and is never blocked by it.
- **One bulk writer job at a time.** Run only one of `index-local`, `index-raw`, `scrape`,
  `features` and `fetch-metadata` at once. Two writers do not corrupt the file, but they queue on
  the write lock, and their summaries (and the `existing` snapshot taken by `scrape`) stop
  meaning anything.

WAL is persistent: it is switched on the first time a process opens the manifest and no other
process has it open. Keep read transactions short (call `fetchall()` before slow per-row work).

## Environment failures

ffmpeg and ffprobe must be on PATH. `index-local`, `index-raw`, `scrape` and `features` check
this at start and exit with code 2 if either is missing. A missing tool, a full disk or a locked
database aborts the command; it is never recorded as a verdict on a map (`unreadable`,
`no_audio`). `features` also stops with a message when free space in the features folder drops
below 10 GB.

`features --workers N` (default `min(8, cpus - 2)`, at least 1) decodes in N worker processes.
Workers only write `.npy` files; the main process alone reads and writes the manifest.

`scrape` indexes only the maps it downloaded in that run plus maps marked `done` that have no
difficulty rows (left unindexed by a crashed run); use `index-raw` to re-index everything
in the raw folder.

## Song-level split

Train/val/test is assigned per song, not per map key: maps of the same song must never land in
different splits. `bsmaker data resplit` groups all maps in the manifest (title+artist equal
after normalisation, or equal title with durations within 2 s, transitively; an empty
normalised title is its own song), stores the smallest key of the group in `maps.song_id` and
sets every `difficulties.split` to `assign_split(song_id)`. It is idempotent. Re-indexing keeps
an existing row's split; run `resplit` after indexing new maps. The manifest schema version
lives in `PRAGMA user_version` (2); older files are migrated by adding `maps.song_id`.

Splits are frozen for a model generation: `index.json` of the event cache stores the split of every
difficulty at `data build-events` time. After any `data resplit` that changes a split, rebuild the
event cache with `data build-events` and retrain both models (`resplit` prints a warning).
