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

`scrape` indexes only the maps it downloaded in that run plus maps marked `done` that have no
difficulty rows (left unindexed by a crashed run); use `index-raw` to re-index everything
in the raw folder.
