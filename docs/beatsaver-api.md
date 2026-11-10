# BeatSaver API notes

Verified 2026-11-10 against `https://api.beatsaver.com/docs` (Swagger UI) and its spec
`https://api.beatsaver.com/docs/swagger.json`, plus a handful of small real requests
(by-ids, search). Only `api.beatsaver.com` was contacted; nothing was downloaded.

## Endpoints used

- `GET /maps/ids/{ids}`: up to 50 comma-separated ids ("Up to 50 ids seperated by commas").
  Returns a JSON object `{id: map}`; ids that do not exist (e.g. `zzzz`) are simply
  omitted (no error, still HTTP 200). Plan matched.
- `GET /search/text/{page}` (solr search; `/search/v1/{page}` is the older variant with fewer
  filters). Returns `{"docs": [map...], "info": {"total", "pages", "duration"}}`; page size
  is 20 in practice (`info.pages` showed total/20 although `pageSize` is listed).
- `GET /maps/hash/{hash}`: not used.

## Search parameters (plan -> reality)

| Plan wants | API parameter |
|---|---|
| ranked / BeatLeader-ranked | `leaderboard=Ranked` (ScoreSaber) or `BeatLeader`; also `All`, `ScoreSaber` |
| curated | `curated=true` |
| minimum rating | `minRating` (0-1 float; also `maxRating`, `minUpVotes`, `minVotes`, ...) |
| exclude automapper | `automapper=false`. Doc says "options are a little weird". Observed: `automapper=false&leaderboard=Ranked` returned 0 results, so combine with care. |
| exclude AI-declared | no filter; check `declaredAi` on the result (`"None"`/absent = human) and drop client-side |
| sort | `order` in `Latest, Relevance, Rating, Curated, Random, Duration`; `ascending` bool on `/search/text` |
| paging | `{page}` path segment, 0-based |
| tags | `tags=a,b` (and), `a|b` (or), `!x` excluded |

## Response fields (map document)

- `id` (hex string, the "key"), `name`, `description`, `uploader.name`,
  `metadata.{bpm,duration,songName,songSubName,songAuthorName,levelAuthorName}`.
- `stats.{upvotes,downvotes,score,...}`: `score` is 0-1.
- `tags`: list of strings; style tags: `tech`, `dance-style`, `speed`, `accuracy`,
  `balanced`, `challenge`, `fitness`, plus genre tags (`pop`, `rock`, `anime`, ...).
- Flags: `ranked` (ScoreSaber), `blRanked` (BeatLeader), `qualified`, `blQualified`,
  `automapper`, `declaredAi` (string), `nsfw`.
- `curatedAt` / `curator`: present when curated. There is no `curated` boolean in the
  document; curated := `curatedAt` is non-empty (nearest equivalent used).
- `versions[]` (last = newest): `hash`, `key`, `state`, `downloadURL`, `coverURL`, `previewURL`,
  `diffs[]`. Hash and download URL live on the version, not on the map.

## Download host

`downloadURL` values observed: `https://r2cdn.beatsaver.com/<sha1>.zip`. `download()`
accepts only `https` and host `r2cdn.beatsaver.com`, refuses other hosts and does not follow
redirects. (Only this host was observed; extend the allow-list deliberately if others appear.)

## Rate limiting / etiquette

The documentation states no rate limit, and no `RateLimit-*` or `Retry-After` headers were
seen on 200 responses. The API description asks mirrors to use the websocket API
(`wss://ws.beatsaver.com/maps`) instead of making hundreds of requests. It does not ask
clients for contact details (the User-Agent therefore carries none). We keep
`min_interval_s = 0.25` (at most 4 req/s; the client refuses lower values), send only
`User-Agent: BeatSaberMaker/0.1`, and retry 429/5xx with backoff. The real run is ~28 requests.

## Fixtures

`tests/fixtures/beatsaver/maps_by_ids.json` (2 maps, `/maps/ids/2,3a,zzzz`) and
`search_page.json` (3 maps from `/search/text/0?leaderboard=Ranked&order=Rating`), trimmed to
metadata: descriptions emptied, uploader reduced to the name, no avatars, difficulties dropped.
