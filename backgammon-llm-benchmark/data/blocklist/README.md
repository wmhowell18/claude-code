# data/blocklist/

Versioned dedup blocklist: normalized XGID keys of **known public** positions
(GNU BG example DBs, public match archives scraped *only to exclude*, opening
books, textbook/quiz positions). Candidates matching the blocklist are rejected
(PLAN.md §2.3). Committed so exclusions are reproducible. Empty in Phase 0.

## File format

Each list is a `*.json` file in this directory. Loading is handled by
`generate.dedup.load_blocklist()` (all `*.json` files, sorted by name, merged).
A file is a JSON object with **any** of these keys:

| Key | Value | Meaning |
|-----|-------|---------|
| `xgids` | `["XGID=...", ...]` | Raw XGID strings; normalized on load. |
| `keys` | `["<canonical-key>", ...]` | Pre-normalized keys (from `dedup.mirror_key`). |
| `entries` | `[{"xgid": "..."} \| {"key": "..."}, ...]` | Annotated entries (may carry extra provenance fields, which are ignored by the matcher). |

Recommended top-level metadata (ignored by the matcher, kept for provenance):
`name`, `version`, `source`, `source_url`, `retrieved`, `license`, `note`.

### Normalization (how matching works)

Both the blocklist entries **and** the candidate being tested are reduced to the
same key before comparison: `min(canonical_key(board), canonical_key(flip(board)))`
— the mover-relative canonical form folded with its color/symmetry mirror
(`generate.dedup.mirror_key`). So a blocked position matches regardless of which
seat/color it is presented in or its dice ordering (PLAN.md §2.3).

### Example

```json
{
  "name": "gnubg-example-positions",
  "version": "2026.07",
  "source": "GNU Backgammon distributed example database",
  "source_url": "https://www.gnu.org/software/gnubg/",
  "retrieved": "2026-07-12",
  "license": "GPL-3.0 (positions used only to EXCLUDE, never redistributed)",
  "note": "Known-public positions; any candidate matching these is rejected.",
  "xgids": [
    "XGID=-b----E-C---eE---c-e----B-:0:0:1:31:0:0:0:0:10"
  ],
  "entries": [
    { "xgid": "XGID=-a----DDC---bB---bbb-c-B--:1:1:1:52:0:0:0:7:10",
      "why": "famous prime-vs-prime reference position" }
  ]
}
```
