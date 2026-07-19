# CONTAMINATION — anti-contamination policy + audit steps

> Phase 0 placeholder. See PLAN.md §2 for the authoritative policy.

Positions must be **in-distribution** (real strong play) yet **out-of-corpus**
(not in pretraining). Controls:

1. **Generate, don't harvest** — fresh strong-bot self-play, sampled on the
   creation date (PLAN.md §2.1).
2. **Exclude famous/canonical** — textbook/quiz/opening-book positions and
   anything web-findable (PLAN.md §2.2).
3. **Deduplicate** — versioned blocklist of public-source hashes + intra-set
   dedup via normalized XGID (PLAN.md §2.3).
4. **Verify unfindability** — rate-limited web search of exact XGID / GNU BG ID,
   check date recorded (PLAN.md §2.4).
5. **Held-out + canaries + date stamps** — private authoritative set never
   published in full; only SHA-256 hashes + canary token + creation date are
   released (PLAN.md §2.5; CANARY.md).

Residual, accepted risks (format familiarity, distributional leakage, public-dev
decay) are documented in PLAN.md §2.6.

## Held-out tooling (`scripts/heldout.py`)

The private set's lifecycle is operationalized by `scripts/heldout.py`
(stdlib-only; `openssl` via subprocess for encryption):

- `hash` — writes `positions/heldout/hashes.json` (the only published artifact):
  a per-record SHA-256 over canonical JSON plus a set-level manifest (count,
  combined hash, `dataset_hash`, creation date, canary). Canonical JSON =
  `json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
  UTF-8, so any third party can reproduce the hashes.
- `verify` — recomputes from local plaintext and diffs the committed
  `hashes.json` (nonzero exit on mismatch). `verify --results <file>` instead
  checks that a results file's `manifest.dataset_hash` matches the committed
  manifest — provable, offline, that a run scored against this exact set.
- `encrypt` / `decrypt` — package the plaintext into one AES-encrypted tar for
  backup. `openssl enc` cannot do AEAD/GCM on stock OpenSSL 3.x, so the tool
  detects support and otherwise uses `aes-256-cbc -pbkdf2 -iter 600000`. The key
  (`GAMMONBENCH_HELDOUT_KEY` or `--key-file`) is never written to disk; the
  ciphertext's SHA-256 and cipher mode are recorded in
  `data/manifests/heldout-blobs.json`.

See `positions/heldout/README.md` for the full generate → hash → commit →
encrypt workflow.
