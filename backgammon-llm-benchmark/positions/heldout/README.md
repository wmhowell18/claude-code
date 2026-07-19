# positions/heldout/ — the PRIVATE authoritative set

This directory holds the real leaderboard positions. **It is never committed.**
`.gitignore` ignores everything here except `hashes.json` and this `README.md`.
The plaintext records (and any `rollouts/` subdirectory) exist only on the
maintainer's machine and inside the encrypted archive.

Only two things are ever published from this set (PLAN.md §2.5):

1. `hashes.json` — SHA-256 hashes of each held-out record plus a set-level
   manifest. It proves *what* is in the set (and lets a model's training cutoff
   be checked against a creation date) without revealing the positions.
2. The canary token, embedded in every record and in `CANARY.md`.

## Workflow (build → publish → archive)

All commands run from the repo root via `scripts/heldout.py`:

```
# 1. generate + rollout the held-out positions (needs gnubg) into:
#      positions/heldout/*.json                 (position records)
#      positions/heldout/rollouts/*.json         (or rollouts/heldout/*.json)
python3 scripts/build_dataset.py ...            # produces the plaintext here

# 2. hash → writes positions/heldout/hashes.json (COMMIT this file)
python3 scripts/heldout.py hash

# 3. verify the committed hashes match local plaintext (CI / before publish)
python3 scripts/heldout.py verify

# 4. encrypt the plaintext into a single tar.enc for backup / cold storage
export GAMMONBENCH_HELDOUT_KEY='<a strong passphrase>'
python3 scripts/heldout.py encrypt        # -> data/heldout/heldout-<date>.tar.enc
                                          #    registered in data/manifests/heldout-blobs.json

# 5. later: restore the plaintext on a fresh machine
python3 scripts/heldout.py decrypt --in data/heldout/heldout-<date>.tar.enc
```

## Canonical-JSON hash spec (independently verifiable)

Each record's hash is `sha256` of its canonical JSON:

```python
json.dumps(record, sort_keys=True, separators=(",", ":"),
           ensure_ascii=False).encode("utf-8")
```

Keys are sorted recursively, there is no insignificant whitespace, and non-ASCII
characters are kept as UTF-8 (not `\uXXXX`-escaped). The manifest's
`combined_hash` is the `sha256` over the sorted per-record hex digests, each
followed by a newline (independent of file order). `dataset_hash` is the same
order-independent hash the evaluation harness writes into every results file
(`harness.runner.dataset_hash`), so `heldout.py verify --results <file>` can
confirm a run was scored against this exact set — offline, from `hashes.json`
alone.

## Encryption

`openssl enc` cannot do AEAD/GCM on stock OpenSSL 3.x ("AEAD ciphers not
supported"), so the tool detects GCM support and otherwise uses
`aes-256-cbc -pbkdf2 -iter 600000`. The key is taken from
`GAMMONBENCH_HELDOUT_KEY` or `--key-file` and is **never written to disk**. The
ciphertext's SHA-256 is printed and recorded so the blob's integrity is checkable.
