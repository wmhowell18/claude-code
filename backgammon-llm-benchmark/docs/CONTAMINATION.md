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
