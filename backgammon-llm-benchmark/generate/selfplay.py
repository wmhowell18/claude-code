"""Drive strong-bot self-play (PLAN.md §2.1).

Generates brand-new match/money games via GNU BG (and later XG) self-play on the
dataset creation date, so sampled positions cannot be in any pretraining corpus.
Writes raw games to ``data/selfplay/``.
"""
