"""Tests: cost tracking, budget guard, self-consistency vote (PLAN.md §5.2-5.3)."""

from bgcore.board import Board
from harness import parse
from harness.cost import BudgetGuard, CostTracker, majority_vote


def test_cost_tracker_totals():
    ct = CostTracker()
    ct.add("a/b", 0.01)
    ct.add("a/b", 0.02)
    ct.add("c/d", 0.05)
    assert ct.total == 0.08
    assert ct.by_model() == {"a/b": 0.03, "c/d": 0.05}
    assert ct.requests_by_model() == {"a/b": 2, "c/d": 1}


def test_budget_untracked_always_spends():
    g = BudgetGuard(None)
    assert g.can_spend(100.0)
    g.spend(100.0)
    assert not g.exhausted
    assert g.remaining == float("inf")


def test_budget_guard_stops_exactly_at_cap():
    g = BudgetGuard(0.10)
    spent = 0.0
    issued = 0
    # each request costs 0.03; we should issue while budget allows, then stop
    while g.can_spend():
        g.spend(0.03)
        spent += 0.03
        issued += 1
        if issued > 100:
            break
    # 0.03 * 4 = 0.12 > 0.10; guard stops after the charge that crosses the cap
    assert issued == 4
    assert g.exhausted
    assert not g.can_spend()
    assert g.spent > g.budget_usd


def test_budget_can_spend_with_estimate():
    g = BudgetGuard(0.05)
    assert g.can_spend(0.05)
    assert not g.can_spend(0.06)  # estimate exceeds remaining


def test_majority_vote_by_equivalent_class():
    b = Board.starting_position([3, 1])
    outs = [
        parse.parse_answer(b, "MOVE: 8/5 6/5"),
        parse.parse_answer(b, "MOVE: 6/5 8/5"),   # equivalent to the above
        parse.parse_answer(b, "MOVE: 24/21/20"),  # a distinct legal play
    ]
    winner, tally = majority_vote(b, outs)
    assert winner is not None
    assert tally[0][1] == 2  # the 8/5 6/5 class won with 2 votes
    # only two distinct classes
    assert sum(c for _, c in tally) == 3
    assert len(tally) == 2


def test_majority_vote_no_parse():
    b = Board.starting_position([3, 1])
    outs = [parse.parse_answer(b, "dunno"), parse.parse_answer(b, "nope")]
    winner, tally = majority_vote(b, outs)
    assert winner is None and tally == []


def test_majority_vote_cube():
    b = Board.starting_position()
    outs = [
        parse.parse_answer(b, "ACTION: no double"),
        parse.parse_answer(b, "ACTION: double, take"),
        parse.parse_answer(b, "ACTION: no double"),
    ]
    winner, tally = majority_vote(b, outs)
    assert winner.cube_action == ("no double", None)
    assert tally[0][1] == 2
