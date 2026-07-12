"""Tests: async runner end-to-end on a fake transport (PLAN.md §5.2)."""

import asyncio
import json

import httpx

from bgcore.board import Board
from harness import report
from harness.cache import ResponseCache
from harness.client import OpenRouterClient
from harness.cost import BudgetGuard, CostTracker
from harness.runner import (
    dataset_hash,
    load_positions,
    load_rollouts,
    parse_simple_yaml,
    run_config,
    run_model,
)


def _run(coro):
    return asyncio.run(coro)


# -- fixtures (hand-built, NOT under positions/) --------------------------


def _checker_position():
    b = Board.starting_position([3, 1])
    return {
        "position_id": "chk-1",
        "xgid": "XGID-CHK-1",
        "board_json": b.to_json(),
        "ascii": "ASCII-CHK",
        "tier": "T2",
        "decision_type": "checker",
    }


def _cube_position():
    b = Board.starting_position()  # no dice -> cube
    return {
        "position_id": "cube-1",
        "xgid": "XGID-CUBE-1",
        "board_json": b.to_json(),
        "ascii": "ASCII-CUBE",
        "tier": "T3",
        "decision_type": "cube",
    }


ROLLOUTS = {
    "chk-1": {
        "position_id": "chk-1",
        "decision_type": "checker",
        "checker": {"moves": [
            {"move": "8/5 6/5", "equity": 0.1, "error_mp": 0.0, "std_err": 0.002, "rank": 1},
            {"move": "24/23 8/5", "equity": -0.2, "error_mp": 250.0, "std_err": 0.003, "rank": 2},
        ]},
        "best_move": "8/5 6/5",
    },
    "cube-1": {
        "position_id": "cube-1",
        "decision_type": "cube",
        "cube": {"best_action": "No double",
                 "error_mp": {"No double": 0.0, "Double, Take": 80.0, "Double, Pass": 500.0}},
    },
}


def _answer_for(request, *, cost=0.001):
    """Return the 'right' answer based on which contract the prompt asked for.

    Discriminate on the *user* message only (the system prompt mentions both
    MOVE: and ACTION:, so it cannot be used to tell the decision type apart).
    """
    body = json.loads(request.content)
    user = body["messages"][-1]["content"]
    user_text = user if isinstance(user, str) else json.dumps(user)
    if "CUBE decision" in user_text:
        content = "Reasoning...\nACTION: no double"
    else:
        content = "I build the five point.\nMOVE: 8/5 6/5"
    return httpx.Response(200, json={
        "id": "gen", "model": body["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "cost": cost,
                  "completion_tokens_details": {"reasoning_tokens": 1}},
    })


# -- tests ----------------------------------------------------------------


def test_run_model_end_to_end_and_results_schema(tmp_path):
    positions = [_checker_position(), _cube_position()]
    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(_answer_for))
    cache = ResponseCache(tmp_path / "cache")
    cost = CostTracker()

    mr = _run(run_model(
        client, positions, ROLLOUTS, model="fake/model", track="text",
        sampling={"temperature": 0, "n": 1, "mode": "greedy", "max_retries": 2},
        cache=cache, cost=cost, raw_dir=tmp_path / "raw",
        dataset_hash_str=dataset_hash(positions),
    ))
    _run(client.aclose())

    assert len(mr.decisions) == 2
    # both answered best -> zero loss, full accuracy
    assert all(d.is_best for d in mr.decisions)
    assert all(d.equity_loss == 0.0 for d in mr.decisions)
    assert mr.cost_usd > 0

    obj = report.build_results(mr, run_id="testrun")
    errs = report.validate_results(obj)
    assert errs == [], errs
    assert obj["aggregate"]["benchpr"] == 0.0
    assert obj["aggregate"]["best_move_accuracy"] == 1.0
    assert obj["manifest"]["dataset_hash"] == dataset_hash(positions)
    assert set(obj["aggregate"]["per_decision_type"]) == {"checker", "cube"}
    # raw responses were written
    assert (tmp_path / "raw" / "fake_model" / "chk-1.json").is_file()


def test_cache_makes_rerun_free(tmp_path):
    positions = [_checker_position()]
    calls = {"n": 0}

    def counting(request):
        calls["n"] += 1
        return _answer_for(request)

    cache = ResponseCache(tmp_path / "cache")
    sampling = {"temperature": 0, "n": 1, "mode": "greedy", "max_retries": 0}

    c1 = OpenRouterClient(api_key="k", transport=httpx.MockTransport(counting))
    _run(run_model(c1, positions, ROLLOUTS, model="m", track="text",
                   sampling=sampling, cache=cache))
    _run(c1.aclose())
    assert calls["n"] == 1

    # second run hits the cache: no new requests
    c2 = OpenRouterClient(api_key="k", transport=httpx.MockTransport(counting))
    mr2 = _run(run_model(c2, positions, ROLLOUTS, model="m", track="text",
                         sampling=sampling, cache=cache))
    _run(c2.aclose())
    assert calls["n"] == 1
    assert mr2.decisions[0].is_best


def test_reparse_retry_then_success(tmp_path):
    positions = [_checker_position()]

    def handler(request):
        body = json.loads(request.content)
        text = json.dumps(body["messages"])
        if "Reply again" in text:  # this is a re-ask
            return httpx.Response(200, json={
                "id": "g", "model": body["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": "MOVE: 8/5 6/5"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            })
        return httpx.Response(200, json={  # first answer: unparseable
            "id": "g", "model": body["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": "hmm not sure"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
        })

    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    mr = _run(run_model(client, positions, ROLLOUTS, model="m", track="text",
                        sampling={"temperature": 0, "max_retries": 2}))
    _run(client.aclose())
    d = mr.decisions[0]
    assert d.attempts == 2  # one failure + one successful re-ask
    assert d.is_best and not d.parse_failed


def test_budget_exhaustion_scores_unanswered_worst(tmp_path):
    positions = [_checker_position(),
                 dict(_checker_position(), position_id="chk-2"),
                 dict(_checker_position(), position_id="chk-3")]
    rollouts = {p["position_id"]: dict(ROLLOUTS["chk-1"], position_id=p["position_id"])
                for p in positions}

    def handler(request):
        return _answer_for(request, cost=0.03)

    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    budget = BudgetGuard(0.05)
    mr = _run(run_model(client, positions, rollouts, model="m", track="text",
                        sampling={"temperature": 0, "max_retries": 0},
                        budget=budget, concurrency=1))
    _run(client.aclose())

    answered = [d for d in mr.decisions if not d.parse_failed]
    unanswered = [d for d in mr.decisions if d.parse_failed]
    assert len(answered) == 2      # 0.03 * 2 = 0.06 crosses 0.05
    assert len(unanswered) == 1
    # the unanswered one was scored worst-legal (250 mpt)
    assert unanswered[0].equity_loss_mp == 250.0
    assert unanswered[0].attempts == 0


def test_run_config_writes_valid_results(tmp_path):
    positions = [_checker_position(), _cube_position()]
    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(_answer_for))
    config = {
        "run_id": "e2e",
        "track": "text",
        "models": ["fake/a"],
        "sampling": {"temperature": 0, "n": 1, "mode": "greedy", "max_retries": 2},
        "budget": {"usd": None},
        "concurrency": 4,
        "results_dir": "results",
    }
    results = _run(run_config(config, base_dir=tmp_path, client=client,
                              positions=positions, rollouts=ROLLOUTS))
    _run(client.aclose())
    assert len(results) == 1

    out = tmp_path / "results" / "e2e__fake_a__text.json"
    assert out.is_file()
    obj = json.loads(out.read_text())
    assert report.validate_results(obj) == []
    # working tree written
    assert (tmp_path / "runs" / "e2e" / "scores" / "fake_a__text.jsonl").is_file()
    assert (tmp_path / "runs" / "e2e" / "cache").is_dir()


def test_dataset_loading_from_dir_and_jsonl(tmp_path):
    posdir = tmp_path / "positions"
    posdir.mkdir()
    (posdir / "a.json").write_text(json.dumps(_checker_position()))
    (posdir / "b.json").write_text(json.dumps(_cube_position()))
    loaded = load_positions(posdir)
    assert {p["position_id"] for p in loaded} == {"chk-1", "cube-1"}

    rl = tmp_path / "rollouts.jsonl"
    rl.write_text("\n".join(json.dumps(v) for v in ROLLOUTS.values()))
    rolls = load_rollouts(rl)
    assert set(rolls) == {"chk-1", "cube-1"}


def test_config_mini_yaml_matches_expected():
    text = (
        "run_id: r1\n"
        "track: text\n"
        "models:\n"
        "  - m/one\n"
        "  - m/two\n"
        "sampling:\n"
        "  temperature: 0\n"
        "  n: 3\n"
        "  mode: self-consistency\n"
        "budget:\n"
        "  usd: null\n"
    )
    cfg = parse_simple_yaml(text)
    assert cfg["run_id"] == "r1"
    assert cfg["models"] == ["m/one", "m/two"]
    assert cfg["sampling"] == {"temperature": 0, "n": 3, "mode": "self-consistency"}
    assert cfg["budget"] == {"usd": None}
