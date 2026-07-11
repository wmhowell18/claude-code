"""Tests for exact bearoff evaluation wired into search and self-play.

Covers TODO item 129: when the shared bearoff database is enabled, the
evaluation funnels (search._batch_evaluate, self_play._batched_inference,
NeuralNetworkAgent.get_equity_estimate) return EXACT values for
mutual-bearoff positions where gammons are impossible, and fall back to
the network everywhere else.

Uses small database builds (max_checkers <= 6) so runtime stays negligible.
"""

import numpy as np
import pytest
import jax

from backgammon.core.board import (
    empty_board,
    initial_board,
    apply_move,
    generate_legal_moves,
    is_game_over,
)
from backgammon.core.types import Player
from backgammon.encoding.encoder import enhanced_encoding_config
from backgammon.training.train import TrainingConfig, create_train_state
from backgammon.evaluation.bearoff import (
    BearoffDatabase,
    enable_exact_bearoff,
    disable_exact_bearoff,
    get_exact_bearoff_db,
    exact_bearoff_value,
    exact_bearoff_equity6,
)
from backgammon.evaluation.search import (
    _batch_evaluate,
    _batch_evaluate_network,
    select_move_0ply,
)
from backgammon.evaluation.network_agent import NeuralNetworkAgent
from backgammon.training.self_play import (
    _batched_inference,
    _batched_inference_network,
    _get_jit_inference,
    play_games_batched,
)


@pytest.fixture(scope="module")
def db6():
    """Database covering positions with up to 6 checkers per side."""
    return BearoffDatabase.build(max_checkers=6)


@pytest.fixture(scope="module")
def small_state():
    """Small training state for exercising the network paths."""
    config = TrainingConfig(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        train_policy=False,
    )
    rng = jax.random.PRNGKey(42)
    return create_train_state(config, rng)


@pytest.fixture(autouse=True)
def _reset_shared_db():
    """The shared database is process-global state; never leak it."""
    disable_exact_bearoff()
    yield
    disable_exact_bearoff()


def _race_board(white_pts, black_dists, player_to_move=Player.WHITE):
    """Mutual bearoff race board.

    Args:
        white_pts: {point: count} for White (points 1-6); rest borne off.
        black_dists: {distance: count} for Black (point 25-d); rest off.
        player_to_move: Who is on roll.
    """
    board = empty_board()
    for pt, cnt in white_pts.items():
        board.set_checkers(Player.WHITE, pt, cnt)
    board.set_checkers(Player.WHITE, 25, 15 - sum(white_pts.values()))
    for d, cnt in black_dists.items():
        board.set_checkers(Player.BLACK, 25 - d, cnt)
    board.set_checkers(Player.BLACK, 25, 15 - sum(black_dists.values()))
    board.player_to_move = player_to_move
    return board


# ============================================================
# Shared database accessor
# ============================================================


class TestSharedDatabaseAccessor:
    def test_disabled_by_default(self):
        assert get_exact_bearoff_db() is None

    def test_enable_with_explicit_db(self, db6):
        returned = enable_exact_bearoff(db=db6)
        assert returned is db6
        assert get_exact_bearoff_db() is db6

    def test_disable(self, db6):
        enable_exact_bearoff(db=db6)
        disable_exact_bearoff()
        assert get_exact_bearoff_db() is None

    def test_training_config_flag_defaults(self):
        config = TrainingConfig()
        assert config.use_exact_bearoff is False
        assert config.bearoff_max_checkers == 15


# ============================================================
# Exactness gate
# ============================================================


class TestExactnessGate:
    def test_not_bearing_off_returns_none(self, db6):
        board = initial_board()
        assert exact_bearoff_value(db6, board) is None
        assert exact_bearoff_equity6(db6, board) is None

    def test_gammon_possible_returns_none(self, db6):
        # Black has borne off nothing: a gammon is still possible, so the
        # gammon-free database equity would not be exact.
        board = _race_board({1: 2}, {1: 15})
        assert board.black_checkers[25] == 0
        assert exact_bearoff_value(db6, board) is None

    def test_beyond_db_limit_returns_none(self, db6):
        # 8 checkers on White's side exceeds the db6 build limit.
        board = _race_board({1: 4, 2: 4}, {2: 1})
        assert exact_bearoff_value(db6, board) is None

    def test_exact_value_matches_win_probability(self, db6):
        board = _race_board({1: 1, 2: 1, 5: 1}, {2: 1, 3: 1, 4: 1})
        ours = (1, 1, 0, 0, 1, 0)
        theirs = (0, 1, 1, 1, 0, 0)
        p = db6.win_probability(ours, theirs)
        val = exact_bearoff_value(db6, board)
        assert val == pytest.approx(2.0 * p - 1.0)

    def test_exact_equity6_structure(self, db6):
        board = _race_board({1: 1, 2: 1, 5: 1}, {2: 1, 3: 1, 4: 1})
        eq = exact_bearoff_equity6(db6, board)
        assert eq.shape == (6,)
        assert eq.dtype == np.float32
        # Gammon/backgammon slots exactly zero, distribution sums to 1
        assert eq[1] == eq[2] == eq[4] == eq[5] == 0.0
        assert float(eq.sum()) == pytest.approx(1.0)
        # Consistent with the scalar value (no gammons: value = 2p - 1)
        val = exact_bearoff_value(db6, board)
        assert float(eq[0] - eq[3]) == pytest.approx(val, abs=1e-6)

    def test_black_on_roll_uses_black_counts_first(self, db6):
        board = _race_board(
            {1: 1, 2: 1, 5: 1}, {2: 1, 3: 1, 4: 1},
            player_to_move=Player.BLACK,
        )
        p = db6.win_probability((0, 1, 1, 1, 0, 0), (1, 1, 0, 0, 1, 0))
        assert exact_bearoff_value(db6, board) == pytest.approx(2.0 * p - 1.0)


# ============================================================
# Search funnel (_batch_evaluate)
# ============================================================


class TestSearchInterception:
    def test_disabled_matches_network_path(self, small_state):
        cfg = enhanced_encoding_config()
        boards = [initial_board(), _race_board({1: 2, 3: 1}, {2: 2})]
        np.testing.assert_allclose(
            _batch_evaluate(small_state, boards, cfg),
            _batch_evaluate_network(small_state, boards, cfg),
        )

    def test_mixed_batch_exact_and_network(self, small_state, db6):
        enable_exact_bearoff(db=db6)
        cfg = enhanced_encoding_config()

        exact_a = _race_board({1: 2, 3: 1}, {2: 2})
        midgame = initial_board()
        exact_b = _race_board({2: 1, 6: 2}, {1: 1, 5: 1})
        values = _batch_evaluate(small_state, [exact_a, midgame, exact_b], cfg)

        assert values[0] == pytest.approx(exact_bearoff_value(db6, exact_a))
        assert values[2] == pytest.approx(exact_bearoff_value(db6, exact_b))
        # The midgame board still goes through the network.
        net = _batch_evaluate_network(small_state, [midgame], cfg)
        assert values[1] == pytest.approx(float(net[0]))

    def test_all_exact_batch_makes_no_network_call(self, db6):
        enable_exact_bearoff(db=db6)
        cfg = enhanced_encoding_config()
        boards = [
            _race_board({1: 2, 3: 1}, {2: 2}),
            _race_board({2: 1, 6: 2}, {1: 1, 5: 1}),
        ]
        # state=None would crash inside _batch_evaluate_network, so this
        # passing proves the network path is never entered.
        values = _batch_evaluate(None, boards, cfg)
        for b, v in zip(boards, values):
            assert v == pytest.approx(exact_bearoff_value(db6, b))

    def test_select_move_0ply_plays_max_win_probability_move(
        self, small_state, db6
    ):
        enable_exact_bearoff(db=db6)
        cfg = enhanced_encoding_config()
        board = _race_board({1: 1, 2: 1, 5: 1}, {2: 1, 3: 1, 4: 1})
        dice = (5, 2)
        legal = generate_legal_moves(board, Player.WHITE, dice)
        assert len(legal) > 1

        chosen, chosen_val = select_move_0ply(
            small_state, board, Player.WHITE, legal, cfg
        )

        # Every after-state is exactly evaluable (both sides still have
        # checkers off), so the chosen move must maximize the exact value.
        best = -np.inf
        for mv in legal:
            after = apply_move(board, Player.WHITE, mv)
            if is_game_over(after):
                continue
            # After our move the opponent is on roll; negate for us.
            best = max(best, -exact_bearoff_value(db6, after))
        assert chosen_val == pytest.approx(best)


# ============================================================
# Self-play funnel (_batched_inference) and TD targets
# ============================================================


class TestSelfPlayInterception:
    def test_disabled_matches_network_path(self, small_state):
        jit_fn = _get_jit_inference(small_state.apply_fn)
        boards = [initial_board(), _race_board({1: 2, 3: 1}, {2: 2})]
        np.testing.assert_allclose(
            _batched_inference(jit_fn, small_state.params, boards),
            _batched_inference_network(jit_fn, small_state.params, boards),
        )

    def test_mixed_batch_exact_rows(self, small_state, db6):
        enable_exact_bearoff(db=db6)
        jit_fn = _get_jit_inference(small_state.apply_fn)

        exact_a = _race_board({1: 2, 3: 1}, {2: 2})
        midgame = initial_board()
        equity = _batched_inference(
            jit_fn, small_state.params, [exact_a, midgame]
        )

        np.testing.assert_allclose(
            equity[0], exact_bearoff_equity6(db6, exact_a)
        )
        net = _batched_inference_network(jit_fn, small_state.params, [midgame])
        np.testing.assert_allclose(equity[1], net[0])

    def test_td_value_estimates_are_exact_in_bearoff(self, small_state, db6):
        enable_exact_bearoff(db=db6)
        start = _race_board({1: 1, 2: 1, 4: 1}, {1: 1, 2: 1, 3: 1})
        rng = np.random.default_rng(7)

        results = play_games_batched(
            num_games=2,
            state=small_state,
            variants=[start],
            temperature=0.0,
            max_moves=50,
            rng=rng,
            record_value_estimates=True,
        )

        checked = 0
        for game in results:
            assert game.outcome is not None  # tiny races always finish
            assert len(game.value_estimates) == len(game.steps)
            for step, est in zip(game.steps, game.value_estimates):
                exact = exact_bearoff_equity6(db6, step.board)
                if exact is not None:
                    np.testing.assert_allclose(est, exact)
                    checked += 1
        # The starting position qualifies, so at least the first step of
        # each game must have been checked.
        assert checked >= 2


# ============================================================
# Neural agent equity estimates
# ============================================================


class TestAgentEquityInterception:
    def test_exact_in_bearoff(self, small_state, db6):
        enable_exact_bearoff(db=db6)
        agent = NeuralNetworkAgent(small_state)
        board = _race_board({1: 2, 3: 1}, {2: 2})
        np.testing.assert_allclose(
            agent.get_equity_estimate(board),
            exact_bearoff_equity6(db6, board),
        )

    def test_network_outside_bearoff(self, small_state, db6):
        enable_exact_bearoff(db=db6)
        agent = NeuralNetworkAgent(small_state)
        board = initial_board()
        estimate = agent.get_equity_estimate(board)
        assert estimate.shape == (6,)
        disable_exact_bearoff()
        np.testing.assert_allclose(
            estimate, agent.get_equity_estimate(board)
        )
