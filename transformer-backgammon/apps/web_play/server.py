#!/usr/bin/env python3
"""Web server for playing backgammon against your trained AI.

Based on MateiCosa/backgammon-ai architecture but adapted for JAX/Flax models.

Usage:
    python server.py --checkpoint /path/to/checkpoint

Then open: http://localhost:8002
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flask import Flask, render_template, request, jsonify
from flax.training import checkpoints

# Import your backgammon modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backgammon.core.board import (
    initial_board,
    generate_legal_moves,
    apply_move,
    is_game_over,
    winner,
    Board,
)
from backgammon.core.types import Player, Move
from backgammon.core.dice import all_possible_dice
from backgammon.encoding.encoder import encode_board
from backgammon.network.network import BackgammonTransformer
from backgammon.evaluation.network_agent import create_neural_agent


# ==============================================================================
# FLASK APP
# ==============================================================================

app = Flask(__name__)

# Global state
GAME_STATE = {
    'board': None,
    'current_player': Player.WHITE,
    'dice': None,
    'legal_moves': [],
    'game_over': False,
    'winner': None,
}

AI_AGENT = None


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(checkpoint_path: str):
    """Load trained JAX/Flax model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or file

    Returns:
        Loaded model agent
    """
    global AI_AGENT

    print(f"Loading model from: {checkpoint_path}")

    # Create agent from checkpoint
    AI_AGENT = create_neural_agent(checkpoint_path, temperature=0.1)

    print(f"✅ Model loaded successfully!")
    return AI_AGENT


# ==============================================================================
# GAME LOGIC
# ==============================================================================

def new_game():
    """Initialize a new game."""
    global GAME_STATE

    GAME_STATE = {
        'board': initial_board(),
        'current_player': Player.WHITE,
        'dice': None,
        'legal_moves': [],
        'game_over': False,
        'winner': None,
    }

    return GAME_STATE


def board_to_dict(board: Board) -> Dict:
    """Convert board state to JSON-serializable dictionary.

    Args:
        board: Board object

    Returns:
        Dictionary with board state
    """
    return {
        'white_checkers': board.white_checkers.tolist(),
        'black_checkers': board.black_checkers.tolist(),
        'player_to_move': board.player_to_move.name,
    }


def move_to_dict(move: Move) -> List[Dict]:
    """Convert move to JSON-serializable format.

    Args:
        move: Move tuple (from, to) pairs

    Returns:
        List of move steps
    """
    if not move:
        return []

    return [{'from': step[0], 'to': step[1]} for step in move]


def dict_to_move(move_dict: List[Dict]) -> Move:
    """Convert dictionary back to move tuple.

    Args:
        move_dict: List of move step dicts

    Returns:
        Move tuple
    """
    if not move_dict:
        return ()

    return tuple((step['from'], step['to']) for step in move_dict)


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def api_new_game():
    """Start a new game.

    Returns:
        JSON with initial game state
    """
    state = new_game()

    return jsonify({
        'success': True,
        'board': board_to_dict(state['board']),
        'current_player': state['current_player'].name,
        'game_over': state['game_over'],
    })


@app.route('/api/roll_dice', methods=['POST'])
def api_roll_dice():
    """Roll dice for current player.

    Returns:
        JSON with dice roll and legal moves
    """
    global GAME_STATE

    # Roll dice
    dice = tuple(np.random.randint(1, 7, size=2))
    GAME_STATE['dice'] = dice

    # Generate legal moves
    legal_moves = generate_legal_moves(
        GAME_STATE['board'],
        GAME_STATE['current_player'],
        dice
    )
    GAME_STATE['legal_moves'] = legal_moves

    return jsonify({
        'success': True,
        'dice': dice,
        'num_legal_moves': len(legal_moves),
        'legal_moves': [move_to_dict(move) for move in legal_moves[:10]],  # First 10
    })


@app.route('/api/make_move', methods=['POST'])
def api_make_move():
    """Apply a move to the board.

    Request JSON:
        move: Move in dictionary format

    Returns:
        JSON with updated game state
    """
    global GAME_STATE

    data = request.get_json()
    move_dict = data.get('move', [])
    move = dict_to_move(move_dict)

    # Validate move is legal
    if move not in GAME_STATE['legal_moves']:
        return jsonify({
            'success': False,
            'error': 'Illegal move',
        })

    # Apply move
    GAME_STATE['board'] = apply_move(
        GAME_STATE['board'],
        GAME_STATE['current_player'],
        move
    )

    # Check game over
    if is_game_over(GAME_STATE['board']):
        GAME_STATE['game_over'] = True
        outcome = winner(GAME_STATE['board'])
        GAME_STATE['winner'] = outcome.winner.name if outcome else None
    else:
        # Switch player
        GAME_STATE['current_player'] = GAME_STATE['current_player'].opponent()

    # Clear dice and moves
    GAME_STATE['dice'] = None
    GAME_STATE['legal_moves'] = []

    return jsonify({
        'success': True,
        'board': board_to_dict(GAME_STATE['board']),
        'current_player': GAME_STATE['current_player'].name,
        'game_over': GAME_STATE['game_over'],
        'winner': GAME_STATE['winner'],
    })


@app.route('/api/ai_move', methods=['POST'])
def api_ai_move():
    """Let AI make a move.

    Returns:
        JSON with AI's move and updated state
    """
    global GAME_STATE, AI_AGENT

    if AI_AGENT is None:
        return jsonify({
            'success': False,
            'error': 'AI model not loaded',
        })

    # Roll dice if needed
    if GAME_STATE['dice'] is None:
        dice = tuple(np.random.randint(1, 7, size=2))
        GAME_STATE['dice'] = dice
        GAME_STATE['legal_moves'] = generate_legal_moves(
            GAME_STATE['board'],
            GAME_STATE['current_player'],
            dice
        )

    # AI selects move
    move = AI_AGENT.select_move(
        GAME_STATE['board'],
        GAME_STATE['current_player'],
        GAME_STATE['dice'],
        GAME_STATE['legal_moves']
    )

    # Apply move
    GAME_STATE['board'] = apply_move(
        GAME_STATE['board'],
        GAME_STATE['current_player'],
        move
    )

    # Check game over
    if is_game_over(GAME_STATE['board']):
        GAME_STATE['game_over'] = True
        outcome = winner(GAME_STATE['board'])
        GAME_STATE['winner'] = outcome.winner.name if outcome else None
    else:
        # Switch player
        GAME_STATE['current_player'] = GAME_STATE['current_player'].opponent()

    # Clear dice and moves
    dice_used = GAME_STATE['dice']
    GAME_STATE['dice'] = None
    GAME_STATE['legal_moves'] = []

    return jsonify({
        'success': True,
        'dice': dice_used,
        'move': move_to_dict(move),
        'board': board_to_dict(GAME_STATE['board']),
        'current_player': GAME_STATE['current_player'].name,
        'game_over': GAME_STATE['game_over'],
        'winner': GAME_STATE['winner'],
    })


@app.route('/api/get_hint', methods=['POST'])
def api_get_hint():
    """Get AI suggestion for current position.

    Returns:
        JSON with suggested move
    """
    global GAME_STATE, AI_AGENT

    if AI_AGENT is None:
        return jsonify({
            'success': False,
            'error': 'AI model not loaded',
        })

    if not GAME_STATE['legal_moves']:
        return jsonify({
            'success': False,
            'error': 'No legal moves available. Roll dice first.',
        })

    # AI suggests move
    suggested_move = AI_AGENT.select_move(
        GAME_STATE['board'],
        GAME_STATE['current_player'],
        GAME_STATE['dice'],
        GAME_STATE['legal_moves']
    )

    return jsonify({
        'success': True,
        'suggested_move': move_to_dict(suggested_move),
    })


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backgammon web interface')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8002,
        help='Port to bind to (default: 8002)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )

    args = parser.parse_args()

    # Load model
    load_model(args.checkpoint)

    # Initialize game
    new_game()

    # Start server
    print(f"\n🎲 Backgammon server starting...")
    print(f"   Model: {args.checkpoint}")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"\n✅ Open the URL in your browser to play!\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
