// ============================================================================
// BACKGAMMON WEB INTERFACE - GAME LOGIC
// ============================================================================

// Game state
let gameState = {
    board: null,
    currentPlayer: 'WHITE',
    dice: null,
    legalMoves: [],
    gameOver: false,
    humanPlayer: 'WHITE', // User plays as white by default
};

// ============================================================================
// API CALLS
// ============================================================================

async function apiCall(endpoint, data = {}) {
    try {
        const response = await fetch(`/api/${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        if (!result.success) {
            console.error(`API error: ${result.error}`);
            alert(`Error: ${result.error}`);
        }
        return result;
    } catch (error) {
        console.error('API call failed:', error);
        alert('Network error. Please check the server is running.');
        return { success: false };
    }
}

async function newGame() {
    const result = await apiCall('new_game');
    if (result.success) {
        gameState.board = result.board;
        gameState.currentPlayer = result.current_player;
        gameState.gameOver = result.game_over;
        gameState.dice = null;
        gameState.legalMoves = [];

        updateUI();

        // If AI plays first, make AI move
        if (gameState.currentPlayer !== gameState.humanPlayer) {
            setTimeout(aiMove, 500);
        }
    }
}

async function rollDice() {
    const result = await apiCall('roll_dice');
    if (result.success) {
        gameState.dice = result.dice;
        gameState.legalMoves = result.legal_moves || [];

        updateUI();

        // If no legal moves and it's AI's turn, let AI play
        if (gameState.legalMoves.length === 0 && gameState.currentPlayer !== gameState.humanPlayer) {
            alert('No legal moves! Turn passes.');
            setTimeout(aiMove, 1000);
        }
    }
}

async function makeMove(move) {
    const result = await apiCall('make_move', { move });
    if (result.success) {
        gameState.board = result.board;
        gameState.currentPlayer = result.current_player;
        gameState.gameOver = result.game_over;
        gameState.dice = null;
        gameState.legalMoves = [];

        updateUI();

        if (gameState.gameOver) {
            showGameOver(result.winner);
        } else if (gameState.currentPlayer !== gameState.humanPlayer) {
            // AI's turn
            setTimeout(aiMove, 500);
        }
    }
}

async function aiMove() {
    // Disable buttons during AI move
    document.getElementById('roll-dice-btn').disabled = true;
    document.getElementById('ai-move-btn').disabled = true;

    const result = await apiCall('ai_move');
    if (result.success) {
        gameState.board = result.board;
        gameState.currentPlayer = result.current_player;
        gameState.gameOver = result.game_over;
        gameState.dice = null;
        gameState.legalMoves = [];

        // Show AI's move
        console.log('AI rolled:', result.dice);
        console.log('AI played:', result.move);

        updateUI();

        if (gameState.gameOver) {
            showGameOver(result.winner);
        }
    }

    // Re-enable buttons
    document.getElementById('roll-dice-btn').disabled = false;
    document.getElementById('ai-move-btn').disabled = false;
}

async function getHint() {
    const result = await apiCall('get_hint');
    if (result.success) {
        const move = result.suggested_move;
        const moveStr = move.map(step => `${step.from}→${step.to}`).join(', ');

        document.getElementById('hint-text').textContent = `AI suggests: ${moveStr}`;
        document.getElementById('hint-section').style.display = 'block';

        // Hide hint after 5 seconds
        setTimeout(() => {
            document.getElementById('hint-section').style.display = 'none';
        }, 5000);
    }
}

// ============================================================================
// UI RENDERING
// ============================================================================

function updateUI() {
    renderBoard();
    updateGameInfo();
    updateButtons();
}

function renderBoard() {
    if (!gameState.board) return;

    const { white_checkers, black_checkers } = gameState.board;

    // Clear all points
    document.querySelectorAll('.point').forEach(point => {
        point.innerHTML = '';
    });

    // Render checkers on each point
    for (let pointNum = 1; pointNum <= 24; pointNum++) {
        const whiteCount = white_checkers[pointNum];
        const blackCount = black_checkers[pointNum];

        const pointEl = document.querySelector(`.point[data-point="${pointNum}"]`);
        if (!pointEl) continue;

        // Render white checkers
        for (let i = 0; i < whiteCount; i++) {
            const checker = createChecker('white', i < 5 ? null : whiteCount);
            pointEl.appendChild(checker);
        }

        // Render black checkers
        for (let i = 0; i < blackCount; i++) {
            const checker = createChecker('black', i < 5 ? null : blackCount);
            pointEl.appendChild(checker);
        }
    }

    // Render bar (point 0)
    renderBar(white_checkers[0], black_checkers[0]);

    // Render borne off (point 25)
    renderBorneOff(white_checkers[25], black_checkers[25]);
}

function createChecker(color, count = null) {
    const checker = document.createElement('div');
    checker.className = `checker ${color}`;

    // Show count if > 5 checkers stacked
    if (count !== null && count > 5) {
        const countEl = document.createElement('div');
        countEl.className = 'checker-count';
        countEl.textContent = count;
        checker.appendChild(countEl);
    }

    return checker;
}

function renderBar(whiteCount, blackCount) {
    // Bar rendering can be added here if needed
    // For now, we'll skip it to keep things simple
}

function renderBorneOff(whiteCount, blackCount) {
    const whiteOffEl = document.getElementById('white-off');
    const blackOffEl = document.getElementById('black-off');

    whiteOffEl.innerHTML = '';
    blackOffEl.innerHTML = '';

    // Render white borne off
    for (let i = 0; i < Math.min(whiteCount, 15); i++) {
        whiteOffEl.appendChild(createChecker('white'));
    }

    // Render black borne off
    for (let i = 0; i < Math.min(blackCount, 15); i++) {
        blackOffEl.appendChild(createChecker('black'));
    }
}

function updateGameInfo() {
    // Current player
    document.getElementById('current-player').textContent = gameState.currentPlayer;
    document.getElementById('current-player').style.color =
        gameState.currentPlayer === 'WHITE' ? '#f5f5f5' : '#2c2c2c';

    // Dice
    if (gameState.dice) {
        const diceSymbols = gameState.dice.map(d => ['⚀','⚁','⚂','⚃','⚄','⚅'][d-1]);
        document.getElementById('dice-display').textContent =
            `${diceSymbols.join(' ')} (${gameState.dice.join(', ')})`;
    } else {
        document.getElementById('dice-display').textContent = '--';
    }

    // Legal moves
    document.getElementById('legal-moves-count').textContent = gameState.legalMoves.length;
}

function updateButtons() {
    const isHumanTurn = gameState.currentPlayer === gameState.humanPlayer;
    const hasDice = gameState.dice !== null;
    const hasLegalMoves = gameState.legalMoves.length > 0;

    // Roll dice button
    document.getElementById('roll-dice-btn').disabled =
        !isHumanTurn || hasDice || gameState.gameOver;

    // AI move button
    document.getElementById('ai-move-btn').disabled =
        isHumanTurn || gameState.gameOver;

    // Hint button
    document.getElementById('hint-btn').disabled =
        !isHumanTurn || !hasLegalMoves || gameState.gameOver;

    // New game button always enabled
    document.getElementById('new-game-btn').disabled = false;
}

function showGameOver(winner) {
    document.getElementById('winner-text').textContent =
        `${winner} wins!`;
    document.getElementById('game-over-section').style.display = 'block';
}

// ============================================================================
// MOVE INPUT (SIMPLIFIED)
// ============================================================================

// For now, we'll use a simple approach: show legal moves and let user select
// A more sophisticated UI would allow drag-and-drop or click-to-move

function showLegalMoves() {
    if (gameState.legalMoves.length === 0) {
        alert('No legal moves available. Roll dice first!');
        return;
    }

    // Create a simple text representation of legal moves
    let moveText = 'Legal moves:\n\n';
    gameState.legalMoves.slice(0, 10).forEach((move, idx) => {
        const moveStr = move.map(step => `${step.from}→${step.to}`).join(', ');
        moveText += `${idx + 1}. ${moveStr}\n`;
    });

    if (gameState.legalMoves.length > 10) {
        moveText += `\n... and ${gameState.legalMoves.length - 10} more`;
    }

    const choice = prompt(moveText + '\n\nEnter move number (1-' +
        Math.min(10, gameState.legalMoves.length) + ') or cancel:');

    if (choice) {
        const moveIdx = parseInt(choice) - 1;
        if (moveIdx >= 0 && moveIdx < gameState.legalMoves.length) {
            makeMove(gameState.legalMoves[moveIdx]);
        }
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Button listeners
    document.getElementById('new-game-btn').addEventListener('click', newGame);
    document.getElementById('roll-dice-btn').addEventListener('click', rollDice);
    document.getElementById('ai-move-btn').addEventListener('click', aiMove);
    document.getElementById('hint-btn').addEventListener('click', getHint);

    // Player color selection
    document.querySelectorAll('input[name="player-color"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            gameState.humanPlayer = e.target.value.toUpperCase();
            newGame();
        });
    });

    // Point click to show legal moves (simplified move input)
    document.querySelectorAll('.point').forEach(point => {
        point.addEventListener('click', () => {
            if (gameState.currentPlayer === gameState.humanPlayer &&
                gameState.legalMoves.length > 0) {
                showLegalMoves();
            }
        });
    });

    // Initialize game
    newGame();
});

// ============================================================================
// UTILITY
// ============================================================================

function formatMove(move) {
    return move.map(step => `${step.from}→${step.to}`).join(', ');
}

console.log('🎲 Backgammon game loaded!');
