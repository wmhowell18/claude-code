# Backgammon Web Interface

**Play against your trained transformer model in a beautiful web interface!**

This web app is adapted from [MateiCosa/backgammon-ai](https://github.com/MateiCosa/backgammon-ai) but modified to work with JAX/Flax models instead of PyTorch.

## Features

- 🎨 **Visual board** - Beautiful backgammon board with checkers
- 🤖 **AI opponent** - Play against your trained transformer
- 💡 **Hints** - Ask AI for move suggestions
- 🎲 **Full game rules** - Legal move generation and validation
- 📱 **Responsive** - Works on desktop and mobile

## Quick Start

### 1. Install Dependencies

```bash
pip install flask
# Your JAX/Flax dependencies should already be installed
```

### 2. Run the Server

```bash
cd transformer-backgammon/apps/web_play

# Point to your trained model checkpoint
python server.py --checkpoint /path/to/your/checkpoint/directory
```

**Example:**
```bash
# Using Colab checkpoint from Google Drive
python server.py --checkpoint "/content/drive/MyDrive/backgammon_training/checkpoints"

# Using local checkpoint
python server.py --checkpoint "../../checkpoints/checkpoint_3200"
```

### 3. Open in Browser

Navigate to: **http://localhost:8002**

## How to Play

1. **Choose your color:**
   - White (moves 24→1)
   - Black (moves 1→24)

2. **Roll dice:** Click "Roll Dice" button

3. **Make move:** Click any point on the board to see legal moves, then select one

4. **AI plays:** AI automatically plays when it's their turn

5. **Get hints:** Click "Get Hint" to see what the AI would do

## Command-Line Options

```bash
python server.py [OPTIONS]

Options:
  --checkpoint PATH   Path to model checkpoint (required)
  --host HOST         Host to bind to (default: localhost)
  --port PORT         Port to bind to (default: 8002)
  --debug             Run in debug mode
```

**Examples:**

```bash
# Custom port
python server.py --checkpoint /path/to/checkpoint --port 8080

# Allow external connections
python server.py --checkpoint /path/to/checkpoint --host 0.0.0.0

# Debug mode (auto-reload on code changes)
python server.py --checkpoint /path/to/checkpoint --debug
```

## Architecture

### Backend (`server.py`)
- **Flask** web server
- Loads JAX/Flax checkpoint
- Provides REST API for game logic
- Handles move generation and AI inference

### Frontend
- **HTML** (`templates/index.html`) - Board layout
- **CSS** (`static/styles.css`) - Visual styling
- **JavaScript** (`static/game.js`) - Game logic and API calls

### API Endpoints

- `POST /api/new_game` - Start new game
- `POST /api/roll_dice` - Roll dice for current player
- `POST /api/make_move` - Apply a move to board
- `POST /api/ai_move` - Let AI make a move
- `POST /api/get_hint` - Get AI move suggestion

## Troubleshooting

### "Model not loaded" error
Make sure you're pointing to a valid checkpoint directory with the `--checkpoint` flag.

### "No legal moves" message
This is normal! Sometimes there are no legal moves with certain dice rolls. The turn automatically passes to the opponent.

### Port already in use
Change the port with `--port`:
```bash
python server.py --checkpoint /path/to/checkpoint --port 8003
```

### Can't connect from another device
Use `--host 0.0.0.0` to allow external connections:
```bash
python server.py --checkpoint /path/to/checkpoint --host 0.0.0.0
```

Then access from other device: `http://YOUR_IP:8002`

## Customization

### Change AI Difficulty

Edit `server.py`, line ~63:
```python
# Lower temperature = stronger play (less random)
AI_AGENT = create_neural_agent(checkpoint_path, temperature=0.1)  # Strong
AI_AGENT = create_neural_agent(checkpoint_path, temperature=1.0)  # Weaker
```

### Change Board Colors

Edit `static/styles.css`, lines 9-15:
```css
:root {
    --board-bg: #8b4513;         /* Board background */
    --point-dark: #654321;       /* Dark points */
    --point-light: #d4a574;      /* Light points */
    --checker-white: #f5f5f5;    /* White checkers */
    --checker-black: #2c2c2c;    /* Black checkers */
}
```

## Credits

- **Architecture adapted from:** [MateiCosa/backgammon-ai](https://github.com/MateiCosa/backgammon-ai)
- **Model training:** transformer-backgammon project
- **Game engine:** backgammon core modules (JAX/Flax)

## License

MIT (same as transformer-backgammon project)
