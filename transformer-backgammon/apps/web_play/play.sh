#!/bin/bash
# Quick launch script for backgammon web interface
#
# Usage:
#   ./play.sh /path/to/checkpoint
#
# Example:
#   ./play.sh ../../checkpoints/checkpoint_3200

if [ -z "$1" ]; then
    echo "❌ Error: Please provide checkpoint path"
    echo ""
    echo "Usage: ./play.sh /path/to/checkpoint"
    echo ""
    echo "Example:"
    echo "  ./play.sh ../../checkpoints/checkpoint_3200"
    exit 1
fi

CHECKPOINT_PATH="$1"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "🎲 Starting Backgammon Web Interface..."
echo "   Checkpoint: $CHECKPOINT_PATH"
echo ""

python server.py --checkpoint "$CHECKPOINT_PATH"
