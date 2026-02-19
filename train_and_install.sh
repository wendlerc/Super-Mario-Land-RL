#!/bin/bash
# Install dependencies and train
export PATH="/home/node/.local/bin:$PATH"
export PIP_BREAK_SYSTEM_PACKAGES=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy

cd /data/workspace/mario-land-rl

echo "Installing dependencies..."
pip install --user torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5
pip install --user stable-baselines3 opencv-python 2>&1 | tail -5

echo "Starting training..."
python3 train_sb3.py --rom "/data/workspace/roms/Super Mario Land (World) (Rev 1).gb" --timesteps 50000 2>&1 | tee training.log

echo "Training complete!"
