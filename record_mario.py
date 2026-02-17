#!/usr/bin/env python3
"""Record a short clip of Mario gameplay"""
import os
import sys
import numpy as np
from PIL import Image

# Add path for ROM
os.environ['ROM_PATH'] = '/data/workspace/roms'

import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

def record_mario_clip(duration_seconds=10, output_path='mario_clip.mp4'):
    """Record Mario gameplay for a short clip"""
    
    # Create environment
    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Video writer setup
    fps = 30
    frame_width = 256
    frame_height = 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Reset and play
    state = env.reset()
    frames_recorded = 0
    max_frames = duration_seconds * fps
    
    print(f"Recording {duration_seconds}s clip...")
    
    while frames_recorded < max_frames:
        # Random action (no trained agent, just showing the game works)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
        frames_recorded += 1
        
        if done:
            state = env.reset()
    
    out.release()
    env.close()
    
    print(f"Saved clip to: {output_path}")
    print(f"Frames: {frames_recorded}")
    return output_path

if __name__ == '__main__':
    output = '/data/workspace/mario_clip.mp4'
    record_mario_clip(duration_seconds=15, output_path=output)
    print(f"Clip ready at: {output}")
