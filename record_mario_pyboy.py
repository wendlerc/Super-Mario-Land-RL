#!/usr/bin/env python3
"""Record a short clip of Super Mario Land using PyBoy"""
import os
import sys
from pathlib import Path

# Add pyboy to path if needed
try:
    from pyboy import PyBoy
except ImportError:
    sys.path.insert(0, '/home/node/.local/lib/python3.11/site-packages')
    from pyboy import PyBoy

from pyboy.utils import WindowEvent
import numpy as np
import cv2

def record_mario_clip(rom_path, duration_seconds=15, output_path='mario_clip.mp4'):
    """Record Mario gameplay for a short clip"""
    
    # Create PyBoy instance (headless)
    pyboy = PyBoy(rom_path, window="null")
    
    # Get screen dimensions
    screen = pyboy.screen
    height, width = screen.ndarray.shape[:2]
    print(f"Screen size: {width}x{height}")
    
    # Video writer setup
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Button mapping for PyBoy
    buttons = {
        'right': WindowEvent.PRESS_ARROW_RIGHT,
        'left': WindowEvent.PRESS_ARROW_LEFT,
        'up': WindowEvent.PRESS_ARROW_UP,
        'down': WindowEvent.PRESS_ARROW_DOWN,
        'a': WindowEvent.PRESS_BUTTON_A,
        'b': WindowEvent.PRESS_BUTTON_B,
        'start': WindowEvent.PRESS_BUTTON_START,
        'select': WindowEvent.PRESS_BUTTON_SELECT,
    }
    releases = {
        'right': WindowEvent.RELEASE_ARROW_RIGHT,
        'left': WindowEvent.RELEASE_ARROW_LEFT,
        'up': WindowEvent.RELEASE_ARROW_UP,
        'down': WindowEvent.RELEASE_ARROW_DOWN,
        'a': WindowEvent.RELEASE_BUTTON_A,
        'b': WindowEvent.RELEASE_BUTTON_B,
        'start': WindowEvent.RELEASE_BUTTON_START,
        'select': WindowEvent.RELEASE_BUTTON_SELECT,
    }
    
    # Press Start to begin
    pyboy.send_input(buttons['start'])
    pyboy.tick()
    pyboy.send_input(releases['start'])
    
    frames_recorded = 0
    max_frames = duration_seconds * fps
    button_list = ['right', 'left', 'up', 'down', 'a', 'b']
    current_button = None
    button_duration = 0
    
    print(f"Recording {duration_seconds}s clip...")
    
    while frames_recorded < max_frames and pyboy.tick():
        # Get screen as numpy array
        screen_array = screen.ndarray
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
        frames_recorded += 1
        
        # Random button presses every few frames
        if button_duration <= 0:
            if current_button:
                pyboy.send_input(releases[current_button])
            current_button = button_list[frames_recorded % len(button_list)]
            pyboy.send_input(buttons[current_button])
            button_duration = 15  # Hold for 15 frames (0.5s)
        else:
            button_duration -= 1
    
    # Release any held buttons
    if current_button:
        pyboy.send_input(releases[current_button])
    
    out.release()
    pyboy.stop()
    
    print(f"Saved clip to: {output_path}")
    print(f"Frames: {frames_recorded}")
    return output_path

if __name__ == '__main__':
    rom_path = '/data/workspace/roms/Super Mario Land (World) (Rev 1).gb'
    output = '/data/workspace/mario_clip.mp4'
    record_mario_clip(rom_path, duration_seconds=15, output_path=output)
    print(f"Clip ready at: {output}")
