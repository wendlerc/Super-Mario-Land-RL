#!/usr/bin/env python3
"""Record actual Super Mario Land gameplay using PyBoy"""
import os
import numpy as np
from PIL import Image

# Set up environment for headless SDL2
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['ROM_PATH'] = '/data/workspace/roms'

from pyboy import PyBoy
from pyboy.utils import WindowEvent

def record_gameplay(rom_path, duration_seconds=10, output_path='actual_mario.gif'):
    """Record actual Game Boy gameplay"""
    
    print(f'Loading ROM: {rom_path}')
    pyboy = PyBoy(rom_path, window='null', sound=False)
    
    # Get screen dimensions
    screen = pyboy.screen
    print(f'Screen shape: {screen.ndarray.shape}')
    
    # Wait for boot screen
    print('Booting...')
    for _ in range(120):
        pyboy.tick()
    
    # Press Start to begin game
    print('Pressing Start...')
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
    
    # Wait for game to load (longer wait)
    print('Waiting for game to load...')
    for _ in range(180):  # ~3 seconds
        pyboy.tick()
    
    frames = []
    fps = 30
    total_frames = duration_seconds * fps
    
    # Button mappings
    buttons = {
        'right': (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
        'left': (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        'a': (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        'b': (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    }
    
    current_btn = None
    frame_count = 0
    action_timer = 0
    
    print(f'Recording {duration_seconds}s at {fps}fps...')
    
    while frame_count < total_frames and pyboy.tick():
        # Get screen as numpy array (RGBA)
        screen_array = screen.ndarray
        
        # Convert RGBA to RGB (drop alpha)
        rgb_array = screen_array[:, :, :3]
        
        # Convert to PIL Image
        img = Image.fromarray(rgb_array, 'RGB')
        
        # Scale up 3x for visibility (480x432)
        img_scaled = img.resize((480, 432), Image.NEAREST)
        frames.append(img_scaled)
        
        frame_count += 1
        
        # Simple AI: mostly run right, sometimes jump
        action_timer -= 1
        if action_timer <= 0:
            # Change action
            if current_btn:
                pyboy.send_input(buttons[current_btn][1])  # Release
            
            # Random action weighted toward right
            import random
            actions = ['right', 'right', 'right', 'a', 'right', 'a']
            current_btn = random.choice(actions)
            pyboy.send_input(buttons[current_btn][0])  # Press
            action_timer = random.randint(15, 30)  # Hold for 0.5-1s
        
        if frame_count % 30 == 0:
            print(f'Progress: {frame_count}/{total_frames}')
    
    # Release any held button
    if current_btn:
        pyboy.send_input(buttons[current_btn][1])
    
    pyboy.stop()
    
    # Save as GIF (every 2nd frame = 15fps for smaller file)
    print(f'Saving GIF with {len(frames[::2])} frames...')
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1::2],
        duration=67,  # 67ms = ~15fps
        loop=0
    )
    
    print(f'Saved: {output_path}')
    print(f'Size: {os.path.getsize(output_path)} bytes')
    return output_path

if __name__ == '__main__':
    rom = '/data/workspace/roms/Super Mario Land (World) (Rev 1).gb'
    output = '/data/workspace/actual_mario_gameplay.gif'
    record_gameplay(rom, duration_seconds=10, output_path=output)
