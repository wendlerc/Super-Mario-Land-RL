#!/usr/bin/env python3
"""Record actual Super Mario Land gameplay scenarios using PyBoy"""
import os
import numpy as np
from PIL import Image

# Set up environment for headless SDL2
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['ROM_PATH'] = '/data/workspace/roms'

from pyboy import PyBoy
from pyboy.utils import WindowEvent

def capture_frame(screen):
    """Capture and scale a frame"""
    screen_array = screen.ndarray
    rgb_array = screen_array[:, :, :3]
    img = Image.fromarray(rgb_array, 'RGB')
    return img.resize((480, 432), Image.NEAREST)

def wait_frames(pyboy, count):
    """Wait for N frames"""
    for _ in range(count):
        pyboy.tick()

def start_game(pyboy):
    """Get past title screen into actual gameplay"""
    print('Booting...')
    wait_frames(pyboy, 120)
    
    print('Pressing Start...')
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
    
    print('Waiting for level to load...')
    wait_frames(pyboy, 180)

def record_jump_scenario():
    """Record Mario jumping over an obstacle"""
    print('\n=== Recording Jump Scenario ===')
    pyboy = PyBoy('/data/workspace/roms/Super Mario Land (World) (Rev 1).gb', window='null', sound=False)
    screen = pyboy.screen
    
    start_game(pyboy)
    
    frames = []
    
    # Run right for a bit then jump
    print('Running and jumping...')
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    
    for i in range(120):  # 4 seconds
        pyboy.tick()
        
        # Jump at frame 40
        if i == 40:
            pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        if i == 50:
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        
        frames.append(capture_frame(screen))
        
        if i % 30 == 0:
            print(f'  Frame {i}/120')
    
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    pyboy.stop()
    
    # Save GIF
    print('Saving jump.gif...')
    frames[0].save(
        '/data/workspace/real_mario_jump.gif',
        save_all=True,
        append_images=frames[1::2],
        duration=67,
        loop=0
    )
    print('Saved!')

def record_stomp_scenario():
    """Record Mario stomping an enemy"""
    print('\n=== Recording Stomp Scenario ===')
    pyboy = PyBoy('/data/workspace/roms/Super Mario Land (World) (Rev 1).gb', window='null', sound=False)
    screen = pyboy.screen
    
    start_game(pyboy)
    
    frames = []
    
    # Run right, then jump on enemy
    print('Approaching enemy...')
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    
    for i in range(150):  # 5 seconds
        pyboy.tick()
        
        # Jump to stomp enemy around frame 70
        if i == 70:
            pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        if i == 85:
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        
        frames.append(capture_frame(screen))
        
        if i % 30 == 0:
            print(f'  Frame {i}/150')
    
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    pyboy.stop()
    
    print('Saving stomp.gif...')
    frames[0].save(
        '/data/workspace/real_mario_stomp.gif',
        save_all=True,
        append_images=frames[1::2],
        duration=67,
        loop=0
    )
    print('Saved!')

def record_brick_scenario():
    """Record Mario hitting a block"""
    print('\n=== Recording Brick Hit Scenario ===')
    pyboy = PyBoy('/data/workspace/roms/Super Mario Land (World) (Rev 1).gb', window='null', sound=False)
    screen = pyboy.screen
    
    start_game(pyboy)
    
    frames = []
    
    # Run right, jump to hit block
    print('Running toward block...')
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    
    for i in range(120):  # 4 seconds
        pyboy.tick()
        
        # Jump to hit block at frame 50
        if i == 50:
            pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        if i == 65:
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        
        frames.append(capture_frame(screen))
        
        if i % 30 == 0:
            print(f'  Frame {i}/120')
    
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    pyboy.stop()
    
    print('Saving brick.gif...')
    frames[0].save(
        '/data/workspace/real_mario_brick.gif',
        save_all=True,
        append_images=frames[1::2],
        duration=67,
        loop=0
    )
    print('Saved!')

def record_long_gameplay():
    """Record longer gameplay"""
    print('\n=== Recording Long Gameplay ===')
    pyboy = PyBoy('/data/workspace/roms/Super Mario Land (World) (Rev 1).gb', window='null', sound=False)
    screen = pyboy.screen
    
    start_game(pyboy)
    
    frames = []
    action = 'right'
    action_timer = 0
    
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    
    for i in range(300):  # 10 seconds
        pyboy.tick()
        
        # Change actions periodically
        action_timer -= 1
        if action_timer <= 0:
            # Release current
            if action == 'right':
                pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            elif action == 'left':
                pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
            elif action == 'a':
                pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            
            # Pick new action
            import random
            choices = ['right', 'right', 'right', 'a', 'b']
            action = random.choice(choices)
            action_timer = random.randint(20, 40)
            
            if action == 'right':
                pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            elif action == 'left':
                pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            elif action == 'a':
                pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        
        frames.append(capture_frame(screen))
        
        if i % 60 == 0:
            print(f'  Frame {i}/300')
    
    # Release all
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    pyboy.stop()
    
    print('Saving gameplay.gif...')
    frames[0].save(
        '/data/workspace/real_mario_gameplay.gif',
        save_all=True,
        append_images=frames[1::2],
        duration=67,
        loop=0
    )
    print('Saved!')

if __name__ == '__main__':
    print('Recording real Super Mario Land gameplay with PyBoy...')
    print('This will take a few minutes...')
    
    record_jump_scenario()
    record_stomp_scenario()
    record_brick_scenario()
    record_long_gameplay()
    
    print('\n=== All clips recorded! ===')
    import os
    for f in ['real_mario_jump.gif', 'real_mario_stomp.gif', 'real_mario_brick.gif', 'real_mario_gameplay.gif']:
        path = f'/data/workspace/{f}'
        if os.path.exists(path):
            print(f'{f}: {os.path.getsize(path)} bytes')
