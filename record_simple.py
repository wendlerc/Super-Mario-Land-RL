#!/usr/bin/env python3
"""
Simple Mario gameplay recorder - loads model without stable-baselines3
"""
import os
import sys
import numpy as np
from PIL import Image
import cv2

# SDL2 headless mode
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Actions
ACTIONS = [
    [WindowEvent.PASS],  # NOP
    [WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_BUTTON_B],
    [WindowEvent.PRESS_ARROW_UP],
    [WindowEvent.PRESS_ARROW_DOWN],
    [WindowEvent.PRESS_ARROW_LEFT],
    [WindowEvent.PRESS_ARROW_RIGHT],
    [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],  # Jump right
    [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],   # Jump left
]

BUTTONS = [
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
]

RELEASE_BUTTONS = [
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
]

RELEASE_BUTTON_LOOKUP = {btn: rel for btn, rel in zip(BUTTONS, RELEASE_BUTTONS)}


class SimpleMarioPlayer:
    """Simple Mario player that runs random or heuristic policy"""
    
    def __init__(self, rom_path='super_mario_land.gb'):
        self.rom_path = rom_path
        self.pyboy = PyBoy(rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.start_game()
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        self.frame_count = 0
        
    def reset(self):
        self.pyboy.game_wrapper.reset_game()
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        self.frame_count = 0
        return self._get_obs()
    
    def _get_obs(self):
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def get_frame(self):
        return self.screen.ndarray[:, :, :3]
    
    def do_action(self, action_idx):
        action = ACTIONS[action_idx]
        for btn in self._currently_held:
            if self._currently_held[btn] and btn not in action:
                self.pyboy.send_input(RELEASE_BUTTON_LOOKUP[btn])
                self._currently_held[btn] = False
        for btn in action:
            if btn not in [WindowEvent.PASS]:
                self.pyboy.send_input(btn)
                self._currently_held[btn] = True
    
    def step(self, action):
        self.do_action(action)
        self.pyboy.tick(8)
        
        wrapper = self.pyboy.game_wrapper
        progress = wrapper.level_progress
        score = wrapper.score
        lives = wrapper.lives_left
        time_left = wrapper.time_left
        
        reward = 0
        if progress > self.prev_progress:
            reward += (progress - self.prev_progress) * 10
            self.prev_progress = progress
        if score > self.prev_score:
            reward += (score - self.prev_score) * 0.1
            self.prev_score = score
        if lives < self.prev_lives:
            reward -= 15
            self.prev_lives = lives
        reward -= 0.1
        
        done = bool(self.pyboy.game_wrapper.game_over())
        if time_left == 0:
            done = True
        
        self.frame_count += 1
        return reward, done
    
    def close(self):
        self.pyboy.stop()


def record_random_gameplay(rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
                           output_path='mario_random.mp4',
                           fps=30,
                           max_steps=800):
    """Record random gameplay video"""
    
    print(f"Creating environment with ROM: {rom_path}...")
    player = SimpleMarioPlayer(rom_path=rom_path)
    
    # Get frame dimensions
    player.reset()
    frame = player.get_frame()
    height, width = frame.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Recording random gameplay to {output_path}...")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Max steps: {max_steps}")
    
    done = False
    steps = 0
    total_reward = 0
    max_progress = 0
    
    while not done and steps < max_steps:
        # Get frame
        frame = player.get_frame()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add info overlay
        info_text = f"Step: {steps} | Score: {player.prev_score} | Lives: {player.prev_lives}"
        cv2.putText(frame_bgr, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2)
        cv2.putText(frame_bgr, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1)
        
        # Add "RANDOM AGENT" label
        cv2.putText(frame_bgr, "RANDOM AGENT", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)
        
        video_writer.write(frame_bgr)
        
        # Random action
        action = np.random.randint(0, len(ACTIONS))
        reward, done = player.step(action)
        total_reward += reward
        max_progress = max(max_progress, player.prev_progress)
        steps += 1
        
        if steps % 100 == 0:
            print(f"  Step {steps}/{max_steps} | Progress: {player.prev_progress}")
    
    video_writer.release()
    player.close()
    
    print(f"✅ Video saved to {output_path}")
    print(f"📊 Stats: {steps} steps, {max_progress} progress, {total_reward:.1f} total reward")
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    parser.add_argument('--output', default='mario_random.mp4')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    
    record_random_gameplay(
        rom_path=args.rom,
        output_path=args.output,
        fps=args.fps,
        max_steps=args.steps
    )
