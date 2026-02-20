#!/usr/bin/env python3
"""
Corrected Trained Agent Recorder
Uses SAME environment setup as training (with VecTransposeImage)
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import gymnasium as gym
from gymnasium import spaces

ACTION_NAMES = ["NOP", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "JUMP_RIGHT", "JUMP_LEFT"]
ACTIONS = [
    [WindowEvent.PASS], [WindowEvent.PRESS_BUTTON_A], [WindowEvent.PRESS_BUTTON_B],
    [WindowEvent.PRESS_ARROW_UP], [WindowEvent.PRESS_ARROW_DOWN],
    [WindowEvent.PRESS_ARROW_LEFT], [WindowEvent.PRESS_ARROW_RIGHT],
    [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
]

BUTTONS = [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, 
           WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_LEFT,
           WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B]

RELEASE_BUTTONS = [WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_DOWN,
                   WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_LEFT,
                   WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B]

RELEASE_LOOKUP = {btn: rel for btn, rel in zip(BUTTONS, RELEASE_BUTTONS)}


# Same as train_sb3.py
class SimpleMarioEnv(gym.Env):
    def __init__(self, rom_path='super_mario_land.gb'):
        super().__init__()
        self.rom_path = rom_path
        self.pyboy = None
        self.screen = None
        self._currently_held = None
        
    def start(self):
        self.pyboy = PyBoy(self.rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.start_game()
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.pyboy:
            self.start()
        self.pyboy.game_wrapper.reset_game()
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        return self._get_obs(), {}
    
    def _get_obs(self):
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def get_frame_rgb(self):
        return self.screen.ndarray[:, :, :3]
    
    def step(self, action):
        # action is int
        action_idx = int(action)
        action_list = ACTIONS[action_idx]
        
        for btn in self._currently_held:
            if self._currently_held[btn] and btn not in action_list:
                self.pyboy.send_input(RELEASE_LOOKUP[btn])
                self._currently_held[btn] = False
        for btn in action_list:
            if btn != WindowEvent.PASS:
                self.pyboy.send_input(btn)
                self._currently_held[btn] = True
        
        self.pyboy.tick(8)
        obs = self._get_obs()
        
        wrapper = self.pyboy.game_wrapper
        progress = wrapper.level_progress
        score = wrapper.score
        lives = wrapper.lives_left
        time_left = wrapper.time_left
        world, level = wrapper.world
        
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
        
        done = wrapper.game_over() or time_left == 0
        truncated = False
        
        info = {
            'world': world, 'level': level, 'progress': progress,
            'score': score, 'lives': lives, 'time_left': time_left,
        }
        
        return obs, reward, done, truncated, info
    
    def close(self):
        if self.pyboy:
            self.pyboy.stop()


def record_with_wrapped_env(
    model_path='/data/workspace/mario-land-rl/mario_ppo_model.zip',
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    output_path='/data/workspace/trained_corrected.mp4',
    max_frames=3000,
    fps=30
):
    """Record with SAME setup as training"""
    
    print("🧠 Loading trained PPO model...")
    model = PPO.load(model_path)
    print("   ✅ Model loaded")
    
    print("\n🎮 Creating environment (SAME as training)...")
    # EXACT same setup as train_sb3.py line 145-146
    env = SimpleMarioEnv(rom_path=rom_path)
    env.start()
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)  # THIS IS THE KEY!
    
    print("   ✅ Environment ready with VecTransposeImage")
    
    obs = env.reset()
    
    # Get frame dimensions from underlying env
    raw_env = env.envs[0]
    frame = raw_env.get_frame_rgb()
    h, w = frame.shape[:2]
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    print(f"\n⏺️  Recording {max_frames} frames...")
    
    actions_count = {i: 0 for i in range(9)}
    
    for i in range(max_frames):
        # obs is already wrapped (channel-first, batched)
        action, _ = model.predict(obs, deterministic=False)
        action_idx = int(action[0])  # Unbatch
        actions_count[action_idx] += 1
        
        action_name = ACTION_NAMES[action_idx]
        
        # Visualize
        frame = raw_env.get_frame_rgb()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        text = f"TRAINED F{i} {action_name}"
        cv2.putText(frame_bgr, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        writer.write(frame_bgr)
        
        obs, reward, done, info = env.step(action)
        
        if i % 100 == 0:
            print(f"   Frame {i}: action={action_name}, progress={info[0]['progress']}")
        
        if done:
            print(f"   Episode ended at frame {i}")
            break
    
    writer.release()
    env.close()
    
    print(f"\n✅ Video saved to {output_path}")
    print("\n📊 Action distribution:")
    for idx, count in actions_count.items():
        if count > 0:
            print(f"   {ACTION_NAMES[idx]}: {count}")


if __name__ == '__main__':
    record_with_wrapped_env()
