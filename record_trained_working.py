#!/usr/bin/env python3
"""
WORKING Trained Agent Recorder
Uses stochastic sampling (deterministic=False) which actually works!
"""
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
from PIL import Image
import cv2

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


class MarioEnv(gym.Env):
    def __init__(self, rom_path):
        super().__init__()
        self.rom_path = rom_path
        self.pyboy = PyBoy(rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.start_game()
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(72, 80, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        
        w = self.pyboy.game_wrapper
        progress = w.level_progress
        score = w.score
        lives = w.lives_left
        time_left = w.time_left
        world, level = w.world
        
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
        
        done = w.game_over() or time_left == 0
        
        return obs, reward, done, False, {
            'world': world, 'level': level, 'progress': progress,
            'score': score, 'lives': lives, 'time_left': time_left,
        }
    
    def close(self):
        self.pyboy.stop()


def main():
    model_path = '/data/workspace/mario-land-rl/mario_ppo_model.zip'
    rom_path = '/data/workspace/roms/Super Mario Land (World) (Rev 1).gb'
    output_path = '/data/workspace/trained_working.mp4'
    
    print("🧠 Loading PPO model...")
    model = PPO.load(model_path)
    print("✅ Model loaded")
    print(f"   Num timesteps: {model.num_timesteps}")
    
    print("\n🎮 Creating environment...")
    env = MarioEnv(rom_path)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)
    print("✅ Environment ready")
    
    print("\n🎬 Starting recording...")
    obs = vec_env.reset()
    print(f"   Initial obs shape: {obs.shape}")
    
    frame = env.get_frame_rgb()
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    actions_taken = []
    
    for i in range(2000):
        # KEY: Use deterministic=False for varied actions!
        action, _ = model.predict(obs, deterministic=False)
        action_idx = int(action[0])
        actions_taken.append(action_idx)
        
        frame = env.get_frame_rgb()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        action_name = ACTION_NAMES[action_idx]
        text = f"TRAINED F{i} {action_name}"
        cv2.putText(frame_bgr, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        writer.write(frame_bgr)
        
        obs, reward, done, info = vec_env.step(action)
        
        if i % 200 == 0:
            print(f"   Frame {i}: action={action_name}, progress={info[0]['progress']}, lives={info[0]['lives']}")
        
        if done:
            print(f"   Episode ended at frame {i}")
            break
    
    writer.release()
    vec_env.close()
    
    print(f"\n✅ Video saved to {output_path}")
    
    from collections import Counter
    print("\n📊 Action distribution:")
    for act, cnt in Counter(actions_taken).most_common():
        pct = cnt / len(actions_taken) * 100
        print(f"   {ACTION_NAMES[act]:12s}: {cnt:4d} ({pct:5.1f}%)")
    
    unique = len(set(actions_taken))
    if unique > 3:
        print(f"\n🎉 SUCCESS: Model using {unique}/9 different actions!")
    else:
        print(f"\n⚠️  Only {unique} unique actions - model may still be stuck")


if __name__ == '__main__':
    main()
