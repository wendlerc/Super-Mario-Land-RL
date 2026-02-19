#!/usr/bin/env python3
"""
Trained Agent Recorder - Uses same env as training
Records gameplay with the trained PPO model
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import time

os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from stable_baselines3 import PPO
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import gymnasium as gym
from gymnasium import spaces

# Actions from train_sb3.py
ACTIONS = [
    [WindowEvent.PASS],
    [WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_BUTTON_B],
    [WindowEvent.PRESS_ARROW_UP],
    [WindowEvent.PRESS_ARROW_DOWN],
    [WindowEvent.PRESS_ARROW_LEFT],
    [WindowEvent.PRESS_ARROW_RIGHT],
    [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
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

ACTION_NAMES = ["NOP", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "JUMP_RIGHT", "JUMP_LEFT"]


class SimpleMarioEnv(gym.Env):
    """Exact same env as train_sb3.py"""
    
    def __init__(self, rom_path='super_mario_land.gb', render_mode=None):
        super().__init__()
        
        self.rom_path = rom_path
        self.render_mode = render_mode
        
        win = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=win, sound=False)
        self.pyboy.game_wrapper.start_game()
        
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        
        # Same observation space as training
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(72, 80, 1), dtype=np.uint8
        )
        
        self.action_space = spaces.Discrete(len(ACTIONS))
        
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pyboy.game_wrapper.reset_game()
        
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def _get_obs(self):
        # Exact same preprocessing as training
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def get_frame_rgb(self):
        """Get full RGB frame for video"""
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
        
        done = bool(self.pyboy.game_wrapper.game_over())
        if time_left == 0:
            done = True
            reward -= 10
        
        truncated = False
        info = {
            'world': world,
            'level': level,
            'progress': progress,
            'score': score,
            'lives': lives,
            'time_left': time_left,
        }
        
        return obs, reward, done, truncated, info
    
    def close(self):
        self.pyboy.stop()


def record_trained_agent(
    model_path='/data/workspace/mario-land-rl/mario_ppo_model.zip',
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    output_dir='/data/workspace/mario_trained_clips',
    num_episodes=10,
    fps=30
):
    """Record episodes with trained PPO agent"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧠 Loading trained PPO model...")
    model = PPO.load(model_path)
    print(f"   ✅ Model loaded")
    
    print(f"\n🎮 Recording {num_episodes} episodes with trained agent...")
    
    for ep in range(num_episodes):
        print(f"\n[{ep+1}/{num_episodes}] Starting episode...")
        
        env = SimpleMarioEnv(rom_path=rom_path)
        obs, _ = env.reset()
        
        frame = env.get_frame_rgb()
        h, w = frame.shape[:2]
        
        video_path = os.path.join(output_dir, f"trained_ep{ep:03d}.mp4")
        json_path = os.path.join(output_dir, f"trained_ep{ep:03d}.json")
        
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        annotation = {
            'metadata': {
                'episode': ep,
                'agent': 'trained_ppo',
                'fps': fps,
                'created_at': datetime.now().isoformat(),
            },
            'frames': []
        }
        
        done = False
        frame_count = 0
        max_frames = 30000
        
        while not done and frame_count < max_frames:
            action, _ = model.predict(obs, deterministic=False)
            action_idx = int(action)
            action_name = ACTION_NAMES[action_idx]
            
            frame = env.get_frame_rgb()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            text = f"TRAINED EP{ep} F{frame_count} {action_name}"
            cv2.putText(frame_bgr, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            writer.write(frame_bgr)
            
            obs, reward, done, truncated, info = env.step(action_idx)
            
            annotation['frames'].append({
                'frame': frame_count,
                'action': {'index': action_idx, 'name': action_name},
                'reward': float(reward),
                'state': {
                    'world': int(info['world']),
                    'level': int(info['level']),
                    'progress': int(info['progress']),
                    'score': int(info['score']),
                    'lives': int(info['lives']),
                    'time_left': int(info['time_left'])
                },
                'done': bool(done)
            })
            
            frame_count += 1
            
            if frame_count % (fps * 60) == 0:
                minutes = frame_count / fps / 60
                print(f"      {minutes:.0f}m | Progress: {info['progress']} | Lives: {info['lives']}")
        
        annotation['metadata']['total_frames'] = frame_count
        annotation['metadata']['duration_sec'] = frame_count / fps
        annotation['metadata']['final_progress'] = info['progress']
        annotation['metadata']['final_score'] = info['score']
        
        writer.release()
        env.close()
        
        with open(json_path, 'w') as f:
            json.dump(annotation, f)
        
        print(f"   ✅ Episode {ep}: {frame_count/fps/60:.1f}min, progress: {info['progress']}")
    
    print(f"\n✅ Recorded {num_episodes} trained agent episodes to {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output', default='/data/workspace/mario_trained_clips')
    parser.add_argument('--model', default='/data/workspace/mario-land-rl/mario_ppo_model.zip')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    args = parser.parse_args()
    
    record_trained_agent(
        model_path=args.model,
        rom_path=args.rom,
        output_dir=args.output,
        num_episodes=args.episodes
    )
