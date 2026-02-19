#!/usr/bin/env python3
"""
Trained Agent on Multiple Levels
Records gameplay starting from World 0-3, Level 0-2
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
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import gymnasium as gym
from gymnasium import spaces

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
ACTION_NAMES = ["NOP", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "JUMP_RIGHT", "JUMP_LEFT"]


class MarioEnv(gym.Env):
    def __init__(self, rom_path='super_mario_land.gb'):
        super().__init__()
        self.rom_path = rom_path
        self.pyboy = None
        self.screen = None
        self._currently_held = None
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(72, 80, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))
        
    def start_at_level(self, world, level):
        if self.pyboy:
            self.pyboy.stop()
        
        self.pyboy = PyBoy(self.rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.set_world_level(world, level)
        self.pyboy.game_wrapper.start_game()
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        return self._get_obs()
        
    def _get_obs(self):
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def get_frame_rgb(self):
        return self.screen.ndarray[:, :, :3]
    
    def step(self, action_idx):
        action = ACTIONS[action_idx]
        for btn in self._currently_held:
            if self._currently_held[btn] and btn not in action:
                self.pyboy.send_input(RELEASE_LOOKUP[btn])
                self._currently_held[btn] = False
        for btn in action:
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
        
        info = {'world': world, 'level': level, 'progress': progress,
                'score': score, 'lives': lives, 'time_left': time_left}
        
        return obs, reward, done, info
    
    def close(self):
        if self.pyboy:
            self.pyboy.stop()


def record_from_level(model, env, output_dir, world, level, fps=30):
    """Record one episode from specific world/level"""
    
    obs = env.start_at_level(world, level)
    frame = env.get_frame_rgb()
    h, w = frame.shape[:2]
    
    ep_name = f"trained_w{world}l{level}"
    video_path = os.path.join(output_dir, f"{ep_name}.mp4")
    json_path = os.path.join(output_dir, f"{ep_name}.json")
    
    if os.path.exists(video_path):
        print(f"   ⏭️  {ep_name} already exists, skipping")
        return
    
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    annotation = {
        'metadata': {
            'start_world': world,
            'start_level': level,
            'agent': 'trained_ppo',
            'fps': fps,
            'created_at': datetime.now().isoformat(),
        },
        'frames': []
    }
    
    done = False
    frame_count = 0
    max_frames = 30000
    
    print(f"   🎮 Recording W{world}-L{level}...")
    
    while not done and frame_count < max_frames:
        action, _ = model.predict(obs, deterministic=False)
        action_idx = int(action)
        action_name = ACTION_NAMES[action_idx]
        
        frame = env.get_frame_rgb()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        text = f"TRAINED W{world}-{level} F{frame_count} {action_name}"
        cv2.putText(frame_bgr, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        writer.write(frame_bgr)
        
        obs, reward, done, info = env.step(action_idx)
        
        annotation['frames'].append({
            'frame': frame_count,
            'action': {'index': action_idx, 'name': action_name},
            'reward': float(reward),
            'state': {
                'world': int(info['world']), 'level': int(info['level']),
                'progress': int(info['progress']), 'score': int(info['score']),
                'lives': int(info['lives']), 'time_left': int(info['time_left'])
            },
            'done': bool(done)
        })
        
        frame_count += 1
        
        if frame_count % (fps * 30) == 0:  # Every 30 sec
            print(f"      {frame_count/fps:.0f}s | Progress: {info['progress']} | Lives: {info['lives']}")
    
    annotation['metadata']['total_frames'] = frame_count
    annotation['metadata']['duration_sec'] = frame_count / fps
    annotation['metadata']['final_progress'] = info['progress']
    
    writer.release()
    
    with open(json_path, 'w') as f:
        json.dump(annotation, f)
    
    print(f"   ✅ W{world}-L{level}: {frame_count/fps/60:.1f}min, progress: {info['progress']}")


def record_all_levels(
    model_path='/data/workspace/mario-land-rl/mario_ppo_model.zip',
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    output_dir='/data/workspace/mario_trained_all_levels'
):
    """Record trained agent on all world/level combinations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧠 Loading trained PPO model...")
    model = PPO.load(model_path)
    print("   ✅ Model loaded")
    
    env = MarioEnv(rom_path)
    
    print("\n🎮 Recording trained agent on all levels...")
    
    # Record from each world/level
    for world in range(4):  # Worlds 0-3
        for level in range(3):  # Levels 0-2
            print(f"\n📍 World {world}, Level {level}:")
            try:
                record_from_level(model, env, output_dir, world, level)
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    env.close()
    print("\n✅ All levels recorded!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/data/workspace/mario_trained_all_levels')
    parser.add_argument('--model', default='/data/workspace/mario-land-rl/mario_ppo_model.zip')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    args = parser.parse_args()
    
    record_all_levels(
        model_path=args.model,
        rom_path=args.rom,
        output_dir=args.output
    )
