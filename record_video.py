#!/usr/bin/env python3
"""
Record gameplay video of the trained Mario PPO agent
"""
import os
import sys
import gymnasium as gym
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


class MarioVideoEnv(gym.Env):
    """Mario environment with video recording"""
    
    def __init__(self, rom_path='super_mario_land.gb'):
        super().__init__()
        
        self.rom_path = rom_path
        
        # Create PyBoy with headless window
        self.pyboy = PyBoy(rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.start_game()
        
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(72, 80, 1), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        
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
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def get_frame(self):
        """Get RGB frame for video recording"""
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
            'progress': progress,
            'score': score,
            'lives': lives,
        }
        
        return obs, reward, done, truncated, info
    
    def close(self):
        self.pyboy.stop()


def record_gameplay(model_path='mario_ppo_model.zip', 
                    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
                    output_path='mario_gameplay.mp4',
                    fps=30,
                    max_steps=2000):
    """Record gameplay video of trained agent"""
    from stable_baselines3 import PPO
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    print(f"Creating environment with ROM: {rom_path}...")
    env = MarioVideoEnv(rom_path=rom_path)
    
    # Get frame dimensions
    obs, _ = env.reset()
    frame = env.get_frame()
    height, width = frame.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Recording gameplay to {output_path}...")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Max steps: {max_steps}")
    
    frames = []
    done = False
    steps = 0
    total_reward = 0
    max_progress = 0
    
    while not done and steps < max_steps:
        # Get frame and add to video
        frame = env.get_frame()
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add info overlay
        info_text = f"Step: {steps} | Score: {env.prev_score} | Lives: {env.prev_lives}"
        cv2.putText(frame_bgr, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2)
        cv2.putText(frame_bgr, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1)
        
        video_writer.write(frame_bgr)
        
        # Take action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        max_progress = max(max_progress, info['progress'])
        steps += 1
        
        if steps % 100 == 0:
            print(f"  Step {steps}/{max_steps} | Progress: {info['progress']} | Reward: {total_reward:.1f}")
    
    video_writer.release()
    env.close()
    
    print(f"✅ Video saved to {output_path}")
    print(f"📊 Stats: {steps} steps, {max_progress} progress, {total_reward:.1f} total reward")
    
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mario_ppo_model.zip')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    parser.add_argument('--output', default='mario_gameplay.mp4')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    
    record_gameplay(
        model_path=args.model,
        rom_path=args.rom,
        output_path=args.output,
        fps=args.fps,
        max_steps=args.steps
    )
