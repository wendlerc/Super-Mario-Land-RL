#!/usr/bin/env python3
"""
Behavioral Cloning Trainer
Learn from the collected random agent data to create a better policy
"""
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import json
import numpy as np
from PIL import Image
import cv2
import glob
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from pyboy import PyBoy
from pyboy.utils import WindowEvent

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


class MarioCNN(BaseFeaturesExtractor):
    """CNN for Mario environment"""
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 72, 80)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))


def load_dataset(data_dir='/data/workspace/mario_long_episodes'):
    """Load the collected dataset"""
    print("Loading dataset...")
    json_files = glob.glob(f'{data_dir}/*.json')
    print(f"Found {len(json_files)} JSON files")
    
    observations = []
    actions = []
    
    for i, json_file in enumerate(json_files[:100]):  # Load first 100 episodes for now
        if i % 10 == 0:
            print(f"  Loading {i}/{len(json_files)}: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for frame in data['frames']:
            # We don't have the actual observation images saved, only metadata
            # For now, we'll train on synthetic data or need to regenerate observations
            # This is a placeholder - real implementation needs actual screen data
            pass
    
    print(f"Dataset loading complete")
    return observations, actions


def train_behavioral_cloning():
    """Train a policy using behavioral cloning on collected data"""
    rom_path = '/data/workspace/roms/Super Mario Land (World) (Rev 1).gb'
    
    print("=" * 60)
    print("BEHAVIORAL CLONING TRAINER")
    print("=" * 60)
    
    # Create environment
    env = MarioEnv(rom_path)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)
    
    # Create policy with custom CNN
    policy_kwargs = dict(
        features_extractor_class=MarioCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    
    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    print("\nStarting behavioral cloning training...")
    print("(Note: This is a skeleton - needs actual data loading implementation)")
    
    # For now, just do regular RL training
    print("\nFalling back to regular RL training for now...")
    model.learn(total_timesteps=100000)
    
    # Save
    model.save('/data/workspace/mario_bc_model.zip')
    print("\n✅ Model saved to /data/workspace/mario_bc_model.zip")
    
    vec_env.close()


if __name__ == '__main__':
    train_behavioral_cloning()
