#!/usr/bin/env python3
"""
Simple PPO training for Super Mario Land using stable-baselines3
This is a lighter alternative to PufferLib
"""
import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

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


class SimpleMarioEnv(gym.Env):
    """Simplified Super Mario Land environment for SB3"""
    
    def __init__(self, rom_path='super_mario_land.gb', render_mode=None):
        super().__init__()
        
        self.rom_path = rom_path
        self.render_mode = render_mode
        
        # Create PyBoy instance
        win = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=win, sound=False)
        self.pyboy.game_wrapper.start_game()
        
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        
        # Observation: grayscale screen (downscaled)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(72, 80, 1), dtype=np.uint8
        )
        
        # Action space: discrete actions
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
        # Get screen, crop top bar, convert to grayscale
        rgb = self.screen.ndarray[18:, :, :3]  # Remove top UI bar
        # Convert to grayscale using PIL
        img = Image.fromarray(rgb, 'RGB').convert('L')
        # Downscale for faster training
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def do_action(self, action_idx):
        action = ACTIONS[action_idx]
        
        # Release buttons not in new action
        for btn in self._currently_held:
            if self._currently_held[btn] and btn not in action:
                self.pyboy.send_input(RELEASE_BUTTON_LOOKUP[btn])
                self._currently_held[btn] = False
        
        # Press new buttons
        for btn in action:
            if btn not in [WindowEvent.PASS]:
                self.pyboy.send_input(btn)
                self._currently_held[btn] = True
    
    def step(self, action):
        self.do_action(action)
        self.pyboy.tick(8)  # 8 frames per action
        
        obs = self._get_obs()
        
        # Calculate reward
        wrapper = self.pyboy.game_wrapper
        progress = wrapper.level_progress
        score = wrapper.score
        lives = wrapper.lives_left
        time_left = wrapper.time_left
        
        reward = 0
        
        # Progress reward (main objective)
        if progress > self.prev_progress:
            reward += (progress - self.prev_progress) * 10
            self.prev_progress = progress
        
        # Score reward
        if score > self.prev_score:
            reward += (score - self.prev_score) * 0.1
            self.prev_score = score
        
        # Death penalty
        if lives < self.prev_lives:
            reward -= 15
            self.prev_lives = lives
        
        # Time penalty (encourage faster completion)
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
    
    def render(self):
        if self.render_mode == "human":
            return self.screen.ndarray
        return None
    
    def close(self):
        self.pyboy.stop()


def train_mario(rom_path='super_mario_land.gb', total_timesteps=100000):
    """Train a PPO agent on Super Mario Land"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print("Creating environment...")
    env = SimpleMarioEnv(rom_path=rom_path)
    env = DummyVecEnv([lambda: env])
    
    print("Creating PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
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
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    model_path = "mario_ppo_model.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()
    return model


def evaluate_model(model_path='mario_ppo_model.zip', rom_path='super_mario_land.gb', episodes=5):
    """Evaluate a trained model"""
    from stable_baselines3 import PPO
    
    env = SimpleMarioEnv(rom_path=rom_path)
    model = PPO.load(model_path)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}, Progress={info['progress']}")
    
    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    parser.add_argument('--timesteps', type=int, default=100000)
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mario(rom_path=args.rom, total_timesteps=args.timesteps)
    else:
        evaluate_model(rom_path=args.rom)
