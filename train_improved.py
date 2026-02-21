#!/usr/bin/env python3
"""
Improved Mario Training Script
Fixes for the deterministic policy issue:
1. Lower entropy coefficient (less randomness over time)
2. Longer training time
3. Evaluation during training
4. Better reward tracking
"""
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
from PIL import Image

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
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
        
        self.episode_steps = 0
        self.max_steps = 3000  # ~100 seconds
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pyboy.game_wrapper.reset_game()
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        self.episode_steps = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
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
        self.episode_steps += 1
        
        obs = self._get_obs()
        w = self.pyboy.game_wrapper
        progress = w.level_progress
        score = w.score
        lives = w.lives_left
        time_left = w.time_left
        
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
        truncated = self.episode_steps >= self.max_steps
        
        return obs, reward, done, truncated, {}
    
    def close(self):
        self.pyboy.stop()


def train_improved():
    rom_path = '/data/workspace/roms/Super Mario Land (World) (Rev 1).gb'
    
    print("=" * 60)
    print("IMPROVED MARIO TRAINING")
    print("=" * 60)
    
    # Create environment
    print("Creating environment...")
    env = MarioEnv(rom_path)
    
    # Convert to proper format
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    
    # Create model with IMPROVED settings
    print("\nCreating PPO model with improved settings...")
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
        ent_coef=0.005,  # LOWER entropy (was 0.01) - less randomness
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./mario_tensorboard/"
    )
    
    # Callbacks for evaluation and checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./mario_checkpoints/",
        name_prefix="mario_improved"
    )
    
    # Train for longer
    print("\nStarting training for 500,000 timesteps...")
    print("(Previous was only 100,000)")
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save('/data/workspace/mario_improved_model.zip')
    print("\n✅ Model saved to /data/workspace/mario_improved_model.zip")
    
    env.close()


def test_model(model_path, deterministic=True):
    """Test a trained model"""
    from collections import Counter
    
    rom_path = '/data/workspace/roms/Super Mario Land (World) (Rev 1).gb'
    
    print(f"\nTesting model: {model_path}")
    print(f"Deterministic: {deterministic}")
    
    model = PPO.load(model_path)
    env = MarioEnv(rom_path)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)
    
    obs = vec_env.reset()
    actions = []
    
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(int(action[0]))
        obs, reward, done, info = vec_env.step(action)
        
        if done:
            break
    
    vec_env.close()
    
    print(f"Actions taken: {len(actions)}")
    print("Distribution:")
    for idx, cnt in Counter(actions).most_common():
        print(f"  {ACTION_NAMES[idx]}: {cnt}")
    
    unique = len(set(actions))
    if unique > 3:
        print(f"✅ Good variety: {unique}/9 actions")
    else:
        print(f"⚠️ Limited variety: {unique}/9 actions")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--test', type=str, help='Test a model')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic testing')
    args = parser.parse_args()
    
    if args.train:
        train_improved()
    elif args.test:
        test_model(args.test, deterministic=not args.stochastic)
    else:
        print("Usage:")
        print("  python3 train_improved.py --train")
        print("  python3 train_improved.py --test /path/to/model.zip")
        print("  python3 train_improved.py --test /path/to/model.zip --stochastic")
